"""ADR-0011 v5 subword evaluation and training-fitted byte baselines."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aptivra.resources import aptivra_resource

V5_EVALUATION_SCHEMA_VERSION = "aptivra.model_card_eval.v5"
V5_METRIC_SCHEMA_VERSION = "aptivra.bpb_metric.v5"
DEFAULT_STANDARD_EVALUATION_CONFIG = aptivra_resource(
    "configs", "smollm2_evaluation.yaml"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_LN_2 = math.log(2.0)

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluationFeatureSettings(_StrictModel):
    enabled: bool


class BpbProtocolSettings(_StrictModel):
    protocol_id: Literal["aptivra.subword_bpb.v1"]
    boundary_policy: Literal["encode_prompt_and_scored_text_separately"]
    special_token_policy: Literal["context_only_not_scored"]
    byte_encoding: Literal["utf-8"]


class ByteUnigramSettings(_StrictModel):
    smoothing: Literal["add_k"]
    add_k: PositiveFinite


class ReleaseGateSettings(_StrictModel):
    margin_bits_per_byte: FiniteNonNegative
    required_cells: tuple[Literal["heldout_answers", "heldout_text"], ...]

    @field_validator("required_cells", mode="before")
    @classmethod
    def validate_required_cells(
        cls, value: object
    ) -> tuple[Literal["heldout_answers", "heldout_text"], ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("required_cells must be a list")
        cells = tuple(cast(Sequence[str], value))
        if set(cells) != {"heldout_answers", "heldout_text"} or len(cells) != 2:
            raise ValueError("required_cells must contain heldout_answers and heldout_text once")
        return cast(tuple[Literal["heldout_answers", "heldout_text"], ...], cells)


class ContaminationSettings(_StrictModel):
    required_training_sources: tuple[str, ...]

    @field_validator("required_training_sources", mode="before")
    @classmethod
    def validate_sources(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("required_training_sources must be a list")
        sources = tuple(cast(Sequence[str], value))
        if not sources or any(not isinstance(source, str) or not source for source in sources):
            raise ValueError("required_training_sources must contain non-empty names")
        if len(sources) != len(set(sources)):
            raise ValueError("required_training_sources must be unique")
        return sources


class StandardEvaluationSettings(_StrictModel):
    schema_version: Literal["aptivra.smollm2_evaluation_config.v1"]
    feature: EvaluationFeatureSettings
    bpb: BpbProtocolSettings
    byte_unigram: ByteUnigramSettings
    release_gate: ReleaseGateSettings
    contamination: ContaminationSettings


def load_standard_evaluation_settings(
    path: Path | str | Traversable = DEFAULT_STANDARD_EVALUATION_CONFIG,
) -> StandardEvaluationSettings:
    """Load the strict, feature-disabled v5 evaluation configuration."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"smollm2_evaluation"}:
        raise ValueError("evaluation config root must contain only 'smollm2_evaluation'")
    return StandardEvaluationSettings.model_validate(raw["smollm2_evaluation"])


@dataclass(frozen=True)
class BpbMeasurement:
    """Aggregate cross-entropy divided by exact scored UTF-8 bytes."""

    total_cross_entropy_bits: float
    scored_utf8_bytes: int
    scored_tokens: int
    bits_per_byte: float


def aggregate_bits_per_byte(
    token_nlls_nats: Sequence[Sequence[float]],
    scored_texts: Sequence[str],
) -> BpbMeasurement:
    """Aggregate token NLLs before dividing by the exact payload byte count."""
    if not token_nlls_nats or len(token_nlls_nats) != len(scored_texts):
        raise ValueError("token losses and scored texts must be non-empty and aligned")
    total_tokens = 0
    losses: list[float] = []
    total_bytes = 0
    for item_losses, text in zip(token_nlls_nats, scored_texts, strict=True):
        if not item_losses:
            raise ValueError("each scored text must have at least one scored token")
        encoded = text.encode("utf-8")
        if not encoded:
            raise ValueError("scored text must not be empty")
        for loss in item_losses:
            if not math.isfinite(loss) or loss < 0.0:
                raise ValueError("token negative log-likelihoods must be finite and non-negative")
            losses.append(loss)
        total_tokens += len(item_losses)
        total_bytes += len(encoded)
    total_bits = math.fsum(losses) / _LN_2
    return BpbMeasurement(
        total_cross_entropy_bits=total_bits,
        scored_utf8_bytes=total_bytes,
        scored_tokens=total_tokens,
        bits_per_byte=total_bits / total_bytes,
    )


class SubwordTokenizer(Protocol):
    bos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text without injecting protocol tokens."""

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        """Decode token IDs exactly."""


def _exact_encode(tokenizer: SubwordTokenizer, text: str, description: str) -> list[int]:
    token_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
    if not token_ids:
        raise ValueError(f"{description} tokenized to zero tokens")
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError(f"{description} failed exact encode/decode validation")
    return token_ids


def _target_nlls(
    model: torch.nn.Module,
    *,
    prefix_ids: list[int],
    target_ids: list[int],
    device: torch.device,
) -> list[float]:
    if not prefix_ids:
        raise ValueError("a non-scored prefix token is required for causal scoring")
    combined = torch.tensor([prefix_ids + target_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(combined)
    with torch.inference_mode():
        output = cast(
            Any,
            model(input_ids=combined, attention_mask=attention_mask, return_dict=True),
        )
    logits = output.logits if hasattr(output, "logits") else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("causal model must return rank-three logits")
    start = len(prefix_ids) - 1
    selected = logits[0, start : start + len(target_ids), :]
    targets = combined[0, len(prefix_ids) :]
    losses = F.cross_entropy(selected, targets, reduction="none")
    return [float(value) for value in losses.detach().cpu().tolist()]


def score_subword_answers(
    model: torch.nn.Module,
    tokenizer: SubwordTokenizer,
    prompt_answers: Sequence[tuple[str, str]],
    *,
    device: torch.device,
) -> BpbMeasurement:
    """Score answer tokens with prompt/answer encoding boundaries kept separate."""
    if not prompt_answers:
        raise ValueError("prompt_answers must not be empty")
    all_nlls: list[list[float]] = []
    answers: list[str] = []
    for prompt, answer in prompt_answers:
        prompt_ids = _exact_encode(tokenizer, prompt, "prompt")
        answer_ids = _exact_encode(tokenizer, answer, "answer")
        all_nlls.append(
            _target_nlls(
                model,
                prefix_ids=prompt_ids,
                target_ids=answer_ids,
                device=device,
            )
        )
        answers.append(answer)
    return aggregate_bits_per_byte(all_nlls, answers)


def score_subword_text(
    model: torch.nn.Module,
    tokenizer: SubwordTokenizer,
    text: str,
    *,
    device: torch.device,
) -> BpbMeasurement:
    """Score every held-out payload token with BOS used only as causal context."""
    text_ids = _exact_encode(tokenizer, text, "held-out text")
    bos_token_id = tokenizer.bos_token_id
    if not isinstance(bos_token_id, int):
        raise ValueError("tokenizer must define an integer BOS ID for whole-text scoring")
    nlls = _target_nlls(
        model,
        prefix_ids=[bos_token_id],
        target_ids=text_ids,
        device=device,
    )
    return aggregate_bits_per_byte([nlls], [text])


def canonical_corpus_payload(texts: Sequence[str]) -> bytes:
    """Serialize ordered text records without losing UTF-8 identity or boundaries."""
    if not texts or any(not text for text in texts):
        raise ValueError("corpus must contain non-empty text records")
    return json.dumps(
        list(texts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def corpus_sha256(texts: Sequence[str]) -> str:
    return hashlib.sha256(canonical_corpus_payload(texts)).hexdigest()


def _validate_sha256(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


@dataclass(frozen=True)
class ByteUnigramModel:
    """Frozen add-k byte distribution fitted only from declared training targets."""

    probabilities: tuple[float, ...]
    training_corpus_sha256: str
    training_utf8_bytes: int
    add_k: float

    def score(self, texts: Sequence[str]) -> BpbMeasurement:
        payloads = [text.encode("utf-8") for text in texts]
        if not payloads or any(not payload for payload in payloads):
            raise ValueError("unigram evaluation texts must be non-empty")
        losses = [
            [-math.log(self.probabilities[byte]) for byte in payload]
            for payload in payloads
        ]
        return aggregate_bits_per_byte(losses, texts)


def fit_byte_unigram(
    training_texts: Sequence[str], *, add_k: float
) -> ByteUnigramModel:
    """Fit and freeze a 256-way byte unigram using training-target bytes only."""
    if not math.isfinite(add_k) or add_k <= 0.0:
        raise ValueError("add_k must be finite and greater than zero")
    canonical = canonical_corpus_payload(training_texts)
    raw = b"".join(text.encode("utf-8") for text in training_texts)
    counts = Counter(raw)
    denominator = len(raw) + add_k * 256
    probabilities = tuple((counts[value] + add_k) / denominator for value in range(256))
    return ByteUnigramModel(
        probabilities=probabilities,
        training_corpus_sha256=hashlib.sha256(canonical).hexdigest(),
        training_utf8_bytes=len(raw),
        add_k=add_k,
    )


class CorpusEvidence(_StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str
    records: PositiveInt
    utf8_bytes: PositiveInt

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value)


class MetricEvidence(_StrictModel):
    schema_version: Literal["aptivra.bpb_metric.v5"]
    metric_kind: Literal["subword_model", "training_fitted_byte_unigram"]
    corpus_name: str = Field(min_length=1)
    corpus_sha256: str
    total_cross_entropy_bits: FiniteNonNegative
    scored_utf8_bytes: PositiveInt
    scored_tokens: PositiveInt
    bits_per_byte: FiniteNonNegative

    @field_validator("corpus_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value)

    @model_validator(mode="after")
    def validate_ratio(self) -> MetricEvidence:
        expected = self.total_cross_entropy_bits / self.scored_utf8_bytes
        if not math.isclose(self.bits_per_byte, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("bits_per_byte does not equal aggregate bits divided by UTF-8 bytes")
        return self


class UnigramFitEvidence(_StrictModel):
    training_corpus_name: str = Field(min_length=1)
    training_corpus_sha256: str
    training_utf8_bytes: PositiveInt
    smoothing: Literal["add_k"]
    add_k: PositiveFinite
    vocabulary_size: Literal[256]

    @field_validator("training_corpus_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value)


class ContaminationAuditEvidence(_StrictModel):
    heldout_corpus_name: str = Field(min_length=1)
    heldout_corpus_sha256: str
    training_source_sha256: dict[str, str]
    overlap_count: Literal[0]
    passed: Literal[True]

    @field_validator("heldout_corpus_sha256")
    @classmethod
    def validate_heldout_hash(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("training_source_sha256")
    @classmethod
    def validate_training_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("contamination audit requires training sources")
        for source, digest in value.items():
            if not source:
                raise ValueError("contamination source names must be non-empty")
            _validate_sha256(digest)
        return value


class ReleaseComparisonEvidence(_StrictModel):
    cell: Literal["heldout_answers", "heldout_text"]
    model_metric: MetricEvidence
    baseline_metric: MetricEvidence
    margin_bits_per_byte: FiniteNonNegative
    passed: bool

    @model_validator(mode="after")
    def validate_comparison(self) -> ReleaseComparisonEvidence:
        expected = release_gate_passes(
            self.model_metric,
            self.baseline_metric,
            margin_bits_per_byte=self.margin_bits_per_byte,
        )
        if self.passed is not expected:
            raise ValueError("release comparison pass flag does not match the strict gate")
        return self


class V5EvaluationRecord(_StrictModel):
    """Strict evidence boundary for the standard-base path; v0/v4 is invalid."""

    schema_version: Literal["aptivra.model_card_eval.v5"]
    artifact_identity: Literal["smollm2-standard-base-v1"]
    base_repository: Literal["HuggingFaceTB/SmolLM2-360M"]
    base_revision: Literal["f8027fd0eaeea54caa13c31d31b9fdc459c38b49"]
    scoring_protocol: BpbProtocolSettings
    corpora: tuple[CorpusEvidence, ...]
    unigram_fit: UnigramFitEvidence
    contamination_audits: tuple[ContaminationAuditEvidence, ...]
    comparisons: tuple[ReleaseComparisonEvidence, ...]
    release_gate_passed: bool

    @field_validator("corpora", "contamination_audits", "comparisons", mode="before")
    @classmethod
    def deserialize_sequences(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("v5 record collections must be lists")
        return tuple(cast(Sequence[object], value))

    @model_validator(mode="after")
    def validate_links(self) -> V5EvaluationRecord:
        corpora = {item.name: item for item in self.corpora}
        if len(corpora) != len(self.corpora):
            raise ValueError("corpus names must be unique")
        if self.unigram_fit.training_corpus_name not in corpora:
            raise ValueError("unigram fit corpus is missing from the evidence record")
        fit_corpus = corpora[self.unigram_fit.training_corpus_name]
        if (
            fit_corpus.sha256 != self.unigram_fit.training_corpus_sha256
            or fit_corpus.utf8_bytes != self.unigram_fit.training_utf8_bytes
        ):
            raise ValueError("unigram fit evidence does not match its training corpus")
        required_cells = {"heldout_answers", "heldout_text"}
        cells = {item.cell for item in self.comparisons}
        if cells != required_cells or len(self.comparisons) != 2:
            raise ValueError("v5 evidence requires exactly both held-out release cells")
        for comparison in self.comparisons:
            corpus = corpora.get(comparison.model_metric.corpus_name)
            if corpus is None:
                raise ValueError("release metric corpus is missing from the evidence record")
            for metric in (comparison.model_metric, comparison.baseline_metric):
                if (
                    metric.corpus_name != corpus.name
                    or metric.corpus_sha256 != corpus.sha256
                    or metric.scored_utf8_bytes != corpus.utf8_bytes
                ):
                    raise ValueError("release metrics do not identify the exact held-out corpus")
        if self.release_gate_passed is not all(item.passed for item in self.comparisons):
            raise ValueError("overall release flag must be the conjunction of both cells")
        return self


def metric_evidence(
    measurement: BpbMeasurement,
    *,
    metric_kind: Literal["subword_model", "training_fitted_byte_unigram"],
    corpus_name: str,
    corpus_hash: str,
) -> MetricEvidence:
    """Bind a computed aggregate to its v5 schema and exact corpus identity."""
    return MetricEvidence(
        schema_version="aptivra.bpb_metric.v5",
        metric_kind=metric_kind,
        corpus_name=corpus_name,
        corpus_sha256=corpus_hash,
        total_cross_entropy_bits=measurement.total_cross_entropy_bits,
        scored_utf8_bytes=measurement.scored_utf8_bytes,
        scored_tokens=measurement.scored_tokens,
        bits_per_byte=measurement.bits_per_byte,
    )


def release_gate_passes(
    model_metric: MetricEvidence | object,
    baseline_metric: MetricEvidence | object,
    *,
    margin_bits_per_byte: float,
) -> bool:
    """Compare like-for-like v5 metrics with a strict training-unigram gate."""
    model_value = MetricEvidence.model_validate(model_metric)
    baseline_value = MetricEvidence.model_validate(baseline_metric)
    if (
        isinstance(margin_bits_per_byte, bool)
        or not math.isfinite(margin_bits_per_byte)
        or margin_bits_per_byte < 0.0
    ):
        raise ValueError("margin_bits_per_byte must be finite and non-negative")
    if model_value.metric_kind != "subword_model":
        raise ValueError("release candidate must be a v5 subword-model metric")
    if baseline_value.metric_kind != "training_fitted_byte_unigram":
        raise ValueError("release baseline must be a v5 training-fitted byte unigram")
    if (
        model_value.corpus_name != baseline_value.corpus_name
        or model_value.corpus_sha256 != baseline_value.corpus_sha256
        or model_value.scored_utf8_bytes != baseline_value.scored_utf8_bytes
    ):
        raise ValueError("release metrics must score the exact same UTF-8 corpus")
    return (
        model_value.bits_per_byte + margin_bits_per_byte
        < baseline_value.bits_per_byte
    )


def audit_heldout_contamination(
    heldout_texts: Sequence[str],
    *,
    heldout_name: str,
    expected_heldout_sha256: str,
    training_sources: Mapping[str, Sequence[str]],
    expected_training_sha256: Mapping[str, str],
    required_training_sources: Sequence[str],
) -> ContaminationAuditEvidence:
    """Verify hashes and reject any training/held-out overlap, failing closed."""
    required = tuple(required_training_sources)
    if not required or len(required) != len(set(required)):
        raise ValueError("required training sources must be non-empty and unique")
    if set(training_sources) != set(required) or set(expected_training_sha256) != set(
        required
    ):
        raise ValueError("all and only required training sources must be supplied")
    heldout_hash = corpus_sha256(heldout_texts)
    if heldout_hash != _validate_sha256(expected_heldout_sha256):
        raise ValueError("held-out corpus SHA-256 mismatch")
    heldout = tuple(heldout_texts)
    source_hashes: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for source in required:
        training = tuple(training_sources[source])
        actual_hash = corpus_sha256(training)
        expected_hash = _validate_sha256(expected_training_sha256[source])
        if actual_hash != expected_hash:
            raise ValueError(f"training source {source!r} SHA-256 mismatch")
        source_hashes[source] = actual_hash
        for training_text in training:
            for heldout_text in heldout:
                if training_text in heldout_text or heldout_text in training_text:
                    overlaps.append((source, training_text, heldout_text))
    if overlaps:
        raise ValueError(
            f"held-out contamination detected in {len(overlaps)} source/record pairs"
        )
    return ContaminationAuditEvidence(
        heldout_corpus_name=heldout_name,
        heldout_corpus_sha256=heldout_hash,
        training_source_sha256=source_hashes,
        overlap_count=0,
        passed=True,
    )


def validate_v5_evaluation_record(
    record: object,
    *,
    corpus_payloads: Mapping[str, Sequence[str]],
    required_training_sources: Sequence[str],
) -> V5EvaluationRecord:
    """Validate v5 schema, exact corpus hashes, and complete contamination coverage."""
    validated = V5EvaluationRecord.model_validate(record)
    corpora = {item.name: item for item in validated.corpora}
    if set(corpus_payloads) != set(corpora):
        raise ValueError("all and only evidence-record corpora must be supplied")
    for name, evidence in corpora.items():
        texts = corpus_payloads[name]
        if corpus_sha256(texts) != evidence.sha256:
            raise ValueError(f"corpus {name!r} SHA-256 mismatch")
        if len(texts) != evidence.records:
            raise ValueError(f"corpus {name!r} record count mismatch")
        if sum(len(text.encode("utf-8")) for text in texts) != evidence.utf8_bytes:
            raise ValueError(f"corpus {name!r} UTF-8 byte count mismatch")
    required_sequence = tuple(required_training_sources)
    if not required_sequence or len(required_sequence) != len(set(required_sequence)):
        raise ValueError("required training sources must be non-empty and unique")
    required = set(required_sequence)
    heldout_names = {item.heldout_corpus_name for item in validated.contamination_audits}
    if heldout_names != {"heldout_answers", "heldout_text"}:
        raise ValueError("both held-out corpora require contamination audits")
    for audit in validated.contamination_audits:
        corpus = corpora.get(audit.heldout_corpus_name)
        if corpus is None or audit.heldout_corpus_sha256 != corpus.sha256:
            raise ValueError("contamination audit held-out hash mismatch")
        if set(audit.training_source_sha256) != required:
            raise ValueError("contamination audit does not cover every required source")
        for source, digest in audit.training_source_sha256.items():
            source_corpus = corpora.get(source)
            if source_corpus is None or source_corpus.sha256 != digest:
                raise ValueError("contamination audit training-source hash mismatch")
        recomputed = audit_heldout_contamination(
            corpus_payloads[audit.heldout_corpus_name],
            heldout_name=audit.heldout_corpus_name,
            expected_heldout_sha256=audit.heldout_corpus_sha256,
            training_sources={
                source: corpus_payloads[source] for source in required_sequence
            },
            expected_training_sha256=audit.training_source_sha256,
            required_training_sources=required_sequence,
        )
        if recomputed != audit:
            raise ValueError("contamination audit does not match recomputed evidence")
    return validated
