"""Acceptance tests for ADR-0011 lane-2 stopping and v5 evidence."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace
from importlib.resources.abc import Traversable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
import torch
from pydantic import ValidationError

from aptivra.eval.standard_evidence import (
    BpbMeasurement,
    ContaminationAuditEvidence,
    ContaminationSettings,
    CorpusEvidence,
    EvaluationFeatureSettings,
    MetricEvidence,
    ReleaseComparisonEvidence,
    ReleaseGateSettings,
    SubwordTokenizer,
    UnigramFitEvidence,
    V5EvaluationRecord,
    _exact_encode,
    _target_nlls,
    aggregate_bits_per_byte,
    audit_heldout_contamination,
    canonical_corpus_payload,
    corpus_sha256,
    fit_byte_unigram,
    load_standard_evaluation_settings,
    metric_evidence,
    release_gate_passes,
    score_subword_answers,
    score_subword_text,
    validate_v5_evaluation_record,
)
from aptivra.inference.standard_base import (
    StandardBase,
    VerifiedSnapshot,
    generate_task_instance_answer,
    load_standard_base_settings,
)
from aptivra.tasks.base import TaskInstance

_ROOT = Path(__file__).resolve().parents[1]
_TRAINING_SOURCES = (
    "identity_chat",
    "task_stream_v2",
    "replay",
    "chronological_issue_stream",
)


class _BoundaryTokenizer:
    eos_token_id = 0
    bos_token_id = 0

    def __init__(self, answer: str, terminator: str) -> None:
        self.answer = answer
        self.terminator = terminator

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return [41]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        if token_ids == [5]:
            return self.answer
        if token_ids == [5, 6]:
            return self.answer
        if token_ids[:3] == [5, 6, 7]:
            return f"{self.answer}{self.terminator}leaked-in-token"
        return ""


class _CriteriaAwareGenerator:
    def generate(self, **kwargs: object) -> torch.Tensor:
        output = cast(torch.Tensor, kwargs["input_ids"])
        criteria = cast(Any, kwargs["stopping_criteria"])
        for token_id in (5, 6, 7, 8):
            token = torch.tensor([[token_id]], dtype=torch.long, device=output.device)
            output = torch.cat((output, token), dim=1)
            if bool(criteria(output, None)[0].item()):
                break
        return output


def _generation_base(answer: str, terminator: str) -> StandardBase:
    settings = load_standard_base_settings(_ROOT / "configs/smollm2_base.yaml")
    return StandardBase(
        model=cast(Any, _CriteriaAwareGenerator()),
        tokenizer=cast(Any, _BoundaryTokenizer(answer, terminator)),
        settings=settings,
        snapshot=VerifiedSnapshot(Path("."), "owner/model", "0" * 40, 1, 1),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("instance", "terminator"),
    [
        (
            TaskInstance("single", "single", "prompt", "answer"),
            "\n",
        ),
        (
            TaskInstance(
                "code",
                "code",
                "prompt",
                "line1\nline2",
                answer_stop_characters=("\x00",),
                training_terminator="\x00",
            ),
            "\x00",
        ),
    ],
)
def test_task_generation_stops_on_decoded_multitoken_lf_or_nul_without_leakage(
    instance: TaskInstance,
    terminator: str,
) -> None:
    result = generate_task_instance_answer(
        _generation_base(instance.target_solution, terminator),
        instance,
        max_new_tokens=4,
    )

    assert result.text == instance.target_solution
    assert result.generated_token_ids == (5, 6, 7)
    assert result.stop_reason == "terminator"
    assert result.matched_terminator == terminator
    assert terminator not in result.text
    assert "leaked-in-token" not in result.text


def test_task_generation_validates_decoded_stop_sequences() -> None:
    base = _generation_base("answer", "\n")
    duplicate = replace(
        TaskInstance("bad", "single", "prompt", "answer"),
        answer_stop_characters=("\n", "\n"),
    )
    empty = replace(duplicate, answer_stop_characters=("",))

    with pytest.raises(ValueError, match="unique"):
        generate_task_instance_answer(base, duplicate, max_new_tokens=4)
    with pytest.raises(ValueError, match="non-empty"):
        generate_task_instance_answer(base, empty, max_new_tokens=4)


def test_analytic_byte_subword_and_multibyte_bpb_are_equal() -> None:
    byte_level = aggregate_bits_per_byte(
        [[math.log(2.0), math.log(4.0)], [math.log(4.0), math.log(4.0)]],
        ["ab", "é"],
    )
    subword = aggregate_bits_per_byte(
        [[math.log(8.0)], [math.log(16.0)]],
        ["ab", "é"],
    )

    assert byte_level.total_cross_entropy_bits == pytest.approx(
        subword.total_cross_entropy_bits
    )
    assert byte_level.scored_utf8_bytes == subword.scored_utf8_bytes
    assert byte_level.bits_per_byte == pytest.approx(subword.bits_per_byte)
    assert subword.total_cross_entropy_bits == pytest.approx(7.0)
    assert subword.scored_utf8_bytes == 4
    assert subword.scored_tokens == 2
    assert subword.bits_per_byte == pytest.approx(1.75)


def test_bpb_aggregates_bits_and_bytes_before_division() -> None:
    result = aggregate_bits_per_byte(
        [[4.0 * math.log(2.0)], [0.0]],
        ["a", "bbb"],
    )

    assert result.bits_per_byte == pytest.approx(1.0)
    assert result.bits_per_byte != pytest.approx((4.0 + 0.0) / 2.0)


@pytest.mark.parametrize(
    ("losses", "texts", "message"),
    [([], [], "non-empty and aligned"), ([[0.0]], ["a", "b"], "aligned"),
     ([[]], ["a"], "at least one"), ([[0.0]], [""], "must not be empty"),
     ([[float("nan")]], ["a"], "finite"), ([[-1.0]], ["a"], "non-negative")],
)
def test_bpb_rejects_invalid_inputs(
    losses: list[list[float]], texts: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_bits_per_byte(losses, texts)


def test_bpb_rejects_reverse_alignment_mismatch() -> None:
    with pytest.raises(ValueError, match="aligned"):
        aggregate_bits_per_byte([[0.0], [0.0]], ["a"])


class _ExactTokenizer:
    bos_token_id = 4

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self._texts: dict[tuple[int, ...], str] = {}
        self._next = 5

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded.append(text)
        token_ids = [self._next, self._next + 1] if len(text) > 1 else [self._next]
        self._next += len(token_ids)
        self._texts[tuple(token_ids)] = text
        return token_ids

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return self._texts[tuple(token_ids)]


class _PerfectTargetModel(torch.nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict is True
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        vocab = 64
        logits = torch.zeros((*input_ids.shape, vocab), dtype=torch.float32)
        for position in range(input_ids.shape[1] - 1):
            logits[0, position, input_ids[0, position + 1]] = math.log(3.0)
        return SimpleNamespace(logits=logits)


def test_subword_scorers_keep_boundaries_and_score_exact_utf8_payload() -> None:
    tokenizer = _ExactTokenizer()
    model = _PerfectTargetModel()

    answers = score_subword_answers(
        model,
        tokenizer,
        [("prompt", "é"), ("question", "ok")],
        device=torch.device("cpu"),
    )
    whole_text = score_subword_text(
        model,
        tokenizer,
        "éx",
        device=torch.device("cpu"),
    )

    assert tokenizer.encoded[:4] == ["prompt", "é", "question", "ok"]
    assert answers.scored_utf8_bytes == len("éok".encode())
    assert whole_text.scored_utf8_bytes == len("éx".encode())
    assert answers.scored_tokens == 3
    assert whole_text.scored_tokens == 2


def test_subword_scorers_fail_closed_on_non_roundtripping_or_missing_context() -> None:
    tokenizer = _ExactTokenizer()
    model = _PerfectTargetModel()
    tokenizer.decode = lambda _ids, **_kwargs: "wrong"  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="exact encode/decode"):
        score_subword_answers(
            model,
            tokenizer,
            [("prompt", "answer")],
            device=torch.device("cpu"),
        )

    missing_bos = _ExactTokenizer()
    missing_bos.bos_token_id = None
    with pytest.raises(ValueError, match="BOS"):
        score_subword_text(
            model,
            missing_bos,
            "heldout",
            device=torch.device("cpu"),
        )


def test_exact_encoding_rejects_lexicographically_lower_but_distinct_text() -> None:
    tokenizer = _ExactTokenizer()
    tokenizer.decode = lambda _ids, **_kwargs: "\x00"  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="exact encode/decode"):
        _exact_encode(tokenizer, "prompt", "prompt")

    original = "prompt"
    equal_but_distinct = "".join(["pro", "mpt"])
    assert equal_but_distinct == original
    assert equal_but_distinct is not original
    tokenizer.decode = (  # type: ignore[method-assign]
        lambda _ids, **_kwargs: equal_but_distinct
    )
    assert _exact_encode(tokenizer, "prompt", "prompt")


class _PositionSensitiveModel(torch.nn.Module):
    def forward(self, **kwargs: object) -> SimpleNamespace:
        input_ids = cast(torch.Tensor, kwargs["input_ids"])
        logits = torch.full((*input_ids.shape, 16), -8.0)
        for position in range(input_ids.shape[1] - 1):
            logits[0, position, input_ids[0, position + 1]] = float(position + 1)
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("rank", [2, 4])
def test_target_nlls_rejects_every_non_rank_three_tensor(rank: int) -> None:
    class _BadRankModel(torch.nn.Module):
        def forward(self, **_kwargs: object) -> torch.Tensor:
            return torch.zeros((1,) * rank)

    with pytest.raises(TypeError, match="rank-three"):
        _target_nlls(
            _BadRankModel(),
            prefix_ids=[1],
            target_ids=[2],
            device=torch.device("cpu"),
        )


def test_target_nlls_rejects_non_tensor_and_uses_last_prefix_position() -> None:
    class _NotTensorModel(torch.nn.Module):
        def forward(self, **_kwargs: object) -> str:
            return "not logits"

    with pytest.raises(TypeError, match="rank-three"):
        _target_nlls(
            _NotTensorModel(),
            prefix_ids=[1],
            target_ids=[2],
            device=torch.device("cpu"),
        )

    losses = _target_nlls(
        _PositionSensitiveModel(),
        prefix_ids=[1, 2, 3],
        target_ids=[4, 5],
        device=torch.device("cpu"),
    )
    expected = torch.nn.functional.cross_entropy(
        torch.tensor(
            [[-8.0] * 4 + [3.0] + [-8.0] * 11, [-8.0] * 5 + [4.0] + [-8.0] * 10]
        ),
        torch.tensor([4, 5]),
        reduction="none",
    ).tolist()
    assert losses == pytest.approx(expected)


def _measurement(bits_per_byte: float, *, byte_count: int = 4) -> BpbMeasurement:
    return BpbMeasurement(bits_per_byte * byte_count, byte_count, 2, bits_per_byte)


def _metric(
    bits_per_byte: float,
    kind: str,
    *,
    corpus_name: str = "heldout_answers",
    corpus_hash: str | None = None,
) -> MetricEvidence:
    return metric_evidence(
        _measurement(bits_per_byte),
        metric_kind=cast(Any, kind),
        corpus_name=corpus_name,
        corpus_hash=corpus_hash or ("a" * 64),
    )


def test_training_fitted_unigram_is_frozen_and_scores_heldout_bytes() -> None:
    model = fit_byte_unigram(["aaaa", "ab"], add_k=1.0)
    score = model.score(["ab", "é"])

    assert len(model.probabilities) == 256
    assert sum(model.probabilities) == pytest.approx(1.0)
    assert model.training_utf8_bytes == 6
    assert model.training_corpus_sha256 == corpus_sha256(["aaaa", "ab"])
    assert score.scored_utf8_bytes == 4
    assert score.scored_tokens == 4
    assert score.bits_per_byte > 0.0
    with pytest.raises(ValueError, match="non-empty"):
        model.score([])
    with pytest.raises(ValueError, match="non-empty"):
        model.score([""])
    with pytest.raises(AttributeError):
        model.add_k = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("add_k", [0.0, -1.0, float("inf")])
def test_unigram_rejects_invalid_smoothing(add_k: float) -> None:
    with pytest.raises(ValueError, match="add_k"):
        fit_byte_unigram(["training"], add_k=add_k)


def test_unigram_release_gate_has_explicit_pass_and_fail_boundaries() -> None:
    baseline = _metric(3.0, "training_fitted_byte_unigram")

    assert release_gate_passes(_metric(2.999, "subword_model"), baseline, margin_bits_per_byte=0)
    assert not release_gate_passes(_metric(3.0, "subword_model"), baseline, margin_bits_per_byte=0)
    assert not release_gate_passes(
        _metric(2.9, "subword_model"), baseline, margin_bits_per_byte=0.1
    )
    assert release_gate_passes(_metric(2.899, "subword_model"), baseline, margin_bits_per_byte=0.1)


def test_release_gate_rejects_v0_or_noncomparable_metrics() -> None:
    baseline = _metric(3.0, "training_fitted_byte_unigram")
    v0_metric = {
        "schema_version": "aptivra.model_card_eval.v4",
        "metric_kind": "subword_model",
        "corpus_name": "heldout_answers",
        "corpus_sha256": "a" * 64,
        "total_cross_entropy_bits": 8.0,
        "scored_utf8_bytes": 4,
        "scored_tokens": 2,
        "bits_per_byte": 2.0,
    }

    with pytest.raises(ValidationError):
        release_gate_passes(v0_metric, baseline, margin_bits_per_byte=0.0)
    with pytest.raises(ValueError, match="same UTF-8 corpus"):
        release_gate_passes(
            _metric(2.0, "subword_model", corpus_hash="b" * 64),
            baseline,
            margin_bits_per_byte=0.0,
        )
    with pytest.raises(ValueError, match="subword-model"):
        release_gate_passes(baseline, baseline, margin_bits_per_byte=0.0)
    with pytest.raises(ValueError, match="training-fitted"):
        release_gate_passes(
            _metric(2.0, "subword_model"),
            _metric(3.0, "subword_model"),
            margin_bits_per_byte=0.0,
        )
    with pytest.raises(ValueError, match="margin_bits_per_byte"):
        release_gate_passes(
            _metric(2.0, "subword_model"),
            baseline,
            margin_bits_per_byte=cast(Any, True),
        )
    with pytest.raises(ValueError, match="margin_bits_per_byte"):
        release_gate_passes(
            _metric(2.0, "subword_model"),
            baseline,
            margin_bits_per_byte=-0.5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_name", "aa_heldout_answers"),
        ("corpus_name", "zz_heldout_answers"),
        ("corpus_sha256", "0" * 64),
        ("corpus_sha256", "f" * 64),
        ("scored_utf8_bytes", 3),
        ("scored_utf8_bytes", 5),
    ],
)
def test_release_gate_rejects_both_directions_of_corpus_mismatch(
    field: str, value: object
) -> None:
    baseline = _metric(3.0, "training_fitted_byte_unigram")
    changed = baseline.model_copy(update={field: value})
    if field == "scored_utf8_bytes":
        changed = changed.model_copy(
            update={"total_cross_entropy_bits": 3.0 * cast(int, value)}
        )
    with pytest.raises(ValueError, match="same UTF-8 corpus"):
        release_gate_passes(
            _metric(2.0, "subword_model"),
            changed,
            margin_bits_per_byte=0.0,
        )


def _source_payloads() -> dict[str, list[str]]:
    return {
        "identity_chat": ["identity training"],
        "task_stream_v2": ["synthetic training"],
        "replay": ["replay training"],
        "chronological_issue_stream": ["issue training"],
    }


def test_heldout_audit_passes_only_with_complete_verified_clean_inputs() -> None:
    heldout = ["held out alpha", "held out beta"]
    sources = _source_payloads()
    audit = audit_heldout_contamination(
        heldout,
        heldout_name="heldout_answers",
        expected_heldout_sha256=corpus_sha256(heldout),
        training_sources=sources,
        expected_training_sha256={
            source: corpus_sha256(texts) for source, texts in sources.items()
        },
        required_training_sources=_TRAINING_SOURCES,
    )

    assert audit.passed is True
    assert audit.overlap_count == 0
    assert set(audit.training_source_sha256) == set(_TRAINING_SOURCES)


def test_heldout_audit_fails_closed_on_missing_hash_or_contamination() -> None:
    heldout = ["held out alpha"]
    sources = _source_payloads()
    hashes = {source: corpus_sha256(texts) for source, texts in sources.items()}

    with pytest.raises(ValueError, match="all and only"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256=corpus_sha256(heldout),
            training_sources={key: value for key, value in sources.items() if key != "replay"},
            expected_training_sha256=hashes,
            required_training_sources=_TRAINING_SOURCES,
        )


@pytest.mark.parametrize("required", [(), ("identity_chat", "identity_chat")])
def test_heldout_audit_rejects_invalid_required_source_sets(
    required: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        audit_heldout_contamination(
            ["heldout"],
            heldout_name="heldout_answers",
            expected_heldout_sha256=corpus_sha256(["heldout"]),
            training_sources={},
            expected_training_sha256={},
            required_training_sources=required,
        )


@pytest.mark.parametrize("replacement", ["0" * 64, "f" * 64])
def test_heldout_audit_rejects_hash_mismatch_in_either_order(
    replacement: str,
) -> None:
    heldout = ["heldout"]
    sources = _source_payloads()
    hashes = {source: corpus_sha256(texts) for source, texts in sources.items()}
    with pytest.raises(ValueError, match="held-out corpus SHA-256"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256=replacement,
            training_sources=sources,
            expected_training_sha256=hashes,
            required_training_sources=_TRAINING_SOURCES,
        )
    hashes["replay"] = replacement
    with pytest.raises(ValueError, match="replay.*SHA-256"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256=corpus_sha256(heldout),
            training_sources=sources,
            expected_training_sha256=hashes,
            required_training_sources=_TRAINING_SOURCES,
        )
    with pytest.raises(ValueError, match="held-out corpus SHA-256"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256="0" * 64,
            training_sources=sources,
            expected_training_sha256=hashes,
            required_training_sources=_TRAINING_SOURCES,
        )
    bad_hashes = {**hashes, "replay": "0" * 64}
    with pytest.raises(ValueError, match="replay.*SHA-256"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256=corpus_sha256(heldout),
            training_sources=sources,
            expected_training_sha256=bad_hashes,
            required_training_sources=_TRAINING_SOURCES,
        )
    contaminated = {**sources, "replay": ["prefix heldout suffix"]}
    contaminated_hashes = {
        source: corpus_sha256(texts) for source, texts in contaminated.items()
    }
    with pytest.raises(ValueError, match="contamination detected"):
        audit_heldout_contamination(
            heldout,
            heldout_name="heldout_answers",
            expected_heldout_sha256=corpus_sha256(heldout),
            training_sources=contaminated,
            expected_training_sha256=contaminated_hashes,
            required_training_sources=_TRAINING_SOURCES,
        )


def _corpus(name: str, texts: list[str]) -> CorpusEvidence:
    return CorpusEvidence(
        name=name,
        version="v2",
        sha256=corpus_sha256(texts),
        records=len(texts),
        utf8_bytes=sum(len(text.encode("utf-8")) for text in texts),
    )


def _valid_v5_record() -> tuple[dict[str, object], dict[str, list[str]]]:
    payloads = {
        **_source_payloads(),
        "training_targets": ["all declared training targets"],
        "heldout_answers": ["ans1"],
        "heldout_text": ["text"],
    }
    corpora = [_corpus(name, texts) for name, texts in payloads.items()]
    training = next(item for item in corpora if item.name == "training_targets")
    source_hashes = {
        name: corpus_sha256(payloads[name]) for name in _TRAINING_SOURCES
    }
    audits = tuple(
        ContaminationAuditEvidence(
            heldout_corpus_name=name,
            heldout_corpus_sha256=corpus_sha256(payloads[name]),
            training_source_sha256=source_hashes,
            overlap_count=0,
            passed=True,
        )
        for name in ("heldout_answers", "heldout_text")
    )
    comparisons = []
    for cell in ("heldout_answers", "heldout_text"):
        corpus = next(item for item in corpora if item.name == cell)
        model_metric = _metric(
            2.0,
            "subword_model",
            corpus_name=cell,
            corpus_hash=corpus.sha256,
        ).model_copy(
            update={
                "scored_utf8_bytes": corpus.utf8_bytes,
                "total_cross_entropy_bits": 2.0 * corpus.utf8_bytes,
            }
        )
        baseline = _metric(
            3.0,
            "training_fitted_byte_unigram",
            corpus_name=cell,
            corpus_hash=corpus.sha256,
        ).model_copy(
            update={
                "scored_utf8_bytes": corpus.utf8_bytes,
                "total_cross_entropy_bits": 3.0 * corpus.utf8_bytes,
            }
        )
        comparisons.append(
            ReleaseComparisonEvidence(
                cell=cast(Any, cell),
                model_metric=model_metric,
                baseline_metric=baseline,
                margin_bits_per_byte=0.0,
                passed=True,
            )
        )
    record = V5EvaluationRecord(
        schema_version="aptivra.model_card_eval.v5",
        artifact_identity="smollm2-standard-base-v1",
        base_repository="HuggingFaceTB/SmolLM2-360M",
        base_revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        scoring_protocol=load_standard_evaluation_settings().bpb,
        corpora=tuple(corpora),
        unigram_fit=UnigramFitEvidence(
            training_corpus_name=training.name,
            training_corpus_sha256=training.sha256,
            training_utf8_bytes=training.utf8_bytes,
            smoothing="add_k",
            add_k=1.0,
            vocabulary_size=256,
        ),
        contamination_audits=audits,
        comparisons=tuple(comparisons),
        release_gate_passed=True,
    )
    return record.model_dump(mode="json"), payloads


def test_evidence_models_enforce_strict_bounds_and_frozen_values() -> None:
    digest = "a" * 64
    corpus = CorpusEvidence(
        name="x", version="y", sha256=digest, records=1, utf8_bytes=1
    )
    assert corpus.name == "x"
    with pytest.raises(ValidationError):
        CorpusEvidence(
            name="", version="y", sha256=digest, records=1, utf8_bytes=1
        )
    with pytest.raises(ValidationError):
        CorpusEvidence(
            name="x", version="", sha256=digest, records=1, utf8_bytes=1
        )
    with pytest.raises(ValidationError):
        CorpusEvidence(
            name="x", version="y", sha256=digest, records=0, utf8_bytes=1
        )
    with pytest.raises(ValidationError):
        CorpusEvidence.model_validate(
            {
                "name": "x",
                "version": "y",
                "sha256": digest,
                "records": 1.0,
                "utf8_bytes": 1,
            }
        )
    with pytest.raises(ValidationError):
        CorpusEvidence.model_validate(
            {
                "name": "x",
                "version": "y",
                "sha256": digest,
                "records": 1,
                "utf8_bytes": 1,
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError):
        corpus.name = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        EvaluationFeatureSettings(enabled=cast(Any, 1))
    measurement = _measurement(1.0)
    with pytest.raises(AttributeError):
        measurement.scored_tokens = 3  # type: ignore[misc]

    metric = _metric(1.0, "subword_model", corpus_name="x")
    assert metric.corpus_name == "x"
    with pytest.raises(ValidationError):
        MetricEvidence.model_validate({**metric.model_dump(), "corpus_name": ""})
    with pytest.raises(ValidationError):
        MetricEvidence.model_validate(
            {**metric.model_dump(), "total_cross_entropy_bits": -0.5}
        )
    with pytest.raises(ValidationError):
        MetricEvidence.model_validate(
            {**metric.model_dump(), "bits_per_byte": float("inf")}
        )

    fit = UnigramFitEvidence(
        training_corpus_name="x",
        training_corpus_sha256=digest,
        training_utf8_bytes=1,
        smoothing="add_k",
        add_k=1.0,
        vocabulary_size=256,
    )
    assert fit.training_corpus_name == "x"
    with pytest.raises(ValidationError):
        UnigramFitEvidence.model_validate(
            {**fit.model_dump(), "training_corpus_name": ""}
        )
    with pytest.raises(ValidationError):
        UnigramFitEvidence.model_validate({**fit.model_dump(), "add_k": -0.5})
    with pytest.raises(ValidationError):
        UnigramFitEvidence.model_validate(
            {**fit.model_dump(), "add_k": float("inf")}
        )


@pytest.mark.parametrize(
    "cells",
    ["heldout_answers", [], ["heldout_answers"], ["heldout_answers"] * 2],
)
def test_release_settings_require_exactly_both_cells(cells: object) -> None:
    with pytest.raises(ValidationError):
        ReleaseGateSettings.model_validate(
            {"margin_bits_per_byte": 0.0, "required_cells": cells}
        )


@pytest.mark.parametrize("margin", [-0.5, float("inf")])
def test_release_settings_reject_negative_or_nonfinite_margin(margin: float) -> None:
    with pytest.raises(ValidationError):
        ReleaseGateSettings(
            margin_bits_per_byte=margin,
            required_cells=("heldout_answers", "heldout_text"),
        )


@pytest.mark.parametrize(
    "sources", ["identity_chat", [], [""], [1], ["identity_chat"] * 2]
)
def test_contamination_settings_require_unique_nonempty_names(sources: object) -> None:
    with pytest.raises(ValidationError):
        ContaminationSettings.model_validate({"required_training_sources": sources})


def test_evidence_hash_fields_reject_invalid_values_and_empty_source_names() -> None:
    with pytest.raises(ValidationError):
        CorpusEvidence(
            name="x", version="v", sha256="A" * 64, records=1, utf8_bytes=1
        )
    with pytest.raises(ValidationError):
        ContaminationAuditEvidence(
            heldout_corpus_name="x",
            heldout_corpus_sha256="a" * 64,
            training_source_sha256={"": "b" * 64},
            overlap_count=0,
            passed=True,
        )
    with pytest.raises(ValidationError):
        ContaminationAuditEvidence(
            heldout_corpus_name="x",
            heldout_corpus_sha256="bad",
            training_source_sha256={"training": "b" * 64},
            overlap_count=0,
            passed=True,
        )
    with pytest.raises(ValidationError):
        MetricEvidence.model_validate(
            {**_metric(1.0, "subword_model").model_dump(), "corpus_sha256": "bad"}
        )
    fit = UnigramFitEvidence(
        training_corpus_name="x",
        training_corpus_sha256="a" * 64,
        training_utf8_bytes=1,
        smoothing="add_k",
        add_k=1.0,
        vocabulary_size=256,
    )
    with pytest.raises(ValidationError):
        UnigramFitEvidence.model_validate(
            {**fit.model_dump(), "training_corpus_sha256": "bad"}
        )

    audit = ContaminationAuditEvidence(
        heldout_corpus_name="x",
        heldout_corpus_sha256="a" * 64,
        training_source_sha256={"training": "b" * 64},
        overlap_count=0,
        passed=True,
    )
    assert audit.heldout_corpus_name == "x"
    with pytest.raises(ValidationError):
        ContaminationAuditEvidence.model_validate(
            {**audit.model_dump(), "heldout_corpus_name": ""}
        )


def test_runtime_type_and_keyword_only_contracts_are_intact() -> None:
    load_hints = get_type_hints(load_standard_evaluation_settings)
    assert load_hints["path"] == Path | str | Traversable
    gate_hints = get_type_hints(release_gate_passes)
    assert gate_hints["model_metric"] == MetricEvidence | object
    assert gate_hints["baseline_metric"] == MetricEvidence | object

    keyword_only = {
        SubwordTokenizer.encode: {"add_special_tokens"},
        _target_nlls: {"prefix_ids", "target_ids", "device"},
        score_subword_answers: {"device"},
        score_subword_text: {"device"},
        fit_byte_unigram: {"add_k"},
        metric_evidence: {"metric_kind", "corpus_name", "corpus_hash"},
        release_gate_passes: {"margin_bits_per_byte"},
        audit_heldout_contamination: {
            "heldout_name",
            "expected_heldout_sha256",
            "training_sources",
            "expected_training_sha256",
            "required_training_sources",
        },
        validate_v5_evaluation_record: {
            "corpus_payloads",
            "required_training_sources",
        },
    }
    for function, expected_names in keyword_only.items():
        parameters = inspect.signature(function).parameters
        assert {
            name
            for name, parameter in parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        } == expected_names
    with pytest.raises(ValidationError):
        ContaminationAuditEvidence(
            heldout_corpus_name="x",
            heldout_corpus_sha256="a" * 64,
            training_source_sha256={"training": "bad"},
            overlap_count=0,
            passed=True,
        )


def test_metric_ratio_rejects_material_absolute_and_relative_errors() -> None:
    metric = _metric(1.0, "subword_model")
    with pytest.raises(ValidationError, match="aggregate bits"):
        MetricEvidence.model_validate({**metric.model_dump(), "bits_per_byte": 1.5})


def test_release_comparison_rejects_both_incorrect_boolean_directions() -> None:
    model = _metric(2.0, "subword_model")
    baseline = _metric(3.0, "training_fitted_byte_unigram")
    with pytest.raises(ValidationError, match="pass flag"):
        ReleaseComparisonEvidence(
            cell="heldout_answers",
            model_metric=model,
            baseline_metric=baseline,
            margin_bits_per_byte=0.0,
            passed=False,
        )
    with pytest.raises(ValidationError, match="pass flag"):
        ReleaseComparisonEvidence(
            cell="heldout_answers",
            model_metric=baseline.model_copy(
                update={"metric_kind": "subword_model"}
            ),
            baseline_metric=baseline,
            margin_bits_per_byte=0.0,
            passed=True,
        )


def test_v5_record_rejects_v0_and_validates_payload_hashes_fail_closed() -> None:
    record, payloads = _valid_v5_record()
    validated = validate_v5_evaluation_record(
        record,
        corpus_payloads=payloads,
        required_training_sources=_TRAINING_SOURCES,
    )
    assert validated.release_gate_passed is True

    v0 = copy.deepcopy(record)
    v0["schema_version"] = "aptivra.model_card_eval.v4"
    with pytest.raises(ValidationError):
        V5EvaluationRecord.model_validate(v0)

    tampered = {**payloads, "heldout_text": ["changed"]}
    with pytest.raises(ValueError, match="heldout_text.*SHA-256"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads=tampered,
            required_training_sources=_TRAINING_SOURCES,
        )
    with pytest.raises(ValueError, match="all and only"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads={key: value for key, value in payloads.items() if key != "replay"},
            required_training_sources=_TRAINING_SOURCES,
        )

    contaminated_record, contaminated_payloads = _valid_v5_record()
    contaminated_payloads["replay"] = ["prefix text suffix"]
    replay_hash = corpus_sha256(contaminated_payloads["replay"])
    for corpus in cast(list[dict[str, object]], contaminated_record["corpora"]):
        if corpus["name"] == "replay":
            corpus["sha256"] = replay_hash
            corpus["utf8_bytes"] = len(b"prefix text suffix")
    for audit in cast(
        list[dict[str, object]], contaminated_record["contamination_audits"]
    ):
        cast(dict[str, str], audit["training_source_sha256"])["replay"] = replay_hash
    with pytest.raises(ValueError, match="contamination detected"):
        validate_v5_evaluation_record(
            contaminated_record,
            corpus_payloads=contaminated_payloads,
            required_training_sources=_TRAINING_SOURCES,
        )


def test_v5_schema_rejects_broken_corpus_and_cell_links() -> None:
    record, _ = _valid_v5_record()

    duplicate_corpus = copy.deepcopy(record)
    cast(list[object], duplicate_corpus["corpora"]).append(
        copy.deepcopy(cast(list[object], duplicate_corpus["corpora"])[0])
    )
    with pytest.raises(ValidationError, match="corpus names must be unique"):
        V5EvaluationRecord.model_validate(duplicate_corpus)

    missing_fit = copy.deepcopy(record)
    cast(dict[str, object], missing_fit["unigram_fit"])["training_corpus_name"] = (
        "missing"
    )
    with pytest.raises(ValidationError, match="fit corpus is missing"):
        V5EvaluationRecord.model_validate(missing_fit)

    duplicate_cell = copy.deepcopy(record)
    comparisons = cast(list[object], duplicate_cell["comparisons"])
    duplicate_cell["comparisons"] = [comparisons[0], copy.deepcopy(comparisons[0])]
    with pytest.raises(ValidationError, match="exactly both"):
        V5EvaluationRecord.model_validate(duplicate_cell)

    extra_cell = copy.deepcopy(record)
    cast(list[object], extra_cell["comparisons"]).append(
        copy.deepcopy(cast(list[object], extra_cell["comparisons"])[0])
    )
    with pytest.raises(ValidationError, match="exactly both"):
        V5EvaluationRecord.model_validate(extra_cell)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corpus_name", "zz_missing"),
        ("corpus_name", "aa_missing"),
        ("corpus_sha256", "0" * 64),
        ("corpus_sha256", "f" * 64),
        ("scored_utf8_bytes", 1),
        ("scored_utf8_bytes", 99),
    ],
)
def test_v5_schema_rejects_metric_corpus_mismatch_in_both_directions(
    field: str, value: object
) -> None:
    record, _ = _valid_v5_record()
    comparison = cast(list[dict[str, object]], record["comparisons"])[0]
    for metric_name in ("model_metric", "baseline_metric"):
        metric = cast(dict[str, object], comparison[metric_name])
        metric[field] = value
        if field == "scored_utf8_bytes":
            metric["total_cross_entropy_bits"] = (
                cast(float, metric["bits_per_byte"]) * cast(int, value)
            )
    with pytest.raises(ValidationError, match="exact held-out corpus|corpus is missing"):
        V5EvaluationRecord.model_validate(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [("training_corpus_sha256", "0" * 64), ("training_corpus_sha256", "f" * 64),
     ("training_utf8_bytes", 1), ("training_utf8_bytes", 99)],
)
def test_v5_schema_rejects_unigram_fit_mismatch_in_both_directions(
    field: str, value: object
) -> None:
    record, _ = _valid_v5_record()
    cast(dict[str, object], record["unigram_fit"])[field] = value
    with pytest.raises(ValidationError, match="fit evidence does not match"):
        V5EvaluationRecord.model_validate(record)


def test_v5_schema_rejects_overall_release_flag_in_both_directions() -> None:
    record, _ = _valid_v5_record()
    record["release_gate_passed"] = False
    with pytest.raises(ValidationError, match="overall release flag"):
        V5EvaluationRecord.model_validate(record)

    record, _ = _valid_v5_record()
    for comparison in cast(list[dict[str, object]], record["comparisons"]):
        model_metric = cast(dict[str, object], comparison["model_metric"])
        model_metric["bits_per_byte"] = 4.0
        model_metric["total_cross_entropy_bits"] = 4.0 * cast(
            int, model_metric["scored_utf8_bytes"]
        )
        comparison["passed"] = False
    with pytest.raises(ValidationError, match="overall release flag"):
        V5EvaluationRecord.model_validate(record)


@pytest.mark.parametrize("required", [(), ("identity_chat", "identity_chat")])
def test_v5_validation_rejects_invalid_required_source_sets(
    required: tuple[str, ...],
) -> None:
    record, payloads = _valid_v5_record()
    with pytest.raises(ValueError, match="non-empty and unique"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads=payloads,
            required_training_sources=required,
        )


def test_v5_validation_rejects_record_and_byte_count_mismatches() -> None:
    record, payloads = _valid_v5_record()
    training = next(
        item
        for item in cast(list[dict[str, object]], record["corpora"])
        if item["name"] == "training_targets"
    )
    training["records"] = 2
    with pytest.raises(ValueError, match="record count mismatch"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads=payloads,
            required_training_sources=_TRAINING_SOURCES,
        )

    for claimed_bytes in (1, 99):
        record, payloads = _valid_v5_record()
        training = next(
            item
            for item in cast(list[dict[str, object]], record["corpora"])
            if item["name"] == "training_targets"
        )
        training["utf8_bytes"] = claimed_bytes
        cast(dict[str, object], record["unigram_fit"])["training_utf8_bytes"] = (
            claimed_bytes
        )
        with pytest.raises(ValueError, match="UTF-8 byte count mismatch"):
            validate_v5_evaluation_record(
                record,
                corpus_payloads=payloads,
                required_training_sources=_TRAINING_SOURCES,
            )


def test_v5_validation_rejects_incomplete_or_mismatched_audits() -> None:
    record, payloads = _valid_v5_record()
    cast(list[object], record["contamination_audits"]).pop()
    with pytest.raises(ValueError, match="both held-out corpora"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads=payloads,
            required_training_sources=_TRAINING_SOURCES,
        )

    record, payloads = _valid_v5_record()
    audit = cast(list[dict[str, object]], record["contamination_audits"])[0]
    cast(dict[str, str], audit["training_source_sha256"]).pop("replay")
    with pytest.raises(ValueError, match="every required source"):
        validate_v5_evaluation_record(
            record,
            corpus_payloads=payloads,
            required_training_sources=_TRAINING_SOURCES,
        )

    for digest in ("0" * 64, "f" * 64):
        record, payloads = _valid_v5_record()
        audit = cast(list[dict[str, object]], record["contamination_audits"])[0]
        audit["heldout_corpus_sha256"] = digest
        with pytest.raises(ValueError, match="held-out hash mismatch"):
            validate_v5_evaluation_record(
                record,
                corpus_payloads=payloads,
                required_training_sources=_TRAINING_SOURCES,
            )

        record, payloads = _valid_v5_record()
        audit = cast(list[dict[str, object]], record["contamination_audits"])[0]
        cast(dict[str, str], audit["training_source_sha256"])["replay"] = digest
        with pytest.raises(ValueError, match="training-source hash mismatch"):
            validate_v5_evaluation_record(
                record,
                corpus_payloads=payloads,
                required_training_sources=_TRAINING_SOURCES,
            )


def test_lane2_config_is_disabled_strict_and_packaged_byte_identical(tmp_path: Path) -> None:
    settings = load_standard_evaluation_settings()
    assert settings.feature.enabled is False
    assert settings.byte_unigram.add_k == 1.0
    assert settings.release_gate.required_cells == (
        "heldout_answers",
        "heldout_text",
    )
    repository = _ROOT / "configs/smollm2_evaluation.yaml"
    packaged = _ROOT / "aptivra/resources/configs/smollm2_evaluation.yaml"
    assert repository.read_bytes() == packaged.read_bytes()
    assert load_standard_evaluation_settings(str(repository)) == settings
    assert (
        load_standard_base_settings().schema_boundary.new_evaluation_schema
        == "aptivra.model_card_eval.v5"
    )

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only 'smollm2_evaluation'"):
        load_standard_evaluation_settings(invalid)


def test_corpus_canonicalization_is_ordered_utf8_and_rejects_empty() -> None:
    assert canonical_corpus_payload(["é", "x"]) == '["é","x"]'.encode()
    assert corpus_sha256(["a", "b"]) != corpus_sha256(["b", "a"])
    with pytest.raises(ValueError, match="non-empty"):
        canonical_corpus_payload([])
