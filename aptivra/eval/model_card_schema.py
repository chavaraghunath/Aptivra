"""Strict schema and cross-file validation for model-card evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVALUATION_SCHEMA_VERSION = "aptivra.model_card_eval.v4"
PROVENANCE_SCHEMA_VERSION = "aptivra.conversion_provenance.v1"
PROVENANCE_FILENAME = "conversion-provenance.json"

FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


def _sha256_file(path: Path | Traversable) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MetricSummary(_StrictModel):
    """Finite aggregate over a non-empty collection of scored items."""

    mean: FiniteNonNegative
    minimum: FiniteNonNegative = Field(alias="min")
    maximum: FiniteNonNegative = Field(alias="max")
    samples: PositiveInt

    @model_validator(mode="after")
    def validate_order(self) -> MetricSummary:
        if not self.minimum <= self.mean <= self.maximum:
            raise ValueError("metric summary must satisfy min <= mean <= max")
        return self


class MetricBaselines(_StrictModel):
    """Like-for-like baselines summarized over the same scored items."""

    uniform_bpb: MetricSummary
    unigram_bpb: MetricSummary
    unigram_fit: Literal["evaluated_bytes"]


class AnswerSpanMetric(MetricSummary):
    """Answer-span score plus baselines fitted to those answer bytes."""

    evaluated_bytes: PositiveInt
    baselines: MetricBaselines

    @model_validator(mode="after")
    def validate_baseline_counts(self) -> AnswerSpanMetric:
        if self.baselines.uniform_bpb.samples != self.samples:
            raise ValueError("uniform baseline sample count must match metric samples")
        if self.baselines.unigram_bpb.samples != self.samples:
            raise ValueError("unigram baseline sample count must match metric samples")
        return self


class ScalarBaselines(_StrictModel):
    """Baselines for a single aggregate score over one exact byte sequence."""

    uniform_bpb: FiniteNonNegative
    unigram_bpb: FiniteNonNegative
    unigram_fit: Literal["evaluated_bytes"]


class WholeTextMetric(_StrictModel):
    """Whole-text score over all predicted bytes after the first input byte."""

    value: FiniteNonNegative
    evaluated_bytes: PositiveInt
    baselines: ScalarBaselines


class PromptSet(_StrictModel):
    name: str = Field(min_length=1)
    sha256: str
    samples: PositiveInt

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validated_sha256(value)


class TextSet(_StrictModel):
    """Identity and exact byte size of a versioned held-out text corpus."""

    name: str = Field(min_length=1)
    sha256: str
    bytes: PositiveInt

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validated_sha256(value)


class Evaluation(_StrictModel):
    moe_enabled: bool
    heldout_prompt_set: PromptSet
    heldout_text_set: TextSet
    trained_prompt_answer_bpb: AnswerSpanMetric
    heldout_prompt_answer_bpb: AnswerSpanMetric
    heldout_text_bpb: WholeTextMetric

    @model_validator(mode="after")
    def validate_heldout_counts(self) -> Evaluation:
        if self.heldout_prompt_set.samples != self.heldout_prompt_answer_bpb.samples:
            raise ValueError("held-out prompt-set and metric sample counts must match")
        if self.heldout_text_set.bytes - 1 != self.heldout_text_bpb.evaluated_bytes:
            raise ValueError("held-out text-set and metric byte counts must match")
        return self


class ModelMeasurements(_StrictModel):
    backbone_parameters: NonNegativeInt
    router_parameters: NonNegativeInt
    expert_parameters: NonNegativeInt
    total_parameters: PositiveInt
    config_parameter_count: PositiveInt
    context_length: PositiveInt
    vocab_size: Annotated[int, Field(strict=True, gt=1)]
    position_embedding: Literal["learned_absolute"]

    @model_validator(mode="after")
    def validate_parameter_totals(self) -> ModelMeasurements:
        component_total = (
            self.backbone_parameters + self.router_parameters + self.expert_parameters
        )
        if component_total != self.total_parameters:
            raise ValueError("model parameter components do not equal total_parameters")
        if self.config_parameter_count != self.total_parameters:
            raise ValueError("config parameter_count does not equal measured total_parameters")
        return self


class ConversionProvenance(_StrictModel):
    schema_version: Literal["aptivra.conversion_provenance.v1"]
    source_layout: Literal["native_huggingface", "legacy_multitier"]
    source_manifest_version: str | None
    source_shards_sha256: dict[str, str]
    source_hub_revision: str | None
    converter_version: str = Field(min_length=1)
    model_safetensors_sha256: str

    @field_validator("source_shards_sha256")
    @classmethod
    def validate_source_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for path, digest in value.items():
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("provenance source shard paths must be safe relative paths")
            _validated_sha256(digest)
        return value

    @field_validator("model_safetensors_sha256")
    @classmethod
    def validate_model_hash(cls, value: str) -> str:
        return _validated_sha256(value)

    @model_validator(mode="after")
    def validate_source_layout(self) -> ConversionProvenance:
        backbone_path = "dense/backbone.pt"
        if self.source_layout == "legacy_multitier":
            if self.source_manifest_version is None:
                raise ValueError("legacy provenance requires source_manifest_version")
            if backbone_path not in self.source_shards_sha256:
                raise ValueError("legacy provenance requires dense/backbone.pt hash")
        elif self.source_manifest_version is not None or self.source_shards_sha256:
            raise ValueError("native provenance must not claim legacy source shards")
        return self


class CheckpointEvidence(_StrictModel):
    model_safetensors_sha256: str
    config_sha256: str
    provenance_sha256: str
    backbone_sha256: str | None
    source_hub_revision: str | None
    provenance: ConversionProvenance

    @field_validator(
        "model_safetensors_sha256",
        "config_sha256",
        "provenance_sha256",
        "backbone_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validated_sha256(value)

    @model_validator(mode="after")
    def validate_provenance_consistency(self) -> CheckpointEvidence:
        provenance = self.provenance
        if provenance.model_safetensors_sha256 != self.model_safetensors_sha256:
            raise ValueError("provenance and checkpoint model hashes do not match")
        if provenance.source_hub_revision != self.source_hub_revision:
            raise ValueError("provenance and checkpoint source revisions do not match")
        expected_backbone = provenance.source_shards_sha256.get("dense/backbone.pt")
        if self.backbone_sha256 != expected_backbone:
            raise ValueError("checkpoint backbone hash does not match provenance")
        return self


class TokenizerEvidence(_StrictModel):
    padding_token_id: NonNegativeInt
    end_of_sequence_token_ids: list[NonNegativeInt]
    newline_stop_token_id: NonNegativeInt

    @model_validator(mode="after")
    def validate_stop_ids(self) -> TokenizerEvidence:
        if not self.end_of_sequence_token_ids:
            raise ValueError("end_of_sequence_token_ids must not be empty")
        if self.padding_token_id not in self.end_of_sequence_token_ids:
            raise ValueError("padding token must be an end-of-sequence token")
        if self.newline_stop_token_id not in self.end_of_sequence_token_ids:
            raise ValueError("newline stop token must be an end-of-sequence token")
        return self


class EvaluationRecord(_StrictModel):
    """Validated v4 record that is safe to use for release decisions."""

    schema_version: Literal["aptivra.model_card_eval.v4"]
    timestamp_utc: str
    checkpoint: CheckpointEvidence
    model: ModelMeasurements
    evaluation: Evaluation
    tokenizer: TokenizerEvidence

    @field_validator("timestamp_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("timestamp_utc must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp_utc must include a timezone")
        return value


def validate_evaluation_record(
    record: object,
    *,
    minimum_trained_samples: int,
    minimum_heldout_samples: int,
    minimum_heldout_text_bytes: int,
    export_dir: Path | None = None,
) -> EvaluationRecord:
    """Validate schema, bundled evidence identity, counts, and optional export files."""
    validated = EvaluationRecord.model_validate(record)

    from aptivra.eval.model_card_eval import (
        DEFAULT_HELDOUT_PROMPTS_PATH,
        DEFAULT_HELDOUT_TEXT_PATH,
    )

    prompt_set = validated.evaluation.heldout_prompt_set
    if prompt_set.name != DEFAULT_HELDOUT_PROMPTS_PATH.name:
        raise ValueError("held-out prompt-set name does not match the bundled set")
    if prompt_set.sha256 != _sha256_file(DEFAULT_HELDOUT_PROMPTS_PATH):
        raise ValueError("held-out prompt-set SHA-256 does not match the bundled set")
    if validated.evaluation.trained_prompt_answer_bpb.samples < minimum_trained_samples:
        raise ValueError("trained prompt sample count is below the configured minimum")
    if prompt_set.samples < minimum_heldout_samples:
        raise ValueError("held-out prompt sample count is below the configured minimum")
    text_set = validated.evaluation.heldout_text_set
    if text_set.name != DEFAULT_HELDOUT_TEXT_PATH.name:
        raise ValueError("held-out text-set name does not match the bundled set")
    if text_set.sha256 != _sha256_file(DEFAULT_HELDOUT_TEXT_PATH):
        raise ValueError("held-out text-set SHA-256 does not match the bundled set")
    actual_text_bytes = len(DEFAULT_HELDOUT_TEXT_PATH.read_bytes())
    if text_set.bytes != actual_text_bytes:
        raise ValueError("held-out text-set byte count does not match the bundled set")
    if text_set.bytes < minimum_heldout_text_bytes:
        raise ValueError("held-out text byte count is below the configured minimum")

    if export_dir is not None:
        _validate_export_files(validated, export_dir)
    return validated


def _validate_export_files(record: EvaluationRecord, export_dir: Path) -> None:
    from aptivra.eval.model_card_eval import count_export_parameters

    weights_path = export_dir / "model.safetensors"
    config_path = export_dir / "config.json"
    provenance_path = export_dir / PROVENANCE_FILENAME
    if _sha256_file(weights_path) != record.checkpoint.model_safetensors_sha256:
        raise ValueError("results record model.safetensors SHA-256 does not match export")
    if _sha256_file(config_path) != record.checkpoint.config_sha256:
        raise ValueError("results record config.json SHA-256 does not match export")
    if _sha256_file(provenance_path) != record.checkpoint.provenance_sha256:
        raise ValueError("results record provenance SHA-256 does not match export")

    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance = ConversionProvenance.model_validate(provenance_payload)
    if provenance != record.checkpoint.provenance:
        raise ValueError("results record provenance does not match export provenance")

    counts = count_export_parameters(export_dir)
    measured = record.model
    for field_name, expected in counts.items():
        if getattr(measured, field_name) != expected:
            raise ValueError(f"results record {field_name} does not match export")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("config.json must contain an object")
    if config_payload.get("parameter_count") != measured.config_parameter_count:
        raise ValueError("results record parameter count does not match config.json")
    if config_payload.get("max_seq_len") != measured.context_length:
        raise ValueError("results record context length does not match config.json")
    if config_payload.get("vocab_size") != measured.vocab_size:
        raise ValueError("results record vocabulary size does not match config.json")


def _validated_sha256(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError("SHA-256 values must contain exactly 64 hexadecimal characters")
    return value.lower()
