"""Verified offline stock-Llama loading for the ADR-0011 standard-base path."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

import torch
import yaml
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from aptivra.resources import aptivra_resource
from aptivra.tasks.base import TaskInstance

DEFAULT_STANDARD_BASE_CONFIG = aptivra_resource("configs", "smollm2_base.yaml")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_PATTERN = re.compile(r"[^/\s]+/[^/\s]+\Z")
_CACHE_REPOSITORY_PREFIX = "models--"
_IGNORE_INDEX = -100

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
TokenId = Annotated[int, Field(strict=True, ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FeatureSettings(_StrictModel):
    """Fail-closed feature switch for the unaccepted production path."""

    enabled: bool


class SchemaBoundary(_StrictModel):
    """Non-overlapping legacy and standard-base artifact identities."""

    legacy_identity: Literal["v0-byte-level"]
    legacy_evaluation_schema: Literal["aptivra.model_card_eval.v4"]
    legacy_provenance_schema: Literal["aptivra.conversion_provenance.v1"]
    new_identity: Literal["smollm2-standard-base-v1"]
    new_fingerprint_schema: Literal["aptivra.standard_base_fingerprint.v1"]
    new_evaluation_schema: Literal["aptivra.model_card_eval.v5"] = (
        "aptivra.model_card_eval.v5"
    )


class FileFingerprint(_StrictModel):
    """Content address for one required local snapshot file."""

    path: str = Field(min_length=1)
    bytes: PositiveInt
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1 or path.name != value:
            raise ValueError("fingerprint paths must be single safe filenames")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("fingerprint sha256 must be 64 lowercase hexadecimal characters")
        return value


class SourceFingerprint(_StrictModel):
    """Immutable repository identity and required local file manifest."""

    repository: str
    revision: str
    config: FileFingerprint
    weights: FileFingerprint
    tokenizer_assets: list[FileFingerprint] = Field(min_length=1)

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        if _REPOSITORY_PATTERN.fullmatch(value) is None:
            raise ValueError("source repository must be an owner/name identifier")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _REVISION_PATTERN.fullmatch(value) is None:
            raise ValueError("source revision must be a 40-character lowercase commit hash")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> SourceFingerprint:
        if self.config.path != "config.json":
            raise ValueError("source config path must be config.json")
        if self.weights.path != "model.safetensors":
            raise ValueError("source weights path must be model.safetensors")
        paths = [asset.path for asset in self.tokenizer_assets]
        if len(paths) != len(set(paths)):
            raise ValueError("tokenizer asset paths must be unique")
        if "tokenizer_config.json" not in paths:
            raise ValueError("tokenizer assets must include tokenizer_config.json")
        return self


class ArchitectureFingerprint(_StrictModel):
    """Allocation-free Llama architecture and safetensors fingerprint."""

    architectures: list[Literal["LlamaForCausalLM"]]
    model_type: Literal["llama"]
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    vocab_size: PositiveInt
    max_position_embeddings: PositiveInt
    bos_token_id: TokenId
    eos_token_id: TokenId
    tie_word_embeddings: Literal[True]
    unique_parameter_count: PositiveInt
    tensor_count: PositiveInt
    tensor_dtype: Literal["BF16", "F32"]
    separate_lm_head_weight: Literal[False]

    @model_validator(mode="after")
    def validate_attention_shape(self) -> ArchitectureFingerprint:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return self


class TokenizerFingerprint(_StrictModel):
    """Pinned tokenizer class and special-token semantics."""

    tokenizer_class: str = Field(min_length=1)
    bos_token: str = Field(min_length=1)
    eos_token: str = Field(min_length=1)
    unk_token: str = Field(min_length=1)
    pad_token: None
    chat_template: None
    standalone_chat_template: Literal[False]
    required_added_tokens: dict[TokenId, str] = Field(min_length=1)

    @field_validator("required_added_tokens")
    @classmethod
    def validate_added_tokens(cls, value: dict[int, str]) -> dict[int, str]:
        if any(not token for token in value.values()):
            raise ValueError("required added-token strings must be non-empty")
        return value


class RuntimeSettings(_StrictModel):
    """Operator-owned fp32 length and stopping bounds."""

    dtype: Literal["float32"]
    max_prompt_tokens: PositiveInt
    max_sequence_tokens: PositiveInt
    max_new_tokens: PositiveInt
    stop_policy: Literal["eos_only"]
    task_stop_policy: Literal["decoded_text"] = "decoded_text"

    @model_validator(mode="after")
    def validate_lengths(self) -> RuntimeSettings:
        if self.max_prompt_tokens >= self.max_sequence_tokens:
            raise ValueError("max_prompt_tokens must be less than max_sequence_tokens")
        if self.max_prompt_tokens + self.max_new_tokens > self.max_sequence_tokens:
            raise ValueError(
                "max_prompt_tokens plus max_new_tokens must not exceed max_sequence_tokens"
            )
        return self


class StandardBaseSettings(_StrictModel):
    """Complete versioned contract for the standard-base vertical slice."""

    schema_version: Literal["aptivra.standard_base_config.v1"]
    feature: FeatureSettings
    schema_boundary: SchemaBoundary
    source: SourceFingerprint
    architecture: ArchitectureFingerprint
    tokenizer: TokenizerFingerprint
    runtime: RuntimeSettings

    @model_validator(mode="after")
    def validate_context_contract(self) -> StandardBaseSettings:
        if self.runtime.max_sequence_tokens > self.architecture.max_position_embeddings:
            raise ValueError("max_sequence_tokens exceeds the architecture context length")
        return self


@dataclass(frozen=True)
class VerifiedSnapshot:
    """Snapshot identity proven without constructing model tensors."""

    path: Path
    repository: str
    revision: str
    unique_parameter_count: int
    tensor_count: int


@dataclass(frozen=True)
class StandardBase:
    """Loaded stock model/tokenizer pair and the settings that authorize it."""

    model: LlamaForCausalLM
    tokenizer: PreTrainedTokenizerBase
    settings: StandardBaseSettings
    snapshot: VerifiedSnapshot
    device: torch.device


@dataclass(frozen=True)
class CausalBatch:
    """Right-padded target-only causal-language-model batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    """Observable deterministic generation outcome."""

    text: str
    generated_token_ids: tuple[int, ...]
    stop_reason: Literal["eos", "terminator", "length"]
    matched_terminator: str | None = None


def _validated_stop_sequences(stop_sequences: tuple[str, ...]) -> tuple[str, ...]:
    if any(not sequence for sequence in stop_sequences):
        raise ValueError("decoded stop sequences must be non-empty")
    if len(stop_sequences) != len(set(stop_sequences)):
        raise ValueError("decoded stop sequences must be unique")
    return stop_sequences


def _first_decoded_terminator(
    text: str, stop_sequences: tuple[str, ...]
) -> tuple[int, str] | None:
    payload = text.encode("utf-8")
    matches = (
        (payload.find(sequence.encode("utf-8")), index, sequence)
        for index, sequence in enumerate(stop_sequences)
    )
    present = [match for match in matches if match[0] >= 0]
    if not present:
        return None
    byte_offset, _, sequence = min(present)
    return byte_offset, sequence


class DecodedTextStoppingCriteria(StoppingCriteria):
    """Stop after a complete decoded terminator, independent of token boundaries."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        prompt_length: int,
        stop_sequences: tuple[str, ...],
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        self._tokenizer = tokenizer
        self._prompt_length = prompt_length
        self._stop_sequences = _validated_stop_sequences(stop_sequences)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: tuple[torch.FloatTensor, ...] | None,
        **kwargs: object,
    ) -> torch.BoolTensor:
        del scores, kwargs
        decisions: list[bool] = []
        for row in input_ids:
            generated_ids = [int(token_id) for token_id in row[self._prompt_length :]]
            decoded = cast(
                str,
                self._tokenizer.decode(generated_ids, skip_special_tokens=False),
            )
            decisions.append(
                _first_decoded_terminator(decoded, self._stop_sequences) is not None
            )
        return cast(
            torch.BoolTensor,
            torch.tensor(decisions, dtype=torch.bool, device=input_ids.device),
        )


class _GreedyGenerator(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor:
        """Return generated token IDs for a stock causal language model."""
        ...


def load_standard_base_settings(
    path: Path | str | Traversable = DEFAULT_STANDARD_BASE_CONFIG,
) -> StandardBaseSettings:
    """Load and strictly validate the versioned standard-base YAML."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("standard-base config root must be a YAML mapping")
    if set(raw) != {"standard_base"}:
        raise ValueError("standard-base config root must contain only 'standard_base'")
    return StandardBaseSettings.model_validate(raw["standard_base"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(snapshot: Path, fingerprint: FileFingerprint) -> Path:
    path = snapshot / fingerprint.path
    if not path.is_file():
        raise ValueError(f"required snapshot file is missing: {fingerprint.path}")
    size = path.stat().st_size
    if size != fingerprint.bytes:
        raise ValueError(
            f"snapshot file size mismatch for {fingerprint.path}: "
            f"expected {fingerprint.bytes}, got {size}"
        )
    digest = _sha256(path)
    if digest != fingerprint.sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch for {fingerprint.path}: "
            f"expected {fingerprint.sha256}, got {digest}"
        )
    return path


def _repository_from_cache_path(snapshot: Path) -> str:
    if snapshot.parent.name != "snapshots":
        raise ValueError("snapshot path must be an immutable Hugging Face cache snapshot")
    cache_name = snapshot.parent.parent.name
    if not cache_name.startswith(_CACHE_REPOSITORY_PREFIX):
        raise ValueError("snapshot path does not identify a Hugging Face repository")
    encoded = cache_name.removeprefix(_CACHE_REPOSITORY_PREFIX)
    parts = encoded.split("--")
    if len(parts) != 2 or not all(parts):
        raise ValueError("snapshot cache repository identity is malformed")
    return "/".join(parts)


def _require_json_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return cast(dict[str, object], raw)


def _verify_architecture(
    config: dict[str, object], expected: ArchitectureFingerprint
) -> None:
    if config.get("auto_map") is not None:
        raise ValueError("snapshot config must not declare remote-code auto_map entries")
    expected_values: dict[str, object] = {
        "architectures": list(expected.architectures),
        "model_type": expected.model_type,
        "hidden_size": expected.hidden_size,
        "intermediate_size": expected.intermediate_size,
        "num_hidden_layers": expected.num_hidden_layers,
        "num_attention_heads": expected.num_attention_heads,
        "num_key_value_heads": expected.num_key_value_heads,
        "vocab_size": expected.vocab_size,
        "max_position_embeddings": expected.max_position_embeddings,
        "bos_token_id": expected.bos_token_id,
        "eos_token_id": expected.eos_token_id,
        "tie_word_embeddings": expected.tie_word_embeddings,
    }
    for key, expected_value in expected_values.items():
        actual = config.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"architecture fingerprint mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )


def _verify_tokenizer(
    snapshot: Path,
    tokenizer_config: dict[str, object],
    expected: TokenizerFingerprint,
) -> None:
    expected_values: dict[str, object] = {
        "tokenizer_class": expected.tokenizer_class,
        "bos_token": expected.bos_token,
        "eos_token": expected.eos_token,
        "unk_token": expected.unk_token,
        "pad_token": expected.pad_token,
        "chat_template": expected.chat_template,
    }
    for key, expected_value in expected_values.items():
        actual = tokenizer_config.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"tokenizer fingerprint mismatch for {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    added_tokens = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(added_tokens, dict):
        raise ValueError("tokenizer added_tokens_decoder must be a JSON object")
    for token_id, token in expected.required_added_tokens.items():
        record = added_tokens.get(str(token_id))
        if not isinstance(record, dict) or record.get("content") != token:
            raise ValueError(f"tokenizer fingerprint mismatch for added token {token_id}")
    standalone_present = (snapshot / "chat_template.jinja").is_file()
    if standalone_present != expected.standalone_chat_template:
        raise ValueError("tokenizer standalone chat-template fingerprint mismatch")


def _verify_tensor_inventory(
    weights_path: Path, expected: ArchitectureFingerprint
) -> tuple[int, int]:
    parameter_count = 0
    tensor_count = 0
    with safe_open(weights_path, framework="pt", device="cpu") as tensors:
        names = set(tensors.keys())
        for name in names:
            tensor = tensors.get_slice(name)
            if str(tensor.get_dtype()) != expected.tensor_dtype:
                raise ValueError(
                    f"weights dtype mismatch for {name}: expected {expected.tensor_dtype}, "
                    f"got {tensor.get_dtype()}"
                )
            parameter_count += math.prod(tensor.get_shape())
            tensor_count += 1
    separate_lm_head = "lm_head.weight" in names
    if separate_lm_head != expected.separate_lm_head_weight:
        raise ValueError("weights tied-head fingerprint mismatch")
    if parameter_count != expected.unique_parameter_count:
        raise ValueError(
            "weights parameter-count fingerprint mismatch: "
            f"expected {expected.unique_parameter_count}, got {parameter_count}"
        )
    if tensor_count != expected.tensor_count:
        raise ValueError(
            f"weights tensor-count fingerprint mismatch: expected {expected.tensor_count}, "
            f"got {tensor_count}"
        )
    return parameter_count, tensor_count


def verify_snapshot(
    settings: StandardBaseSettings,
    snapshot_path: Path | str | None = None,
) -> VerifiedSnapshot:
    """Verify identity, bytes, architecture, and tokenizer before model allocation."""
    if snapshot_path is None:
        snapshot_path = snapshot_download(
            repo_id=settings.source.repository,
            revision=settings.source.revision,
            local_files_only=True,
        )
    snapshot = Path(snapshot_path).absolute()
    if snapshot.name != settings.source.revision:
        raise ValueError(
            f"snapshot revision mismatch: expected {settings.source.revision}, "
            f"got {snapshot.name}"
        )
    repository = _repository_from_cache_path(snapshot)
    if repository != settings.source.repository:
        raise ValueError(
            f"snapshot repository mismatch: expected {settings.source.repository}, "
            f"got {repository}"
        )

    config_path = _verify_file(snapshot, settings.source.config)
    weights_path = _verify_file(snapshot, settings.source.weights)
    tokenizer_paths = {
        asset.path: _verify_file(snapshot, asset)
        for asset in settings.source.tokenizer_assets
    }
    config = _require_json_mapping(config_path, "model config")
    _verify_architecture(config, settings.architecture)
    tokenizer_config = _require_json_mapping(
        tokenizer_paths["tokenizer_config.json"], "tokenizer config"
    )
    _verify_tokenizer(snapshot, tokenizer_config, settings.tokenizer)
    parameter_count, tensor_count = _verify_tensor_inventory(
        weights_path, settings.architecture
    )
    return VerifiedSnapshot(
        path=snapshot,
        repository=repository,
        revision=snapshot.name,
        unique_parameter_count=parameter_count,
        tensor_count=tensor_count,
    )


def resolve_standard_base_device(requested: str) -> torch.device:
    """Resolve auto/explicit placement without silently changing explicit requests."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_standard_base(
    settings: StandardBaseSettings,
    *,
    snapshot_path: Path | str | None = None,
    device: str = "auto",
) -> StandardBase:
    """Load the verified stock Llama/tokenizer pair locally in fp32."""
    if not settings.feature.enabled:
        raise RuntimeError(
            "ADR-0011 standard-base path is disabled until final local acceptance"
        )
    snapshot = verify_snapshot(settings, snapshot_path)
    common: dict[str, object] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(snapshot.path, **common)
    if not isinstance(config, LlamaConfig):
        raise ValueError("verified snapshot did not load as stock LlamaConfig")
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(snapshot.path, use_fast=False, **common),
    )
    if type(tokenizer).__name__ != settings.tokenizer.tokenizer_class:
        raise ValueError(
            "loaded tokenizer class mismatch: "
            f"expected {settings.tokenizer.tokenizer_class}, got {type(tokenizer).__name__}"
        )
    if len(tokenizer) != settings.architecture.vocab_size:
        raise ValueError("loaded tokenizer vocabulary does not match the fingerprint")
    if tokenizer.eos_token_id != settings.architecture.eos_token_id:
        raise ValueError("loaded tokenizer EOS ID does not match the architecture")
    if tokenizer.pad_token_id is not None:
        raise ValueError("standard-base tokenizer must not define a pad token")

    loaded = AutoModelForCausalLM.from_pretrained(
        snapshot.path,
        config=config,
        dtype=torch.float32,
        **common,
    )
    if not isinstance(loaded, LlamaForCausalLM):
        raise ValueError("verified snapshot did not load as stock LlamaForCausalLM")
    resolved_device = resolve_standard_base_device(device)
    module = cast(torch.nn.Module, loaded)
    module.to(device=resolved_device, dtype=torch.float32)
    module.eval()
    model = loaded
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise ValueError("standard-base loader failed to cast every floating parameter to fp32")
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.lm_head
    if input_embeddings.weight is not output_embeddings.weight:
        raise ValueError("standard-base loader did not preserve tied input/output weights")
    loaded_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if loaded_parameter_count != snapshot.unique_parameter_count:
        raise ValueError("loaded unique parameter count does not match the verified snapshot")
    return StandardBase(
        model=model,
        tokenizer=tokenizer,
        settings=settings,
        snapshot=snapshot,
        device=resolved_device,
    )


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded]


def collate_task_instances(
    base: StandardBase,
    instances: list[TaskInstance],
) -> CausalBatch:
    """Materialize target-only labels with EOS padding masked from attention/loss."""
    if not instances:
        raise ValueError("cannot collate an empty task batch")
    eos_id = base.tokenizer.eos_token_id
    if not isinstance(eos_id, int):
        raise ValueError("standard-base tokenizer must define an integer EOS ID")

    materialized: list[tuple[TaskInstance, list[int], list[int]]] = []
    for instance in instances:
        prompt_ids = _encode(base.tokenizer, instance.prompt)
        target_ids = _encode(base.tokenizer, instance.training_target())
        if not prompt_ids:
            raise ValueError(f"task {instance.task_id!r} prompt tokenized to zero tokens")
        if not target_ids or target_ids[-1] != eos_id:
            target_ids.append(eos_id)
        prompt_length = len(prompt_ids)
        target_length = len(target_ids)
        total_length = prompt_length + target_length
        if (
            prompt_length > base.settings.runtime.max_prompt_tokens
            or total_length > base.settings.runtime.max_sequence_tokens
        ):
            raise ValueError(
                f"task {instance.task_id!r} exceeds configured token limits: "
                f"prompt={prompt_length}, target={target_length}, total={total_length}, "
                f"max_prompt={base.settings.runtime.max_prompt_tokens}, "
                f"max_sequence={base.settings.runtime.max_sequence_tokens}"
            )
        ids = [*prompt_ids, *target_ids]
        labels = [_IGNORE_INDEX] * prompt_length + target_ids
        materialized.append((instance, ids, labels))

    batch_length = max(len(ids) for _, ids, _ in materialized)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for _, ids, labels in materialized:
        padding = batch_length - len(ids)
        input_rows.append(ids + [eos_id] * padding)
        mask_rows.append([1] * len(ids) + [0] * padding)
        label_rows.append(labels + [_IGNORE_INDEX] * padding)
    return CausalBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long, device=base.device),
        attention_mask=torch.tensor(mask_rows, dtype=torch.long, device=base.device),
        labels=torch.tensor(label_rows, dtype=torch.long, device=base.device),
        task_ids=tuple(instance.task_id for instance, _, _ in materialized),
    )


def forward_task_batch(base: StandardBase, batch: CausalBatch) -> CausalLMOutputWithPast:
    """Run the stock raw-task causal forward path with target-only labels."""
    return cast(
        CausalLMOutputWithPast,
        base.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            return_dict=True,
        ),
    )


def generate_task_answer(
    base: StandardBase,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    stop_sequences: tuple[str, ...] = (),
) -> GenerationResult:
    """Greedily generate until EOS, a complete decoded terminator, or the limit."""
    prompt_ids = _encode(base.tokenizer, prompt)
    if not prompt_ids:
        raise ValueError("generation prompt tokenized to zero tokens")
    requested_tokens = (
        base.settings.runtime.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    if isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= requested_tokens <= base.settings.runtime.max_new_tokens:
        raise ValueError(
            f"max_new_tokens must be between 1 and {base.settings.runtime.max_new_tokens}"
        )
    prompt_length = len(prompt_ids)
    if (
        prompt_length > base.settings.runtime.max_prompt_tokens
        or prompt_length + requested_tokens > base.settings.runtime.max_sequence_tokens
    ):
        raise ValueError(
            "generation input exceeds configured token limits: "
            f"prompt={prompt_length}, requested_new={requested_tokens}, "
            f"max_prompt={base.settings.runtime.max_prompt_tokens}, "
            f"max_sequence={base.settings.runtime.max_sequence_tokens}"
        )
    eos_id = base.tokenizer.eos_token_id
    if not isinstance(eos_id, int):
        raise ValueError("standard-base tokenizer must define an integer EOS ID")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=base.device)
    attention_mask = torch.ones_like(input_ids)
    validated_stops = _validated_stop_sequences(stop_sequences)
    stopping_criteria = (
        StoppingCriteriaList(
            [
                DecodedTextStoppingCriteria(
                    base.tokenizer,
                    prompt_length=prompt_length,
                    stop_sequences=validated_stops,
                )
            ]
        )
        if validated_stops
        else None
    )
    generation_options: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": False,
        "max_new_tokens": requested_tokens,
        "eos_token_id": eos_id,
        "pad_token_id": eos_id,
        "use_cache": True,
    }
    if stopping_criteria is not None:
        generation_options["stopping_criteria"] = stopping_criteria
    with torch.inference_mode():
        generator = cast(_GreedyGenerator, base.model)
        output = generator.generate(**generation_options)
    generated = tuple(int(token_id) for token_id in output[0, prompt_length:].tolist())
    ended_with_eos = bool(generated and generated[-1] == eos_id)
    text_ids = generated[:-1] if ended_with_eos else generated
    decoded = cast(
        str,
        base.tokenizer.decode(list(text_ids), skip_special_tokens=False),
    )
    terminator = _first_decoded_terminator(decoded, validated_stops)
    matched_terminator: str | None = None
    if terminator is not None:
        byte_offset, matched_terminator = terminator
        text = decoded.encode("utf-8")[:byte_offset].decode("utf-8")
        stop_reason: Literal["eos", "terminator", "length"] = "terminator"
    else:
        text = decoded
        stop_reason = "eos" if ended_with_eos else "length"
    return GenerationResult(
        text=text,
        generated_token_ids=generated,
        stop_reason=stop_reason,
        matched_terminator=matched_terminator,
    )


def generate_task_instance_answer(
    base: StandardBase,
    instance: TaskInstance,
    *,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    """Generate one task answer with that task family's decoded-text stop contract."""
    return generate_task_answer(
        base,
        instance.prompt,
        max_new_tokens=max_new_tokens,
        stop_sequences=instance.answer_stop_characters,
    )
