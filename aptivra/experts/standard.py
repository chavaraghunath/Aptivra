"""Content-addressed PEFT LoRA experts for the ADR-0011 standard-base path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aptivra.inference.standard_base import (
    ArchitectureFingerprint,
    StandardBase,
    StandardBaseSettings,
)
from aptivra.resources import aptivra_resource

DEFAULT_STANDARD_EXPERT_CONFIG = aptivra_resource("configs", "smollm2_experts.yaml")
_ALL_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_ADAPTER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
FilesystemPath: TypeAlias = Path | str
ReadablePath: TypeAlias = FilesystemPath | Traversable


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExpertFeatureSettings(_StrictModel):
    """Fail-closed feature switch for the unaccepted expert path."""

    enabled: bool


class PeftDependencySettings(_StrictModel):
    """Reviewed runtime dependency identity and compatible license."""

    version: Literal["0.21.2"]
    license: Literal["Apache-2.0"]


class ExpertLoRASettings(_StrictModel):
    """The selected rank-16, all-seven PEFT LoRA contract."""

    rank: Literal[16]
    alpha: PositiveInt
    dropout: float = Field(strict=True, ge=0.0, le=1.0)
    bias: Literal["none"]
    target_modules: list[str] = Field(min_length=7, max_length=7)
    max_adapters: PositiveInt

    @model_validator(mode="after")
    def validate_lora_contract(self) -> ExpertLoRASettings:
        if self.dropout != 0.0:
            raise ValueError("ADR-0011 lane 4 requires deterministic LoRA dropout=0.0")
        if tuple(self.target_modules) != _ALL_LORA_TARGETS:
            raise ValueError("PEFT expert must target all seven Llama projection families")
        return self


class RouterSettings(_StrictModel):
    """Registered sequence-level router fitting and abstention controls."""

    confidence_threshold: float = Field(strict=True, gt=0.0, le=1.0)
    learning_rate: float = Field(strict=True, gt=0.0, le=1.0)
    weight_decay: float = Field(strict=True, ge=0.0, le=1.0)
    adam_beta1: float = Field(strict=True, gt=0.0, lt=1.0)
    adam_beta2: float = Field(strict=True, gt=0.0, lt=1.0)
    adam_epsilon: float = Field(strict=True, gt=0.0, le=1.0)
    max_optimizer_steps: PositiveInt = Field(le=10_000)
    replay_examples_per_expert: PositiveInt = Field(le=10_000)


class GrowthPolicySettings(_StrictModel):
    """Registered real-evidence expert-growth thresholds."""

    verifier_failure_threshold: float = Field(strict=True, ge=0.0, le=1.0)
    mean_entropy_threshold: float = Field(strict=True, ge=0.0)
    minimum_observations: PositiveInt = Field(le=100_000)


class ConsolidationSettings(_StrictModel):
    """Bounded adapter/replay/promotion transaction controls."""

    adapter_learning_rate: float = Field(strict=True, gt=0.0, le=1.0)
    adapter_weight_decay: float = Field(strict=True, ge=0.0, le=1.0)
    adapter_adam_beta1: float = Field(strict=True, gt=0.0, lt=1.0)
    adapter_adam_beta2: float = Field(strict=True, gt=0.0, lt=1.0)
    adapter_adam_epsilon: float = Field(strict=True, gt=0.0, le=1.0)
    max_adapter_optimizer_steps: PositiveInt = Field(le=100_000)
    max_target_tokens: PositiveInt
    replay_capacity: PositiveInt = Field(le=50_000)
    replay_ratio: float = Field(strict=True, gt=0.0, le=1.0)
    replay_sampling_seed: NonNegativeInt = Field(le=2**31 - 1)
    replay_per_family_minimum: PositiveInt
    replay_byte_limit: PositiveInt
    replay_eviction_policy: Literal["stratified_hash_reservoir"]
    promotion_alpha: float = Field(strict=True, gt=0.0, lt=1.0)
    minimum_regression_samples: PositiveInt = Field(ge=2)
    minimum_validation_gain: float = Field(strict=True, ge=0.0, le=1.0)
    loss_noninferiority_margin: float = Field(strict=True, ge=0.0)

    @model_validator(mode="after")
    def validate_replay_minimum(self) -> ConsolidationSettings:
        if self.replay_per_family_minimum > self.replay_capacity:
            raise ValueError("consolidation replay per-family minimum exceeds capacity")
        return self


class StandardExpertSettings(_StrictModel):
    """Complete disabled-by-default lane-4/5 expert configuration."""

    schema_version: Literal["aptivra.standard_expert_config.v2"]
    feature: ExpertFeatureSettings
    peft: PeftDependencySettings
    lora: ExpertLoRASettings
    router: RouterSettings
    growth: GrowthPolicySettings
    consolidation: ConsolidationSettings


class ExpertManifest(_StrictModel):
    """Immutable content and compatibility contract for one promoted expert."""

    schema_version: Literal["aptivra.peft_expert_manifest.v1"]
    adapter_name: str
    adapter_subdir: str
    base_repository: str = Field(min_length=1)
    base_revision: str
    base_config_sha256: str
    base_weights_sha256: str
    base_fingerprint: str
    rank: Literal[16]
    target_modules: tuple[str, ...]
    parameter_count: PositiveInt
    peft_version: Literal["0.21.2"]
    peft_license: Literal["Apache-2.0"]
    content_sha256: str

    @field_validator("target_modules", mode="before")
    @classmethod
    def normalize_target_modules(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("adapter_name", "adapter_subdir")
    @classmethod
    def validate_adapter_name(cls, value: str) -> str:
        if _ADAPTER_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("adapter names must contain only letters, digits, underscore, or dash")
        return value

    @field_validator("base_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("base revision must be a 40-character lowercase commit hash")
        return value

    @field_validator(
        "base_config_sha256",
        "base_weights_sha256",
        "base_fingerprint",
        "content_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("manifest hashes must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> ExpertManifest:
        if self.adapter_subdir != self.adapter_name:
            raise ValueError("adapter subdirectory must equal the adapter name")
        if self.target_modules != _ALL_LORA_TARGETS:
            raise ValueError("manifest target modules must contain the all-seven contract")
        return self


class _LlamaArchitecture(Protocol):
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


class _AdapterLayerController(Protocol):
    def enable_adapter_layers(self) -> None:
        """Enable adapter contributions in the forward pass."""

    def disable_adapter_layers(self) -> None:
        """Disable adapter contributions in the forward pass."""


def load_standard_expert_settings(
    path: ReadablePath = DEFAULT_STANDARD_EXPERT_CONFIG,
) -> StandardExpertSettings:
    """Load and strictly validate the packaged lane-4 expert configuration."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("standard expert config root must be a YAML mapping")
    if set(raw) != {"smollm2_experts"}:
        raise ValueError("standard expert config root must contain only 'smollm2_experts'")
    return StandardExpertSettings.model_validate(raw["smollm2_experts"])


def expected_lora_parameter_count(
    architecture: _LlamaArchitecture | ArchitectureFingerprint,
    *,
    rank: int,
) -> int:
    """Return exact all-seven LoRA parameters for a Llama architecture."""
    if isinstance(rank, bool) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    hidden = architecture.hidden_size
    if hidden % architecture.num_attention_heads != 0:
        raise ValueError("hidden size must be divisible by query heads")
    head_dim = hidden // architecture.num_attention_heads
    key_value_width = head_dim * architecture.num_key_value_heads
    attention_per_rank = 6 * hidden + 2 * key_value_width
    mlp_per_rank = 3 * (hidden + architecture.intermediate_size)
    return architecture.num_hidden_layers * (attention_per_rank + mlp_per_rank) * rank


def standard_base_fingerprint(settings: StandardBaseSettings) -> str:
    """Bind an expert to the immutable repository, revision, config, and weights."""
    payload = {
        "schema_version": settings.schema_boundary.new_fingerprint_schema,
        "repository": settings.source.repository,
        "revision": settings.source.revision,
        "config_sha256": settings.source.config.sha256,
        "weights_sha256": settings.source.weights.sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_adapter_name(name: str) -> str:
    if _ADAPTER_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("adapter names must contain only letters, digits, underscore, or dash")
    return name


def _lora_config(
    settings: StandardExpertSettings,
) -> LoraConfig:
    return LoraConfig(
        r=settings.lora.rank,
        lora_alpha=settings.lora.alpha,
        lora_dropout=settings.lora.dropout,
        target_modules=list(settings.lora.target_modules),
        bias=settings.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def _bind_adapter_provenance(
    model: PeftModel,
    adapter_name: str,
    base_settings: StandardBaseSettings,
) -> None:
    config = model.peft_config[adapter_name]
    config.base_model_name_or_path = base_settings.source.repository
    config.revision = base_settings.source.revision


def _tensor_digest(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _artifact_content_sha256(adapter_dir: Path) -> str:
    digest = hashlib.sha256()
    for filename in _ADAPTER_FILES:
        path = adapter_dir / filename
        if not path.is_file():
            raise ValueError(f"promoted expert is missing required PEFT file: {filename}")
        payload = path.read_bytes()
        digest.update(filename.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _read_json_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return cast(dict[str, object], raw)


def _adapter_config_contract(adapter_dir: Path, settings: StandardExpertSettings) -> None:
    config = _read_json_mapping(adapter_dir / "adapter_config.json", "PEFT adapter config")
    if config.get("r") != settings.lora.rank:
        raise ValueError("PEFT adapter config rank does not match the registered rank")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or set(targets) != set(settings.lora.target_modules):
        raise ValueError("PEFT adapter config target modules do not match the registered set")
    if config.get("lora_alpha") != settings.lora.alpha:
        raise ValueError("PEFT adapter config alpha does not match the registered value")
    if config.get("lora_dropout") != settings.lora.dropout:
        raise ValueError("PEFT adapter config dropout does not match the registered value")
    if config.get("bias") != settings.lora.bias:
        raise ValueError("PEFT adapter config bias does not match the registered value")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("PEFT adapter config is not a causal-language-model LoRA adapter")


@dataclass
class PeftExpertBank:
    """One stock base with isolated PEFT adapters and at most one trainable candidate."""

    base: StandardBase
    settings: StandardExpertSettings
    model: PeftModel
    mature_adapters: set[str]
    trainable_adapter: str | None

    @classmethod
    def start_candidate(
        cls,
        base: StandardBase,
        settings: StandardExpertSettings,
        adapter_name: str,
    ) -> PeftExpertBank:
        """Attach the first candidate only after the disabled flag and pin pass."""
        if not settings.feature.enabled:
            raise RuntimeError("ADR-0011 PEFT expert path is disabled by configuration")
        if not isinstance(base, StandardBase):
            raise TypeError("PEFT experts require the verified ADR-0011 StandardBase")
        if version("peft") != settings.peft.version:
            raise RuntimeError("installed PEFT version does not match the reviewed exact pin")
        name = _validate_adapter_name(adapter_name)
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        attached = get_peft_model(
            base.model,
            _lora_config(settings),
            adapter_name=name,
        )
        if not isinstance(attached, PeftModel):
            raise TypeError("PEFT did not return a standard PeftModel")
        bank = cls(base, settings, attached, set(), name)
        _bind_adapter_provenance(attached, name, base.settings)
        bank._enforce_trainability()
        bank._verify_parameter_count(name)
        return bank

    @property
    def adapter_names(self) -> tuple[str, ...]:
        """Return stable adapter IDs in sorted order."""
        return tuple(sorted(self.model.peft_config))

    def _enforce_trainability(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.trainable_adapter is not None:
            for parameter in self.adapter_parameters(self.trainable_adapter):
                parameter.requires_grad_(True)

    def adapter_parameters(self, adapter_name: str) -> tuple[torch.nn.Parameter, ...]:
        """Return only the named adapter tensors, never base or sibling tensors."""
        _validate_adapter_name(adapter_name)
        marker = f".{adapter_name}."
        parameters = tuple(
            parameter
            for name, parameter in self.model.named_parameters()
            if "lora_" in name and marker in name
        )
        if not parameters:
            raise KeyError(f"PEFT adapter '{adapter_name}' does not exist")
        return parameters

    def actual_parameter_count(self, adapter_name: str) -> int:
        """Count the exact tensors owned by one adapter."""
        return sum(parameter.numel() for parameter in self.adapter_parameters(adapter_name))

    def _verify_parameter_count(self, adapter_name: str) -> None:
        expected = expected_lora_parameter_count(
            self.base.settings.architecture,
            rank=self.settings.lora.rank,
        )
        actual = self.actual_parameter_count(adapter_name)
        if actual != expected:
            raise ValueError(
                f"PEFT adapter parameter count mismatch: expected {expected}, got {actual}"
            )

    def activate(self, adapter_name: str) -> None:
        """Select one real PEFT forward path without unfreezing mature adapters."""
        if adapter_name not in self.model.peft_config:
            raise KeyError(f"PEFT adapter '{adapter_name}' does not exist")
        controller = cast(_AdapterLayerController, self.model.base_model)
        controller.enable_adapter_layers()
        self.model.set_adapter(
            adapter_name,
            inference_mode=adapter_name in self.mature_adapters,
        )
        self._enforce_trainability()

    def deactivate(self) -> None:
        """Bypass every adapter while leaving all base tensors frozen."""
        controller = cast(_AdapterLayerController, self.model.base_model)
        controller.disable_adapter_layers()
        self._enforce_trainability()

    def add_candidate(self, adapter_name: str) -> None:
        """Add one isolated trainable adapter while every mature adapter stays frozen."""
        if self.trainable_adapter is not None:
            raise RuntimeError("an expert bank may have only one trainable candidate")
        name = _validate_adapter_name(adapter_name)
        if name in self.model.peft_config:
            raise ValueError(f"PEFT adapter '{name}' already exists")
        if len(self.model.peft_config) >= self.settings.lora.max_adapters:
            raise RuntimeError("PEFT expert adapter-count bound exceeded")
        self.model.add_adapter(name, _lora_config(self.settings))
        _bind_adapter_provenance(self.model, name, self.base.settings)
        self.trainable_adapter = name
        self.activate(name)
        self._verify_parameter_count(name)

    def base_parameter_sha256(self) -> str:
        """Hash every non-adapter parameter to prove byte isolation."""
        tensors = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if "lora_" not in name
        ]
        return _tensor_digest(tensors)

    def adapter_parameter_sha256(self, adapter_name: str) -> str:
        """Hash one adapter's live tensor bytes independent of its trainability."""
        marker = f".{adapter_name}."
        tensors = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if "lora_" in name and marker in name
        ]
        if not tensors:
            raise KeyError(f"PEFT adapter '{adapter_name}' does not exist")
        return _tensor_digest(tensors)

    def _manifest(self, adapter_name: str, content_sha256: str) -> ExpertManifest:
        source = self.base.settings.source
        return ExpertManifest(
            schema_version="aptivra.peft_expert_manifest.v1",
            adapter_name=adapter_name,
            adapter_subdir=adapter_name,
            base_repository=source.repository,
            base_revision=source.revision,
            base_config_sha256=source.config.sha256,
            base_weights_sha256=source.weights.sha256,
            base_fingerprint=standard_base_fingerprint(self.base.settings),
            rank=self.settings.lora.rank,
            target_modules=tuple(self.settings.lora.target_modules),
            parameter_count=self.actual_parameter_count(adapter_name),
            peft_version=self.settings.peft.version,
            peft_license=self.settings.peft.license,
            content_sha256=content_sha256,
        )

    def persist_adapter(
        self, adapter_name: str, output_dir: FilesystemPath
    ) -> ExpertManifest:
        """Atomically persist one attached adapter without changing bank lifecycle state."""
        name = _validate_adapter_name(adapter_name)
        if name not in self.model.peft_config:
            raise KeyError(f"PEFT adapter '{name}' does not exist")
        output = Path(output_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            self.model.save_pretrained(
                str(stage),
                safe_serialization=True,
                selected_adapters=[name],
                save_embedding_layers=False,
            )
            adapter_dir = stage / name
            _adapter_config_contract(adapter_dir, self.settings)
            manifest = self._manifest(
                name,
                _artifact_content_sha256(adapter_dir),
            )
            manifest_path = stage / "expert_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if output.exists():
                existing = _load_expert_manifest(output)
                if existing != manifest:
                    raise FileExistsError(
                        "refusing to replace a different promoted expert artifact"
                    )
                if (
                    _artifact_content_sha256(output / existing.adapter_subdir)
                    != existing.content_sha256
                ):
                    raise ValueError("existing promoted expert content SHA-256 mismatch")
            else:
                os.replace(stage, output)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def promote_candidate(self, output_dir: FilesystemPath) -> ExpertManifest:
        """Atomically persist, content-address, and freeze the current candidate."""
        if self.trainable_adapter is None:
            raise RuntimeError("there is no trainable PEFT candidate to promote")
        adapter_name = self.trainable_adapter
        manifest = self.persist_adapter(adapter_name, output_dir)
        self.trainable_adapter = None
        self.mature_adapters.add(adapter_name)
        self.activate(adapter_name)
        return manifest


def _load_expert_manifest(artifact_dir: Path) -> ExpertManifest:
    path = artifact_dir / "expert_manifest.json"
    payload = _read_json_mapping(path, "expert manifest")
    return ExpertManifest.model_validate(payload)


def _validate_manifest_for_base(
    manifest: ExpertManifest,
    base: StandardBase,
    settings: StandardExpertSettings,
    adapter_dir: Path,
) -> None:
    source = base.settings.source
    if manifest.base_fingerprint != standard_base_fingerprint(base.settings):
        raise ValueError("promoted expert base fingerprint does not match the loaded base")
    if (
        manifest.base_repository != source.repository
        or manifest.base_revision != source.revision
        or manifest.base_config_sha256 != source.config.sha256
        or manifest.base_weights_sha256 != source.weights.sha256
    ):
        raise ValueError("promoted expert base identity fields do not match the loaded base")
    if manifest.rank != settings.lora.rank:
        raise ValueError("promoted expert rank does not match the registered rank")
    if manifest.target_modules != tuple(settings.lora.target_modules):
        raise ValueError("promoted expert target modules do not match the registered set")
    expected_parameters = expected_lora_parameter_count(
        base.settings.architecture,
        rank=settings.lora.rank,
    )
    if manifest.parameter_count != expected_parameters:
        raise ValueError("promoted expert parameter count does not match the base architecture")
    if manifest.peft_version != settings.peft.version:
        raise ValueError("promoted expert PEFT version does not match the reviewed pin")
    if manifest.peft_license != settings.peft.license:
        raise ValueError("promoted expert PEFT license does not match the reviewed license")
    _adapter_config_contract(adapter_dir, settings)
    if _artifact_content_sha256(adapter_dir) != manifest.content_sha256:
        raise ValueError("promoted expert content SHA-256 mismatch")


def load_promoted_expert(
    base: StandardBase,
    settings: StandardExpertSettings,
    artifact_dir: FilesystemPath,
) -> PeftExpertBank:
    """Validate every compatibility/hash boundary before standard PEFT loading."""
    if not settings.feature.enabled:
        raise RuntimeError("ADR-0011 PEFT expert path is disabled by configuration")
    if version("peft") != settings.peft.version:
        raise RuntimeError("installed PEFT version does not match the reviewed exact pin")
    artifact = Path(artifact_dir)
    manifest = _load_expert_manifest(artifact)
    adapter_dir = artifact / manifest.adapter_subdir
    _validate_manifest_for_base(manifest, base, settings, adapter_dir)
    for parameter in base.model.parameters():
        parameter.requires_grad_(False)
    loaded = PeftModel.from_pretrained(
        base.model,
        adapter_dir,
        adapter_name=manifest.adapter_name,
        is_trainable=False,
        local_files_only=True,
    )
    bank = PeftExpertBank(
        base=base,
        settings=settings,
        model=loaded,
        mature_adapters={manifest.adapter_name},
        trainable_adapter=None,
    )
    bank.activate(manifest.adapter_name)
    bank._verify_parameter_count(manifest.adapter_name)
    return bank


def load_additional_promoted_expert(
    bank: PeftExpertBank,
    artifact_dir: FilesystemPath,
) -> ExpertManifest:
    """Validate and attach another frozen expert to an already-loaded expert bank."""
    artifact = Path(artifact_dir)
    manifest = _load_expert_manifest(artifact)
    adapter_dir = artifact / manifest.adapter_subdir
    _validate_manifest_for_base(manifest, bank.base, bank.settings, adapter_dir)
    if manifest.adapter_name in bank.model.peft_config:
        raise ValueError(f"PEFT adapter '{manifest.adapter_name}' is already loaded")
    if len(bank.model.peft_config) >= bank.settings.lora.max_adapters:
        raise RuntimeError("PEFT expert adapter-count bound exceeded")
    bank.model.load_adapter(
        adapter_dir,
        adapter_name=manifest.adapter_name,
        is_trainable=False,
        local_files_only=True,
    )
    bank.mature_adapters.add(manifest.adapter_name)
    bank.activate(manifest.adapter_name)
    bank._verify_parameter_count(manifest.adapter_name)
    return manifest
