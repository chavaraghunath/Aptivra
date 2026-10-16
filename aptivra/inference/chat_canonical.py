"""Evidence-bound canonical identity records for the Aptivra chat adapter."""

from __future__ import annotations

import hashlib
import json
import operator
import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Optional, Union, assert_never

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from aptivra.experts.standard import (
    StandardExpertSettings,
    load_standard_expert_settings,
    standard_base_fingerprint,
)
from aptivra.inference.chat_prompt import (
    DEFAULT_IDENTITY_CONFIG,
    IdentitySettings,
    format_chat_prompt,
    load_identity_settings,
)
from aptivra.inference.standard_base import (
    StandardBaseSettings,
    load_standard_base_settings,
)
from aptivra.resources import aptivra_resource

DEFAULT_IDENTITY_MANIFEST = aptivra_resource("configs", "identity_manifest.yaml")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_REQUIRED_LIMITATIONS = (
    "An ordinary conversation changes only ephemeral context and does not persist "
    "weights, replay, or episodic memory.",
    "Aptivra may be wrong.",
    "Under automatic routing an adapter is active only when selected; manual adapter "
    "selection is an explicit evaluation or runtime mode.",
    "Capability is limited to the named evaluations in bound release evidence.",
)

PositiveInt = Annotated[int, Field(gt=0)]


def _validate_repository(value: str) -> str:
    if re.fullmatch(r"[^/\s]+/[^/\s]+", value) is None:
        raise ValueError("identity base repository must be exactly owner/name")
    return value


def _validate_revision(value: str) -> str:
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError("identity base revision must be a full lowercase commit hash")
    return value


def _validate_base_hash(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("identity base hashes must be lowercase SHA-256 values")
    return value


def _validate_metric_name(value: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise ValueError("release metric name must be lowercase snake case")
    return value


def _validate_metric_value(value: float) -> float:
    if not (-1.0e12 < value < 1.0e12):
        raise ValueError("release metric value must be finite and bounded")
    return value


def _list_to_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_seeds(value: tuple[int, ...]) -> tuple[int, ...]:
    if any(isinstance(seed, bool) or seed < 0 for seed in value):
        raise ValueError("release seeds must be non-negative integers")
    if operator.ne(len(value), len(set(value))):
        raise ValueError("release seeds must be unique")
    return value


def _validate_release_hash(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("release evidence hashes must be lowercase SHA-256 values")
    return value


def _validate_manifest_hash(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("identity manifest content hash must be lowercase SHA-256")
    return value


Repository = Annotated[str, AfterValidator(_validate_repository)]
Revision = Annotated[str, AfterValidator(_validate_revision)]
BaseHash = Annotated[str, AfterValidator(_validate_base_hash)]
MetricName = Annotated[
    str,
    Field(max_length=80),
    AfterValidator(_validate_metric_name),
]
MetricValue = Annotated[float, AfterValidator(_validate_metric_value)]
SeedSet = Annotated[
    tuple[int, ...],
    Field(min_length=1),
    BeforeValidator(_list_to_tuple),
    AfterValidator(_validate_seeds),
]
ReleaseHash = Annotated[str, AfterValidator(_validate_release_hash)]
ManifestHash = Annotated[str, AfterValidator(_validate_manifest_hash)]
Limitations = Annotated[tuple[str, ...], BeforeValidator(_list_to_tuple)]

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  Classic Optional and
# Union preserve the public types while keeping the inventory behavioral.


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CreatorAttribution(_StrictModel):
    name: Literal["Raghunath Chava"]
    attribution_kind: Literal["owner_attribution"]
    approval: Literal["approved by the owner on 2026-10-17"]


class ModelIdentity(_StrictModel):
    name: Literal["Aptivra"]
    creator: CreatorAttribution


class BaseIdentity(_StrictModel):
    repository: Repository
    revision: Revision
    config_sha256: BaseHash
    weights_sha256: BaseHash
    fingerprint: BaseHash
    unique_parameter_count: PositiveInt


class RoutedLoRAClaim(_StrictModel):
    statement: Literal["frozen base plus separately routed LoRA adapters"]
    required_adapter_name: Literal["identity"]
    required_rank: Literal[16]
    required_router: Literal["sequence-level top-1"]
    require_release_enabled_config: Literal[True]


class ComponentClaims(_StrictModel):
    routed_lora: RoutedLoRAClaim


class ReleaseMetric(_StrictModel):
    name: MetricName
    value: MetricValue
    unit: str = Field(min_length=1, max_length=40)


class ReleaseEvaluationRecord(_StrictModel):
    metric: ReleaseMetric
    stream_version: str = Field(min_length=1, max_length=40)
    seed_set: SeedSet
    denominator: PositiveInt
    date: date
    base_fingerprint: ReleaseHash
    content_sha256: ReleaseHash

    @model_validator(mode="after")
    def validate_content_hash(self) -> ReleaseEvaluationRecord:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        _require_equal(
            _canonical_sha256(payload),
            self.content_sha256,
            "release evaluation content SHA-256 is stale",
        )
        return self


class ReleaseEvaluationBinding(_StrictModel):
    expected_stream_version: str = Field(min_length=1, max_length=40)
    record: Optional[ReleaseEvaluationRecord]  # noqa: UP045


class IdentityManifest(_StrictModel):
    schema_version: Literal["aptivra.identity_manifest.v1"]
    content_sha256: ManifestHash
    model: ModelIdentity
    base: BaseIdentity
    components: ComponentClaims
    research_status: str = Field(min_length=1, max_length=240)
    required_limitations: Limitations
    release_evaluation: ReleaseEvaluationBinding

    @model_validator(mode="after")
    def validate_manifest_contract(self) -> IdentityManifest:
        _require_equal(
            self.required_limitations,
            _REQUIRED_LIMITATIONS,
            "identity manifest required limitations do not match ADR-0011",
        )
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        _require_equal(
            _canonical_sha256(payload),
            self.content_sha256,
            "identity manifest content SHA-256 is stale",
        )
        return self


class PromptIntent(StrEnum):
    """Closed set of identity prompts generated for lane-6 chat records."""

    WHAT_ARE_YOU = "what_are_you"
    ARCHITECTURE = "architecture"
    CHAT_LEARNING = "chat_learning"
    CREATOR = "creator"
    CAPABILITIES = "capabilities"


@dataclass(frozen=True)
class CanonicalIdentityRecord:
    """One generated prompt intent and its evidence-bound exact target."""

    prompt_intent: PromptIntent
    question: str
    answer: str


def allowed_release_metric(
    prompt_intent: PromptIntent,
    release_record: Optional[ReleaseEvaluationRecord],  # noqa: UP045
) -> Optional[ReleaseMetric]:  # noqa: UP045
    """Bind release metrics only to the capabilities intent, exhaustively."""
    match prompt_intent:
        case PromptIntent.CAPABILITIES:
            return None if release_record is None else release_record.metric
        case (
            PromptIntent.WHAT_ARE_YOU
            | PromptIntent.ARCHITECTURE
            | PromptIntent.CHAT_LEARNING
            | PromptIntent.CREATOR
        ):
            return None
        case unreachable:
            assert_never(unreachable)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def release_record_sha256(payload: dict[str, object]) -> str:
    """Return the required content hash for a release record without its hash field."""
    content = dict(payload)
    content.pop("content_sha256", None)
    return _canonical_sha256(content)


def identity_manifest_sha256(payload: dict[str, object]) -> str:
    """Return the manifest hash over every field except the hash itself."""
    content = dict(payload)
    content.pop("content_sha256", None)
    return _canonical_sha256(content)


def _require_equal(actual: object, expected: object, message: str) -> None:
    if operator.eq(actual, expected):
        return
    raise ValueError(message)


def _read_manifest(
    path: Union[Path, str, Traversable],  # noqa: UP007
) -> IdentityManifest:
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("identity manifest root must contain only 'identity_manifest'")
    if set(raw) != {"identity_manifest"}:
        raise ValueError("identity manifest root must contain only 'identity_manifest'")
    return IdentityManifest.model_validate(raw["identity_manifest"])


def load_identity_manifest(
    path: Union[Path, str, Traversable] = DEFAULT_IDENTITY_MANIFEST,  # noqa: UP007
    *,
    base_settings: Optional[StandardBaseSettings] = None,  # noqa: UP045
) -> IdentityManifest:
    """Validate manifest content and bind it to the pinned standard-base fingerprint."""
    manifest = _read_manifest(path)
    base = load_standard_base_settings() if base_settings is None else base_settings
    expected_fingerprint = standard_base_fingerprint(base)
    source = base.source
    manifest_base = (
        manifest.base.repository,
        manifest.base.revision,
        manifest.base.config_sha256,
        manifest.base.weights_sha256,
        manifest.base.fingerprint,
        manifest.base.unique_parameter_count,
    )
    expected_base = (
        source.repository,
        source.revision,
        source.config.sha256,
        source.weights.sha256,
        expected_fingerprint,
        base.architecture.unique_parameter_count,
    )
    _require_equal(
        manifest_base,
        expected_base,
        "identity manifest base fingerprint is stale",
    )
    record = manifest.release_evaluation.record
    if record is not None:
        _require_equal(
            record.base_fingerprint,
            expected_fingerprint,
            "release evaluation base fingerprint is stale",
        )
        _require_equal(
            record.stream_version,
            manifest.release_evaluation.expected_stream_version,
            "release evaluation stream version is stale",
        )
    return manifest


def load_configured_identity_contract(
    config_path: Union[Path, str, Traversable] = DEFAULT_IDENTITY_CONFIG,  # noqa: UP007
    *,
    base_settings: Optional[StandardBaseSettings] = None,  # noqa: UP045
) -> tuple[IdentitySettings, IdentityManifest, StandardExpertSettings]:
    """Resolve and validate the configured adjacent manifest and expert/router config."""
    settings = load_identity_settings(config_path)
    config_source = Path(str(config_path))
    manifest_path = config_source.parent / settings.template.manifest_path
    expert_path = config_source.parent / settings.components.standard_expert_config_path
    manifest = load_identity_manifest(manifest_path, base_settings=base_settings)
    experts = load_standard_expert_settings(expert_path)
    return settings, manifest, experts


def _routed_lora_enabled(
    manifest: IdentityManifest,
    expert_settings: Optional[StandardExpertSettings],  # noqa: UP045
) -> bool:
    if expert_settings is None:
        return False
    if not expert_settings.feature.enabled:
        return False
    claim = manifest.components.routed_lora
    return (
        operator.eq(expert_settings.lora.rank, claim.required_rank)
        and expert_settings.router is not None
    )


def default_system_message(
    manifest: Optional[IdentityManifest] = None,  # noqa: UP045
) -> str:
    """Generate the system identity from the validated manifest."""
    bound = load_configured_identity_contract()[1] if manifest is None else manifest
    return (
        f"You are {bound.model.name}, a research language model. "
        "Answer identity questions only from the bound identity manifest."
    )


def _capability_answer(manifest: IdentityManifest) -> str:
    _, _, _, capability_limitation = manifest.required_limitations
    record = manifest.release_evaluation.record
    if record is None:
        return (
            "No release evidence is available, so I do not claim measured capabilities. "
            f"{capability_limitation}"
        )
    metric = record.metric
    seeds = ", ".join(str(seed) for seed in record.seed_set)
    return (
        f"Bound release evidence reports {metric.name}={metric.value:g} {metric.unit} "
        f"on stream {record.stream_version}, across seeds {seeds}, with denominator "
        f"{record.denominator}, dated {record.date.isoformat()}. "
        f"{capability_limitation}"
    )


def _forbidden_target_reason(
    target: str,
    *,
    allowed_metric: Optional[ReleaseMetric] = None,  # noqa: UP045
) -> Optional[str]:  # noqa: UP045
    lowered = " ".join(target.lower().split())
    if "github" in lowered:
        return "source-host reference"
    if "autonomous" in lowered:
        return "autonomous"
    if re.search(r"\b200\s*m(?:illion)?\b", lowered):
        return "200M"
    if "zero forgetting" in lowered or "zero catastrophic forgetting" in lowered:
        return "zero forgetting"
    if re.search(r"\b\d[\d,]*\+?\s+(?:mcp\s+)?skills\b", lowered):
        return "skill count"
    if "c-engine" in lowered or "native execution" in lowered:
        return "native execution"
    if "throughput" in lowered or re.search(r"\b(?:mb|gb)/s\b", lowered):
        return "throughput"
    without_negated_top2 = re.sub(r"\bnot\s+(?:a\s+)?top-2\b", "", lowered)
    if "top-2" in without_negated_top2:
        return "Top-2 MoE"
    metric_terms = re.findall(r"\b(?:accuracy|bpb|bwt|fwt|metric|score)\b", lowered)
    if metric_terms and allowed_metric is None:
        return "unbound metric"
    if allowed_metric is not None and allowed_metric.name.lower() not in lowered:
        return "release metric binding"
    return None


def assert_permitted_identity_target(
    target: str,
    *,
    allowed_metric: Optional[ReleaseMetric] = None,  # noqa: UP045
) -> None:
    """Fail generation when any forbidden identity claim reaches a target."""
    reason = _forbidden_target_reason(target, allowed_metric=allowed_metric)
    if reason is not None:
        raise ValueError(f"identity target contains forbidden claim: {reason}")


def canonical_identity_records(
    manifest: Optional[IdentityManifest] = None,  # noqa: UP045
    *,
    expert_settings: Optional[StandardExpertSettings] = None,  # noqa: UP045
) -> tuple[CanonicalIdentityRecord, ...]:
    """Generate every canonical record solely from validated manifest evidence."""
    if manifest is None:
        _, bound, configured_experts = load_configured_identity_contract()
    else:
        bound = manifest
        configured_experts = load_standard_expert_settings()
    experts = configured_experts if expert_settings is None else expert_settings
    routed = _routed_lora_enabled(bound, experts)
    (
        _context_limitation,
        fallibility_limitation,
        routing_limitation,
        _capability_limitation,
    ) = bound.required_limitations
    component_phrase = (
        " and adapted with separately routed LoRA modules"
        if routed
        else ""
    )
    architecture_suffix = (
        " Aptivra adds rank-16 LoRA experts and a sequence-level top-1 router; "
        "it is not a Top-2 mixture-of-experts model."
        if routed
        else " No routed LoRA component is enabled for this release."
    )
    records = (
        CanonicalIdentityRecord(
            PromptIntent.WHAT_ARE_YOU,
            "What are you?",
            f"I am {bound.model.name}, a research language model built on "
            f"`{bound.base.repository}` at revision `{bound.base.revision}`"
            f"{component_phrase}. My measured abilities and limitations are listed in "
            f"my model card. {fallibility_limitation}",
        ),
        CanonicalIdentityRecord(
            PromptIntent.ARCHITECTURE,
            "What is your architecture?",
            f"My frozen base is a {bound.base.unique_parameter_count:,}-parameter Llama "
            f"causal language model.{architecture_suffix}",
        ),
        CanonicalIdentityRecord(
            PromptIntent.CHAT_LEARNING,
            "Do you learn from this chat?",
            f"This conversation changes my temporary context, but it does not persistently "
            f"update my weights, replay buffer, or episodic memory. Persistent learning "
            f"requires the explicit verifier-gated training and promotion process. "
            f"{routing_limitation}",
        ),
        CanonicalIdentityRecord(
            PromptIntent.CREATOR,
            f"Who created {bound.model.name}?",
            f"The owner-approved creator attribution names {bound.model.creator.name} as "
            f"{bound.model.name}'s creator.",
        ),
        CanonicalIdentityRecord(
            PromptIntent.CAPABILITIES,
            "What can you do?",
            _capability_answer(bound),
        ),
    )
    release_record = bound.release_evaluation.record
    for record in records:
        assert_permitted_identity_target(
            record.answer,
            allowed_metric=allowed_release_metric(
                record.prompt_intent,
                release_record,
            ),
        )
    return records


def identity_chat_templates(
    manifest: Optional[IdentityManifest] = None,  # noqa: UP045
    *,
    expert_settings: Optional[StandardExpertSettings] = None,  # noqa: UP045
) -> list[tuple[str, str, str]]:
    """Return question, formatted open-assistant prompt, and exact target triples."""
    bound = load_configured_identity_contract()[1] if manifest is None else manifest
    system = default_system_message(bound)
    return [
        (
            record.question,
            format_chat_prompt([], record.question, system_message=system),
            record.answer,
        )
        for record in canonical_identity_records(bound, expert_settings=expert_settings)
    ]


_DEFAULT_MANIFEST = load_configured_identity_contract()[1]
_DEFAULT_RECORDS = canonical_identity_records(_DEFAULT_MANIFEST)
_DEFAULT_RECORD_BY_INTENT = {
    record.prompt_intent: record for record in _DEFAULT_RECORDS
}
APTIVRA_INTRO = _DEFAULT_RECORD_BY_INTENT[PromptIntent.WHAT_ARE_YOU].answer
NAME_REPLY = f"My name is {_DEFAULT_MANIFEST.model.name}."
CREATOR_REPLY = _DEFAULT_RECORD_BY_INTENT[PromptIntent.CREATOR].answer
CREATOR_BIO_REPLY = CREATOR_REPLY
CAPABILITIES_SUMMARY = _DEFAULT_RECORD_BY_INTENT[PromptIntent.CAPABILITIES].answer
IDENTITY_CHAT_TEMPLATES = identity_chat_templates(_DEFAULT_MANIFEST)
