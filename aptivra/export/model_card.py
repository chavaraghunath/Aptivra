"""Configuration and evidence-bound model cards for standard PEFT releases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from aptivra.resources import aptivra_resource

DEFAULT_RELEASE_GATE_CONFIG = aptivra_resource(
    "configs", "model_card_release_gate.yaml"
)

ConfigSource = Union[Path, str, Traversable]  # noqa: UP007


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReleaseFeatureSettings(_StrictModel):
    """Fail-closed switch for the unaccepted standard publication path."""

    enabled: bool


class RepositorySettings(_StrictModel):
    """Owner-controlled destination; null deliberately has no fallback."""

    repository_id: Optional[str]  # noqa: UP045

    @field_validator("repository_id")
    @classmethod
    def validate_repository_id(cls, value: Optional[str]) -> Optional[str]:  # noqa: UP045
        if value is None:
            return None
        if re.fullmatch(r"[^/\s]+/[^/\s]+", value) is None:
            raise ValueError("repository_id must be null or an owner/name identifier")
        return value


class PublicationSettings(_StrictModel):
    """Closed public-adapter provenance allowlist."""

    allowed_adapter_kinds: tuple[Literal["synthetic", "identity"], ...]

    @field_validator("allowed_adapter_kinds", mode="before")
    @classmethod
    def normalize_kinds(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_kinds(self) -> PublicationSettings:
        if self.allowed_adapter_kinds != ("synthetic", "identity"):
            raise ValueError(
                "allowed_adapter_kinds must be exactly synthetic then identity"
            )
        return self


class ReleaseGateSettings(_StrictModel):
    """Complete lane-7 staging and publication policy."""

    schema_version: Literal["aptivra.model_card_release_gate.v1"]
    feature: ReleaseFeatureSettings
    repository: RepositorySettings
    publication: PublicationSettings
    forbidden_references: tuple[str, ...]

    @field_validator("forbidden_references", mode="before")
    @classmethod
    def normalize_references(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("forbidden_references")
    @classmethod
    def validate_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("forbidden references must be non-empty strings")
        folded = tuple(item.casefold() for item in value)
        if len(set(folded)) < len(folded):
            raise ValueError("forbidden references must be unique case-insensitively")
        return value


def _read_text(source: ConfigSource) -> str:
    if isinstance(source, str):
        return Path(source).read_text(encoding="utf-8")
    return source.read_text(encoding="utf-8")


def load_release_gate_settings(
    config_path: ConfigSource = DEFAULT_RELEASE_GATE_CONFIG,
) -> ReleaseGateSettings:
    """Load the strict disabled-by-default standard publication policy."""
    raw: object = yaml.safe_load(_read_text(config_path))
    if not isinstance(raw, dict) or set(raw) != {"model_card_release_gate"}:
        raise ValueError(
            "release gate config root must contain only 'model_card_release_gate'"
        )
    return ReleaseGateSettings.model_validate(raw["model_card_release_gate"])


@dataclass(frozen=True)
class ModelCardData:
    """Validated immutable release facts consumed by the renderer."""

    model_name: str
    creator_name: str
    creator_approval: str
    base_repository: str
    base_revision: str
    research_status: str
    limitations: tuple[str, ...]
    adapters: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"staging manifest {field} must be an object")
    return cast(dict[str, Any], value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"staging manifest {field} must be non-empty text")
    return value


def _card_data(manifest: dict[str, Any]) -> ModelCardData:
    identity = _mapping(manifest.get("identity"), "identity")
    creator = _mapping(identity.get("creator"), "identity.creator")
    base = _mapping(manifest.get("base"), "base")
    raw_adapters = manifest.get("adapters")
    if not isinstance(raw_adapters, list) or not raw_adapters:
        raise ValueError("staging manifest adapters must be a non-empty list")
    adapters = tuple(_mapping(item, "adapters[]") for item in raw_adapters)
    raw_limitations = identity.get("limitations")
    if not isinstance(raw_limitations, list) or not raw_limitations:
        raise ValueError("staging manifest identity limitations must be a non-empty list")
    limitations = tuple(_text(item, "identity.limitations[]") for item in raw_limitations)
    return ModelCardData(
        model_name=_text(identity.get("model_name"), "identity.model_name"),
        creator_name=_text(creator.get("name"), "identity.creator.name"),
        creator_approval=_text(creator.get("approval"), "identity.creator.approval"),
        base_repository=_text(base.get("repository"), "base.repository"),
        base_revision=_text(base.get("revision"), "base.revision"),
        research_status=_text(identity.get("research_status"), "identity.research_status"),
        limitations=limitations,
        adapters=adapters,
        evaluation=_mapping(manifest.get("evaluation"), "evaluation"),
    )


def _adapter_rows(adapters: tuple[dict[str, Any], ...]) -> str:
    rows: list[str] = []
    for adapter in adapters:
        adapter_id = _text(adapter.get("adapter_id"), "adapters[].adapter_id")
        kind = _text(adapter.get("kind"), "adapters[].kind")
        task_family = _text(adapter.get("task_family"), "adapters[].task_family")
        rows.append(
            f"| `{adapter_id}` | {kind} | {task_family} | Derivative of the pinned base |"
        )
    return "\n".join(rows)


def _capability_section(evaluation: dict[str, Any]) -> str:
    record = evaluation.get("record")
    if record is None:
        absence = _text(evaluation.get("absence"), "evaluation.absence")
        return (
            "No capability is claimed because no bound release-evaluation record is "
            f"available. {absence}"
        )
    record_mapping = _mapping(record, "evaluation.record")
    metric = _mapping(record_mapping.get("metric"), "evaluation.record.metric")
    seeds = record_mapping.get("seed_set")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("evaluation record seed_set must be a non-empty list")
    return (
        "The only release capability statement is the bound record: "
        f"`{_text(metric.get('name'), 'evaluation.record.metric.name')}` = "
        f"`{metric.get('value')}` `{_text(metric.get('unit'), 'evaluation.record.metric.unit')}` "
        "on stream `"
        f"{_text(record_mapping.get('stream_version'), 'evaluation.record.stream_version')}`, "
        f"seeds `{json.dumps(seeds)}`, denominator `{record_mapping.get('denominator')}`, "
        f"dated `{record_mapping.get('date')}`. No broader capability follows from it."
    )


def generate_model_card_readme(manifest: dict[str, Any]) -> str:
    """Render only facts present in the immutable package and identity manifests."""
    data = _card_data(manifest)
    first_adapter = _text(data.adapters[0].get("adapter_id"), "adapters[0].adapter_id")
    limitations = "\n".join(f"- {item}" for item in data.limitations)
    return f"""---
license: apache-2.0
library_name: peft
base_model: {data.base_repository}
pipeline_tag: text-generation
tags:
- aptivra
- peft
- research
---

# {data.model_name}

## Status and evidence

{data.research_status}

{_capability_section(data.evaluation)}

## Stock loading with an explicit adapter

The base weights are not included in this repository. Load the exact upstream base and revision,
then select one standard PEFT adapter explicitly:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "{data.base_repository}",
    revision="{data.base_revision}",
)
model = PeftModel.from_pretrained(
    base,
    ".",
    subfolder="adapters/{first_adapter}",
    adapter_name="{first_adapter}",
)
model.set_adapter("{first_adapter}")
```

Automatic routing is separate from stock base and PEFT loading. Automatic routing requires the
installed Aptivra orchestrator to read the router sidecar and select an adapter. The router is not
a built-in Llama layer, and the bundle does not claim that stock Transformers performs automatic
routing.

## Adapters

| Adapter | Publication kind | Task family | Relationship |
|---|---|---|---|
{_adapter_rows(data.adapters)}

Every listed adapter is an Aptivra modification and a derivative of
`{data.base_repository}` at `{data.base_revision}`. The router sidecar and `aptivra-chat-v1`
template are also Aptivra modifications. Base weights are referenced, not redistributed.

## Provenance, attribution, and licence

Owner-approved creator attribution: **{data.creator_name}** ({data.creator_approval}). Legal
approval for publication remains with the repository owner and is an explicit provenance state.

SmolLM2 is licensed under Apache-2.0. Redistribution must include the licence, retain applicable
upstream copyright, patent, trademark, and attribution notices, mark Aptivra-modified files, and
retain any readable attribution notices from an upstream NOTICE if one is supplied. The staged
NOTICE identifies the exact base, revision, Aptivra modifications, and derivative adapters. No
copyright holder or year is invented where the offline source evidence supplies none.

## Limitations

{limitations}

The model may be wrong. Results outside a named, bound release-evaluation record are unmeasured.
"""
