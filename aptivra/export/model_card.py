"""Measured-results model card renderer for Aptivra Hub exports."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from aptivra.eval.model_card_schema import EvaluationRecord, validate_evaluation_record
from aptivra.model.config import ModelConfig

DEFAULT_RELEASE_GATE_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "model_card_release_gate.yaml"
)


def load_results_record(results_path: Path | str) -> dict[str, Any]:
    """Loads one machine-readable model-card evaluation record."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("model-card results record must be a JSON object")
    return data


@dataclass(frozen=True)
class ReleaseGateSettings:
    """Validated operator settings for release evidence and comparison."""

    heldout_bpb_unigram_margin_bits: float
    minimum_trained_prompt_samples: int
    minimum_heldout_prompt_samples: int
    minimum_heldout_text_bytes: int
    supplied_record_metric_tolerance: float


def load_release_gate_settings(
    config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG,
) -> ReleaseGateSettings:
    """Load and validate all model-card release-gate settings."""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("release gate config must be a YAML mapping")
    section = data.get("model_card_release_gate")
    if not isinstance(section, dict):
        raise ValueError("release gate config missing model_card_release_gate")
    raw_margin = section.get("heldout_bpb_unigram_margin_bits")
    if not isinstance(raw_margin, int | float) or isinstance(raw_margin, bool):
        raise ValueError("heldout_bpb_unigram_margin_bits must be numeric")
    if raw_margin < 0:
        raise ValueError("heldout_bpb_unigram_margin_bits must be non-negative")
    trained_minimum = _positive_config_int(section, "minimum_trained_prompt_samples")
    heldout_minimum = _positive_config_int(section, "minimum_heldout_prompt_samples")
    text_minimum = _positive_config_int(section, "minimum_heldout_text_bytes")
    raw_tolerance = section.get("supplied_record_metric_tolerance")
    if not isinstance(raw_tolerance, int | float) or isinstance(raw_tolerance, bool):
        raise ValueError("supplied_record_metric_tolerance must be numeric")
    if not 0.0 <= raw_tolerance <= 0.001:
        raise ValueError(
            "supplied_record_metric_tolerance must be between 0.0 and 0.001"
        )
    return ReleaseGateSettings(
        heldout_bpb_unigram_margin_bits=float(raw_margin),
        minimum_trained_prompt_samples=trained_minimum,
        minimum_heldout_prompt_samples=heldout_minimum,
        minimum_heldout_text_bytes=text_minimum,
        supplied_record_metric_tolerance=float(raw_tolerance),
    )


def load_release_gate_config(config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG) -> float:
    """Return the configured held-out BpB margin for backward-compatible callers."""
    return load_release_gate_settings(config_path).heldout_bpb_unigram_margin_bits


def release_gate_allows_general_language_model(
    results_record: dict[str, Any],
    config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG,
    *,
    fresh_evaluation: EvaluationRecord | None = None,
) -> bool:
    """Return true only for freshly measured evidence that beats the configured gate."""
    try:
        settings = load_release_gate_settings(config_path)
        validated = validate_evaluation_record(
            results_record,
            minimum_trained_samples=settings.minimum_trained_prompt_samples,
            minimum_heldout_samples=settings.minimum_heldout_prompt_samples,
            minimum_heldout_text_bytes=settings.minimum_heldout_text_bytes,
        )
    except (OSError, ValueError):
        return False
    if fresh_evaluation is None or fresh_evaluation != validated:
        return False
    heldout = validated.evaluation.heldout_prompt_answer_bpb
    return (
        heldout.mean + settings.heldout_bpb_unigram_margin_bits
        < heldout.baselines.unigram_bpb.mean
    )


def generate_model_card_readme(
    repo_id: str,
    model_config: ModelConfig,
    num_experts: int,
    *,
    results_record: dict[str, Any] | None = None,
    results_path: Path | str | None = None,
    fresh_evaluation: EvaluationRecord | None = None,
    release_gate_config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG,
) -> str:
    """Render a card; only publication's fresh validated evidence may clear its notice."""
    _ = (model_config, num_experts)
    record = results_record if results_record is not None else {}
    if results_path is not None:
        record = load_results_record(results_path)

    settings = load_release_gate_settings(release_gate_config_path)
    validated: EvaluationRecord | None = None
    with suppress(ValueError):
        validated = validate_evaluation_record(
            record,
            minimum_trained_samples=settings.minimum_trained_prompt_samples,
            minimum_heldout_samples=settings.minimum_heldout_prompt_samples,
            minimum_heldout_text_bytes=settings.minimum_heldout_text_bytes,
        )
    display_record = (
        validated.model_dump(mode="json", by_alias=True) if validated is not None else {}
    )
    gate_passed = (
        validated is not None
        and fresh_evaluation is not None
        and fresh_evaluation == validated
        and release_gate_allows_general_language_model(
            record,
            release_gate_config_path,
            fresh_evaluation=fresh_evaluation,
        )
    )
    trained_bpb = _bpb_range_summary(
        display_record, ("evaluation", "trained_prompt_answer_bpb")
    )
    heldout_bpb = _bpb_summary(
        display_record, ("evaluation", "heldout_prompt_answer_bpb")
    )
    prompt_set_name = _text_at(
        display_record, ("evaluation", "heldout_prompt_set", "name")
    )
    prompt_set_sha256 = _text_at(
        display_record, ("evaluation", "heldout_prompt_set", "sha256")
    )
    prompt_set_samples = _number_text_at(
        display_record, ("evaluation", "heldout_prompt_set", "samples")
    )
    text_set_name = _text_at(display_record, ("evaluation", "heldout_text_set", "name"))
    text_set_sha256 = _text_at(
        display_record, ("evaluation", "heldout_text_set", "sha256")
    )
    text_set_bytes = _number_text_at(
        display_record, ("evaluation", "heldout_text_set", "bytes")
    )
    trained_uniform = _bpb_summary(
        display_record,
        ("evaluation", "trained_prompt_answer_bpb", "baselines", "uniform_bpb"),
    )
    trained_unigram = _bpb_summary(
        display_record,
        ("evaluation", "trained_prompt_answer_bpb", "baselines", "unigram_bpb"),
    )
    heldout_uniform = _bpb_summary(
        display_record,
        ("evaluation", "heldout_prompt_answer_bpb", "baselines", "uniform_bpb"),
    )
    heldout_unigram = _bpb_summary(
        display_record,
        ("evaluation", "heldout_prompt_answer_bpb", "baselines", "unigram_bpb"),
    )
    total_parameters = _number_at(display_record, ("model", "total_parameters"))
    weights_hash = _text_at(
        display_record, ("checkpoint", "model_safetensors_sha256")
    )
    whole_text_bpb = _number_text_at(
        display_record, ("evaluation", "heldout_text_bpb", "value")
    )
    whole_text_uniform = _number_text_at(
        display_record,
        ("evaluation", "heldout_text_bpb", "baselines", "uniform_bpb"),
    )
    whole_text_unigram = _number_text_at(
        display_record,
        ("evaluation", "heldout_text_bpb", "baselines", "unigram_bpb"),
    )
    context_length = _number_text_at(display_record, ("model", "context_length"))
    padding_token = _number_text_at(
        display_record, ("tokenizer", "padding_token_id")
    )
    newline_token = _number_text_at(
        display_record, ("tokenizer", "newline_stop_token_id")
    )
    generation_note = (
        "Generation does not stop by itself because this checkpoint did not learn to emit an "
        "end byte. Always set `max_new_tokens`. "
        f"Byte {padding_token} doubles as padding and end-of-sequence, while byte "
        f"{newline_token} (newline) is also a stop. Package and Transformers generation both "
        "retain the first emitted stop byte in the returned suffix; no trailing whitespace "
        "is stripped."
    )
    repository_note = ""
    if "200m" in repo_id.lower() and total_parameters is not None:
        repository_note = (
            f"\nThe repository name predates the measured {int(total_parameters):,}-parameter "
            "checkpoint.\n"
        )
    notice = ""
    if not gate_passed:
        notice = (
            "\n> **Not a usable language model:** held-out answer-span bits per byte "
            "does not beat the unigram byte baseline, or that comparison was not measured. "
            "This checkpoint must not be described as a general language model.\n"
        )

    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- aptivra
- research-prototype
- byte-level
- pytorch
- continual-learning-research
---

# Aptivra Research Prototype Model Card
{notice}
## What This Artefact Is

This repository contains an Aptivra byte-level causal-model research checkpoint and
supporting export artefacts. The checkpoint is intended for auditing and research into this
codebase only. It is not evidence of instruction following, tool use, embodied action,
function calling, code synthesis skill, or continual-learning retention.
{repository_note}

## Measured Facts

| Field | Value |
|---|---|
| Repository | `{repo_id}` |
| Evaluation timestamp | {_text_at(display_record, ("timestamp_utc",))} |
| Source Hub revision | {_text_at(display_record, ("checkpoint", "source_hub_revision"))} |
| `model.safetensors` SHA-256 | {weights_hash} |
| Legacy backbone SHA-256 | {_text_at(display_record, ("checkpoint", "backbone_sha256"))} |
| Backbone parameters | {_number_text_at(display_record, ("model", "backbone_parameters"))} |
| Router parameters | {_number_text_at(display_record, ("model", "router_parameters"))} |
| Expert parameters | {_number_text_at(display_record, ("model", "expert_parameters"))} |
| Total parameters | {_number_text_at(display_record, ("model", "total_parameters"))} |
| Position embedding | {_text_at(display_record, ("model", "position_embedding"))} |
| MoE enabled during evaluation | {_text_at(display_record, ("evaluation", "moe_enabled"))} |
| Held-out prompt set | {prompt_set_name} |
| Held-out prompt set SHA-256 | {prompt_set_sha256} |
| Held-out prompt samples | {prompt_set_samples} |
| Held-out text set | {text_set_name} |
| Held-out text set SHA-256 | {text_set_sha256} |
| Held-out text bytes | {text_set_bytes} |
| Trained-prompt answer-span BpB | {trained_bpb} |
| Trained-answer uniform baseline BpB | {trained_uniform} |
| Trained-answer fitted unigram baseline BpB | {trained_unigram} |
| Held-out answer-span BpB | {heldout_bpb} |
| Held-out-answer uniform baseline BpB | {heldout_uniform} |
| Held-out-answer fitted unigram baseline BpB | {heldout_unigram} |
| Held-out whole-text BpB | {whole_text_bpb} |
| Held-out-text uniform baseline BpB | {whole_text_uniform} |
| Held-out-text fitted unigram baseline BpB | {whole_text_unigram} |

Each baseline is computed on exactly the bytes scored by the metric beside it. The empirical
unigram distribution is fitted and evaluated on those same bytes, so it is an optimistic
baseline bound rather than an independently fitted estimate.

Publication recomputes these loss metrics from the exact local export immediately before upload.
The whole-text metric processes fixed-context chunks, retaining the preceding byte at each chunk
boundary. The metrics establish byte-prediction loss only on the named, hashed prompt and text
sets. They do not establish factual accuracy, answer correctness, instruction following, safety,
or performance on other inputs. A supplied record is only a reproducibility cross-check and never
controls this card or its usability notice.

## What It Cannot Do

- It has not been measured as a usable general language model.
- It has not been measured for external tool invocation, API-call planning, physical-action
  control, or autonomous agent behavior.
- It has not been measured to prevent catastrophic forgetting.
- Held-out byte prediction must beat the unigram baseline before any general-language claim is made.

## How It Was Produced

The repository includes identity fine-tuning prompts in `aptivra/tasks/identity.py` and
canonical chat targets in `aptivra/inference/chat_canonical.py`. Those training strings include
claims that the audit found unsupported. This card reports that provenance but does not change
training data.

## Loading

This repository loads through the stock Transformers auto classes with remote code enabled:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
```

Tokenizer: byte-level. Context length: {context_length}.

{generation_note}

## Intended Use

Use this artefact only to inspect, reproduce, or audit the Aptivra codebase and export pipeline.
Do not use it for production language tasks, user-facing assistants, safety-critical systems, or
capability claims not backed by a fresh results record.

## Limitations

If a value above says "not measured", the generator did not receive that field in its results
record and deliberately did not invent it.
"""


def _mapping_at(record: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, dict) else None


def _positive_config_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _value_at(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number_at(record: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value = _value_at(record, path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _text_at(record: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _value_at(record, path)
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float | str):
        return str(value)
    return "not measured"


def _number_text_at(record: dict[str, Any], path: tuple[str, ...]) -> str:
    value = _value_at(record, path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "not measured"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.4g}"


def _bpb_summary(record: dict[str, Any], path: tuple[str, ...]) -> str:
    data = _mapping_at(record, path)
    if data is None:
        return "not measured"
    mean = data.get("mean")
    min_value = data.get("min")
    max_value = data.get("max")
    samples = data.get("samples")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool)
        for item in (mean, min_value, max_value)
    ):
        return "not measured"
    if not isinstance(samples, int) or isinstance(samples, bool):
        return "not measured"
    mean_number = cast(int | float, mean)
    min_number = cast(int | float, min_value)
    max_number = cast(int | float, max_value)
    return (
        f"mean {float(mean_number):.4g}; range "
        f"{float(min_number):.4g}-{float(max_number):.4g}; samples {samples}"
    )


def _bpb_range_summary(record: dict[str, Any], path: tuple[str, ...]) -> str:
    data = _mapping_at(record, path)
    if data is None:
        return "not measured"
    min_value = data.get("min")
    max_value = data.get("max")
    samples = data.get("samples")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool)
        for item in (min_value, max_value)
    ):
        return "not measured"
    if not isinstance(samples, int) or isinstance(samples, bool):
        return "not measured"
    min_number = cast(int | float, min_value)
    max_number = cast(int | float, max_value)
    return f"range {float(min_number):.4g}-{float(max_number):.4g}; samples {samples}"
