"""Measured-results model card renderer for Aptivra Hub exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from aptivra.model.config import ModelConfig

DEFAULT_RELEASE_GATE_CONFIG = Path("configs/model_card_release_gate.yaml")


def load_results_record(results_path: Path | str) -> dict[str, Any]:
    """Loads one machine-readable model-card evaluation record."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("model-card results record must be a JSON object")
    return data


def load_release_gate_config(config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG) -> float:
    """Returns the configured held-out BpB margin required to beat the unigram baseline."""
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
    return float(raw_margin)


def release_gate_allows_general_language_model(
    results_record: dict[str, Any],
    config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG,
) -> bool:
    """Returns whether the checkpoint beats the configured held-out unigram baseline gate."""
    heldout_bpb = _number_at(results_record, ("evaluation", "heldout_prompt_answer_bpb", "mean"))
    unigram_bpb = _number_at(results_record, ("evaluation", "baselines", "unigram_bpb"))
    if heldout_bpb is None or unigram_bpb is None:
        return False
    margin = load_release_gate_config(config_path)
    return heldout_bpb + margin < unigram_bpb


def generate_model_card_readme(
    repo_id: str,
    model_config: ModelConfig,
    num_experts: int,
    *,
    results_record: dict[str, Any] | None = None,
    results_path: Path | str | None = None,
    release_gate_config_path: Path | str = DEFAULT_RELEASE_GATE_CONFIG,
) -> str:
    """Renders a short model card from measured results only."""
    _ = (model_config, num_experts)
    record = results_record if results_record is not None else {}
    if results_path is not None:
        record = load_results_record(results_path)

    gate_passed = release_gate_allows_general_language_model(record, release_gate_config_path)
    trained_bpb = _bpb_summary(record, ("evaluation", "trained_prompt_answer_bpb"))
    heldout_bpb = _bpb_summary(record, ("evaluation", "heldout_prompt_answer_bpb"))
    uniform_bpb = _number_text_at(record, ("evaluation", "baselines", "uniform_bpb"))
    unigram_bpb = _number_text_at(record, ("evaluation", "baselines", "unigram_bpb"))
    notice = ""
    if not gate_passed:
        notice = (
            "\n> **Not a usable language model:** held-out answer-span bits per byte "
            "does not beat the unigram byte baseline, or that comparison was not measured. "
            "This checkpoint must not be described as a general language model.\n"
        )

    return f"""---
license: apache-2.0
library_name: aptivra
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

## Measured Facts

| Field | Value |
|---|---|
| Repository | `{repo_id}` |
| Evaluation timestamp | {_text_at(record, ("timestamp_utc",))} |
| Hub revision | {_text_at(record, ("checkpoint", "hub_revision"))} |
| Backbone SHA-256 | {_text_at(record, ("checkpoint", "backbone_sha256"))} |
| Backbone parameters | {_number_text_at(record, ("model", "backbone_parameters"))} |
| Router parameters | {_number_text_at(record, ("model", "router_parameters"))} |
| Expert parameters | {_number_text_at(record, ("model", "expert_parameters"))} |
| Total parameters | {_number_text_at(record, ("model", "total_parameters"))} |
| Position embedding | {_text_at(record, ("model", "position_embedding"))} |
| MoE enabled during evaluation | {_text_at(record, ("evaluation", "moe_enabled"))} |
| Trained-prompt answer-span BpB | {trained_bpb} |
| Held-out answer-span BpB | {heldout_bpb} |
| Held-out whole-text BpB | {_number_text_at(record, ("evaluation", "heldout_text_bpb"))} |
| Uniform byte baseline BpB | {uniform_bpb} |
| Unigram byte baseline BpB | {unigram_bpb} |

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

## Loading Interface Under Packaging Review

The packaging work is expected to make the following stock Transformers interface true:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
```

Tokenizer: byte-level. Context length: {_number_text_at(record, ("model", "context_length"))}.

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
