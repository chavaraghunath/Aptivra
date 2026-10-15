#!/usr/bin/env python3
"""Compute ADR-0011 candidate memory arithmetic from immutable cached configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

CAP_BYTES = 28_991_029_248
LORA_RANK = 16
RESIDENT_EXPERTS = 12
REPLAY_CACHE_BYTES = 500 * 8192 * 2 * 4


@dataclass(frozen=True)
class Candidate:
    """Immutable ADR-0008 candidate identity and local evidence binding."""

    name: str
    repository: str
    cache_directory: str
    revision: str
    target_suffixes: tuple[str, ...]
    measured_record: str | None


LLAMA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

CANDIDATES = (
    Candidate(
        name="GPT-2",
        repository="openai-community/gpt2",
        cache_directory="models--openai-community--gpt2",
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        target_suffixes=("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"),
        measured_record=(
            "experiments/base_model_candidates/amendment-4c/candidates/"
            "gpt2-238f51b7c7.json"
        ),
    ),
    Candidate(
        name="SmolLM2-135M",
        repository="HuggingFaceTB/SmolLM2-135M",
        cache_directory="models--HuggingFaceTB--SmolLM2-135M",
        revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        target_suffixes=LLAMA_TARGETS,
        measured_record=(
            "experiments/base_model_candidates/amendment-4c/candidates/"
            "smollm2-135m-55985fae75.json"
        ),
    ),
    Candidate(
        name="Pythia-160M",
        repository="EleutherAI/pythia-160m",
        cache_directory="models--EleutherAI--pythia-160m",
        revision="50f5173d932e8e61f858120bcb800b97af589f46",
        target_suffixes=(
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ),
        measured_record=(
            "experiments/base_model_candidates/amendment-4c/candidates/"
            "pythia-160m-c01af38c88.json"
        ),
    ),
    Candidate(
        name="SmolLM2-360M",
        repository="HuggingFaceTB/SmolLM2-360M",
        cache_directory="models--HuggingFaceTB--SmolLM2-360M",
        revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        target_suffixes=LLAMA_TARGETS,
        measured_record=(
            "experiments/base_model_candidates/amendment-4c/candidates/"
            "smollm2-360m-3156f262fc.json"
        ),
    ),
    Candidate(
        name="Qwen2.5-0.5B",
        repository="Qwen/Qwen2.5-0.5B",
        cache_directory="models--Qwen--Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        target_suffixes=LLAMA_TARGETS,
        measured_record=(
            "experiments/base_model_candidates/amendment-4f/candidates/"
            "qwen2.5-0.5b-02d40d8e2a.json"
        ),
    ),
    Candidate(
        name="Qwen2.5-1.5B",
        repository="Qwen/Qwen2.5-1.5B",
        cache_directory="models--Qwen--Qwen2.5-1.5B",
        revision="8faed761d45a263340a0528343f099c05c9a4323",
        target_suffixes=LLAMA_TARGETS,
        measured_record=None,
    ),
    Candidate(
        name="SmolLM2-1.7B",
        repository="HuggingFaceTB/SmolLM2-1.7B",
        cache_directory="models--HuggingFaceTB--SmolLM2-1.7B",
        revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
        target_suffixes=LLAMA_TARGETS,
        measured_record=None,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measured_peak(
    candidate: Candidate, repo_root: Path, parameters: int
) -> dict[str, Any] | None:
    if candidate.measured_record is None:
        return None
    record_path = repo_root / candidate.measured_record
    record = json.loads(record_path.read_text(encoding="utf-8"))
    identity = record["identity"]
    if record["status"] != "completed":
        raise ValueError(f"measured record is not complete: {record_path}")
    expected_identity = (
        candidate.repository,
        candidate.revision,
        parameters,
    )
    actual_identity = (
        identity["repo_id_or_path"],
        identity["resolved_revision"],
        identity["unique_parameters"],
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"measured record identity mismatch for {candidate.name}: {actual_identity!r}"
        )
    return {
        "bytes": int(record["cost"]["memory"]["hard_constraint_peak_bytes"]),
        "record": candidate.measured_record,
        "record_sha256": _sha256(record_path),
    }


def _projection(
    *,
    measured_peak: dict[str, Any] | None,
    parameters: int,
) -> dict[str, Any]:
    full_state = parameters * 16
    ewc_extra = parameters * 8
    if measured_peak is None:
        ewc_lower_bound = full_state + ewc_extra
        return {
            "naive_sequential_full_finetune": {
                "source": "not_computed",
                "static_lower_bound_bytes": full_state,
                "fits_cap": None,
            },
            "replay_full_finetune": {
                "source": "not_computed",
                "static_lower_bound_bytes": full_state + REPLAY_CACHE_BYTES,
                "fits_cap": None,
            },
            "online_ewc_full_finetune": {
                "source": "cached_config_static_lower_bound",
                "static_lower_bound_bytes": ewc_lower_bound,
                "fits_cap": False if ewc_lower_bound > CAP_BYTES else None,
            },
            "every_comparator_fits": False,
        }

    peak = int(measured_peak["bytes"])
    naive_fits = peak <= CAP_BYTES
    replay_peak = peak + REPLAY_CACHE_BYTES
    ewc_peak = peak + ewc_extra
    return {
        "naive_sequential_full_finetune": {
            "source": "measured_adr_0008_peak",
            "bytes": peak,
            "fits_cap": naive_fits,
        },
        "replay_full_finetune": {
            "source": "measured_adr_0008_peak_plus_replay_cache_bound",
            "bytes": replay_peak,
            "fits_cap": replay_peak <= CAP_BYTES,
        },
        "online_ewc_full_finetune": {
            "source": "measured_adr_0008_peak_plus_fisher_and_anchor",
            "bytes": ewc_peak,
            "fits_cap": ewc_peak <= CAP_BYTES,
        },
        "every_comparator_fits": all(
            value <= CAP_BYTES for value in (peak, replay_peak, ewc_peak)
        ),
    }


def _calculate_candidate(
    candidate: Candidate, cache_root: Path, repo_root: Path
) -> dict[str, Any]:
    snapshot = cache_root / candidate.cache_directory / "snapshots" / candidate.revision
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        return {
            "name": candidate.name,
            "repository": candidate.repository,
            "revision": candidate.revision,
            "status": "not_computed_config_not_cached",
        }

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    target_shape_counts: dict[str, int] = {}
    target_count = 0
    lora_parameters = 0
    for name, module in model.named_modules():
        if not any(name.endswith(suffix) for suffix in candidate.target_suffixes):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"LoRA target has no matrix weight: {name}")
        first, second = (int(dimension) for dimension in weight.shape)
        added = LORA_RANK * (first + second)
        lora_parameters += added
        target_count += 1
        shape = f"{first}x{second}"
        target_shape_counts[shape] = target_shape_counts.get(shape, 0) + 1

    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    expected_target_count = layers * len(candidate.target_suffixes)
    if target_count != expected_target_count:
        raise ValueError(
            f"expected {expected_target_count} LoRA targets for {candidate.name}, "
            f"found {target_count}"
        )

    measured_peak = _measured_peak(candidate, repo_root, parameters)
    first_expert_bytes = parameters * 4 + lora_parameters * 16
    twelve_expert_bytes = (
        parameters * 4
        + RESIDENT_EXPERTS * lora_parameters * 4
        + lora_parameters * 12
    )
    return {
        "name": candidate.name,
        "repository": candidate.repository,
        "revision": candidate.revision,
        "status": "computed",
        "config_sha256": _sha256(config_path),
        "model_type": config.model_type,
        "parameters": parameters,
        "lora": {
            "rank": LORA_RANK,
            "target_suffixes": list(candidate.target_suffixes),
            "target_count": target_count,
            "target_weight_shape_counts": target_shape_counts,
            "parameters_per_expert": lora_parameters,
            "formula": "rank * sum(weight_shape[0] + weight_shape[1])",
        },
        "aptivra_static": {
            "one_trainable_expert_bytes": first_expert_bytes,
            "one_trainable_expert_fits_cap": first_expert_bytes <= CAP_BYTES,
            "twelve_resident_one_trainable_bytes": twelve_expert_bytes,
            "twelve_resident_one_trainable_fits_cap": twelve_expert_bytes <= CAP_BYTES,
        },
        "adr_0008_measured_peak": measured_peak,
        "comparators": _projection(
            measured_peak=measured_peak,
            parameters=parameters,
        ),
    }


def calculate(cache_root: Path, repo_root: Path) -> dict[str, Any]:
    """Return deterministic arithmetic without loading or downloading model weights."""
    return {
        "schema_version": "adr-0011-candidate-matrix-arithmetic-v1",
        "cap_bytes": CAP_BYTES,
        "assumptions": {
            "fp32_bytes_per_value": 4,
            "full_finetune_state_bytes_per_parameter": 16,
            "online_ewc_extra_bytes_per_parameter": 8,
            "lora_rank": LORA_RANK,
            "resident_experts": RESIDENT_EXPERTS,
            "replay_cache_upper_bound_bytes": REPLAY_CACHE_BYTES,
            "measured_projection_caveat": (
                "Replay and online-EWC values add explicit persistent state to ADR-0008's "
                "measured full-fine-tuning peak; they are arithmetic projections, not "
                "measurements of those comparators."
            ),
            "static_lower_bound_caveat": (
                "Static bounds omit activations, allocator state, kernels, and temporary "
                "tensors. A bound below the cap does not establish runtime fit; a bound above "
                "the cap establishes non-fit."
            ),
        },
        "candidates": [
            _calculate_candidate(candidate, cache_root, repo_root)
            for candidate in CANDIDATES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
        help="Hugging Face hub cache root; the script never downloads missing snapshots",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing immutable ADR-0008 evidence",
    )
    args = parser.parse_args()
    result = calculate(args.cache_root.resolve(), args.repo_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
