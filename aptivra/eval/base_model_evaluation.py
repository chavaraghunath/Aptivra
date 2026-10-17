"""Pretrained-base evidence collection and deterministic selection policy.

The evaluator deliberately depends only on public Transformers auto classes for
third-party models.  Aptivra's existing byte export is registered with those
auto classes in-process so it can be measured through the same code path without
executing Hub-hosted Python.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import resource
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import HfApi, snapshot_download
from scipy.stats import t as student_t
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from aptivra.eval.harness import ContinualEvalHarness
from aptivra.eval.model_card_eval import load_prompt_answers, unigram_bits_per_bytes
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.v0.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.v0.huggingface.configuration_aptivra import AptivraConfig
from aptivra.v0.huggingface.modeling_aptivra import AptivraForCausalLM
from aptivra.v0.huggingface.tokenization_aptivra import AptivraByteTokenizer

EVALUATION_SCHEMA_VERSION = "pretrained-base-evaluation-v6"
SEED_EVALUATION_SCHEMA_VERSION = "pretrained-base-seed-evaluation-v1"
RANKING_SCHEMA_VERSION = "pretrained-base-ranking-v5"
MANIFEST_SCHEMA_VERSION = "pretrained-base-manifest-v2"
DERIVED_ANALYSIS_SCHEMA_VERSION = "pretrained-base-derived-analysis-v1"
AMENDMENT_3_ANALYSIS_SCHEMA_VERSION = "pretrained-base-amendment-3-analysis-v1"
AMENDMENT_4_EQUIVALENCE_SCHEMA_VERSION = "pretrained-base-amendment-4-equivalence-v1"
AMENDMENT_4A_EQUIVALENCE_SCHEMA_VERSION = "pretrained-base-amendment-4a-equivalence-v1"
AMENDMENT_4B_EQUIVALENCE_SCHEMA_VERSION = "pretrained-base-amendment-4b-equivalence-v1"
UNIFORM_BYTE_BASELINE_BPB = 8.0


class CausalLMOutput(Protocol):
    """Minimum output surface used by the evaluator."""

    logits: torch.Tensor


class CausalLM(Protocol):
    """Minimum causal model surface shared by Transformers implementations."""

    config: Any

    def __call__(self, **kwargs: torch.Tensor | bool) -> CausalLMOutput:
        """Run a causal-language-model forward pass."""

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:
        """Return model parameters."""

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterable[tuple[str, nn.Parameter]]:
        """Return named model parameters."""

    def named_modules(
        self, memo: set[nn.Module] | None = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterable[tuple[str, nn.Module]]:
        """Return named modules."""

    def train(self, mode: bool = True) -> object:
        """Set training mode."""

    def eval(self) -> object:
        """Set evaluation mode."""

    def to(self, device: torch.device | str) -> CausalLM:
        """Move the model to a device."""

    def get_decoder(self) -> nn.Module:
        """Return the decoder backbone without the language-model head."""

    def get_output_embeddings(self) -> nn.Module:
        """Return the output projection used as the language-model head."""


@dataclass(frozen=True)
class CandidateSpec:
    """One configured model source."""

    candidate_id: str
    source: str
    revision: str | None
    license: str | None
    expected_architecture: str | None
    reference_only: bool
    loader: str = "transformers"


@dataclass(frozen=True)
class MeasurementSettings:
    """Bounded and reproducible measurement settings."""

    device: str
    local_files_only: bool
    torch_dtype: str
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    seed: int
    variance_seeds: tuple[int, ...]
    variance_top_candidates: int
    train_items_per_family: int
    test_items_per_family: int
    validation_items_per_family: int
    max_generation_attempts: int
    max_new_tokens: int
    probe_steps: int
    probe_batch_size: int
    probe_learning_rates: tuple[float, ...]
    probe_warmup_steps: int
    probe_weight_decay: float
    evaluation_batch_size: int
    confidence_level: float
    generation_benchmark_prompts: int
    generation_benchmark_new_tokens: int
    generation_benchmark_warmup_runs: int
    generation_benchmark_repetitions: int
    scoring_stride_tokens: int
    load_retries: int
    load_retry_backoff_seconds: float
    memory_sample_interval_seconds: float
    candidate_timeout_seconds: float
    expected_wall_time_seconds: float
    utf8_roundtrip_samples: tuple[str, ...]
    heldout_text_path: Path
    heldout_prompts_path: Path
    pristine_model_device: str = "candidate_device"
    release_pristine_before_training: bool = False
    probe_micro_batch_sizes: tuple[int, ...] = (8, 4, 2, 1)
    target_only_training_loss: bool = True
    gradient_checkpointing_fallback: bool = True
    equivalence_candidate_ids: tuple[str, ...] = ("smollm2-135m", "gpt2")
    equivalence_steps: int = 3
    equivalence_loss_absolute_tolerance: float = 5e-5
    equivalence_update_max_absolute_tolerance: float = 1e-6
    equivalence_update_relative_l2_tolerance: float = 1e-4
    equivalence_exact_loss_absolute_tolerance: float = 1e-12
    equivalence_exact_gradient_relative_l2_tolerance: float = 1e-10
    equivalence_noise_floor_batches: int = 3
    equivalence_noise_floor_multiplier: float = 2.0
    equivalence_trajectory_steps: int = 50
    equivalence_protocol: str = "amendment_4a"
    equivalence_shape_control_chunk_size: int = 1
    equivalence_outcome_seeds: tuple[int, int] = (0, 1)
    swap_sample_interval_seconds: float = 0.5
    environment_minimum_free_memory_percent: float = 10.0
    environment_recovery_timeout_seconds: float = 900.0
    environment_max_attempts: int = 2
    prestart_minimum_free_memory_percent: float = 70.0
    prestart_max_other_process_rss_bytes: int = 2_147_483_648
    prestart_timeout_seconds: float = 900.0
    prestart_poll_interval_seconds: float = 5.0


@dataclass(frozen=True)
class DecisionSettings:
    """Precommitted hard constraints and ordered ranking thresholds."""

    permissive_licenses: tuple[str, ...]
    target_memory_bytes: int
    minimum_memory_headroom_fraction: float
    minimum_mean_learned_accuracy: float
    minimum_mean_accuracy_gain: float
    learnability_tie_margin: float
    clustered_confidence_interval_intersects_tie_region: bool
    maximum_swap_growth_bytes: int = 1_073_741_824
    require_complete_candidate_measurements: bool = False


@dataclass(frozen=True)
class ExecutionSettings:
    """Operational persistence controls for the evidence sweep."""

    output_dir: Path
    resume: bool
    allow_host_restart: bool = False


@dataclass(frozen=True)
class BaseModelEvaluationSettings:
    """Complete candidate evaluation configuration."""

    candidates: tuple[CandidateSpec, ...]
    measurement: MeasurementSettings
    decision: DecisionSettings
    execution: ExecutionSettings | None = None
    equivalence_gate_owner_override: dict[str, str] | None = None


@dataclass(frozen=True)
class LoadedCandidate:
    """A loaded model/tokenizer pair and its resolved source identity."""

    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    resolved_revision: str


@dataclass(frozen=True)
class TrainingRecipe:
    """Candidate-specific Amendment 4 memory recipe."""

    micro_batch_size: int
    gradient_checkpointing: bool


@dataclass
class _VectorMeasurement:
    """One internal loss/gradient/update observation."""

    loss: float
    gradients: list[torch.Tensor]
    updates: list[torch.Tensor]
    dropout_modules_disabled: int


@dataclass(frozen=True)
class SequenceScore:
    """Aggregate token likelihood normalized by represented UTF-8 bytes."""

    bits_per_byte: float
    total_bits: float
    scored_tokens: int
    scored_bytes: int
    windows: int


@dataclass(frozen=True)
class GenerationResult:
    """Decoded generation plus raw workload counts."""

    text: str
    generated_tokens: int


CandidateLoader = Callable[[CandidateSpec, MeasurementSettings], LoadedCandidate]


ROOT_KEYS = {"candidates", "measurement", "decision", "execution", "equivalence_gate"}
CANDIDATE_KEYS = {
    "id",
    "source",
    "revision",
    "license",
    "architecture",
    "reference_only",
    "loader",
    "trust_remote_code",
}
MEASUREMENT_KEYS = {
    "device",
    "pristine_model_device",
    "release_pristine_before_training",
    "local_files_only",
    "torch_dtype",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "seed",
    "variance_seeds",
    "variance_top_candidates",
    "train_items_per_family",
    "test_items_per_family",
    "validation_items_per_family",
    "max_generation_attempts",
    "max_new_tokens",
    "probe_steps",
    "probe_batch_size",
    "probe_micro_batch_sizes",
    "target_only_training_loss",
    "gradient_checkpointing_fallback",
    "equivalence_candidate_ids",
    "equivalence_steps",
    "equivalence_loss_absolute_tolerance",
    "equivalence_update_max_absolute_tolerance",
    "equivalence_update_relative_l2_tolerance",
    "equivalence_exact_loss_absolute_tolerance",
    "equivalence_exact_gradient_relative_l2_tolerance",
    "equivalence_noise_floor_batches",
    "equivalence_noise_floor_multiplier",
    "equivalence_trajectory_steps",
    "equivalence_protocol",
    "equivalence_shape_control_chunk_size",
    "equivalence_outcome_seeds",
    "probe_learning_rates",
    "probe_warmup_steps",
    "probe_weight_decay",
    "evaluation_batch_size",
    "confidence_level",
    "generation_benchmark_prompts",
    "generation_benchmark_new_tokens",
    "generation_benchmark_warmup_runs",
    "generation_benchmark_repetitions",
    "scoring_stride_tokens",
    "load_retries",
    "load_retry_backoff_seconds",
    "memory_sample_interval_seconds",
    "swap_sample_interval_seconds",
    "environment_minimum_free_memory_percent",
    "environment_recovery_timeout_seconds",
    "environment_max_attempts",
    "prestart_minimum_free_memory_percent",
    "prestart_max_other_process_rss_bytes",
    "prestart_timeout_seconds",
    "prestart_poll_interval_seconds",
    "candidate_timeout_seconds",
    "expected_wall_time_seconds",
    "utf8_roundtrip_samples",
    "heldout_text_path",
    "heldout_prompts_path",
}
DECISION_KEYS = {
    "permissive_licenses",
    "target_memory_bytes",
    "minimum_memory_headroom_fraction",
    "minimum_mean_learned_accuracy",
    "minimum_mean_accuracy_gain",
    "learnability_tie_margin",
    "clustered_confidence_interval_intersects_tie_region",
    "maximum_swap_growth_bytes",
    "require_complete_candidate_measurements",
}
EXECUTION_KEYS = {"output_dir", "resume", "allow_host_restart"}
OWNER_OVERRIDE_KEYS = {"by", "date", "adr_section"}


def _reject_unknown_keys(row: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(unknown)}")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an evidence input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_tree(path: Path) -> str:
    """Hash relative paths and bytes for every Python source below a directory."""
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_license(value: str) -> str:
    return value.strip().lower()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a YAML list")
    return value


def _string(value: object, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{name}' must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"'{name}' must be between {minimum} and {maximum}")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"'{name}' must be between {minimum} and {maximum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean")
    return value


def load_base_model_evaluation_settings(path: Path | str) -> BaseModelEvaluationSettings:
    """Load and strictly validate candidate, measurement, and decision configuration."""
    config_path = Path(path).resolve()
    root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "root")
    _reject_unknown_keys(root, ROOT_KEYS, "root")
    candidate_rows = _sequence(root.get("candidates"), "candidates")
    candidates: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for index, raw_candidate in enumerate(candidate_rows):
        row = _mapping(raw_candidate, f"candidates[{index}]")
        _reject_unknown_keys(row, CANDIDATE_KEYS, f"candidates[{index}]")
        candidate_id = cast(str, _string(row.get("id"), f"candidates[{index}].id"))
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        if row.get("trust_remote_code", False) is not False:
            raise ValueError("trust_remote_code must remain false for every candidate")
        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                source=cast(
                    str, _string(row.get("source"), f"candidates[{index}].source")
                ),
                revision=_string(
                    row.get("revision"), f"candidates[{index}].revision", allow_none=True
                ),
                license=_string(
                    row.get("license"), f"candidates[{index}].license", allow_none=True
                ),
                expected_architecture=_string(
                    row.get("architecture"),
                    f"candidates[{index}].architecture",
                    allow_none=True,
                ),
                reference_only=_boolean(
                    row.get("reference_only", False),
                    f"candidates[{index}].reference_only",
                ),
                loader=cast(
                    str,
                    _string(
                        row.get("loader", "transformers"),
                        f"candidates[{index}].loader",
                    ),
                ),
            )
        )
        if candidates[-1].loader not in {"transformers", "aptivra_legacy"}:
            raise ValueError(
                f"candidates[{index}].loader must be transformers or aptivra_legacy"
            )
    if not candidates:
        raise ValueError("at least one candidate is required")

    measurement = _mapping(root.get("measurement"), "measurement")
    decision = _mapping(root.get("decision"), "decision")
    execution = _mapping(root.get("execution"), "execution")
    _reject_unknown_keys(measurement, MEASUREMENT_KEYS, "measurement")
    _reject_unknown_keys(decision, DECISION_KEYS, "decision")
    _reject_unknown_keys(execution, EXECUTION_KEYS, "execution")
    owner_override: dict[str, str] | None = None
    if "equivalence_gate" in root:
        equivalence_gate = _mapping(root["equivalence_gate"], "equivalence_gate")
        _reject_unknown_keys(equivalence_gate, {"owner_override"}, "equivalence_gate")
        if "owner_override" in equivalence_gate:
            raw_override = _mapping(
                equivalence_gate["owner_override"],
                "equivalence_gate.owner_override",
            )
            _reject_unknown_keys(
                raw_override,
                OWNER_OVERRIDE_KEYS,
                "equivalence_gate.owner_override",
            )
            if set(raw_override) != OWNER_OVERRIDE_KEYS:
                raise ValueError(
                    "equivalence_gate.owner_override requires by, date, and adr_section"
                )
            owner_override = {
                key: cast(
                    str,
                    _string(
                        raw_override[key], f"equivalence_gate.owner_override.{key}"
                    ),
                )
                for key in ("by", "date", "adr_section")
            }
    samples = tuple(
        cast(str, _string(item, "measurement.utf8_roundtrip_samples[]"))
        for item in _sequence(
            measurement.get("utf8_roundtrip_samples"),
            "measurement.utf8_roundtrip_samples",
        )
    )
    if not samples:
        raise ValueError("measurement.utf8_roundtrip_samples must not be empty")
    variance_seeds = tuple(
        _integer(item, "measurement.variance_seeds[]", 0, 2**31 - 1)
        for item in _sequence(
            measurement.get("variance_seeds"), "measurement.variance_seeds"
        )
    )
    if len(variance_seeds) < 3 or len(set(variance_seeds)) != len(variance_seeds):
        raise ValueError("measurement.variance_seeds must contain at least three unique seeds")
    primary_seed = _integer(
        measurement.get("seed"), "measurement.seed", 0, 2**31 - 1
    )
    if variance_seeds[0] != primary_seed:
        raise ValueError("measurement.variance_seeds must start with measurement.seed")
    learning_rates = tuple(
        _number(item, "measurement.probe_learning_rates[]", 1e-8, 1.0)
        for item in _sequence(
            measurement.get("probe_learning_rates"),
            "measurement.probe_learning_rates",
        )
    )
    if not learning_rates or len(set(learning_rates)) != len(learning_rates):
        raise ValueError("measurement.probe_learning_rates must be non-empty and unique")
    probe_steps = _integer(
        measurement.get("probe_steps"), "measurement.probe_steps", 1, 10_000
    )
    probe_batch_size = _integer(
        measurement.get("probe_batch_size"),
        "measurement.probe_batch_size",
        1,
        256,
    )
    micro_batch_sizes = tuple(
        _integer(item, "measurement.probe_micro_batch_sizes[]", 1, probe_batch_size)
        for item in _sequence(
            measurement.get("probe_micro_batch_sizes"),
            "measurement.probe_micro_batch_sizes",
        )
    )
    if (
        not micro_batch_sizes
        or micro_batch_sizes[0] != probe_batch_size
        or tuple(sorted(set(micro_batch_sizes), reverse=True)) != micro_batch_sizes
        or micro_batch_sizes[-1] != 1
        or any(probe_batch_size % item != 0 for item in micro_batch_sizes)
    ):
        raise ValueError(
            "measurement.probe_micro_batch_sizes must be unique descending divisors "
            "of probe_batch_size, beginning with probe_batch_size and ending with 1"
        )
    warmup_steps = _integer(
        measurement.get("probe_warmup_steps"),
        "measurement.probe_warmup_steps",
        0,
        10_000,
    )
    if warmup_steps >= probe_steps:
        raise ValueError("measurement.probe_warmup_steps must be less than probe_steps")
    dtype_name = measurement.get("torch_dtype")
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("measurement.torch_dtype must be float32, float16, or bfloat16")
    pristine_model_device = measurement.get(
        "pristine_model_device", "candidate_device"
    )
    if pristine_model_device not in {"host", "candidate_device"}:
        raise ValueError(
            "measurement.pristine_model_device must be host or candidate_device"
        )
    licenses = tuple(
        cast(str, _string(item, "decision.permissive_licenses[]"))
        for item in _sequence(decision.get("permissive_licenses"), "permissive_licenses")
    )
    if not licenses:
        raise ValueError("decision.permissive_licenses must not be empty")
    equivalence_candidate_ids = tuple(
        cast(str, _string(item, "measurement.equivalence_candidate_ids[]"))
        for item in _sequence(
            measurement.get("equivalence_candidate_ids"),
            "measurement.equivalence_candidate_ids",
        )
    )
    selectable_ids = {
        candidate.candidate_id for candidate in candidates if not candidate.reference_only
    }
    if (
        not equivalence_candidate_ids
        or len(set(equivalence_candidate_ids)) != len(equivalence_candidate_ids)
        or any(item not in selectable_ids for item in equivalence_candidate_ids)
    ):
        raise ValueError(
            "measurement.equivalence_candidate_ids must be unique selectable candidates"
        )
    equivalence_protocol = cast(
        str,
        _string(
            measurement.get("equivalence_protocol", "amendment_4a"),
            "measurement.equivalence_protocol",
        ),
    )
    if equivalence_protocol not in {"amendment_4a", "amendment_4b"}:
        raise ValueError(
            "measurement.equivalence_protocol must be amendment_4a or amendment_4b"
        )
    outcome_seeds = tuple(
        _integer(item, "measurement.equivalence_outcome_seeds[]", 0, 2**31 - 1)
        for item in _sequence(
            measurement.get("equivalence_outcome_seeds", [0, 1]),
            "measurement.equivalence_outcome_seeds",
        )
    )
    if len(outcome_seeds) != 2 or outcome_seeds[0] == outcome_seeds[1]:
        raise ValueError(
            "measurement.equivalence_outcome_seeds must contain two distinct seeds"
        )

    return BaseModelEvaluationSettings(
        candidates=tuple(candidates),
        measurement=MeasurementSettings(
            device=cast(str, _string(measurement.get("device"), "measurement.device")),
            pristine_model_device=cast(str, pristine_model_device),
            release_pristine_before_training=_boolean(
                measurement.get("release_pristine_before_training", False),
                "measurement.release_pristine_before_training",
            ),
            local_files_only=_boolean(
                measurement.get("local_files_only"), "measurement.local_files_only"
            ),
            torch_dtype=cast(
                str, _string(measurement.get("torch_dtype"), "measurement.torch_dtype")
            ),
            deterministic_algorithms=_boolean(
                measurement.get("deterministic_algorithms"),
                "measurement.deterministic_algorithms",
            ),
            deterministic_warn_only=_boolean(
                measurement.get("deterministic_warn_only"),
                "measurement.deterministic_warn_only",
            ),
            seed=primary_seed,
            variance_seeds=variance_seeds,
            variance_top_candidates=_integer(
                measurement.get("variance_top_candidates"),
                "measurement.variance_top_candidates",
                1,
                len([item for item in candidates if not item.reference_only]),
            ),
            train_items_per_family=_integer(
                measurement.get("train_items_per_family"),
                "measurement.train_items_per_family",
                1,
                10_000,
            ),
            test_items_per_family=_integer(
                measurement.get("test_items_per_family"),
                "measurement.test_items_per_family",
                1,
                2_000,
            ),
            validation_items_per_family=_integer(
                measurement.get("validation_items_per_family"),
                "measurement.validation_items_per_family",
                1,
                2_000,
            ),
            max_generation_attempts=_integer(
                measurement.get("max_generation_attempts"),
                "measurement.max_generation_attempts",
                1,
                1_000,
            ),
            max_new_tokens=_integer(
                measurement.get("max_new_tokens"), "measurement.max_new_tokens", 1, 2_048
            ),
            probe_steps=probe_steps,
            probe_batch_size=probe_batch_size,
            probe_learning_rates=learning_rates,
            probe_warmup_steps=warmup_steps,
            probe_weight_decay=_number(
                measurement.get("probe_weight_decay"),
                "measurement.probe_weight_decay",
                0.0,
                1.0,
            ),
            evaluation_batch_size=_integer(
                measurement.get("evaluation_batch_size"),
                "measurement.evaluation_batch_size",
                1,
                512,
            ),
            confidence_level=_number(
                measurement.get("confidence_level"),
                "measurement.confidence_level",
                0.8,
                0.999,
            ),
            generation_benchmark_prompts=_integer(
                measurement.get("generation_benchmark_prompts"),
                "measurement.generation_benchmark_prompts",
                1,
                1_000,
            ),
            generation_benchmark_new_tokens=_integer(
                measurement.get("generation_benchmark_new_tokens"),
                "measurement.generation_benchmark_new_tokens",
                1,
                2_048,
            ),
            generation_benchmark_warmup_runs=_integer(
                measurement.get("generation_benchmark_warmup_runs"),
                "measurement.generation_benchmark_warmup_runs",
                0,
                100,
            ),
            generation_benchmark_repetitions=_integer(
                measurement.get("generation_benchmark_repetitions"),
                "measurement.generation_benchmark_repetitions",
                1,
                100,
            ),
            scoring_stride_tokens=_integer(
                measurement.get("scoring_stride_tokens"),
                "measurement.scoring_stride_tokens",
                1,
                32_768,
            ),
            load_retries=_integer(
                measurement.get("load_retries"),
                "measurement.load_retries",
                1,
                20,
            ),
            load_retry_backoff_seconds=_number(
                measurement.get("load_retry_backoff_seconds"),
                "measurement.load_retry_backoff_seconds",
                0.0,
                300.0,
            ),
            memory_sample_interval_seconds=_number(
                measurement.get("memory_sample_interval_seconds"),
                "measurement.memory_sample_interval_seconds",
                0.001,
                10.0,
            ),
            swap_sample_interval_seconds=_number(
                measurement.get("swap_sample_interval_seconds"),
                "measurement.swap_sample_interval_seconds",
                0.05,
                60.0,
            ),
            environment_minimum_free_memory_percent=_number(
                measurement.get("environment_minimum_free_memory_percent", 10.0),
                "measurement.environment_minimum_free_memory_percent",
                0.0,
                100.0,
            ),
            environment_recovery_timeout_seconds=_number(
                measurement.get("environment_recovery_timeout_seconds", 900.0),
                "measurement.environment_recovery_timeout_seconds",
                0.0,
                86_400.0,
            ),
            environment_max_attempts=_integer(
                measurement.get("environment_max_attempts", 2),
                "measurement.environment_max_attempts",
                1,
                20,
            ),
            prestart_minimum_free_memory_percent=_number(
                measurement.get("prestart_minimum_free_memory_percent", 70.0),
                "measurement.prestart_minimum_free_memory_percent",
                0.0,
                100.0,
            ),
            prestart_max_other_process_rss_bytes=_integer(
                measurement.get(
                    "prestart_max_other_process_rss_bytes", 2_147_483_648
                ),
                "measurement.prestart_max_other_process_rss_bytes",
                1,
                2**63 - 1,
            ),
            prestart_timeout_seconds=_number(
                measurement.get("prestart_timeout_seconds", 900.0),
                "measurement.prestart_timeout_seconds",
                0.0,
                86_400.0,
            ),
            prestart_poll_interval_seconds=_number(
                measurement.get("prestart_poll_interval_seconds", 5.0),
                "measurement.prestart_poll_interval_seconds",
                0.05,
                300.0,
            ),
            candidate_timeout_seconds=_number(
                measurement.get("candidate_timeout_seconds"),
                "measurement.candidate_timeout_seconds",
                1.0,
                86_400.0,
            ),
            expected_wall_time_seconds=_number(
                measurement.get("expected_wall_time_seconds"),
                "measurement.expected_wall_time_seconds",
                1.0,
                86_400.0,
            ),
            utf8_roundtrip_samples=samples,
            heldout_text_path=(config_path.parent / Path(
                cast(
                    str,
                    _string(
                        measurement.get("heldout_text_path"),
                        "measurement.heldout_text_path",
                    ),
                )
            )).resolve(),
            heldout_prompts_path=(config_path.parent / Path(
                cast(
                    str,
                    _string(
                        measurement.get("heldout_prompts_path"),
                        "measurement.heldout_prompts_path",
                    ),
                )
            )).resolve(),
            probe_micro_batch_sizes=micro_batch_sizes,
            target_only_training_loss=_boolean(
                measurement.get("target_only_training_loss"),
                "measurement.target_only_training_loss",
            ),
            gradient_checkpointing_fallback=_boolean(
                measurement.get("gradient_checkpointing_fallback"),
                "measurement.gradient_checkpointing_fallback",
            ),
            equivalence_candidate_ids=equivalence_candidate_ids,
            equivalence_steps=_integer(
                measurement.get("equivalence_steps"),
                "measurement.equivalence_steps",
                1,
                100,
            ),
            equivalence_loss_absolute_tolerance=_number(
                measurement.get("equivalence_loss_absolute_tolerance"),
                "measurement.equivalence_loss_absolute_tolerance",
                0.0,
                1.0,
            ),
            equivalence_update_max_absolute_tolerance=_number(
                measurement.get("equivalence_update_max_absolute_tolerance"),
                "measurement.equivalence_update_max_absolute_tolerance",
                0.0,
                1.0,
            ),
            equivalence_update_relative_l2_tolerance=_number(
                measurement.get("equivalence_update_relative_l2_tolerance"),
                "measurement.equivalence_update_relative_l2_tolerance",
                0.0,
                1.0,
            ),
            equivalence_exact_loss_absolute_tolerance=_number(
                measurement.get("equivalence_exact_loss_absolute_tolerance", 1e-12),
                "measurement.equivalence_exact_loss_absolute_tolerance",
                0.0,
                1.0,
            ),
            equivalence_exact_gradient_relative_l2_tolerance=_number(
                measurement.get(
                    "equivalence_exact_gradient_relative_l2_tolerance", 1e-10
                ),
                "measurement.equivalence_exact_gradient_relative_l2_tolerance",
                0.0,
                1.0,
            ),
            equivalence_noise_floor_batches=_integer(
                measurement.get("equivalence_noise_floor_batches", 3),
                "measurement.equivalence_noise_floor_batches",
                3,
                100,
            ),
            equivalence_noise_floor_multiplier=_number(
                measurement.get("equivalence_noise_floor_multiplier", 2.0),
                "measurement.equivalence_noise_floor_multiplier",
                1.0,
                10.0,
            ),
            equivalence_trajectory_steps=_integer(
                measurement.get("equivalence_trajectory_steps", 50),
                "measurement.equivalence_trajectory_steps",
                1,
                1_000,
            ),
            equivalence_protocol=equivalence_protocol,
            equivalence_shape_control_chunk_size=_integer(
                measurement.get("equivalence_shape_control_chunk_size", 1),
                "measurement.equivalence_shape_control_chunk_size",
                1,
                32_768,
            ),
            equivalence_outcome_seeds=outcome_seeds,
        ),
        decision=DecisionSettings(
            permissive_licenses=licenses,
            target_memory_bytes=_integer(
                decision.get("target_memory_bytes"),
                "decision.target_memory_bytes",
                1,
                2**63 - 1,
            ),
            minimum_memory_headroom_fraction=_number(
                decision.get("minimum_memory_headroom_fraction"),
                "decision.minimum_memory_headroom_fraction",
                0.0,
                0.95,
            ),
            minimum_mean_learned_accuracy=_number(
                decision.get("minimum_mean_learned_accuracy"),
                "decision.minimum_mean_learned_accuracy",
                0.0,
                1.0,
            ),
            minimum_mean_accuracy_gain=_number(
                decision.get("minimum_mean_accuracy_gain"),
                "decision.minimum_mean_accuracy_gain",
                0.0,
                1.0,
            ),
            learnability_tie_margin=_number(
                decision.get("learnability_tie_margin"),
                "decision.learnability_tie_margin",
                0.0,
                1.0,
            ),
            clustered_confidence_interval_intersects_tie_region=_boolean(
                decision.get("clustered_confidence_interval_intersects_tie_region"),
                "decision.clustered_confidence_interval_intersects_tie_region",
            ),
            maximum_swap_growth_bytes=_integer(
                decision.get("maximum_swap_growth_bytes"),
                "decision.maximum_swap_growth_bytes",
                0,
                2**63 - 1,
            ),
            require_complete_candidate_measurements=_boolean(
                decision.get("require_complete_candidate_measurements", False),
                "decision.require_complete_candidate_measurements",
            ),
        ),
        execution=ExecutionSettings(
            output_dir=(
                config_path.parent
                / Path(
                    cast(
                        str,
                        _string(execution.get("output_dir"), "execution.output_dir"),
                    )
                )
            ).resolve(),
            resume=_boolean(execution.get("resume"), "execution.resume"),
            allow_host_restart=_boolean(
                execution.get("allow_host_restart", False),
                "execution.allow_host_restart",
            ),
        ),
        equivalence_gate_owner_override=owner_override,
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic device without silently changing explicit requests."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("measurement.device must be auto, cpu, mps, or cuda")
    device = torch.device(requested)
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_preflight_candidate(
    candidate: CandidateSpec, settings: MeasurementSettings
) -> dict[str, object]:
    """Resolve immutable identity and licence before the manifest is frozen."""
    source_path = Path(candidate.source)
    if source_path.is_dir():
        revision = _local_revision(source_path)
        revision_source = "local_content_hash"
        hub_license: str | None = None
    elif candidate.revision is not None and re.fullmatch(r"[0-9a-f]{40}", candidate.revision):
        revision = candidate.revision
        revision_source = "configured_immutable_revision"
        hub_license = None
    else:
        if settings.local_files_only:
            raise ValueError(
                f"candidate {candidate.candidate_id} needs an immutable revision in offline mode"
            )
        last_error: Exception | None = None
        info: Any = None
        for attempt in range(settings.load_retries):
            try:
                info = HfApi().model_info(candidate.source, revision=candidate.revision)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < settings.load_retries:
                    time.sleep(settings.load_retry_backoff_seconds * (2**attempt))
        if last_error is not None:
            raise last_error
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Hub did not return an immutable revision for {candidate.source}")
        revision = sha
        revision_source = "hub_metadata"
        card_data = getattr(info, "card_data", None)
        raw_license = getattr(card_data, "license", None)
        hub_license = raw_license if isinstance(raw_license, str) else None
    supplied_license = candidate.license
    license_name = hub_license or supplied_license
    if license_name is None:
        raise ValueError(f"candidate {candidate.candidate_id} has no licence metadata or fallback")
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "resolved_revision": revision,
        "revision_source": revision_source,
        "license": _canonical_license(license_name),
        "license_source": "hub_metadata" if hub_license is not None else "config_fallback",
        "reference_only": candidate.reference_only,
    }


def prepare_run_manifest(
    config_path: Path | str,
    output_dir: Path | str,
    settings: BaseModelEvaluationSettings,
) -> tuple[dict[str, object], str]:
    """Preflight and atomically freeze the immutable evidence manifest."""
    config_file = Path(config_path).resolve()
    destination = Path(output_dir)
    candidates: list[dict[str, object]] = []
    for candidate in settings.candidates:
        try:
            candidates.append(_resolve_preflight_candidate(candidate, settings.measurement))
        except Exception as error:
            candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source": candidate.source,
                    "resolution_failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "reference_only": candidate.reference_only,
                }
            )
    repository_orchestrator = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluate_base_model_candidates.py"
    )
    installed_orchestrator = Path(__file__).resolve().with_name("vertex_base_sweep.py")
    orchestrator = (
        repository_orchestrator
        if repository_orchestrator.is_file()
        else installed_orchestrator
    )
    static: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_file),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "orchestrator_sha256": sha256_file(orchestrator),
        "task_sources_sha256": sha256_source_tree(
            Path(__file__).resolve().parents[1] / "tasks"
        ),
        "heldout_text_sha256": sha256_file(settings.measurement.heldout_text_path),
        "heldout_prompts_sha256": sha256_file(settings.measurement.heldout_prompts_path),
        "candidates": candidates,
        "seed": settings.measurement.seed,
        "equivalence_gate_owner_override": settings.equivalence_gate_owner_override,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": (
                "ephemeral-worker"
                if settings.execution is not None
                and settings.execution.allow_host_restart
                else socket.gethostname()
            ),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                {
                    "name": torch.cuda.get_device_name(0),
                    "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "runtime_version": torch.version.cuda,
                }
                if torch.cuda.is_available()
                else None
            ),
            "configured_device": settings.measurement.device,
            "pristine_model_device": settings.measurement.pristine_model_device,
        },
    }
    manifest_path = destination / "run_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        existing_static = {key: value for key, value in existing.items() if key != "correlation_id"}
        if existing_static == static:
            manifest = existing
        else:
            raise RuntimeError(
                "existing run manifest does not match current code/config/environment; "
                "use a new output directory so frozen evidence is not overwritten"
            )
    else:
        manifest = {**static, "correlation_id": str(uuid.uuid4())}
        write_json(manifest_path, manifest)
    payload = json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    return manifest, hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for repeatable evaluation and probes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(settings: MeasurementSettings) -> None:
    """Apply the configured Torch determinism policy before model work."""
    torch.use_deterministic_algorithms(
        settings.deterministic_algorithms,
        warn_only=settings.deterministic_warn_only,
    )


def _register_aptivra_auto_classes() -> None:
    """Register local Aptivra classes without enabling remote code execution."""
    AutoConfig.register("aptivra", AptivraConfig, exist_ok=True)
    AutoModelForCausalLM.register(AptivraConfig, AptivraForCausalLM, exist_ok=True)
    register_tokenizer = cast(Callable[..., None], AutoTokenizer.register)
    register_tokenizer(
        AptivraConfig,
        slow_tokenizer_class=AptivraByteTokenizer,
        exist_ok=True,
    )


def _local_revision(path: Path) -> str:
    """Hash a local model manifest and file contents into a stable revision identity."""
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


def load_candidate(
    candidate: CandidateSpec, settings: MeasurementSettings
) -> LoadedCandidate:
    """Load a configured source with standard auto classes and no remote code."""
    if candidate.loader == "aptivra_legacy":
        if Path(candidate.source).is_dir():
            snapshot = candidate.source
        else:
            snapshot = snapshot_download(
                repo_id=candidate.source,
                revision=candidate.revision,
                local_files_only=settings.local_files_only,
                allow_patterns=[
                    "config.json",
                    "multitier_manifest.json",
                    "dense/*.pt",
                    "experts/*.pt",
                ],
            )
        with tempfile.TemporaryDirectory(prefix="aptivra-base-reference-") as temporary:
            converted = Path(temporary) / "converted"
            convert_legacy_export_to_huggingface(
                snapshot,
                converted,
                source_hub_revision=candidate.revision,
            )
            local_candidate = CandidateSpec(
                candidate_id=candidate.candidate_id,
                source=str(converted),
                revision=None,
                license=candidate.license,
                expected_architecture=candidate.expected_architecture,
                reference_only=candidate.reference_only,
                loader="transformers",
            )
            loaded = load_candidate(local_candidate, settings)
        resolved_revision = candidate.revision or loaded.resolved_revision
        return LoadedCandidate(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            resolved_revision=resolved_revision,
        )
    _register_aptivra_auto_classes()
    common: dict[str, object] = {
        "revision": candidate.revision,
        "local_files_only": settings.local_files_only,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(candidate.source, **common)
    tokenizer = AutoTokenizer.from_pretrained(candidate.source, config=config, **common)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[settings.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        candidate.source, config=config, dtype=dtype, **common
    )
    typed_model = cast(CausalLM, model)
    if settings.pristine_model_device == "candidate_device":
        typed_model.to(resolve_device(settings.device))
    else:
        typed_model.to(torch.device("cpu"))
    typed_model.eval()
    commit_hash = getattr(config, "_commit_hash", None)
    source_path = Path(candidate.source)
    if isinstance(commit_hash, str) and commit_hash:
        resolved_revision = commit_hash
    elif source_path.is_dir():
        resolved_revision = _local_revision(source_path)
    elif candidate.revision is not None:
        resolved_revision = candidate.revision
    else:
        raise ValueError(
            f"candidate '{candidate.candidate_id}' did not expose a resolved revision hash"
        )
    architecture = getattr(config, "model_type", None)
    if (
        candidate.expected_architecture is not None
        and architecture != candidate.expected_architecture
    ):
        raise ValueError(
            f"expected architecture {candidate.expected_architecture}, loaded {architecture}"
        )
    if candidate.revision is not None and resolved_revision != candidate.revision:
        raise ValueError(
            f"resolved revision {resolved_revision} differs from pinned {candidate.revision}"
        )
    return LoadedCandidate(
        model=typed_model,
        tokenizer=tokenizer,
        resolved_revision=resolved_revision,
    )


def unique_parameter_count(model: CausalLM) -> int:
    """Count tied parameters once using storage identity plus view metadata."""
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    total = 0
    for _, parameter in model.named_parameters(remove_duplicate=False):
        identity = (
            parameter.untyped_storage().data_ptr(),
            int(parameter.storage_offset()),
            tuple(parameter.size()),
            tuple(parameter.stride()),
        )
        if identity not in seen:
            seen.add(identity)
            total += parameter.numel()
    return total


def model_context_length(model: CausalLM, tokenizer: PreTrainedTokenizerBase) -> int:
    """Read a finite context length from standard configuration fields."""
    for name in ("max_position_embeddings", "n_positions", "max_seq_len"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 1 < value < 10**9:
            return value
    value = tokenizer.model_max_length
    if isinstance(value, int) and 1 < value < 10**9:
        return value
    raise ValueError("model/tokenizer does not expose a finite context length")


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(encoded, list) or any(not isinstance(item, int) for item in encoded):
        raise TypeError("tokenizer.encode must return a list of integer token IDs")
    return cast(list[int], encoded)


def _decode(tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]) -> str:
    decoded = tokenizer.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise TypeError("tokenizer.decode must return a string for one token sequence")
    return decoded


def _decode_skipping_special(
    tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]
) -> str:
    decoded = tokenizer.decode(
        list(ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise TypeError("tokenizer.decode must return a string for one token sequence")
    return decoded


def _seed_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for value in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise ValueError("tokenizer needs BOS, EOS, or padding token to score the first text token")


def score_token_sequence(
    model: CausalLM,
    ids: Sequence[int],
    *,
    target_start: int,
    scored_bytes: int,
    context_length: int,
    device: torch.device,
    stride_tokens: int | None = None,
) -> SequenceScore:
    """Score every target token once in bounded maximum-left-context chunks.

    A window ends after the current target chunk and starts as far left as the
    context bound permits.  The configured stride is capped at
    ``context_length - 1`` so every target has at least one causal predecessor.
    Target chunks are disjoint, so no target token is duplicated or omitted.
    """
    if target_start <= 0 or target_start >= len(ids):
        raise ValueError("target_start must identify a non-empty target after context")
    if scored_bytes <= 0:
        raise ValueError("scored_bytes must be positive")
    if context_length < 2:
        raise ValueError("context_length must be at least two tokens")
    if stride_tokens is not None and stride_tokens < 1:
        raise ValueError("stride_tokens must be positive")
    stride = min(stride_tokens or context_length - 1, context_length - 1)
    cursor = target_start
    total_nats = 0.0
    total_tokens = 0
    windows = 0
    model.eval()
    while cursor < len(ids):
        chunk_end = min(len(ids), cursor + stride)
        window_end = chunk_end
        window_start = max(0, window_end - context_length)
        local_target_start = cursor - window_start
        window_ids = torch.tensor(
            [list(ids[window_start:window_end])], dtype=torch.long, device=device
        )
        with torch.no_grad():
            output = model(input_ids=window_ids, return_dict=True)
        logits = output.logits
        target_count = chunk_end - cursor
        prediction_logits = logits[
            0,
            local_target_start - 1 : local_target_start - 1 + target_count,
            :,
        ]
        targets = window_ids[0, local_target_start : local_target_start + target_count]
        if prediction_logits.shape[0] != targets.shape[0]:
            raise RuntimeError("causal score alignment failed")
        total_nats += float(F.cross_entropy(prediction_logits, targets, reduction="sum").item())
        count = int(targets.numel())
        total_tokens += count
        cursor += count
        windows += 1
    total_bits = total_nats / math.log(2)
    return SequenceScore(
        bits_per_byte=total_bits / scored_bytes,
        total_bits=total_bits,
        scored_tokens=total_tokens,
        scored_bytes=scored_bytes,
        windows=windows,
    )


def whole_text_bits_per_byte(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    context_length: int,
    device: torch.device,
    stride_tokens: int | None = None,
) -> SequenceScore:
    """Score all UTF-8 bytes by conditioning the first token on a local seed token."""
    text_bytes = text.encode("utf-8")
    if not text_bytes:
        raise ValueError("held-out text must not be empty")
    text_ids = _encode(tokenizer, text)
    if not text_ids:
        raise ValueError("held-out text tokenized to an empty sequence")
    ids = [_seed_token_id(tokenizer), *text_ids]
    return score_token_sequence(
        model,
        ids,
        target_start=1,
        scored_bytes=len(text_bytes),
        context_length=context_length,
        device=device,
        stride_tokens=stride_tokens,
    )


def answer_bits_per_byte(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer: str,
    *,
    context_length: int,
    device: torch.device,
    stride_tokens: int | None = None,
) -> SequenceScore:
    """Score losslessly boundary-isolated answer tokens after prompt tokens.

    Prompt and answer are encoded separately to make the scored byte span exact.
    The resulting token sequence must decode to the original concatenated text;
    this is an explicit boundary convention, not canonical joint tokenization.
    """
    answer_bytes = answer.encode("utf-8")
    if not answer_bytes:
        raise ValueError("answer must not be empty")
    prompt_ids = _encode(tokenizer, prompt)
    answer_ids = _encode(tokenizer, answer)
    if not answer_ids:
        raise ValueError("answer tokenized to an empty sequence")
    if not prompt_ids:
        prompt_ids = [_seed_token_id(tokenizer)]
    else:
        if _decode(tokenizer, prompt_ids) != prompt:
            raise ValueError("prompt tokenization is not an exact UTF-8 round trip")
        if _decode(tokenizer, answer_ids) != answer:
            raise ValueError("answer tokenization is not an exact UTF-8 round trip")
        if _decode(tokenizer, [*prompt_ids, *answer_ids]) != prompt + answer:
            raise ValueError("boundary-isolated prompt/answer tokenization is not lossless")
    return score_token_sequence(
        model,
        [*prompt_ids, *answer_ids],
        target_start=len(prompt_ids),
        scored_bytes=len(answer_bytes),
        context_length=context_length,
        device=device,
        stride_tokens=stride_tokens,
    )


def _score_summary(scores: Sequence[SequenceScore]) -> dict[str, object]:
    if not scores:
        raise ValueError("cannot summarize empty sequence scores")
    total_bits = sum(item.total_bits for item in scores)
    scored_bytes = sum(item.scored_bytes for item in scores)
    return {
        "bits_per_byte": total_bits / scored_bytes,
        "total_bits": total_bits,
        "scored_bytes": scored_bytes,
        "scored_tokens": sum(item.scored_tokens for item in scores),
        "windows": sum(item.windows for item in scores),
        "uniform_byte_baseline_bpb": UNIFORM_BYTE_BASELINE_BPB,
        "unigram_byte_baseline_bpb": None,
    }


def evaluate_language_quality(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
) -> dict[str, object]:
    """Measure whole-text and prompt/answer BPB against byte baselines."""
    text = settings.heldout_text_path.read_text(encoding="utf-8")
    text_score = whole_text_bits_per_byte(
        model,
        tokenizer,
        text,
        context_length=context_length,
        device=device,
        stride_tokens=settings.scoring_stride_tokens,
    )
    text_bytes = text.encode("utf-8")
    prompt_answers = load_prompt_answers(settings.heldout_prompts_path)
    answer_scores = [
        answer_bits_per_byte(
            model,
            tokenizer,
            item.prompt,
            item.answer,
            context_length=context_length,
            device=device,
            stride_tokens=settings.scoring_stride_tokens,
        )
        for item in prompt_answers
    ]
    answer_summary = _score_summary(answer_scores)
    answer_bytes = b"".join(item.answer.encode("utf-8") for item in prompt_answers)
    answer_summary["unigram_byte_baseline_bpb"] = unigram_bits_per_bytes(answer_bytes)
    combined_bits = text_score.total_bits + cast(float, answer_summary["total_bits"])
    combined_bytes = text_score.scored_bytes + cast(int, answer_summary["scored_bytes"])
    return {
        "heldout_text": {
            "bits_per_byte": text_score.bits_per_byte,
            "total_bits": text_score.total_bits,
            "scored_bytes": text_score.scored_bytes,
            "scored_tokens": text_score.scored_tokens,
            "windows": text_score.windows,
            "uniform_byte_baseline_bpb": UNIFORM_BYTE_BASELINE_BPB,
            "unigram_byte_baseline_bpb": unigram_bits_per_bytes(text_bytes),
            "windowing_rule": (
                "fixed target stride with maximum available left context; "
                "each target token and represented byte scored once"
            ),
            "first_token_conditioning": "tokenizer BOS, EOS, or padding seed token",
        },
        "heldout_prompt_answers": answer_summary,
        "combined": {
            "bits_per_byte": combined_bits / combined_bytes,
            "total_bits": combined_bits,
            "scored_bytes": combined_bytes,
            "boundary_convention": "lossless separately tokenized prompt and answer spans",
        },
    }


def tokenizer_fitness(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    configured_samples: Sequence[str],
) -> dict[str, object]:
    """Check the amended NFC and exact task-stream round-trip contracts."""
    task_texts: list[str] = []
    for family in stream.families:
        split = stream.get_split(family.family_id)
        for item in (*split.train, *split.val, *split.test):
            task_texts.extend((item.prompt, item.target_solution, item.training_target()))
    samples = [*configured_samples, *task_texts]

    def diagnose(sample: str, decoded: str, token_ids: list[int]) -> dict[str, object]:
        cleaned = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if not isinstance(cleaned, str):
            raise TypeError("tokenizer.decode must return a string for one token sequence")
        normalized_forms = {
            form: unicodedata.normalize(form, sample) for form in ("NFC", "NFD")
        }
        backend = getattr(tokenizer, "backend_tokenizer", None)
        normalizer = getattr(backend, "normalizer", None)
        cause = "unclassified_loss"
        if decoded == normalized_forms["NFC"] and sample != normalized_forms["NFC"]:
            cause = "tokenizer_unicode_normalization_nfc"
        elif decoded == normalized_forms["NFD"] and sample != normalized_forms["NFD"]:
            cause = "tokenizer_unicode_normalization_nfd"
        elif cleaned != decoded:
            cause = "decode_cleanup_setting"
        return {
            "input": sample,
            "decoded": decoded,
            "input_code_points": [f"U+{ord(character):04X}" for character in sample],
            "decoded_code_points": [f"U+{ord(character):04X}" for character in decoded],
            "token_ids": token_ids,
            "add_special_tokens": False,
            "decode_skip_special_tokens": False,
            "decoded_with_cleanup_enabled": cleaned,
            "decoded_with_cleanup_disabled": decoded,
            "backend_normalizer": str(normalizer) if normalizer is not None else None,
            "diagnosed_cause": cause,
        }

    raw_failures: list[dict[str, object]] = []
    for sample in samples:
        token_ids = _encode(tokenizer, sample)
        decoded = _decode(tokenizer, token_ids)
        if decoded != sample:
            raw_failures.append(diagnose(sample, decoded, token_ids))
    nfc_failures: list[dict[str, object]] = []
    for sample in configured_samples:
        normalized = unicodedata.normalize("NFC", sample)
        token_ids = _encode(tokenizer, normalized)
        decoded = _decode(tokenizer, token_ids)
        if decoded != normalized:
            nfc_failures.append(diagnose(normalized, decoded, token_ids))
    task_failures: list[dict[str, object]] = []
    for sample in task_texts:
        token_ids = _encode(tokenizer, sample)
        decoded = _decode(tokenizer, token_ids)
        if decoded != sample:
            task_failures.append(diagnose(sample, decoded, token_ids))
    total_bytes = sum(len(text.encode("utf-8")) for text in task_texts)
    total_tokens = sum(len(_encode(tokenizer, text)) for text in task_texts)
    eos_token_id = tokenizer.eos_token_id
    model_vocab_size = getattr(model.config, "vocab_size", None)
    configured_eos = getattr(model.config, "eos_token_id", None)
    configured_eos_ids = (
        configured_eos if isinstance(configured_eos, list) else [configured_eos]
    )
    eos_usable = (
        isinstance(eos_token_id, int)
        and not isinstance(eos_token_id, bool)
        and 0 <= eos_token_id < len(tokenizer)
        and isinstance(model_vocab_size, int)
        and eos_token_id < model_vocab_size
        and eos_token_id in configured_eos_ids
        and isinstance(tokenizer.eos_token, str)
        and bool(tokenizer.eos_token)
        and _decode(tokenizer, [eos_token_id]) == tokenizer.eos_token
    )
    return {
        "exact_utf8_roundtrip": not raw_failures,
        "exact_roundtrip_after_nfc_normalization": not nfc_failures,
        "exact_task_stream_roundtrip": not task_failures,
        "roundtrip_contract": (
            "configured samples round-trip after NFC normalization and every stored "
            "task-stream prompt, target, and training target round-trips exactly"
        ),
        "non_nfc_input_cost": (
            "canonically equivalent non-NFC input is normalized and its original "
            "code-point sequence is not preserved"
        ),
        "roundtrip_samples": len(samples),
        "roundtrip_failures": raw_failures[:10],
        "nfc_roundtrip_failures": nfc_failures[:10],
        "task_stream_roundtrip_failures": task_failures[:10],
        "task_stream_bytes": total_bytes,
        "task_stream_tokens": total_tokens,
        "mean_bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "eos_token_id": eos_token_id if eos_usable else None,
        "eos_token": tokenizer.eos_token if eos_usable else None,
        "eos_usable_as_answer_terminator": eos_usable,
        "model_eos_token_ids": configured_eos_ids,
    }


def inspect_expert_attachment_structure(
    model: CausalLM, hidden_size: int
) -> dict[str, object]:
    """Discover repeated blocks and expansion/contraction FFNs from tensor shapes.

    No architecture-specific module path or class name is used.  A feed-forward
    sublayer is a descendant group containing both a hidden-to-wider and a
    wider-to-hidden learned matrix for the same intermediate width.
    """
    module_lists = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) > 0
    ]
    repeated = [
        item
        for item in module_lists
        if len({type(child) for child in item[1]}) == 1
    ]
    if not repeated:
        return {
            "transformer_blocks": 0,
            "hidden_size": hidden_size,
            "all_blocks_expose_feed_forward": False,
            "block_container": None,
            "feed_forward_paths": [],
            "expected_transformer_blocks": None,
            "consistent_feed_forward_path": False,
        }
    block_name, blocks = max(repeated, key=lambda item: len(item[1]))
    paths: list[str | None] = []
    for block in blocks:
        groups: dict[str, list[tuple[int, int]]] = {}
        for leaf_name, leaf in block.named_modules():
            weight = getattr(leaf, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            parent = leaf_name.rsplit(".", 1)[0] if "." in leaf_name else "<block>"
            groups.setdefault(parent, []).append((int(weight.shape[0]), int(weight.shape[1])))
        discovered: str | None = None
        for group_name, shapes in groups.items():
            wider = {
                dimension
                for shape in shapes
                for dimension in shape
                if dimension > hidden_size and hidden_size in shape
            }
            if any(
                any(hidden_size in shape and width in shape for shape in shapes)
                and sum(hidden_size in shape and width in shape for shape in shapes) >= 2
                for width in wider
            ):
                discovered = group_name
                break
        paths.append(discovered)
    expected_blocks: int | None = None
    for name in ("num_hidden_layers", "n_layer", "n_layers"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            expected_blocks = value
            break
    consistent_path = bool(paths) and len(set(paths)) == 1 and paths[0] is not None
    count_matches = expected_blocks is None or expected_blocks == len(blocks)
    return {
        "transformer_blocks": len(blocks),
        "expected_transformer_blocks": expected_blocks,
        "hidden_size": hidden_size,
        "all_blocks_expose_feed_forward": (
            bool(paths)
            and all(path is not None for path in paths)
            and consistent_path
            and count_matches
        ),
        "consistent_feed_forward_path": consistent_path,
        "block_container": block_name,
        "feed_forward_paths": paths,
    }


def _hidden_size(model: CausalLM) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError("model config does not expose hidden size")


class TransformersPredictor:
    """Greedy predictor adapting byte stop characters to any reversible tokenizer."""

    def __init__(
        self,
        model: CausalLM,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: torch.device,
        context_length: int,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens

    def generate(
        self, prompt: str, *, stop_token_ids: tuple[int, ...]
    ) -> GenerationResult:
        """Generate greedily until EOS or a decoded family stop character."""
        return self.generate_batch([prompt], stop_token_ids=stop_token_ids)[0]

    def generate_batch(
        self, prompts: Sequence[str], *, stop_token_ids: tuple[int, ...]
    ) -> list[GenerationResult]:
        """Greedily generate a right-padded batch while preserving prompt positions."""
        if not prompts:
            return []
        stop_characters = tuple(chr(value) for value in stop_token_ids)
        seed_id = _seed_token_id(self.tokenizer)
        pad_id = self.tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            pad_id = self.tokenizer.eos_token_id
        if not isinstance(pad_id, int):
            pad_id = seed_id
        currents = []
        for prompt in prompts:
            prompt_ids = _encode(self.tokenizer, prompt) or [seed_id]
            currents.append(prompt_ids[-(self.context_length - 1) :])
        generated: list[list[int]] = [[] for _ in prompts]
        active = [True for _ in prompts]
        self.model.eval()
        for _ in range(self.max_new_tokens):
            maximum = max(len(item) for item in currents)
            input_rows = [item + [pad_id] * (maximum - len(item)) for item in currents]
            mask_rows = [[1] * len(item) + [0] * (maximum - len(item)) for item in currents]
            input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                mask_rows, dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            for index, is_active in enumerate(active):
                if not is_active:
                    continue
                last_index = len(currents[index]) - 1
                next_id = int(torch.argmax(output.logits[index, last_index, :]).item())
                generated[index].append(next_id)
                currents[index] = [*currents[index], next_id][-
                    (self.context_length - 1) :
                ]
                decoded = _decode(self.tokenizer, generated[index])
                if next_id == self.tokenizer.eos_token_id or any(
                    character in decoded for character in stop_characters
                ):
                    active[index] = False
            if not any(active):
                break
        return [
            GenerationResult(
                text=_decode_skipping_special(self.tokenizer, token_ids),
                generated_tokens=len(token_ids),
            )
            for token_ids in generated
        ]

    def __call__(self, prompt: str, *, stop_token_ids: tuple[int, ...]) -> str:
        """Satisfy ``ContinualEvalHarness``'s task-aware prediction protocol."""
        return self.generate(prompt, stop_token_ids=stop_token_ids).text


def wilson_interval(
    successes: int, total: int, confidence_level: float
) -> dict[str, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    if not 0 <= successes <= total:
        raise ValueError("Wilson successes must be between zero and total")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    proportion = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "confidence_level": confidence_level,
        "lower": max(0.0, centre - half_width),
        "upper": min(1.0, centre + half_width),
    }


def student_t_interval(
    values: Sequence[float], confidence_level: float
) -> dict[str, float | int | str]:
    """Estimate a mean interval with the family as the independent unit."""
    if len(values) < 2:
        raise ValueError("Student-t interval requires at least two clusters")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Student-t interval values must be finite")
    mean = statistics.mean(numeric)
    sample_standard_deviation = statistics.stdev(numeric)
    standard_error = sample_standard_deviation / math.sqrt(len(numeric))
    critical_value = float(
        student_t.ppf(0.5 + confidence_level / 2.0, df=len(numeric) - 1)
    )
    half_width = critical_value * standard_error
    return {
        "confidence_level": confidence_level,
        "clusters": len(numeric),
        "degrees_of_freedom": len(numeric) - 1,
        "mean": mean,
        "sample_standard_deviation": sample_standard_deviation,
        "standard_error": standard_error,
        "critical_value": critical_value,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "method": "two-sided Student-t interval across task-family values",
    }


def _family_metric_values(
    record: Mapping[str, object], metric: str
) -> dict[str, float]:
    families = _nested(record, "task_stream_readiness", "families")
    if not isinstance(families, list):
        raise TypeError("task_stream_readiness.families must be a list")
    values: dict[str, float] = {}
    for row in families:
        if not isinstance(row, dict):
            raise TypeError("family rows must be mappings")
        family_id = row.get("family_id")
        value = row.get(metric)
        if not isinstance(family_id, str) or not family_id:
            raise TypeError("every family row requires a non-empty family_id")
        if family_id in values:
            raise ValueError(f"duplicate family row: {family_id}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"family metric {metric} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"family metric {metric} must be finite")
        values[family_id] = numeric
    if len(values) < 2:
        raise ValueError("family-clustered analysis requires at least two families")
    return values


def _exact_two_sided_sign_test(positive: int, negative: int) -> dict[str, float | int | str]:
    non_tied = positive + negative
    if non_tied == 0:
        p_value = 1.0
    else:
        extreme = max(positive, negative)
        tail = sum(math.comb(non_tied, count) for count in range(extreme, non_tied + 1))
        p_value = min(1.0, 2.0 * tail / (2**non_tied))
    return {
        "positive": positive,
        "negative": negative,
        "tied": 0,
        "non_tied": non_tied,
        "two_sided_p_value": p_value,
        "method": "exact two-sided binomial sign test with ties excluded",
    }


def paired_family_comparison(
    first: Mapping[str, object],
    second: Mapping[str, object],
    metric: str,
    confidence_level: float,
) -> dict[str, object]:
    """Compare candidates with paired task-family differences as the sample."""
    first_values = _family_metric_values(first, metric)
    second_values = _family_metric_values(second, metric)
    if first_values.keys() != second_values.keys():
        missing_first = sorted(second_values.keys() - first_values.keys())
        missing_second = sorted(first_values.keys() - second_values.keys())
        raise ValueError(
            "paired family IDs differ: "
            f"missing from first={missing_first}, missing from second={missing_second}"
        )
    family_ids = list(first_values)
    differences = [first_values[item] - second_values[item] for item in family_ids]
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    sign_test = _exact_two_sided_sign_test(positive, negative)
    sign_test["tied"] = len(differences) - positive - negative
    return {
        "first_candidate_id": str(first.get("candidate_id")),
        "second_candidate_id": str(second.get("candidate_id")),
        "difference_definition": "first candidate minus second candidate",
        "metric": metric,
        "family_ids": family_ids,
        "family_differences": differences,
        "mean_difference_interval": student_t_interval(
            differences, confidence_level
        ),
        "sign_test": sign_test,
    }


def _template_signature(instance: TaskInstance) -> tuple[str, str]:
    function_name = instance.metadata.get("fn_name")
    if not isinstance(function_name, str):
        match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", instance.prompt)
        function_name = match.group(1) if match is not None else None
    if function_name is None:
        return instance.prompt, instance.target_solution
    return (
        instance.prompt.replace(function_name, "<function_name>"),
        instance.target_solution.replace(function_name, "<function_name>"),
    )


def compute_task_stream_properties(stream: ContinualTaskStream) -> dict[str, object]:
    """Describe target reuse and renamed-template reuse for a generated stream."""
    families: list[dict[str, object]] = []
    all_ascii = True
    for family in stream.families:
        split = stream.get_split(family.family_id)
        split_rows = {"train": split.train, "validation": split.val, "test": split.test}
        if any(not rows for rows in split_rows.values()):
            raise ValueError(f"task stream family {family.family_id} has an empty split")
        targets = {
            name: {item.target_solution for item in rows}
            for name, rows in split_rows.items()
        }
        templates = {
            name: {_template_signature(item) for item in rows}
            for name, rows in split_rows.items()
        }
        test_target_overlap = targets["test"] & targets["train"]
        test_template_overlap = templates["test"] & templates["train"]
        test_items_with_seen_target = sum(
            item.target_solution in targets["train"] for item in split.test
        )
        test_items_with_seen_template = sum(
            _template_signature(item) in templates["train"] for item in split.test
        )
        family_ascii = all(
            item.prompt.isascii() and item.target_solution.isascii()
            for rows in split_rows.values()
            for item in rows
        )
        all_ascii = all_ascii and family_ascii
        test_target_count = len(targets["test"])
        families.append(
            {
                "family_id": family.family_id,
                "distinct_targets": {
                    name: len(values) for name, values in targets.items()
                },
                "uniform_over_distinct_test_targets_chance_level": (
                    1.0 / test_target_count
                ),
                "test_target_overlap_with_train": {
                    "distinct_targets": len(test_target_overlap),
                    "distinct_target_fraction": len(test_target_overlap)
                    / test_target_count,
                    "items": test_items_with_seen_target,
                    "item_fraction": test_items_with_seen_target / len(split.test),
                },
                "distinct_template_signatures": {
                    name: len(values) for name, values in templates.items()
                },
                "test_template_overlap_with_train": {
                    "distinct_templates": len(test_template_overlap),
                    "distinct_template_fraction": len(test_template_overlap)
                    / len(templates["test"]),
                    "items": test_items_with_seen_template,
                    "item_fraction": test_items_with_seen_template / len(split.test),
                },
                "all_prompt_and_target_text_ascii": family_ascii,
            }
        )
    return {
        "families": families,
        "all_prompt_and_target_text_ascii": all_ascii,
        "chance_level_definition": (
            "uniform random choice over distinct targets present in the test split; "
            "descriptive only and most relevant when the target set is small"
        ),
        "template_signature_definition": (
            "exact prompt/target pair after replacing a generated Python function "
            "identifier with <function_name> when one is present"
        ),
    }


def _accuracy_record(
    successes: int, total: int, confidence_level: float
) -> dict[str, object]:
    return {
        "accuracy": successes / total,
        "successes": successes,
        "total": total,
        "wilson_interval": wilson_interval(successes, total, confidence_level),
    }


def evaluate_batched_accuracy(
    predictor: TransformersPredictor,
    family: TaskFamily,
    instances: Sequence[TaskInstance],
    *,
    batch_size: int,
    confidence_level: float,
) -> dict[str, object]:
    """Evaluate one family in generation batches and retain binomial evidence."""
    if not instances:
        raise ValueError("cannot evaluate an empty task split")
    successes = 0
    item_outcomes: list[dict[str, object]] = []
    for offset in range(0, len(instances), batch_size):
        batch = instances[offset : offset + batch_size]
        outputs = predictor.generate_batch(
            [item.prompt for item in batch], stop_token_ids=family.stop_token_ids
        )
        for instance, generated in zip(batch, outputs, strict=True):
            result = ContinualEvalHarness.verify_generated_output(
                family, instance, generated.text
            )
            successes += int(result.passed)
            item_outcomes.append(
                {
                    "task_id": instance.task_id,
                    "passed": result.passed,
                }
            )
    record = _accuracy_record(successes, len(instances), confidence_level)
    record["item_outcomes"] = item_outcomes
    return record


def evaluate_zero_shot_readiness(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
) -> dict[str, object]:
    """Measure held-out task accuracy before allocating any training state."""
    predictor = TransformersPredictor(
        model,
        tokenizer,
        device=device,
        context_length=context_length,
        max_new_tokens=settings.max_new_tokens,
    )
    families = [
        {
            "family_id": family.family_id,
            "zero_shot": evaluate_batched_accuracy(
                predictor,
                family,
                stream.get_split(family.family_id).test,
                batch_size=settings.evaluation_batch_size,
                confidence_level=settings.confidence_level,
            ),
        }
        for family in stream.families
    ]
    successes = sum(
        cast(int, cast(dict[str, object], item["zero_shot"])["successes"])
        for item in families
    )
    total = sum(
        cast(int, cast(dict[str, object], item["zero_shot"])["total"])
        for item in families
    )
    mean = _accuracy_record(successes, total, settings.confidence_level)
    return {
        "families": families,
        "mean_zero_shot": mean,
        "mean_zero_shot_accuracy": mean["accuracy"],
        "heldout_split": "test",
        "evaluation_generation": "right-padded batched greedy decoding",
        "measurement_timing": "before any learnability-probe training allocation",
    }


def _training_example(
    tokenizer: PreTrainedTokenizerBase,
    instance: TaskInstance,
    context_length: int,
) -> tuple[list[int], list[int], int]:
    prompt_ids = _encode(tokenizer, instance.prompt)
    target_text = instance.training_target()
    target_ids = _encode(tokenizer, target_text)
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, int) and (not target_ids or target_ids[-1] != eos_id):
        target_ids.append(eos_id)
    if not target_ids:
        raise ValueError("training target tokenized to an empty sequence")
    if not prompt_ids:
        prompt_ids = [_seed_token_id(tokenizer)]
    combined = [*prompt_ids, *target_ids]
    if len(combined) > context_length:
        raise ValueError(
            "training example exceeds context length; supervision is not silently truncated"
        )
    prompt_length = len(prompt_ids)
    labels = [-100] * prompt_length + combined[prompt_length:]
    return combined, labels, len(instance.prompt.encode("utf-8")) + len(
        target_text.encode("utf-8")
    )


def _training_batch(
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    examples = [_training_example(tokenizer, item, context_length) for item in instances]
    pad_id = tokenizer.pad_token_id
    if not isinstance(pad_id, int):
        pad_id = tokenizer.eos_token_id
    if not isinstance(pad_id, int):
        raise ValueError("training requires a padding or EOS token")
    max_length = max(len(ids) for ids, _, _ in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ids, labels, _ in examples:
        padding = max_length - len(ids)
        input_rows.append(ids + [pad_id] * padding)
        label_rows.append(labels + [-100] * padding)
        mask_rows.append([1] * len(ids) + [0] * padding)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
        sum(byte_count for _, _, byte_count in examples),
    )


def _target_only_loss_mean(
    model: CausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Compute causal loss after selecting supervised hidden-state positions.

    The decoder runs for the complete prompt and target because causal context is
    required, but the vocabulary projection is applied only where shifted labels
    carry supervision. This avoids materializing prompt-position logits.
    """
    decoder = model.get_decoder()
    output = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("causal decoder did not return last_hidden_state")
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != -100
    target_count = int(supervised.sum().item())
    if target_count == 0:
        raise ValueError("training batch contains no supervised target positions")
    selected_hidden_rows: list[torch.Tensor] = []
    selected_label_rows: list[torch.Tensor] = []
    for row_index in range(shifted_labels.shape[0]):
        positions = torch.nonzero(supervised[row_index], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        start = int(positions[0].item())
        stop = int(positions[-1].item()) + 1
        if positions.numel() != stop - start:
            raise ValueError("supervised target positions must be contiguous per example")
        selected_hidden_rows.append(hidden_states[row_index, start:stop, :])
        selected_label_rows.append(shifted_labels[row_index, start:stop])
    selected_hidden = torch.cat(selected_hidden_rows, dim=0)
    selected_labels = torch.cat(selected_label_rows, dim=0)
    logits = model.get_output_embeddings()(selected_hidden)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("language-model head did not return logits")
    return F.cross_entropy(logits, selected_labels, reduction="mean"), target_count


def _set_gradient_checkpointing(model: CausalLM, enabled: bool) -> None:
    method_name = (
        "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    )
    method = getattr(model, method_name, None)
    if not callable(method):
        if enabled:
            raise TypeError("candidate does not expose gradient checkpointing")
        return
    method()
    if enabled and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def _backward_target_only(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler | None = None,
) -> dict[str, float | int]:
    """Backpropagate one token-weighted target-only effective batch."""
    batches = [
        _training_batch(
            tokenizer,
            instances[offset : offset + recipe.micro_batch_size],
            context_length,
            device,
        )
        for offset in range(0, len(instances), recipe.micro_batch_size)
    ]
    supervised_tokens = sum(int((labels != -100).sum().item()) for _, labels, _, _ in batches)
    if supervised_tokens == 0:
        raise ValueError("effective training batch contains no supervised target tokens")
    loss_sum_value = 0.0
    processed_tokens = 0
    trained_bytes = 0
    for input_ids, labels, attention_mask, byte_count in batches:
        loss_mean, target_count = _target_only_loss_mean(
            model, input_ids, labels, attention_mask
        )
        if target_count != int((labels[:, 1:] != -100).sum().item()):
            raise RuntimeError("target-only loss selected an inconsistent token count")
        token_fraction = target_count / supervised_tokens
        torch.autograd.backward(loss_mean * token_fraction)
        loss_sum_value += float(loss_mean.detach().item()) * target_count
        processed_tokens += int(attention_mask.sum().item())
        trained_bytes += byte_count
        if resource_guard is not None:
            resource_guard.raise_if_swap_limit_exceeded()
    return {
        "loss": loss_sum_value / supervised_tokens,
        "processed_tokens": processed_tokens,
        "supervised_tokens": supervised_tokens,
        "trained_bytes": trained_bytes,
        "accumulation_steps": len(batches),
    }


def _optimizer_step_target_only(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler | None = None,
) -> dict[str, float | int]:
    """Run one effective-batch step with token-weighted gradient accumulation."""
    optimizer.zero_grad(set_to_none=True)
    result = _backward_target_only(
        model,
        tokenizer,
        instances,
        context_length,
        device,
        recipe,
        resource_guard=resource_guard,
    )
    optimizer.step()
    if resource_guard is not None:
        resource_guard.raise_if_swap_limit_exceeded()
    return result


def _backward_legacy(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
) -> float:
    """Backpropagate the frozen Amendment 1 full-logit effective batch."""
    input_ids, labels, attention_mask, _ = _training_batch(
        tokenizer, instances, context_length, device
    )
    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        return_dict=True,
    )
    loss = getattr(output, "loss", None)
    if not isinstance(loss, torch.Tensor):
        raise TypeError("causal model did not return a training loss")
    torch.autograd.backward(loss)
    return float(loss.detach().item())


def _backward_legacy_full_logits_exact(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
) -> float:
    """Backpropagate full-logit cross-entropy without implementation dtype casts."""
    input_ids, labels, attention_mask, _ = _training_batch(
        tokenizer, instances, context_length, device
    )
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("causal model did not return logits")
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != -100
    target_count = int(supervised.sum().item())
    if target_count == 0:
        raise ValueError("training batch contains no supervised target positions")
    loss = F.cross_entropy(
        logits[:, :-1, :][supervised],
        shifted_labels[supervised],
        reduction="sum",
    ) / target_count
    torch.autograd.backward(loss)
    return float(loss.detach().item())


def _backward_legacy_chunked_head(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
    sequence_chunk_size: int,
) -> float:
    """Backpropagate the legacy objective through sequence-shaped head chunks."""
    if sequence_chunk_size < 1:
        raise ValueError("sequence chunk size must be positive")
    input_ids, labels, attention_mask, _ = _training_batch(
        tokenizer, instances, context_length, device
    )
    decoder = model.get_decoder()
    output = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("causal decoder did not return last_hidden_state")
    shifted_labels = labels[:, 1:]
    target_count = int((shifted_labels != -100).sum().item())
    if target_count == 0:
        raise ValueError("training batch contains no supervised target positions")
    loss_sum: torch.Tensor | None = None
    sequence_positions = shifted_labels.shape[1]
    for start in range(0, sequence_positions, sequence_chunk_size):
        stop = min(start + sequence_chunk_size, sequence_positions)
        logits = model.get_output_embeddings()(hidden_states[:, start:stop, :])
        if not isinstance(logits, torch.Tensor):
            raise TypeError("language-model head did not return logits")
        chunk_labels = shifted_labels[:, start:stop]
        chunk_targets = int((chunk_labels != -100).sum().item())
        if chunk_targets == 0:
            continue
        chunk_loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
    if loss_sum is None:
        raise RuntimeError("chunked legacy loss found no supervised target chunk")
    loss = loss_sum / target_count
    torch.autograd.backward(loss)
    return float(loss.detach().item())


def _optimizer_step_legacy(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run the frozen Amendment 1 full-logit batch-8 step for equivalence proof."""
    optimizer.zero_grad(set_to_none=True)
    loss = _backward_legacy(model, tokenizer, instances, context_length, device)
    optimizer.step()
    return loss


def _optimizer_step_for_measurement(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    context_length: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    settings: MeasurementSettings,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler | None,
) -> dict[str, float | int]:
    """Run either the Amendment 4 recipe or frozen Amendment 1 confirmation recipe."""
    if settings.target_only_training_loss:
        return _optimizer_step_target_only(
            model,
            tokenizer,
            instances,
            context_length,
            device,
            optimizer,
            recipe,
            resource_guard=resource_guard,
        )
    if recipe != TrainingRecipe(settings.probe_batch_size, False):
        raise ValueError("Amendment 1 confirmation requires full batch without checkpointing")
    processed_tokens = 0
    supervised_tokens = 0
    trained_bytes = 0
    for instance in instances:
        ids, labels, byte_count = _training_example(tokenizer, instance, context_length)
        processed_tokens += len(ids)
        supervised_tokens += sum(label != -100 for label in labels[1:])
        trained_bytes += byte_count
    loss = _optimizer_step_legacy(
        model, tokenizer, instances, context_length, device, optimizer
    )
    if resource_guard is not None:
        resource_guard.raise_if_swap_limit_exceeded()
    return {
        "loss": loss,
        "processed_tokens": processed_tokens,
        "supervised_tokens": supervised_tokens,
        "trained_bytes": trained_bytes,
        "accumulation_steps": 1,
    }


def synchronize_device(device: torch.device) -> None:
    """Synchronize accelerator work before or after a timed interval."""
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_device(device: torch.device) -> None:
    """Release Python and accelerator caches between isolated workloads."""
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def model_state_digest(model: CausalLM) -> str:
    """Hash parameter names, shapes, dtypes, and exact values."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_pristine_snapshot(model: CausalLM, path: Path, device: torch.device) -> str:
    """Persist one CPU snapshot, leaving no second live model in unified memory."""
    model.to(torch.device("cpu"))
    cleanup_device(device)
    digest = model_state_digest(model)
    torch.save(cast(Any, model).state_dict(), path)
    return digest


def _restore_pristine_snapshot(
    model: CausalLM, path: Path, device: torch.device, expected_digest: str
) -> None:
    """Restore the reusable training object from a disk-backed pristine snapshot."""
    model.to(torch.device("cpu"))
    cleanup_device(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = cast(Any, model).load_state_dict(state, strict=True)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("pristine snapshot did not restore every model parameter")
    if model_state_digest(model) != expected_digest:
        raise RuntimeError("restored pristine model digest does not match the snapshot")
    model.to(device)


def run_learnability_probe(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    family_index: int,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    *,
    learning_rate: float | None = None,
    zero_shot: Mapping[str, object] | None = None,
    evaluate_test: bool = True,
    recipe: TrainingRecipe | None = None,
    resource_guard: MemoryPeakSampler | None = None,
    pristine_snapshot_path: Path | None = None,
    pristine_snapshot_digest: str | None = None,
) -> dict[str, object]:
    """Train an isolated family probe, optionally reserving test for the final LR."""
    seed_everything(settings.seed + family_index)
    family = stream.families[family_index]
    split = stream.get_split(family.family_id)
    selected_learning_rate = (
        settings.probe_learning_rates[0] if learning_rate is None else learning_rate
    )
    selected_recipe = recipe or TrainingRecipe(settings.probe_batch_size, False)
    if settings.probe_batch_size % selected_recipe.micro_batch_size != 0:
        raise ValueError("micro-batch size must divide the effective probe batch")
    if pristine_snapshot_path is not None:
        if pristine_snapshot_digest is None:
            raise ValueError("a pristine snapshot path requires its model digest")
        _restore_pristine_snapshot(
            base_model, pristine_snapshot_path, device, pristine_snapshot_digest
        )
    before: dict[str, object] | None = None
    if evaluate_test:
        base_predictor = TransformersPredictor(
            base_model,
            tokenizer,
            device=device,
            context_length=context_length,
            max_new_tokens=settings.max_new_tokens,
        )
        before = (
            dict(zero_shot)
            if zero_shot is not None
            else evaluate_batched_accuracy(
                base_predictor,
                family,
                split.test,
                batch_size=settings.evaluation_batch_size,
                confidence_level=settings.confidence_level,
            )
        )
    probe = (
        copy.deepcopy(base_model).to(device)
        if pristine_snapshot_path is None
        else base_model
    )
    probe.train()
    _set_gradient_checkpointing(probe, selected_recipe.gradient_checkpointing)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=selected_learning_rate,
        weight_decay=settings.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, settings.probe_warmup_steps), 1.0
        ),
    )
    processed_tokens = 0
    supervised_tokens = 0
    trained_bytes = 0
    losses: list[float] = []
    synchronize_device(device)
    started = time.perf_counter()
    for step in range(settings.probe_steps):
        offset = (step * settings.probe_batch_size) % len(split.train)
        batch = [
            split.train[(offset + index) % len(split.train)]
            for index in range(settings.probe_batch_size)
        ]
        step_result = _optimizer_step_for_measurement(
            probe,
            tokenizer,
            batch,
            context_length,
            device,
            optimizer,
            settings,
            selected_recipe,
            resource_guard=resource_guard,
        )
        scheduler.step()
        losses.append(cast(float, step_result["loss"]))
        processed_tokens += cast(int, step_result["processed_tokens"])
        supervised_tokens += cast(int, step_result["supervised_tokens"])
        trained_bytes += cast(int, step_result["trained_bytes"])
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    probe.eval()
    after_predictor = TransformersPredictor(
        probe,
        tokenizer,
        device=device,
        context_length=context_length,
        max_new_tokens=settings.max_new_tokens,
    )
    validation = evaluate_batched_accuracy(
        after_predictor,
        family,
        split.val,
        batch_size=settings.evaluation_batch_size,
        confidence_level=settings.confidence_level,
    )
    after = (
        evaluate_batched_accuracy(
            after_predictor,
            family,
            split.test,
            batch_size=settings.evaluation_batch_size,
            confidence_level=settings.confidence_level,
        )
        if evaluate_test
        else None
    )
    del optimizer, scheduler, probe
    cleanup_device(device)
    result: dict[str, object] = {
        "family_id": family.family_id,
        "learning_rate": selected_learning_rate,
        "validation": validation,
        "test_evaluated": evaluate_test,
        "steps": settings.probe_steps,
        "batch_size": settings.probe_batch_size,
        "effective_batch_size": settings.probe_batch_size,
        "micro_batch_size": selected_recipe.micro_batch_size,
        "gradient_accumulation_steps": (
            settings.probe_batch_size // selected_recipe.micro_batch_size
        ),
        "gradient_checkpointing": selected_recipe.gradient_checkpointing,
        "loss_materialization": "target positions only before language-model head",
        "warmup_steps": settings.probe_warmup_steps,
        "seconds": elapsed,
        "processed_nonpadding_tokens": processed_tokens,
        "supervised_target_tokens": supervised_tokens,
        "training_bytes": trained_bytes,
        "training_tokens_per_second": processed_tokens / elapsed,
        "training_bytes_per_second": trained_bytes / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    if before is not None and after is not None:
        before_interval = cast(dict[str, float], before["wilson_interval"])
        after_interval = cast(dict[str, float], after["wilson_interval"])
        before_accuracy = cast(float, before["accuracy"])
        after_accuracy = cast(float, after["accuracy"])
        result.update(
            {
                "zero_shot": before,
                "learned": after,
                "zero_shot_accuracy": before_accuracy,
                "learned_accuracy": after_accuracy,
                "accuracy_gain": after_accuracy - before_accuracy,
                "accuracy_gain_interval": {
                    "confidence_level": settings.confidence_level,
                    "lower": after_interval["lower"] - before_interval["upper"],
                    "upper": after_interval["upper"] - before_interval["lower"],
                    "method": "Newcombe interval from Wilson component intervals",
                },
            }
        )
    return result


def evaluate_task_readiness(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    *,
    zero_shot_readiness: Mapping[str, object] | None = None,
    recipe: TrainingRecipe | None = None,
    resource_guard: MemoryPeakSampler | None = None,
    pristine_snapshot_path: Path | None = None,
    pristine_snapshot_digest: str | None = None,
) -> dict[str, object]:
    """Select one candidate-wide LR on validation, then report held-out test results."""
    before_digest = model_state_digest(model)
    selection_records: list[dict[str, object]] = []
    for learning_rate in settings.probe_learning_rates:
        runs = [
            run_learnability_probe(
                model,
                tokenizer,
                stream,
                index,
                settings,
                context_length,
                device,
                learning_rate=learning_rate,
                evaluate_test=False,
                recipe=recipe,
                resource_guard=resource_guard,
                pristine_snapshot_path=pristine_snapshot_path,
                pristine_snapshot_digest=pristine_snapshot_digest,
            )
            for index in range(len(stream.families))
        ]
        validation_successes = sum(
            cast(int, cast(dict[str, object], item["validation"])["successes"])
            for item in runs
        )
        validation_total = sum(
            cast(int, cast(dict[str, object], item["validation"])["total"])
            for item in runs
        )
        selection_records.append(
            {
                "learning_rate": learning_rate,
                "mean_validation": _accuracy_record(
                    validation_successes,
                    validation_total,
                    settings.confidence_level,
                ),
                "families": [
                    {
                        "family_id": item["family_id"],
                        "validation": item["validation"],
                    }
                    for item in runs
                ],
            }
        )
    selected_summary = max(
        selection_records,
        key=lambda item: (
            cast(float, cast(dict[str, object], item["mean_validation"])["accuracy"]),
            -cast(float, item["learning_rate"]),
        ),
    )
    selected_learning_rate = cast(float, selected_summary["learning_rate"])
    zero_shot_summary = (
        dict(zero_shot_readiness)
        if zero_shot_readiness is not None
        else evaluate_zero_shot_readiness(
            model, tokenizer, stream, settings, context_length, device
        )
    )
    zero_shot_rows = zero_shot_summary.get("families")
    if not isinstance(zero_shot_rows, list) or len(zero_shot_rows) != len(stream.families):
        raise ValueError("zero-shot readiness must contain one row per task family")
    zero_shot = []
    for family, row in zip(stream.families, zero_shot_rows, strict=True):
        if not isinstance(row, dict) or row.get("family_id") != family.family_id:
            raise ValueError("zero-shot readiness family order does not match the task stream")
        result = row.get("zero_shot")
        if not isinstance(result, dict):
            raise TypeError("zero-shot readiness family result is malformed")
        zero_shot.append(cast(dict[str, object], result))
    probes = [
        run_learnability_probe(
            model,
            tokenizer,
            stream,
            index,
            settings,
            context_length,
            device,
            learning_rate=selected_learning_rate,
            zero_shot=zero_shot[index],
            evaluate_test=True,
            recipe=recipe,
            resource_guard=resource_guard,
            pristine_snapshot_path=pristine_snapshot_path,
            pristine_snapshot_digest=pristine_snapshot_digest,
        )
        for index in range(len(stream.families))
    ]
    if pristine_snapshot_path is not None:
        if pristine_snapshot_digest is None:
            raise ValueError("a pristine snapshot path requires its model digest")
        _restore_pristine_snapshot(
            model, pristine_snapshot_path, device, pristine_snapshot_digest
        )
    after_digest = model_state_digest(model)
    if before_digest != after_digest:
        raise RuntimeError("learnability probes mutated the original candidate weights")
    zero_successes = sum(
        cast(int, cast(dict[str, object], item["zero_shot"])["successes"])
        for item in probes
    )
    learned_successes = sum(
        cast(int, cast(dict[str, object], item["learned"])["successes"])
        for item in probes
    )
    total = sum(
        cast(int, cast(dict[str, object], item["learned"])["total"])
        for item in probes
    )
    mean_zero = _accuracy_record(
        zero_successes, total, settings.confidence_level
    )
    mean_learned = _accuracy_record(
        learned_successes, total, settings.confidence_level
    )
    zero_interval = cast(dict[str, float], mean_zero["wilson_interval"])
    learned_interval = cast(dict[str, float], mean_learned["wilson_interval"])
    clustered_learned = student_t_interval(
        [cast(float, item["learned_accuracy"]) for item in probes],
        settings.confidence_level,
    )
    clustered_gain = student_t_interval(
        [cast(float, item["accuracy_gain"]) for item in probes],
        settings.confidence_level,
    )
    return {
        "families": probes,
        "learning_rate_selection": {
            "split": "validation",
            "selected_learning_rate": selected_learning_rate,
            "tie_break": "highest pooled validation accuracy, then lower learning rate",
            "test_evaluation_timing": (
                "test was not evaluated for the learning-rate grid; the selected rate "
                "was retrained from pristine base weights before one final test evaluation"
            ),
            "grid": selection_records,
        },
        "mean_zero_shot": mean_zero,
        "mean_learned": mean_learned,
        "mean_zero_shot_accuracy": cast(float, mean_zero["accuracy"]),
        "mean_learned_accuracy": cast(float, mean_learned["accuracy"]),
        "mean_accuracy_gain": cast(float, mean_learned["accuracy"])
        - cast(float, mean_zero["accuracy"]),
        "mean_accuracy_gain_interval": {
            "confidence_level": settings.confidence_level,
            "lower": learned_interval["lower"] - zero_interval["upper"],
            "upper": learned_interval["upper"] - zero_interval["lower"],
            "method": "Newcombe interval from Wilson component intervals",
        },
        "pooled_item_analysis": {
            "mean_zero_shot": mean_zero,
            "mean_learned": mean_learned,
            "mean_accuracy_gain_interval": {
                "confidence_level": settings.confidence_level,
                "lower": learned_interval["lower"] - zero_interval["upper"],
                "upper": learned_interval["upper"] - zero_interval["lower"],
                "method": "Newcombe interval from Wilson component intervals",
            },
            "precision_note": (
                "optimistic: treats items clustered within task families as "
                "independent Bernoulli trials"
            ),
        },
        "family_clustered_analysis": {
            "mean_learned_accuracy": clustered_learned,
            "mean_accuracy_gain": clustered_gain,
            "independent_unit": "task family",
        },
        "base_weights_sha256_before": before_digest,
        "base_weights_sha256_after": after_digest,
        "heldout_split": "test",
        "evaluation_generation": "right-padded batched greedy decoding",
    }


def full_finetuning_memory_feasibility(
    unique_parameters: int,
    settings: MeasurementSettings,
    decision: DecisionSettings,
) -> dict[str, object]:
    """Return a conservative parameter-state lower bound for the frozen probe.

    AdamW allocates trainable weights, gradients, and two optimizer moments. A live
    pristine copy counts on MPS even when labelled as host placement because Apple
    silicon uses unified memory. Activations and allocator overhead are deliberately
    excluded, so this is informational and never an arithmetic exclusion.
    """
    element_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[settings.torch_dtype]
    optimizer_state_bytes = 8
    resolved_device = resolve_device(settings.device)
    pristine_device_bytes = (
        0
        if settings.release_pristine_before_training
        or (
            settings.pristine_model_device == "host"
            and resolved_device.type == "cuda"
        )
        else element_bytes
    )
    bytes_per_parameter = 2 * element_bytes + optimizer_state_bytes + pristine_device_bytes
    components = {
        "pristine_base_weights_on_candidate_device": (
            unique_parameters * pristine_device_bytes
        ),
        "trainable_copy_weights": unique_parameters * element_bytes,
        "gradients": unique_parameters * element_bytes,
        "adamw_first_and_second_moments": unique_parameters * optimizer_state_bytes,
    }
    lower_bound = sum(components.values())
    maximum = int(
        decision.target_memory_bytes * (1.0 - decision.minimum_memory_headroom_fraction)
    )
    return {
        "fits_parameter_state_lower_bound": lower_bound <= maximum,
        "unique_parameters": unique_parameters,
        "torch_dtype": settings.torch_dtype,
        "pristine_model_device": settings.pristine_model_device,
        "release_pristine_before_training": (
            settings.release_pristine_before_training
        ),
        "bytes_per_parameter": bytes_per_parameter,
        "parameter_state_lower_bound_bytes": lower_bound,
        "configured_maximum_peak_bytes": maximum,
        "configured_target_memory_bytes": decision.target_memory_bytes,
        "configured_minimum_headroom_fraction": decision.minimum_memory_headroom_fraction,
        "components_bytes": components,
        "excluded_from_lower_bound": [
            "activations",
            "allocator overhead",
            "tokenizer and Python process state",
        ],
        "method": (
            "parameter state: any simultaneously live pristine weights + trainable "
            "weights + gradients + two float32 AdamW moments; informational only, "
            "with measured candidate peak as the sole feasibility verdict"
        ),
    }


def _rss_high_water_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def system_swap_used_bytes() -> int:
    """Read host swap use without adding a runtime dependency."""
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", completed.stdout)
        if match is None:
            raise RuntimeError("could not parse Darwin vm.swapusage")
        multipliers = {
            "K": 1024,
            "M": 1024**2,
            "G": 1024**3,
            "T": 1024**4,
            "P": 1024**5,
        }
        return int(float(match.group(1)) * multipliers[match.group(2)])
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"SwapTotal", "SwapFree"}:
                fields[name] = int(value.strip().split()[0]) * 1024
        if fields.keys() >= {"SwapTotal", "SwapFree"}:
            return max(0, fields["SwapTotal"] - fields["SwapFree"])
    raise RuntimeError(f"system swap measurement is unsupported on {sys.platform}")


def system_free_memory_percentage() -> float:
    """Return the host free-memory percentage used as environment-health evidence."""
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/usr/bin/memory_pressure"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        match = re.search(
            r"System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%",
            completed.stdout,
        )
        if match is None:
            raise RuntimeError("could not parse Darwin memory_pressure output")
        return float(match.group(1))
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                fields[name] = int(value.strip().split()[0])
        if fields.keys() >= {"MemTotal", "MemAvailable"} and fields["MemTotal"]:
            return 100.0 * fields["MemAvailable"] / fields["MemTotal"]
    raise RuntimeError(f"system memory-pressure measurement is unsupported on {sys.platform}")


class MemoryPeakSampler(AbstractContextManager["MemoryPeakSampler"]):
    """Sample per-process RSS and accelerator allocation high-water values."""

    _cuda_active: list[MemoryPeakSampler] = []
    _cuda_lock = threading.Lock()

    def __init__(
        self,
        device: torch.device,
        interval_seconds: float,
        *,
        maximum_swap_growth_bytes: int | None = None,
        swap_baseline_bytes: int | None = None,
        swap_sample_interval_seconds: float = 0.5,
    ) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_baseline = _rss_high_water_bytes()
        self.rss_peak = self.rss_baseline
        self.mps_baseline: int | None = self._mps_bytes()
        self.mps_peak: int | None = self.mps_baseline
        self.cuda_allocated_baseline: int | None = self._cuda_allocated_bytes()
        self.cuda_reserved_baseline: int | None = self._cuda_reserved_bytes()
        self.cuda_allocated_peak: int | None = self.cuda_allocated_baseline
        self.cuda_reserved_peak: int | None = self.cuda_reserved_baseline
        self.maximum_swap_growth_bytes = maximum_swap_growth_bytes
        self.swap_sample_interval_seconds = swap_sample_interval_seconds
        self.swap_baseline = (
            system_swap_used_bytes()
            if swap_baseline_bytes is None
            else swap_baseline_bytes
        )
        self.swap_peak = self.swap_baseline
        self.swap_after = self.swap_baseline
        self._last_swap_sample = 0.0

    def _mps_bytes(self) -> int | None:
        if self.device.type != "mps" or not hasattr(torch, "mps"):
            return None
        try:
            return int(torch.mps.driver_allocated_memory())
        except RuntimeError:
            return None

    def _cuda_allocated_bytes(self) -> int | None:
        if self.device.type != "cuda":
            return None
        try:
            return int(torch.cuda.memory_allocated(self.device))
        except RuntimeError:
            return None

    def _cuda_reserved_bytes(self) -> int | None:
        if self.device.type != "cuda":
            return None
        try:
            return int(torch.cuda.memory_reserved(self.device))
        except RuntimeError:
            return None

    @classmethod
    def _capture_cuda_peaks(cls, device: torch.device) -> None:
        """Propagate process-global CUDA peaks to nested active samplers."""
        try:
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
        except RuntimeError:
            return
        for sampler in cls._cuda_active:
            if sampler.device == device:
                sampler.cuda_allocated_peak = max(
                    sampler.cuda_allocated_peak or 0, allocated
                )
                sampler.cuda_reserved_peak = max(
                    sampler.cuda_reserved_peak or 0, reserved
                )

    def _sample(self, *, force_swap: bool = False) -> None:
        self.rss_peak = max(self.rss_peak, _rss_high_water_bytes())
        mps_value = self._mps_bytes()
        if mps_value is not None:
            self.mps_peak = max(self.mps_peak or 0, mps_value)
        if self.device.type == "cuda":
            with self._cuda_lock:
                self._capture_cuda_peaks(self.device)
        now = time.monotonic()
        if force_swap or now - self._last_swap_sample >= self.swap_sample_interval_seconds:
            self.swap_after = system_swap_used_bytes()
            self.swap_peak = max(self.swap_peak, self.swap_after)
            self._last_swap_sample = now

    def raise_if_swap_limit_exceeded(self) -> None:
        """Sample swap at legacy cooperative boundaries without judging feasibility."""
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> MemoryPeakSampler:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            with self._cuda_lock:
                self._capture_cuda_peaks(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                self._cuda_active.append(self)
        self._sample(force_swap=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._sample(force_swap=True)
        if self.device.type == "cuda":
            with self._cuda_lock:
                if self in self._cuda_active:
                    self._cuda_active.remove(self)

    def record(self) -> dict[str, object]:
        """Return raw and derived values without double-counting unified memory."""
        if self.device.type == "cuda":
            candidates = [
                value
                for value in (self.cuda_allocated_peak, self.cuda_reserved_peak)
                if value is not None
            ]
            if not candidates:
                candidates = [self.rss_peak]
        else:
            candidates = [self.rss_peak]
            if self.mps_peak is not None:
                candidates.append(self.mps_peak)
        return {
            "process_rss_baseline_bytes": self.rss_baseline,
            "process_rss_peak_bytes": self.rss_peak,
            "process_rss_delta_bytes": max(0, self.rss_peak - self.rss_baseline),
            "mps_driver_baseline_bytes": self.mps_baseline,
            "mps_driver_peak_bytes": self.mps_peak,
            "mps_driver_delta_bytes": (
                None
                if self.mps_baseline is None or self.mps_peak is None
                else max(0, self.mps_peak - self.mps_baseline)
            ),
            "cuda_allocated_baseline_bytes": self.cuda_allocated_baseline,
            "cuda_max_memory_allocated_bytes": self.cuda_allocated_peak,
            "cuda_reserved_baseline_bytes": self.cuda_reserved_baseline,
            "cuda_max_memory_reserved_bytes": self.cuda_reserved_peak,
            "hard_constraint_peak_bytes": max(candidates),
            "hard_constraint_metric": (
                "max(CUDA max_memory_allocated, CUDA max_memory_reserved)"
                if self.device.type == "cuda" and self.cuda_reserved_peak is not None
                else (
                    "max(absolute process RSS peak, absolute MPS driver allocation peak)"
                    if self.mps_peak is not None
                    else "absolute process RSS peak"
                )
            ),
            "sample_interval_seconds": self.interval_seconds,
            "system_swap_before_bytes": self.swap_baseline,
            "system_swap_after_bytes": self.swap_after,
            "system_swap_peak_bytes": self.swap_peak,
            "system_swap_growth_bytes": max(0, self.swap_after - self.swap_baseline),
            "system_swap_peak_growth_bytes": max(
                0, self.swap_peak - self.swap_baseline
            ),
            "configured_maximum_swap_growth_bytes": self.maximum_swap_growth_bytes,
            "swap_sample_interval_seconds": self.swap_sample_interval_seconds,
        }


def _worst_case_training_instances(
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    context_length: int,
    effective_batch_size: int,
) -> list[TaskInstance]:
    """Select the longest supervised examples for one-step memory calibration."""
    scored: list[tuple[int, int, str, TaskInstance]] = []
    for family in stream.families:
        for instance in stream.get_split(family.family_id).train:
            ids, labels, _ = _training_example(tokenizer, instance, context_length)
            scored.append(
                (
                    len(ids),
                    sum(label != -100 for label in labels),
                    instance.task_id,
                    instance,
                )
            )
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if len(scored) < effective_batch_size:
        raise ValueError("task stream has too few examples for memory calibration")
    return [item[3] for item in scored[:effective_batch_size]]


def select_training_recipe(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    decision: DecisionSettings,
    context_length: int,
    device: torch.device,
    *,
    resource_guard: MemoryPeakSampler,
    pristine_snapshot_path: Path | None = None,
    pristine_snapshot_digest: str | None = None,
) -> tuple[TrainingRecipe | None, dict[str, object]]:
    """Choose the largest fitting micro-batch and retain every attempt as evidence."""
    if pristine_snapshot_path is not None and pristine_snapshot_digest is None:
        raise ValueError("a pristine snapshot path requires its model digest")
    instances = _worst_case_training_instances(
        tokenizer,
        stream,
        context_length,
        settings.probe_batch_size,
    )
    maximum_peak = int(
        decision.target_memory_bytes * (1.0 - decision.minimum_memory_headroom_fraction)
    )
    recipes = (
        [TrainingRecipe(size, False) for size in settings.probe_micro_batch_sizes]
        if settings.target_only_training_loss
        else [TrainingRecipe(settings.probe_batch_size, False)]
    )
    if settings.target_only_training_loss and settings.gradient_checkpointing_fallback:
        recipes.append(TrainingRecipe(settings.probe_micro_batch_sizes[-1], True))
    attempts: list[dict[str, object]] = []
    selected: TrainingRecipe | None = None
    for recipe in recipes:
        seed_everything(settings.seed)
        probe: CausalLM | None = None
        optimizer: torch.optim.Optimizer | None = None
        error: BaseException | None = None
        attempt_sampler = MemoryPeakSampler(
            device,
            settings.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=settings.swap_sample_interval_seconds,
        )
        with attempt_sampler:
            try:
                if pristine_snapshot_path is None:
                    probe = copy.deepcopy(base_model).to(device)
                else:
                    _restore_pristine_snapshot(
                        base_model,
                        pristine_snapshot_path,
                        device,
                        cast(str, pristine_snapshot_digest),
                    )
                    probe = base_model
                probe.train()
                _set_gradient_checkpointing(probe, recipe.gradient_checkpointing)
                optimizer = torch.optim.AdamW(
                    probe.parameters(),
                    lr=settings.probe_learning_rates[0],
                    weight_decay=settings.probe_weight_decay,
                )
                _optimizer_step_for_measurement(
                    probe,
                    tokenizer,
                    instances,
                    context_length,
                    device,
                    optimizer,
                    settings,
                    recipe,
                    resource_guard=resource_guard,
                )
                synchronize_device(device)
                resource_guard.raise_if_swap_limit_exceeded()
            except BaseException as caught:
                error = caught
        memory = attempt_sampler.record()
        peak = cast(int, memory["hard_constraint_peak_bytes"])
        feasible = error is None and peak <= maximum_peak
        attempt: dict[str, object] = {
            "micro_batch_size": recipe.micro_batch_size,
            "gradient_checkpointing": recipe.gradient_checkpointing,
            "effective_batch_size": settings.probe_batch_size,
            "calibration_steps": 1,
            "calibration_instance_ids": [item.task_id for item in instances],
            "memory": memory,
            "configured_maximum_peak_bytes": maximum_peak,
            "feasible": feasible,
        }
        if error is not None:
            attempt["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "out_of_memory": _is_out_of_memory(error),
            }
        elif peak > maximum_peak:
            attempt["rejection_reason"] = "calibration peak exceeded the memory cap"
        attempts.append(attempt)
        del optimizer, probe
        cleanup_device(device)
        if error is not None and not _is_out_of_memory(error):
            if pristine_snapshot_path is not None:
                _restore_pristine_snapshot(
                    base_model,
                    pristine_snapshot_path,
                    device,
                    cast(str, pristine_snapshot_digest),
                )
            raise error
        if feasible:
            selected = recipe
            break
    if pristine_snapshot_path is not None:
        _restore_pristine_snapshot(
            base_model,
            pristine_snapshot_path,
            device,
            cast(str, pristine_snapshot_digest),
        )
    return selected, {
        "effective_batch_size": settings.probe_batch_size,
        "candidate_micro_batch_sizes": list(settings.probe_micro_batch_sizes),
        "target_only_training_loss": settings.target_only_training_loss,
        "training_recipe": (
            "amendment_4_target_only"
            if settings.target_only_training_loss
            else "amendment_1_full_logits"
        ),
        "gradient_checkpointing_fallback": settings.gradient_checkpointing_fallback,
        "configured_maximum_peak_bytes": maximum_peak,
        "system_swap_role": "environment-health signal only; not candidate feasibility",
        "attempts": attempts,
        "selected_micro_batch_size": (
            None if selected is None else selected.micro_batch_size
        ),
        "selected_gradient_checkpointing": (
            None if selected is None else selected.gradient_checkpointing
        ),
    }


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    accelerator_state: torch.Tensor | None = None
    if device.type == "mps" and hasattr(torch, "mps"):
        accelerator_state = torch.mps.get_rng_state()
    elif device.type == "cuda":
        accelerator_state = torch.cuda.get_rng_state(device)
    return torch.get_rng_state(), accelerator_state


def _restore_rng_state(
    device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]
) -> None:
    cpu_state, accelerator_state = state
    torch.set_rng_state(cpu_state)
    if accelerator_state is None:
        return
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.set_rng_state(accelerator_state)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(accelerator_state, device)


def _parameter_snapshots(model: CausalLM) -> list[torch.Tensor]:
    return [parameter.detach().to(device="cpu").clone() for parameter in model.parameters()]


def _gradient_snapshots(model: CausalLM) -> list[torch.Tensor]:
    snapshots: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            snapshots.append(torch.zeros_like(parameter, device="cpu"))
        else:
            snapshots.append(parameter.grad.detach().to(device="cpu").clone())
    return snapshots


def _updates_from_snapshots(
    model: CausalLM, before: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    return [
        parameter.detach().to(device="cpu") - old
        for parameter, old in zip(model.parameters(), before, strict=True)
    ]


def _vector_relative_l2_difference(
    reference: Sequence[torch.Tensor], candidate: Sequence[torch.Tensor]
) -> dict[str, float]:
    difference_squared = 0.0
    reference_squared = 0.0
    maximum_absolute = 0.0
    for reference_tensor, candidate_tensor in zip(reference, candidate, strict=True):
        difference = reference_tensor.double() - candidate_tensor.double()
        if difference.numel() > 0:
            maximum_absolute = max(
                maximum_absolute, float(difference.abs().max().item())
            )
        difference_squared += float(torch.sum(difference.square()).item())
        reference_squared += float(
            torch.sum(reference_tensor.double().square()).item()
        )
    difference_l2 = math.sqrt(difference_squared)
    reference_l2 = math.sqrt(reference_squared)
    if not all(
        math.isfinite(value)
        for value in (difference_l2, reference_l2, maximum_absolute)
    ):
        raise FloatingPointError("gradient or update comparison is non-finite")
    return {
        "relative_l2": difference_l2
        / max(reference_l2, torch.finfo(torch.float64).eps),
        "difference_l2": difference_l2,
        "reference_l2": reference_l2,
        "maximum_absolute": maximum_absolute,
    }


def _disable_dropout_only(model: CausalLM) -> int:
    """Remove stochastic-mask confounding without disabling autograd."""
    model.eval()
    disabled = 0
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for _, module in model.named_modules():
        if isinstance(module, dropout_types):
            module.eval()
            disabled += 1
    return disabled


def _move_model_for_measurement(
    model: CausalLM, device: torch.device, dtype: torch.dtype
) -> None:
    """Move device before dtype so MPS never receives an unsupported float64 cast."""
    cast(Any, model).to(device=device)
    cast(Any, model).to(dtype=dtype)


def _measure_gradient_and_update(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    target_only: bool,
    recipe: TrainingRecipe,
    include_update: bool,
    exact_full_logits: bool = False,
    legacy_sequence_chunk_size: int | None = None,
) -> _VectorMeasurement:
    model = copy.deepcopy(base_model)
    _move_model_for_measurement(model, device, dtype)
    _set_gradient_checkpointing(model, target_only and recipe.gradient_checkpointing)
    dropout_modules_disabled = _disable_dropout_only(model)
    seed_everything(settings.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.probe_learning_rates[0],
        weight_decay=settings.probe_weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    before = _parameter_snapshots(model) if include_update else []
    if target_only and legacy_sequence_chunk_size is not None:
        raise ValueError("target-only and legacy chunked-head modes are mutually exclusive")
    if target_only:
        result = _backward_target_only(
            model,
            tokenizer,
            instances,
            context_length,
            device,
            recipe,
        )
        loss = cast(float, result["loss"])
    elif exact_full_logits:
        loss = _backward_legacy_full_logits_exact(
            model, tokenizer, instances, context_length, device
        )
    elif legacy_sequence_chunk_size is not None:
        loss = _backward_legacy_chunked_head(
            model,
            tokenizer,
            instances,
            context_length,
            device,
            legacy_sequence_chunk_size,
        )
    else:
        loss = _backward_legacy(
            model, tokenizer, instances, context_length, device
        )
    if not math.isfinite(loss):
        raise FloatingPointError("loss measurement is non-finite")
    synchronize_device(device)
    gradients = _gradient_snapshots(model)
    updates: list[torch.Tensor] = []
    if include_update:
        optimizer.step()
        synchronize_device(device)
        updates = _updates_from_snapshots(model, before)
    del optimizer, before, model
    cleanup_device(device)
    return _VectorMeasurement(
        loss=loss,
        gradients=gradients,
        updates=updates,
        dropout_modules_disabled=dropout_modules_disabled,
    )


def prove_exact_gradient_equivalence(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[TaskInstance],
    settings: MeasurementSettings,
    context_length: int,
    recipe: TrainingRecipe,
) -> dict[str, object]:
    """Prove full-logit and target-only objectives match in float64 on CPU."""
    if len(instances) != settings.probe_batch_size:
        raise ValueError("exact equivalence requires one complete effective batch")
    device = torch.device("cpu")
    legacy = _measure_gradient_and_update(
        base_model,
        tokenizer,
        instances,
        settings,
        context_length,
        device,
        dtype=torch.float64,
        target_only=False,
        recipe=TrainingRecipe(settings.probe_batch_size, False),
        include_update=False,
        exact_full_logits=True,
    )
    micro_batch_sizes = tuple(dict.fromkeys((recipe.micro_batch_size, 1)))
    comparisons: list[dict[str, object]] = []
    passed = True
    dropout_modules_disabled = legacy.dropout_modules_disabled
    for micro_batch_size in micro_batch_sizes:
        amended = _measure_gradient_and_update(
            base_model,
            tokenizer,
            instances,
            settings,
            context_length,
            device,
            dtype=torch.float64,
            target_only=True,
            recipe=TrainingRecipe(micro_batch_size, recipe.gradient_checkpointing),
            include_update=False,
        )
        difference = _vector_relative_l2_difference(
            legacy.gradients, amended.gradients
        )
        loss_difference = abs(legacy.loss - amended.loss)
        comparison_passed = (
            loss_difference <= settings.equivalence_exact_loss_absolute_tolerance
            and difference["relative_l2"]
            <= settings.equivalence_exact_gradient_relative_l2_tolerance
        )
        passed = passed and comparison_passed
        comparisons.append(
            {
                "micro_batch_size": micro_batch_size,
                "legacy_loss": legacy.loss,
                "target_only_loss": amended.loss,
                "absolute_loss_difference": loss_difference,
                "gradient_relative_l2_difference": difference["relative_l2"],
                "gradient_difference": difference,
                "passed": comparison_passed,
            }
        )
        del amended
    del legacy
    return {
        "passed": passed,
        "device": "cpu",
        "dtype": "float64",
        "dropout_disabled": True,
        "dropout_modules_disabled": dropout_modules_disabled,
        "effective_batch_size": settings.probe_batch_size,
        "comparisons": comparisons,
    }


def assess_noise_floor(
    batches: Sequence[Mapping[str, float]], *, multiplier: float
) -> dict[str, object]:
    """Apply the pre-registered batchwise gradient and update noise-floor gate."""
    if multiplier < 1.0:
        raise ValueError("noise-floor multiplier must be at least 1")
    assessed: list[dict[str, object]] = []
    for batch in batches:
        gradient_limit = multiplier * batch["control_gradient_relative_l2"]
        update_limit = multiplier * batch["control_update_relative_l2"]
        gradient_passed = batch["target_gradient_relative_l2"] <= gradient_limit
        update_passed = batch["target_update_relative_l2"] <= update_limit
        assessed.append(
            {
                **batch,
                "gradient_limit": gradient_limit,
                "update_limit": update_limit,
                "gradient_passed": gradient_passed,
                "update_passed": update_passed,
                "passed": gradient_passed and update_passed,
            }
        )
    return {
        "passed": bool(assessed) and all(row["passed"] is True for row in assessed),
        "multiplier": multiplier,
        "batches": assessed,
    }


def assess_training_trajectory(
    *,
    legacy_losses: Sequence[float],
    control_losses: Sequence[float],
    target_losses: Sequence[float],
    legacy_accuracy: Mapping[str, float],
    target_accuracy: Mapping[str, float],
    noise_multiplier: float,
    accuracy_tie_margin: float,
) -> dict[str, object]:
    """Apply trajectory-noise and per-family final-accuracy gates."""
    if not legacy_losses or not (
        len(legacy_losses) == len(control_losses) == len(target_losses)
    ):
        raise ValueError("trajectory losses must be non-empty and equal length")
    if legacy_accuracy.keys() != target_accuracy.keys() or not legacy_accuracy:
        raise ValueError("trajectory accuracy maps must have identical non-empty keys")
    control_difference = max(
        abs(legacy - control)
        for legacy, control in zip(legacy_losses, control_losses, strict=True)
    )
    target_difference = max(
        abs(legacy - target)
        for legacy, target in zip(legacy_losses, target_losses, strict=True)
    )
    trajectory_limit = noise_multiplier * control_difference
    trajectory_passed = target_difference <= trajectory_limit
    family_rows = []
    for family_id in legacy_accuracy:
        difference = abs(legacy_accuracy[family_id] - target_accuracy[family_id])
        family_rows.append(
            {
                "family_id": family_id,
                "legacy_accuracy": legacy_accuracy[family_id],
                "target_only_accuracy": target_accuracy[family_id],
                "absolute_difference": difference,
                "passed": difference <= accuracy_tie_margin + 1e-12,
            }
        )
    legacy_mean = statistics.fmean(legacy_accuracy.values())
    target_mean = statistics.fmean(target_accuracy.values())
    mean_difference = abs(legacy_mean - target_mean)
    mean_passed = mean_difference <= accuracy_tie_margin + 1e-12
    accuracy_passed = mean_passed and all(
        row["passed"] is True for row in family_rows
    )
    return {
        "passed": trajectory_passed and accuracy_passed,
        "trajectory_passed": trajectory_passed,
        "accuracy_passed": accuracy_passed,
        "maximum_control_loss_difference": control_difference,
        "maximum_target_loss_difference": target_difference,
        "trajectory_limit": trajectory_limit,
        "accuracy_tie_margin": accuracy_tie_margin,
        "family_accuracy": family_rows,
        "mean_accuracy": {
            "legacy": legacy_mean,
            "target_only": target_mean,
            "absolute_difference": mean_difference,
            "passed": mean_passed,
        },
    }


def _equivalence_batches(
    stream: ContinualTaskStream, *, count: int, batch_size: int
) -> list[list[TaskInstance]]:
    """Build the preregistered family-round-robin batch sequence."""
    training_rows = [
        stream.get_split(family.family_id).train for family in stream.families
    ]
    if any(not rows for rows in training_rows):
        raise ValueError("equivalence requires non-empty training rows for every family")
    ordered: list[TaskInstance] = []
    item_index = 0
    required = count * batch_size
    while len(ordered) < required:
        for rows in training_rows:
            ordered.append(rows[item_index % len(rows)])
            if len(ordered) == required:
                break
        item_index += 1
    return [
        ordered[offset : offset + batch_size]
        for offset in range(0, required, batch_size)
    ]


def measure_noise_floor(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    batches: Sequence[Sequence[TaskInstance]],
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
) -> dict[str, object]:
    """Measure recipe differences against within-batch permutation noise."""
    if device.type not in {"mps", "cuda"}:
        raise ValueError("Amendment 4a noise-floor measurement requires an accelerator")
    if len(batches) < settings.equivalence_noise_floor_batches:
        raise ValueError("noise-floor measurement received too few batches")
    rows: list[dict[str, float]] = []
    identities: list[dict[str, object]] = []
    for batch_index, batch in enumerate(
        batches[: settings.equivalence_noise_floor_batches]
    ):
        legacy = _measure_gradient_and_update(
            base_model,
            tokenizer,
            batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=False,
            recipe=TrainingRecipe(settings.probe_batch_size, False),
            include_update=True,
        )
        control_batch = list(reversed(batch))
        control = _measure_gradient_and_update(
            base_model,
            tokenizer,
            control_batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=False,
            recipe=TrainingRecipe(settings.probe_batch_size, False),
            include_update=True,
        )
        target = _measure_gradient_and_update(
            base_model,
            tokenizer,
            batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=True,
            recipe=recipe,
            include_update=True,
        )
        control_gradient = _vector_relative_l2_difference(
            legacy.gradients, control.gradients
        )
        target_gradient = _vector_relative_l2_difference(
            legacy.gradients, target.gradients
        )
        control_update = _vector_relative_l2_difference(
            legacy.updates, control.updates
        )
        target_update = _vector_relative_l2_difference(
            legacy.updates, target.updates
        )
        rows.append(
            {
                "control_gradient_relative_l2": control_gradient["relative_l2"],
                "target_gradient_relative_l2": target_gradient["relative_l2"],
                "control_update_relative_l2": control_update["relative_l2"],
                "target_update_relative_l2": target_update["relative_l2"],
            }
        )
        identities.append(
            {
                "batch": batch_index + 1,
                "instance_ids": [item.task_id for item in batch],
                "control_instance_ids": [item.task_id for item in control_batch],
                "legacy_loss": legacy.loss,
                "control_loss": control.loss,
                "target_only_loss": target.loss,
                "control_gradient_difference": control_gradient,
                "target_gradient_difference": target_gradient,
                "control_update_difference": control_update,
                "target_update_difference": target_update,
            }
        )
        del legacy, control, target
    assessment = assess_noise_floor(
        rows, multiplier=settings.equivalence_noise_floor_multiplier
    )
    assessed_rows = cast(list[dict[str, object]], assessment["batches"])
    for identity, assessed in zip(identities, assessed_rows, strict=True):
        identity.update(
            {
                "gradient_limit": assessed["gradient_limit"],
                "update_limit": assessed["update_limit"],
                "gradient_passed": assessed["gradient_passed"],
                "update_passed": assessed["update_passed"],
                "passed": assessed["passed"],
            }
        )
    return {
        "passed": assessment["passed"],
        "device": str(device),
        "dtype": "float32",
        "dropout_disabled": True,
        "effective_batch_size": settings.probe_batch_size,
        "target_only_micro_batch_size": recipe.micro_batch_size,
        "multiplier": settings.equivalence_noise_floor_multiplier,
        "batches": identities,
    }


def measure_shape_changing_control(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    batches: Sequence[Sequence[TaskInstance]],
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
) -> dict[str, object]:
    """Measure a shape-changing legacy control without making it a gate."""
    if device.type not in {"mps", "cuda"}:
        raise ValueError("Amendment 4b shape-changing control requires an accelerator")
    if not batches:
        raise ValueError("shape-changing control requires at least one batch")
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batches):
        legacy = _measure_gradient_and_update(
            base_model,
            tokenizer,
            batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=False,
            recipe=TrainingRecipe(settings.probe_batch_size, False),
            include_update=True,
        )
        shape_control = _measure_gradient_and_update(
            base_model,
            tokenizer,
            batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=False,
            recipe=TrainingRecipe(settings.probe_batch_size, False),
            include_update=True,
            legacy_sequence_chunk_size=(
                settings.equivalence_shape_control_chunk_size
            ),
        )
        target = _measure_gradient_and_update(
            base_model,
            tokenizer,
            batch,
            settings,
            context_length,
            device,
            dtype=torch.float32,
            target_only=True,
            recipe=recipe,
            include_update=True,
        )
        control_gradient = _vector_relative_l2_difference(
            legacy.gradients, shape_control.gradients
        )
        target_gradient = _vector_relative_l2_difference(
            legacy.gradients, target.gradients
        )
        control_update = _vector_relative_l2_difference(
            legacy.updates, shape_control.updates
        )
        target_update = _vector_relative_l2_difference(
            legacy.updates, target.updates
        )
        rows.append(
            {
                "batch": batch_index + 1,
                "instance_ids": [item.task_id for item in batch],
                "legacy_loss": legacy.loss,
                "shape_control_loss": shape_control.loss,
                "target_only_loss": target.loss,
                "shape_control_gradient_difference": control_gradient,
                "target_only_gradient_difference": target_gradient,
                "shape_control_update_difference": control_update,
                "target_only_update_difference": target_update,
                "target_gradient_within_shape_control": (
                    target_gradient["relative_l2"]
                    <= control_gradient["relative_l2"]
                ),
                "target_update_within_shape_control": (
                    target_update["relative_l2"] <= control_update["relative_l2"]
                ),
            }
        )
        del legacy, shape_control, target
    return {
        "status": "completed",
        "gating": False,
        "device": str(device),
        "dtype": "float32",
        "dropout_disabled": True,
        "effective_batch_size": settings.probe_batch_size,
        "target_only_micro_batch_size": recipe.micro_batch_size,
        "sequence_chunk_size": settings.equivalence_shape_control_chunk_size,
        "batches": rows,
    }


def _probe_accuracy_map(summary: Mapping[str, object]) -> dict[str, float]:
    rows = summary.get("families")
    if not isinstance(rows, list):
        raise TypeError("probe accuracy summary has no family rows")
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("family_id"), str):
            raise TypeError("probe accuracy family row is malformed")
        accuracy = row.get("zero_shot")
        if not isinstance(accuracy, dict) or not isinstance(
            accuracy.get("accuracy"), (int, float)
        ):
            raise TypeError("probe accuracy result is malformed")
        result[row["family_id"]] = float(accuracy["accuracy"])
    return result


def assess_outcome_gate(
    *,
    old_seed_0_losses: Sequence[float],
    old_seed_1_losses: Sequence[float],
    new_seed_0_losses: Sequence[float],
    old_seed_0_accuracy: Mapping[str, float],
    old_seed_1_accuracy: Mapping[str, float],
    new_seed_0_accuracy: Mapping[str, float],
    accuracy_tie_margin: float,
) -> dict[str, object]:
    """Apply Amendment 4b's trajectory and final mean-accuracy outcome rules."""
    if not old_seed_0_losses or not (
        len(old_seed_0_losses)
        == len(old_seed_1_losses)
        == len(new_seed_0_losses)
    ):
        raise ValueError("outcome losses must be non-empty and equal length")
    if not old_seed_0_accuracy or not (
        old_seed_0_accuracy.keys()
        == old_seed_1_accuracy.keys()
        == new_seed_0_accuracy.keys()
    ):
        raise ValueError("outcome accuracy maps must have identical non-empty keys")
    seed_loss_difference = max(
        abs(seed_0 - seed_1)
        for seed_0, seed_1 in zip(
            old_seed_0_losses, old_seed_1_losses, strict=True
        )
    )
    recipe_loss_difference = max(
        abs(old - new)
        for old, new in zip(old_seed_0_losses, new_seed_0_losses, strict=True)
    )
    trajectory_passed = recipe_loss_difference <= seed_loss_difference
    old_seed_0_mean = statistics.fmean(old_seed_0_accuracy.values())
    old_seed_1_mean = statistics.fmean(old_seed_1_accuracy.values())
    new_seed_0_mean = statistics.fmean(new_seed_0_accuracy.values())
    seed_accuracy_difference = abs(old_seed_0_mean - old_seed_1_mean)
    recipe_accuracy_difference = abs(old_seed_0_mean - new_seed_0_mean)
    accuracy_limit = max(accuracy_tie_margin, seed_accuracy_difference)
    accuracy_passed = recipe_accuracy_difference <= accuracy_limit
    family_rows = [
        {
            "family_id": family_id,
            "old_seed_0": old_seed_0_accuracy[family_id],
            "old_seed_1": old_seed_1_accuracy[family_id],
            "new_seed_0": new_seed_0_accuracy[family_id],
        }
        for family_id in old_seed_0_accuracy
    ]
    return {
        "passed": trajectory_passed and accuracy_passed,
        "trajectory_passed": trajectory_passed,
        "maximum_old_seed_to_seed_loss_difference": seed_loss_difference,
        "maximum_new_vs_old_seed_0_loss_difference": recipe_loss_difference,
        "accuracy_passed": accuracy_passed,
        "accuracy_tie_margin": accuracy_tie_margin,
        "old_seed_to_seed_accuracy_difference": seed_accuracy_difference,
        "new_vs_old_seed_0_accuracy_difference": recipe_accuracy_difference,
        "accuracy_limit": accuracy_limit,
        "mean_accuracy": {
            "old_seed_0": old_seed_0_mean,
            "old_seed_1": old_seed_1_mean,
            "new_seed_0": new_seed_0_mean,
        },
        "family_accuracy": family_rows,
    }


def _run_outcome_variant(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    batches: Sequence[Sequence[TaskInstance]],
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    seed: int,
    target_only: bool,
    resource_guard: MemoryPeakSampler,
) -> dict[str, object]:
    model = copy.deepcopy(base_model)
    cast(Any, model).to(device=device, dtype=torch.float32)
    model.train()
    _set_gradient_checkpointing(model, target_only and recipe.gradient_checkpointing)
    seed_everything(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.probe_learning_rates[0],
        weight_decay=settings.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, settings.probe_warmup_steps), 1.0
        ),
    )
    losses: list[float] = []
    for batch in batches:
        if target_only:
            step = _optimizer_step_target_only(
                model,
                tokenizer,
                batch,
                context_length,
                device,
                optimizer,
                recipe,
                resource_guard=resource_guard,
            )
            losses.append(cast(float, step["loss"]))
        else:
            losses.append(
                _optimizer_step_legacy(
                    model,
                    tokenizer,
                    batch,
                    context_length,
                    device,
                    optimizer,
                )
            )
        scheduler.step()
        resource_guard.raise_if_swap_limit_exceeded()
    accuracy = _probe_accuracy_map(
        evaluate_zero_shot_readiness(
            model, tokenizer, stream, settings, context_length, device
        )
    )
    del optimizer, scheduler, model
    cleanup_device(device)
    return {"seed": seed, "losses": losses, "accuracy": accuracy}


def measure_outcome_gate(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    decision: DecisionSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler,
) -> dict[str, object]:
    """Run Amendment 4b's seeded 50-step outcome comparison."""
    if device.type not in {"mps", "cuda"}:
        raise ValueError("Amendment 4b outcome gate requires an accelerator")
    batches = _equivalence_batches(
        stream,
        count=settings.equivalence_trajectory_steps,
        batch_size=settings.probe_batch_size,
    )
    seed_0, seed_1 = settings.equivalence_outcome_seeds
    old_seed_0 = _run_outcome_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        seed=seed_0,
        target_only=False,
        resource_guard=resource_guard,
    )
    old_seed_1 = _run_outcome_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        seed=seed_1,
        target_only=False,
        resource_guard=resource_guard,
    )
    new_seed_0 = _run_outcome_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        seed=seed_0,
        target_only=True,
        resource_guard=resource_guard,
    )
    accuracy_rows = (
        old_seed_0["accuracy"],
        old_seed_1["accuracy"],
        new_seed_0["accuracy"],
    )
    if not all(isinstance(row, dict) for row in accuracy_rows):
        raise TypeError("outcome variants did not return final test accuracy")
    assessment = assess_outcome_gate(
        old_seed_0_losses=cast(list[float], old_seed_0["losses"]),
        old_seed_1_losses=cast(list[float], old_seed_1["losses"]),
        new_seed_0_losses=cast(list[float], new_seed_0["losses"]),
        old_seed_0_accuracy=cast(dict[str, float], old_seed_0["accuracy"]),
        old_seed_1_accuracy=cast(dict[str, float], old_seed_1["accuracy"]),
        new_seed_0_accuracy=cast(dict[str, float], new_seed_0["accuracy"]),
        accuracy_tie_margin=decision.learnability_tie_margin,
    )
    return {
        **assessment,
        "device": str(device),
        "dtype": "float32",
        "steps": settings.equivalence_trajectory_steps,
        "seeds": [seed_0, seed_1],
        "data_order": "identical across all variants",
        "batch_instance_ids": [
            [item.task_id for item in batch] for batch in batches
        ],
        "old_seed_0_losses": old_seed_0["losses"],
        "old_seed_1_losses": old_seed_1["losses"],
        "new_seed_0_losses": new_seed_0["losses"],
    }


def _run_trajectory_variant(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    batches: Sequence[Sequence[TaskInstance]],
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    target_only: bool,
    permute_within_batch: bool,
    evaluate_accuracy: bool,
    resource_guard: MemoryPeakSampler,
) -> dict[str, object]:
    model = copy.deepcopy(base_model)
    cast(Any, model).to(device=device, dtype=torch.float32)
    _set_gradient_checkpointing(model, target_only and recipe.gradient_checkpointing)
    dropout_modules_disabled = _disable_dropout_only(model)
    seed_everything(settings.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.probe_learning_rates[0],
        weight_decay=settings.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, settings.probe_warmup_steps), 1.0
        ),
    )
    losses: list[float] = []
    for batch in batches:
        instances = list(reversed(batch)) if permute_within_batch else list(batch)
        if target_only:
            step = _optimizer_step_target_only(
                model,
                tokenizer,
                instances,
                context_length,
                device,
                optimizer,
                recipe,
                resource_guard=resource_guard,
            )
            losses.append(cast(float, step["loss"]))
        else:
            losses.append(
                _optimizer_step_legacy(
                    model,
                    tokenizer,
                    instances,
                    context_length,
                    device,
                    optimizer,
                )
            )
        scheduler.step()
        resource_guard.raise_if_swap_limit_exceeded()
    accuracy: dict[str, float] | None = None
    if evaluate_accuracy:
        accuracy = _probe_accuracy_map(
            evaluate_zero_shot_readiness(
                model, tokenizer, stream, settings, context_length, device
            )
        )
    del optimizer, scheduler, model
    cleanup_device(device)
    return {
        "losses": losses,
        "accuracy": accuracy,
        "dropout_modules_disabled": dropout_modules_disabled,
    }


def measure_training_trajectory(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    decision: DecisionSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler,
) -> dict[str, object]:
    """Run the preregistered 50-step SmolLM2 trajectory comparison."""
    if device.type not in {"mps", "cuda"}:
        raise ValueError("Amendment 4a trajectory measurement requires an accelerator")
    batches = _equivalence_batches(
        stream,
        count=settings.equivalence_trajectory_steps,
        batch_size=settings.probe_batch_size,
    )
    legacy = _run_trajectory_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        target_only=False,
        permute_within_batch=False,
        evaluate_accuracy=True,
        resource_guard=resource_guard,
    )
    control = _run_trajectory_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        target_only=False,
        permute_within_batch=True,
        evaluate_accuracy=False,
        resource_guard=resource_guard,
    )
    target = _run_trajectory_variant(
        base_model,
        tokenizer,
        stream,
        batches,
        settings,
        context_length,
        device,
        recipe,
        target_only=True,
        permute_within_batch=False,
        evaluate_accuracy=True,
        resource_guard=resource_guard,
    )
    legacy_accuracy = legacy["accuracy"]
    target_accuracy = target["accuracy"]
    if not isinstance(legacy_accuracy, dict) or not isinstance(target_accuracy, dict):
        raise TypeError("trajectory variants did not return final test accuracy")
    assessment = assess_training_trajectory(
        legacy_losses=cast(list[float], legacy["losses"]),
        control_losses=cast(list[float], control["losses"]),
        target_losses=cast(list[float], target["losses"]),
        legacy_accuracy=cast(dict[str, float], legacy_accuracy),
        target_accuracy=cast(dict[str, float], target_accuracy),
        noise_multiplier=settings.equivalence_noise_floor_multiplier,
        accuracy_tie_margin=decision.learnability_tie_margin,
    )
    return {
        **assessment,
        "device": str(device),
        "dtype": "float32",
        "dropout_disabled": True,
        "steps": settings.equivalence_trajectory_steps,
        "batch_instance_ids": [
            [item.task_id for item in batch] for batch in batches
        ],
        "legacy_losses": legacy["losses"],
        "control_losses": control["losses"],
        "target_only_losses": target["losses"],
    }


def _parameter_update_difference(
    legacy_model: CausalLM,
    amended_model: CausalLM,
    legacy_before: Sequence[torch.Tensor],
    amended_before: Sequence[torch.Tensor],
) -> dict[str, float]:
    maximum_absolute = 0.0
    difference_squared = 0.0
    reference_squared = 0.0
    rows = zip(
        legacy_model.parameters(),
        amended_model.parameters(),
        legacy_before,
        amended_before,
        strict=True,
    )
    for legacy, amended, old_legacy, old_amended in rows:
        legacy_update = legacy.detach().to(device="cpu") - old_legacy
        amended_update = amended.detach().to(device="cpu") - old_amended
        difference = legacy_update - amended_update
        maximum_absolute = max(maximum_absolute, float(difference.abs().max().item()))
        difference_squared += float(torch.sum(difference.double().square()).item())
        reference_squared += float(torch.sum(legacy_update.double().square()).item())
    relative_l2 = math.sqrt(difference_squared) / max(
        math.sqrt(reference_squared), torch.finfo(torch.float64).eps
    )
    return {
        "maximum_absolute": maximum_absolute,
        "relative_l2": relative_l2,
        "difference_l2": math.sqrt(difference_squared),
        "legacy_update_l2": math.sqrt(reference_squared),
    }


def prove_training_recipe_equivalence(
    base_model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
    recipe: TrainingRecipe,
    *,
    resource_guard: MemoryPeakSampler,
) -> dict[str, object]:
    """Compare legacy full logits with Amendment 4 from identical train-mode state."""
    legacy_model = copy.deepcopy(base_model).to(device)
    amended_model = copy.deepcopy(base_model).to(device)
    legacy_model.train()
    amended_model.train()
    _set_gradient_checkpointing(amended_model, recipe.gradient_checkpointing)
    legacy_optimizer = torch.optim.AdamW(
        legacy_model.parameters(),
        lr=settings.probe_learning_rates[0],
        weight_decay=settings.probe_weight_decay,
    )
    amended_optimizer = torch.optim.AdamW(
        amended_model.parameters(),
        lr=settings.probe_learning_rates[0],
        weight_decay=settings.probe_weight_decay,
    )
    legacy_scheduler = torch.optim.lr_scheduler.LambdaLR(
        legacy_optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, settings.probe_warmup_steps), 1.0
        ),
    )
    amended_scheduler = torch.optim.lr_scheduler.LambdaLR(
        amended_optimizer,
        lr_lambda=lambda step: min(
            (step + 1) / max(1, settings.probe_warmup_steps), 1.0
        ),
    )
    family = stream.families[0]
    train = stream.get_split(family.family_id).train
    steps: list[dict[str, object]] = []
    passed = True
    for step in range(settings.equivalence_steps):
        offset = (step * settings.probe_batch_size) % len(train)
        instances = [
            train[(offset + index) % len(train)]
            for index in range(settings.probe_batch_size)
        ]
        legacy_before = _parameter_snapshots(legacy_model)
        amended_before = _parameter_snapshots(amended_model)
        state = _rng_state(device)
        legacy_loss = _optimizer_step_legacy(
            legacy_model,
            tokenizer,
            instances,
            context_length,
            device,
            legacy_optimizer,
        )
        legacy_scheduler.step()
        _restore_rng_state(device, state)
        amended_step = _optimizer_step_target_only(
            amended_model,
            tokenizer,
            instances,
            context_length,
            device,
            amended_optimizer,
            recipe,
            resource_guard=resource_guard,
        )
        amended_scheduler.step()
        synchronize_device(device)
        update_difference = _parameter_update_difference(
            legacy_model,
            amended_model,
            legacy_before,
            amended_before,
        )
        loss_difference = abs(legacy_loss - cast(float, amended_step["loss"]))
        step_passed = (
            loss_difference <= settings.equivalence_loss_absolute_tolerance
            and update_difference["maximum_absolute"]
            <= settings.equivalence_update_max_absolute_tolerance
            and update_difference["relative_l2"]
            <= settings.equivalence_update_relative_l2_tolerance
        )
        passed = passed and step_passed
        steps.append(
            {
                "step": step + 1,
                "instance_ids": [item.task_id for item in instances],
                "legacy_loss": legacy_loss,
                "amendment_4_loss": amended_step["loss"],
                "absolute_loss_difference": loss_difference,
                "parameter_update_difference": update_difference,
                "passed": step_passed,
            }
        )
        del legacy_before, amended_before
        resource_guard.raise_if_swap_limit_exceeded()
        if not step_passed:
            break
    del legacy_optimizer, amended_optimizer, legacy_model, amended_model
    cleanup_device(device)
    return {
        "passed": passed,
        "family_id": family.family_id,
        "steps_requested": settings.equivalence_steps,
        "steps_completed": len(steps),
        "effective_batch_size": settings.probe_batch_size,
        "amendment_4_micro_batch_size": recipe.micro_batch_size,
        "amendment_4_gradient_checkpointing": recipe.gradient_checkpointing,
        "training_mode": True,
        "dropout_disabled_for_proof": False,
        "steps": steps,
    }


def run_amendment_4_equivalence(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    loader: CandidateLoader = load_candidate,
) -> dict[str, object]:
    """Persist the mandatory pre-sweep equivalence gate and fail closed."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(
        config_path, destination, settings
    )
    record_path = destination / "equivalence.json"
    if record_path.is_file():
        existing = _read_json(record_path)
        if (
            existing.get("schema_version") == AMENDMENT_4_EQUIVALENCE_SCHEMA_VERSION
            and existing.get("manifest_sha256") == manifest_sha256
            and existing.get("status") in {"passed", "failed"}
        ):
            return existing
        raise RuntimeError(
            "existing equivalence evidence does not match the frozen Amendment 4 manifest"
        )
    record: dict[str, object] = {
        "schema_version": AMENDMENT_4_EQUIVALENCE_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": manifest["correlation_id"],
        "status": "running",
        "tolerances": {
            "loss_absolute": settings.measurement.equivalence_loss_absolute_tolerance,
            "update_maximum_absolute": (
                settings.measurement.equivalence_update_max_absolute_tolerance
            ),
            "update_relative_l2": (
                settings.measurement.equivalence_update_relative_l2_tolerance
            ),
        },
        "candidate_results": [],
    }
    write_json(record_path, record)
    by_id = {candidate.candidate_id: candidate for candidate in settings.candidates}
    device = resolve_device(settings.measurement.device)
    for candidate_id in settings.measurement.equivalence_candidate_ids:
        candidate = by_id[candidate_id]
        sampler = MemoryPeakSampler(
            device,
            settings.measurement.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=(
                settings.measurement.swap_sample_interval_seconds
            ),
        )
        candidate_result: dict[str, object]
        model: CausalLM | None = None
        with sampler:
            try:
                seed_everything(settings.measurement.seed)
                configure_determinism(settings.measurement)
                loaded, load_attempts = _load_with_retries(
                    candidate, settings.measurement, loader
                )
                model = loaded.model
                tokenizer = loaded.tokenizer
                context_length = model_context_length(model, tokenizer)
                stream = ContinualTaskStream(
                    num_train_per_task=settings.measurement.train_items_per_family,
                    num_val_per_task=settings.measurement.validation_items_per_family,
                    num_test_per_task=settings.measurement.test_items_per_family,
                    seed=settings.measurement.seed,
                    max_generation_attempts=(
                        settings.measurement.max_generation_attempts
                    ),
                )
                recipe, selection = select_training_recipe(
                    model,
                    tokenizer,
                    stream,
                    settings.measurement,
                    settings.decision,
                    context_length,
                    device,
                    resource_guard=sampler,
                )
                if recipe is None:
                    raise RuntimeError("equivalence proof model has no feasible memory recipe")
                proof = prove_training_recipe_equivalence(
                    model,
                    tokenizer,
                    stream,
                    settings.measurement,
                    context_length,
                    device,
                    recipe,
                    resource_guard=sampler,
                )
                candidate_result = {
                    "candidate_id": candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "memory_recipe_selection": selection,
                    "equivalence": proof,
                    "passed": proof["passed"],
                }
            except Exception as error:
                candidate_result = {
                    "candidate_id": candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
        candidate_result["memory"] = sampler.record()
        cast(list[dict[str, object]], record["candidate_results"]).append(
            candidate_result
        )
        model = None
        cleanup_device(device)
        if candidate_result["passed"] is not True:
            record["status"] = "failed"
            write_json(record_path, record)
            return record
        write_json(record_path, record)
    record["status"] = "passed"
    write_json(record_path, record)
    return record


def _new_equivalence_stream(settings: MeasurementSettings) -> ContinualTaskStream:
    return ContinualTaskStream(
        num_train_per_task=settings.train_items_per_family,
        num_val_per_task=settings.validation_items_per_family,
        num_test_per_task=settings.test_items_per_family,
        seed=settings.seed,
        max_generation_attempts=settings.max_generation_attempts,
    )


def run_amendment_4a_equivalence(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    loader: CandidateLoader = load_candidate,
) -> dict[str, object]:
    """Run and persist the three-stage Amendment 4a fail-closed gate."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(
        config_path, destination, settings
    )
    record_path = destination / "equivalence-amendment-4a.json"
    if record_path.is_file():
        existing = _read_json(record_path)
        if (
            existing.get("schema_version") == AMENDMENT_4A_EQUIVALENCE_SCHEMA_VERSION
            and existing.get("manifest_sha256") == manifest_sha256
            and existing.get("status") in {"passed", "failed"}
        ):
            return existing
        raise RuntimeError(
            "existing Amendment 4a evidence is incomplete or does not match its manifest"
        )
    device = resolve_device(settings.measurement.device)
    if device.type not in {"mps", "cuda"}:
        raise RuntimeError("Amendment 4a requires the configured real accelerator")
    record: dict[str, object] = {
        "schema_version": AMENDMENT_4A_EQUIVALENCE_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": manifest["correlation_id"],
        "status": "running",
        "protocol": {
            "exact_loss_absolute_tolerance": (
                settings.measurement.equivalence_exact_loss_absolute_tolerance
            ),
            "exact_gradient_relative_l2_tolerance": (
                settings.measurement.equivalence_exact_gradient_relative_l2_tolerance
            ),
            "noise_floor_batches": (
                settings.measurement.equivalence_noise_floor_batches
            ),
            "noise_floor_multiplier": (
                settings.measurement.equivalence_noise_floor_multiplier
            ),
            "trajectory_steps": settings.measurement.equivalence_trajectory_steps,
            "accuracy_tie_margin": settings.decision.learnability_tie_margin,
            "seed": settings.measurement.seed,
            "learning_rate": settings.measurement.probe_learning_rates[0],
            "weight_decay": settings.measurement.probe_weight_decay,
            "warmup_steps": settings.measurement.probe_warmup_steps,
            "control_permutation": "reverse within each effective batch",
            "batch_order": "training item index outer, configured family order inner",
            "dropout_disabled": True,
        },
        "exact_math": {"status": "running", "candidate_results": []},
        "noise_floor": {"status": "pending", "candidate_results": []},
        "training_trajectory": {"status": "pending"},
    }
    write_json(record_path, record)
    by_id = {candidate.candidate_id: candidate for candidate in settings.candidates}
    recipes: dict[str, TrainingRecipe] = {}
    exact_section = cast(dict[str, object], record["exact_math"])
    exact_results = cast(list[dict[str, object]], exact_section["candidate_results"])
    for candidate_id in settings.measurement.equivalence_candidate_ids:
        sampler = MemoryPeakSampler(
            device,
            settings.measurement.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=(
                settings.measurement.swap_sample_interval_seconds
            ),
        )
        result: dict[str, object]
        with sampler:
            try:
                seed_everything(settings.measurement.seed)
                configure_determinism(settings.measurement)
                loaded, load_attempts = _load_with_retries(
                    by_id[candidate_id], settings.measurement, loader
                )
                context_length = model_context_length(loaded.model, loaded.tokenizer)
                stream = _new_equivalence_stream(settings.measurement)
                recipe, selection = select_training_recipe(
                    loaded.model,
                    loaded.tokenizer,
                    stream,
                    settings.measurement,
                    settings.decision,
                    context_length,
                    device,
                    resource_guard=sampler,
                )
                if recipe is None:
                    raise RuntimeError("equivalence proof model has no feasible memory recipe")
                recipes[candidate_id] = recipe
                batch = _equivalence_batches(
                    stream, count=1, batch_size=settings.measurement.probe_batch_size
                )[0]
                proof = prove_exact_gradient_equivalence(
                    loaded.model,
                    loaded.tokenizer,
                    batch,
                    settings.measurement,
                    context_length,
                    recipe,
                )
                result = {
                    "candidate_id": candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "instance_ids": [item.task_id for item in batch],
                    "memory_recipe_selection": selection,
                    "recipe": {
                        "micro_batch_size": recipe.micro_batch_size,
                        "gradient_checkpointing": recipe.gradient_checkpointing,
                    },
                    "proof": proof,
                    "passed": proof["passed"],
                }
            except Exception as error:
                result = {
                    "candidate_id": candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
        result["memory"] = sampler.record()
        exact_results.append(result)
        cleanup_device(device)
        if result["passed"] is not True:
            exact_section["status"] = "failed"
            record["status"] = "failed"
            record["failed_stage"] = "exact_math"
            write_json(record_path, record)
            return record
        write_json(record_path, record)
    exact_section["status"] = "passed"
    noise_section = cast(dict[str, object], record["noise_floor"])
    noise_section["status"] = "running"
    noise_results = cast(list[dict[str, object]], noise_section["candidate_results"])
    write_json(record_path, record)
    for candidate_id in settings.measurement.equivalence_candidate_ids:
        sampler = MemoryPeakSampler(
            device,
            settings.measurement.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=(
                settings.measurement.swap_sample_interval_seconds
            ),
        )
        with sampler:
            try:
                seed_everything(settings.measurement.seed)
                loaded, load_attempts = _load_with_retries(
                    by_id[candidate_id], settings.measurement, loader
                )
                context_length = model_context_length(loaded.model, loaded.tokenizer)
                stream = _new_equivalence_stream(settings.measurement)
                batches = _equivalence_batches(
                    stream,
                    count=settings.measurement.equivalence_noise_floor_batches,
                    batch_size=settings.measurement.probe_batch_size,
                )
                proof = measure_noise_floor(
                    loaded.model,
                    loaded.tokenizer,
                    batches,
                    settings.measurement,
                    context_length,
                    device,
                    recipes[candidate_id],
                )
                result = {
                    "candidate_id": candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "proof": proof,
                    "passed": proof["passed"],
                }
            except Exception as error:
                result = {
                    "candidate_id": candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
        result["memory"] = sampler.record()
        noise_results.append(result)
        cleanup_device(device)
        if result["passed"] is not True:
            noise_section["status"] = "failed"
            record["status"] = "failed"
            record["failed_stage"] = "noise_floor"
            write_json(record_path, record)
            return record
        write_json(record_path, record)
    noise_section["status"] = "passed"
    trajectory_section = cast(dict[str, object], record["training_trajectory"])
    trajectory_section["status"] = "running"
    write_json(record_path, record)
    trajectory_candidate_id = "smollm2-135m"
    sampler = MemoryPeakSampler(
        device,
        settings.measurement.memory_sample_interval_seconds,
        maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
        swap_sample_interval_seconds=settings.measurement.swap_sample_interval_seconds,
    )
    with sampler:
        try:
            seed_everything(settings.measurement.seed)
            loaded, load_attempts = _load_with_retries(
                by_id[trajectory_candidate_id], settings.measurement, loader
            )
            context_length = model_context_length(loaded.model, loaded.tokenizer)
            stream = _new_equivalence_stream(settings.measurement)
            proof = measure_training_trajectory(
                loaded.model,
                loaded.tokenizer,
                stream,
                settings.measurement,
                settings.decision,
                context_length,
                device,
                recipes[trajectory_candidate_id],
                resource_guard=sampler,
            )
            trajectory_section.update(
                {
                    "candidate_id": trajectory_candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "proof": proof,
                    "passed": proof["passed"],
                }
            )
        except Exception as error:
            trajectory_section.update(
                {
                    "candidate_id": trajectory_candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
            )
    trajectory_section["memory"] = sampler.record()
    cleanup_device(device)
    if trajectory_section["passed"] is not True:
        trajectory_section["status"] = "failed"
        record["status"] = "failed"
        record["failed_stage"] = "training_trajectory"
        write_json(record_path, record)
        return record
    trajectory_section["status"] = "passed"
    record["status"] = "passed"
    write_json(record_path, record)
    return record


def run_amendment_4b_equivalence(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    loader: CandidateLoader = load_candidate,
) -> dict[str, object]:
    """Run and persist Amendment 4b's identity proof and outcome gate."""
    settings = load_base_model_evaluation_settings(config_path)
    if settings.measurement.equivalence_protocol != "amendment_4b":
        raise ValueError("Amendment 4b requires equivalence_protocol=amendment_4b")
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(
        config_path, destination, settings
    )
    record_path = destination / "equivalence-amendment-4b.json"
    if record_path.is_file():
        existing = _read_json(record_path)
        if (
            existing.get("schema_version") == AMENDMENT_4B_EQUIVALENCE_SCHEMA_VERSION
            and existing.get("manifest_sha256") == manifest_sha256
            and existing.get("status") in {"passed", "failed"}
        ):
            return existing
        raise RuntimeError(
            "existing Amendment 4b evidence is incomplete or does not match its manifest"
        )
    device = resolve_device(settings.measurement.device)
    if device.type not in {"mps", "cuda"}:
        raise RuntimeError("Amendment 4b requires the configured real accelerator")
    outcome_seed_0, outcome_seed_1 = settings.measurement.equivalence_outcome_seeds
    record: dict[str, object] = {
        "schema_version": AMENDMENT_4B_EQUIVALENCE_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": manifest["correlation_id"],
        "status": "running",
        "protocol": {
            "exact_loss_absolute_tolerance": (
                settings.measurement.equivalence_exact_loss_absolute_tolerance
            ),
            "exact_gradient_relative_l2_tolerance": (
                settings.measurement.equivalence_exact_gradient_relative_l2_tolerance
            ),
            "shape_control_gating": False,
            "shape_control_batches": (
                settings.measurement.equivalence_noise_floor_batches
            ),
            "shape_control_sequence_chunk_size": (
                settings.measurement.equivalence_shape_control_chunk_size
            ),
            "outcome_steps": settings.measurement.equivalence_trajectory_steps,
            "outcome_seeds": [outcome_seed_0, outcome_seed_1],
            "accuracy_tie_margin": settings.decision.learnability_tie_margin,
            "learning_rate": settings.measurement.probe_learning_rates[0],
            "weight_decay": settings.measurement.probe_weight_decay,
            "warmup_steps": settings.measurement.probe_warmup_steps,
            "batch_order": "training item index outer, configured family order inner",
            "data_order": "identical across outcome variants",
        },
        "exact_math": {"status": "running", "candidate_results": []},
        "shape_changing_control": {
            "status": "pending",
            "gating": False,
            "candidate_results": [],
        },
        "outcome_gate": {"status": "pending"},
    }
    write_json(record_path, record)
    by_id = {candidate.candidate_id: candidate for candidate in settings.candidates}
    recipes: dict[str, TrainingRecipe] = {}
    exact_section = cast(dict[str, object], record["exact_math"])
    exact_results = cast(list[dict[str, object]], exact_section["candidate_results"])
    for candidate_id in settings.measurement.equivalence_candidate_ids:
        sampler = MemoryPeakSampler(
            device,
            settings.measurement.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=(
                settings.measurement.swap_sample_interval_seconds
            ),
        )
        result: dict[str, object]
        with sampler:
            try:
                seed_everything(settings.measurement.seed)
                configure_determinism(settings.measurement)
                loaded, load_attempts = _load_with_retries(
                    by_id[candidate_id], settings.measurement, loader
                )
                context_length = model_context_length(loaded.model, loaded.tokenizer)
                stream = _new_equivalence_stream(settings.measurement)
                recipe, selection = select_training_recipe(
                    loaded.model,
                    loaded.tokenizer,
                    stream,
                    settings.measurement,
                    settings.decision,
                    context_length,
                    device,
                    resource_guard=sampler,
                )
                if recipe is None:
                    raise RuntimeError("equivalence proof model has no feasible memory recipe")
                recipes[candidate_id] = recipe
                batch = _equivalence_batches(
                    stream, count=1, batch_size=settings.measurement.probe_batch_size
                )[0]
                proof = prove_exact_gradient_equivalence(
                    loaded.model,
                    loaded.tokenizer,
                    batch,
                    settings.measurement,
                    context_length,
                    recipe,
                )
                result = {
                    "candidate_id": candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "instance_ids": [item.task_id for item in batch],
                    "memory_recipe_selection": selection,
                    "recipe": {
                        "micro_batch_size": recipe.micro_batch_size,
                        "gradient_checkpointing": recipe.gradient_checkpointing,
                    },
                    "proof": proof,
                    "passed": proof["passed"],
                }
            except Exception as error:
                result = {
                    "candidate_id": candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
        result["memory"] = sampler.record()
        exact_results.append(result)
        cleanup_device(device)
        if result["passed"] is not True:
            exact_section["status"] = "failed"
            record["status"] = "failed"
            record["failed_stage"] = "exact_math"
            write_json(record_path, record)
            return record
        write_json(record_path, record)
    exact_section["status"] = "passed"

    control_section = cast(dict[str, object], record["shape_changing_control"])
    control_section["status"] = "running"
    control_results = cast(
        list[dict[str, object]], control_section["candidate_results"]
    )
    write_json(record_path, record)
    for candidate_id in settings.measurement.equivalence_candidate_ids:
        sampler = MemoryPeakSampler(
            device,
            settings.measurement.memory_sample_interval_seconds,
            maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
            swap_sample_interval_seconds=(
                settings.measurement.swap_sample_interval_seconds
            ),
        )
        with sampler:
            try:
                seed_everything(settings.measurement.seed)
                loaded, load_attempts = _load_with_retries(
                    by_id[candidate_id], settings.measurement, loader
                )
                context_length = model_context_length(loaded.model, loaded.tokenizer)
                stream = _new_equivalence_stream(settings.measurement)
                batches = _equivalence_batches(
                    stream,
                    count=settings.measurement.equivalence_noise_floor_batches,
                    batch_size=settings.measurement.probe_batch_size,
                )
                proof = measure_shape_changing_control(
                    loaded.model,
                    loaded.tokenizer,
                    batches,
                    settings.measurement,
                    context_length,
                    device,
                    recipes[candidate_id],
                )
                result = {
                    "candidate_id": candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "status": "completed",
                    "proof": proof,
                }
            except Exception as error:
                result = {
                    "candidate_id": candidate_id,
                    "status": "diagnostic_failed",
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
        result["memory"] = sampler.record()
        control_results.append(result)
        cleanup_device(device)
        write_json(record_path, record)
    control_section["status"] = (
        "completed"
        if all(item["status"] == "completed" for item in control_results)
        else "completed_with_diagnostic_failures"
    )

    outcome_section = cast(dict[str, object], record["outcome_gate"])
    outcome_section["status"] = "running"
    write_json(record_path, record)
    outcome_candidate_id = "smollm2-135m"
    sampler = MemoryPeakSampler(
        device,
        settings.measurement.memory_sample_interval_seconds,
        maximum_swap_growth_bytes=settings.decision.maximum_swap_growth_bytes,
        swap_sample_interval_seconds=settings.measurement.swap_sample_interval_seconds,
    )
    with sampler:
        try:
            seed_everything(settings.measurement.seed)
            loaded, load_attempts = _load_with_retries(
                by_id[outcome_candidate_id], settings.measurement, loader
            )
            context_length = model_context_length(loaded.model, loaded.tokenizer)
            stream = _new_equivalence_stream(settings.measurement)
            proof = measure_outcome_gate(
                loaded.model,
                loaded.tokenizer,
                stream,
                settings.measurement,
                settings.decision,
                context_length,
                device,
                recipes[outcome_candidate_id],
                resource_guard=sampler,
            )
            outcome_section.update(
                {
                    "candidate_id": outcome_candidate_id,
                    "resolved_revision": loaded.resolved_revision,
                    "load_attempts": load_attempts,
                    "proof": proof,
                    "passed": proof["passed"],
                }
            )
        except Exception as error:
            outcome_section.update(
                {
                    "candidate_id": outcome_candidate_id,
                    "passed": False,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "out_of_memory": _is_out_of_memory(error),
                    },
                }
            )
    outcome_section["memory"] = sampler.record()
    cleanup_device(device)
    if outcome_section["passed"] is not True:
        outcome_section["status"] = "failed"
        record["status"] = "failed"
        record["failed_stage"] = "outcome_gate"
        write_json(record_path, record)
        return record
    outcome_section["status"] = "passed"
    record["status"] = "passed"
    write_json(record_path, record)
    return record


def benchmark_generation(
    model: CausalLM,
    tokenizer: PreTrainedTokenizerBase,
    stream: ContinualTaskStream,
    settings: MeasurementSettings,
    context_length: int,
    device: torch.device,
) -> dict[str, object]:
    """Benchmark deterministic greedy generation on fixed stream prompts."""
    predictor = TransformersPredictor(
        model,
        tokenizer,
        device=device,
        context_length=context_length,
        max_new_tokens=settings.generation_benchmark_new_tokens,
    )
    items: list[tuple[TaskInstance, tuple[int, ...]]] = []
    for family in stream.families:
        for instance in stream.get_split(family.family_id).test:
            items.append((instance, family.stop_token_ids))
            if len(items) >= settings.generation_benchmark_prompts:
                break
        if len(items) >= settings.generation_benchmark_prompts:
            break
    for _ in range(settings.generation_benchmark_warmup_runs):
        for item, stops in items:
            predictor.generate(item.prompt, stop_token_ids=stops)
    repetitions: list[dict[str, float | int]] = []
    for _ in range(settings.generation_benchmark_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        token_count = 0
        byte_count = 0
        for item, stops in items:
            result = predictor.generate(item.prompt, stop_token_ids=stops)
            token_count += result.generated_tokens
            byte_count += len(result.text.encode("utf-8"))
        synchronize_device(device)
        elapsed = time.perf_counter() - started
        repetitions.append(
            {
                "seconds": elapsed,
                "generated_tokens": token_count,
                "generated_bytes": byte_count,
                "tokens_per_second": token_count / elapsed,
                "bytes_per_second": byte_count / elapsed,
            }
        )
    return {
        "prompts": len(items),
        "warmup_runs": settings.generation_benchmark_warmup_runs,
        "repetitions": repetitions,
        "median_tokens_per_second": statistics.median(
            cast(float, item["tokens_per_second"]) for item in repetitions
        ),
        "median_bytes_per_second": statistics.median(
            cast(float, item["bytes_per_second"]) for item in repetitions
        ),
        "method": "deterministic uncached greedy adapter",
    }


def _architecture_type(model: CausalLM) -> str:
    value = getattr(model.config, "model_type", None)
    return value if isinstance(value, str) and value else type(model).__name__


def _candidate_record_path(output_dir: Path, candidate_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("._") or "candidate"
    suffix = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:10]
    return output_dir / "candidates" / f"{safe}-{suffix}.json"


def _seed_record_path(output_dir: Path, candidate_id: str, seed: int) -> Path:
    primary_name = _candidate_record_path(output_dir, candidate_id).stem
    return output_dir / "seed_runs" / f"{primary_name}-seed-{seed}.json"


def _is_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _tokenizer_contract_satisfied(fitness: Mapping[str, object]) -> bool:
    """Accept Amendment 3 evidence or stricter legacy exact-round-trip evidence."""
    nfc = fitness.get("exact_roundtrip_after_nfc_normalization")
    task = fitness.get("exact_task_stream_roundtrip")
    if nfc is not None or task is not None:
        return nfc is True and task is True
    return fitness.get("exact_utf8_roundtrip") is True


def _config_fingerprint(
    candidate: CandidateSpec, settings: MeasurementSettings, manifest_sha256: str
) -> str:
    payload = {
        "candidate": candidate.__dict__,
        "measurement": {
            **settings.__dict__,
            "heldout_text_path": str(settings.heldout_text_path),
            "heldout_prompts_path": str(settings.heldout_prompts_path),
        },
        "schema": EVALUATION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()


def _is_transient_load_error(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    message = str(error).lower()
    markers = ("http 500", "http 502", "http 503", "http 504", "timed out", "connection reset")
    return any(marker in message for marker in markers)


def _load_with_retries(
    candidate: CandidateSpec,
    settings: MeasurementSettings,
    loader: CandidateLoader,
) -> tuple[LoadedCandidate, int]:
    last_error: Exception | None = None
    for attempt in range(1, settings.load_retries + 1):
        try:
            return loader(candidate, settings), attempt
        except Exception as error:
            last_error = error
            if (
                attempt >= settings.load_retries
                or _is_out_of_memory(error)
                or not _is_transient_load_error(error)
            ):
                raise
            time.sleep(settings.load_retry_backoff_seconds * (2 ** (attempt - 1)))
    if last_error is None:
        raise RuntimeError("candidate loader did not run")
    raise last_error


def evaluate_candidate(
    candidate: CandidateSpec,
    settings: MeasurementSettings,
    *,
    loader: CandidateLoader = load_candidate,
    decision_settings: DecisionSettings | None = None,
    manifest_sha256: str = "test-manifest",
    correlation_id: str = "test-correlation-id",
    equivalence_gate_owner_override: Mapping[str, str] | None = None,
    progress_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate one candidate, converting every failure into a durable record."""
    fingerprint = _config_fingerprint(candidate, settings, manifest_sha256)
    phase = "load"
    started = time.perf_counter()
    device = resolve_device(settings.device)
    sampler = MemoryPeakSampler(
        device,
        settings.memory_sample_interval_seconds,
        maximum_swap_growth_bytes=(
            None
            if decision_settings is None
            else decision_settings.maximum_swap_growth_bytes
        ),
        swap_sample_interval_seconds=settings.swap_sample_interval_seconds,
    )
    partial: dict[str, object] = {}
    model: CausalLM | None = None
    snapshot_directory: tempfile.TemporaryDirectory[str] | None = None
    pristine_snapshot_path: Path | None = None
    pristine_snapshot_digest: str | None = None
    generation: dict[str, object] | None = None
    with sampler:
        try:
            seed_everything(settings.seed)
            configure_determinism(settings)
            loaded, load_attempts = _load_with_retries(candidate, settings, loader)
            sampler.raise_if_swap_limit_exceeded()
            model = loaded.model
            tokenizer = loaded.tokenizer
            context_length = model_context_length(model, tokenizer)
            hidden_size = _hidden_size(model)
            architecture = _architecture_type(model)
            model_vocab_size = getattr(model.config, "vocab_size", None)
            identity: dict[str, object] = {
                "repo_id_or_path": candidate.source,
                "resolved_revision": loaded.resolved_revision,
                "license": (
                    _canonical_license(candidate.license) if candidate.license is not None else None
                ),
                "license_source": "config_fallback",
                "unique_parameters": unique_parameter_count(model),
                "architecture": architecture,
                "tokenizer_vocabulary_size": len(tokenizer),
                "model_vocabulary_size": model_vocab_size,
                "vocabulary_size_mismatch": model_vocab_size != len(tokenizer),
                "context_length": context_length,
                "tokenizer_type": type(tokenizer).__name__,
                "trust_remote_code": False,
                "load_attempts": load_attempts,
                "torch_dtype": settings.torch_dtype,
            }
            partial["identity"] = identity
            phase = "task_stream"
            stream = ContinualTaskStream(
                num_train_per_task=settings.train_items_per_family,
                num_val_per_task=settings.validation_items_per_family,
                num_test_per_task=settings.test_items_per_family,
                seed=settings.seed,
                max_generation_attempts=settings.max_generation_attempts,
            )
            partial["task_stream_properties"] = compute_task_stream_properties(stream)
            phase = "tokenizer_fitness"
            token_fitness = tokenizer_fitness(
                model, tokenizer, stream, settings.utf8_roundtrip_samples
            )
            partial["tokenizer_fitness"] = token_fitness
            phase = "structural_fitness"
            structure = inspect_expert_attachment_structure(model, hidden_size)
            partial["structural_fitness"] = structure
            sampler.raise_if_swap_limit_exceeded()
            cheap_reasons: list[str] = []
            if decision_settings is not None:
                allowed = {
                    _canonical_license(item)
                    for item in decision_settings.permissive_licenses
                }
                if (
                    candidate.license is None
                    or _canonical_license(candidate.license) not in allowed
                ):
                    cheap_reasons.append(
                        f"licence {candidate.license!r} is not in the permissive allowlist"
                    )
            if not _tokenizer_contract_satisfied(token_fitness):
                cheap_reasons.append(
                    "tokenizer failed NFC-normalized or exact task-stream round-trip"
                )
            if token_fitness["eos_usable_as_answer_terminator"] is not True:
                cheap_reasons.append(
                    "tokenizer/model EOS is not usable as an answer terminator"
                )
            if structure["all_blocks_expose_feed_forward"] is not True:
                cheap_reasons.append(
                    "not every transformer block exposes an attachable feed-forward sublayer"
                )
            if cheap_reasons:
                result = {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "manifest_sha256": manifest_sha256,
                    "correlation_id": correlation_id,
                    "config_fingerprint": fingerprint,
                    "status": "hard_constraint_rejected",
                    "candidate_id": candidate.candidate_id,
                    "reference_only": candidate.reference_only,
                    **partial,
                    "hard_constraint_reasons": cheap_reasons,
                    "probe_skipped": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            elif candidate.reference_only:
                result = {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "manifest_sha256": manifest_sha256,
                    "correlation_id": correlation_id,
                    "config_fingerprint": fingerprint,
                    "status": "reference_only",
                    "candidate_id": candidate.candidate_id,
                    "reference_only": True,
                    **partial,
                    "probe_skipped": True,
                    "probe_skip_reason": (
                        "reference-only record is excluded from selection before probe"
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            else:
                phase = "language_quality"
                if settings.pristine_model_device == "host":
                    model.to(device)
                try:
                    language = evaluate_language_quality(
                        model, tokenizer, settings, context_length, device
                    )
                    partial["language_modeling"] = language
                    sampler.raise_if_swap_limit_exceeded()
                    phase = "zero_shot_readiness"
                    zero_shot = evaluate_zero_shot_readiness(
                        model, tokenizer, stream, settings, context_length, device
                    )
                finally:
                    if settings.pristine_model_device == "host":
                        model.to(torch.device("cpu"))
                        cleanup_device(device)
                partial["zero_shot_readiness"] = zero_shot
                if settings.release_pristine_before_training:
                    phase = "generation_benchmark"
                    if settings.pristine_model_device == "host":
                        model.to(device)
                    generation = benchmark_generation(
                        model, tokenizer, stream, settings, context_length, device
                    )
                    sampler.raise_if_swap_limit_exceeded()
                    snapshot_directory = tempfile.TemporaryDirectory(
                        prefix="aptivra-pristine-snapshot-"
                    )
                    pristine_snapshot_path = (
                        Path(snapshot_directory.name) / "model-state.pt"
                    )
                    pristine_snapshot_digest = _write_pristine_snapshot(
                        model, pristine_snapshot_path, device
                    )
                partial["pristine_model_placement"] = {
                    "configured": settings.pristine_model_device,
                    "training_copy_source": (
                        "cpu" if settings.pristine_model_device == "host" else str(device)
                    ),
                    "mathematical_role": (
                        "disk-backed restart source with one live training object"
                        if settings.release_pristine_before_training
                        else "placement only; pristine weights and copied trainable values "
                        "are unchanged"
                    ),
                    "released_before_training": (
                        settings.release_pristine_before_training
                    ),
                    "snapshot_sha256": pristine_snapshot_digest,
                }
                if settings.release_pristine_before_training and progress_path is not None:
                    write_json(
                        progress_path,
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "manifest_sha256": manifest_sha256,
                            "correlation_id": correlation_id,
                            "config_fingerprint": fingerprint,
                            "status": "pristine_evaluation_persisted",
                            "candidate_id": candidate.candidate_id,
                            "reference_only": candidate.reference_only,
                            **partial,
                            "generation_benchmark": generation,
                        },
                    )
                sampler.raise_if_swap_limit_exceeded()
                parameter_count = cast(int, identity["unique_parameters"])
                feasibility = (
                    full_finetuning_memory_feasibility(
                        parameter_count, settings, decision_settings
                    )
                    if decision_settings is not None
                    else None
                )
                if feasibility is not None:
                    feasibility["verdict_role"] = (
                        "informational lower bound; Amendment 4c feasibility uses only "
                        "the measured candidate peak"
                    )
                    partial["full_finetuning_memory_feasibility"] = feasibility
                if not candidate.reference_only:
                    phase = "memory_recipe_selection"
                    recipe: TrainingRecipe | None
                    if decision_settings is None:
                        recipe = TrainingRecipe(settings.probe_batch_size, False)
                        recipe_evidence: dict[str, object] = {
                            "selected_micro_batch_size": settings.probe_batch_size,
                            "selected_gradient_checkpointing": False,
                            "selection_skipped": "decision settings were not supplied",
                        }
                    else:
                        recipe, recipe_evidence = select_training_recipe(
                            model,
                            tokenizer,
                            stream,
                            settings,
                            decision_settings,
                            context_length,
                            device,
                            resource_guard=sampler,
                            pristine_snapshot_path=pristine_snapshot_path,
                            pristine_snapshot_digest=pristine_snapshot_digest,
                        )
                    partial["training_memory_recipe"] = recipe_evidence
                    if recipe is None:
                        attempts = cast(
                            list[dict[str, object]], recipe_evidence.get("attempts", [])
                        )
                        maximum_peak = cast(
                            int, recipe_evidence["configured_maximum_peak_bytes"]
                        )
                        measured_over_cap = any(
                            cast(int, cast(dict[str, object], item["memory"])[
                                "hard_constraint_peak_bytes"
                            ])
                            > maximum_peak
                            for item in attempts
                        )
                        if measured_over_cap:
                            reason = (
                                "candidate's measured accelerator/process peak exceeded the "
                                "configured memory cap"
                            )
                            result = {
                                "schema_version": EVALUATION_SCHEMA_VERSION,
                                "manifest_sha256": manifest_sha256,
                                "correlation_id": correlation_id,
                                "config_fingerprint": fingerprint,
                                "status": "hard_constraint_rejected",
                                "rejection_category": "candidate_footprint",
                                "candidate_id": candidate.candidate_id,
                                "reference_only": candidate.reference_only,
                                **partial,
                                "hard_constraint_reasons": [reason],
                                "probe_skipped": True,
                                "probe_skip_reason": reason,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        else:
                            reason = (
                                "no training recipe completed and no measured candidate "
                                "peak exceeded the configured memory cap"
                            )
                            result = {
                                "schema_version": EVALUATION_SCHEMA_VERSION,
                                "manifest_sha256": manifest_sha256,
                                "correlation_id": correlation_id,
                                "config_fingerprint": fingerprint,
                                "status": "failed",
                                "candidate_id": candidate.candidate_id,
                                "reference_only": candidate.reference_only,
                                "partial_measurements": partial,
                                "failure": {
                                    "phase": "memory_recipe_selection",
                                    "type": "TrainingRecipeMeasurementFailure",
                                    "message": reason,
                                    "out_of_memory": True,
                                },
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                    else:
                        phase = "task_readiness"
                        readiness = evaluate_task_readiness(
                            model,
                            tokenizer,
                            stream,
                            settings,
                            context_length,
                            device,
                            zero_shot_readiness=zero_shot,
                            recipe=recipe,
                            resource_guard=sampler,
                            pristine_snapshot_path=pristine_snapshot_path,
                            pristine_snapshot_digest=pristine_snapshot_digest,
                        )
                        partial["task_stream_readiness"] = readiness
                        sampler.raise_if_swap_limit_exceeded()
                        phase = "cost"
                        if generation is None:
                            if settings.pristine_model_device == "host":
                                model.to(device)
                            try:
                                generation = benchmark_generation(
                                    model,
                                    tokenizer,
                                    stream,
                                    settings,
                                    context_length,
                                    device,
                                )
                            finally:
                                if settings.pristine_model_device == "host":
                                    model.to(torch.device("cpu"))
                                    cleanup_device(device)
                        sampler.raise_if_swap_limit_exceeded()
                        families = cast(list[dict[str, object]], readiness["families"])
                        cost: dict[str, object] = {
                            "device": str(device),
                            "generation": generation,
                            "median_probe_training_tokens_per_second": statistics.median(
                                cast(float, item["training_tokens_per_second"])
                                for item in families
                            ),
                            "median_probe_training_bytes_per_second": statistics.median(
                                cast(float, item["training_bytes_per_second"])
                                for item in families
                            ),
                            "probe_throughput_definition": (
                                "median selected-LR family rate: all non-padding tokens divided "
                                "by synchronized elapsed seconds across the configured probe steps"
                            ),
                            "total_seconds": time.perf_counter() - started,
                        }
                        result = {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "manifest_sha256": manifest_sha256,
                            "correlation_id": correlation_id,
                            "config_fingerprint": fingerprint,
                            "status": "completed",
                            "candidate_id": candidate.candidate_id,
                            "reference_only": candidate.reference_only,
                            **partial,
                            "cost": cost,
                            "measurement_protocol": {
                                "seed": settings.seed,
                                "variance_seeds": list(settings.variance_seeds),
                                "train_items_per_family": settings.train_items_per_family,
                                "validation_items_per_family": (
                                    settings.validation_items_per_family
                                ),
                                "test_items_per_family": settings.test_items_per_family,
                                "probe_steps": settings.probe_steps,
                                "probe_batch_size": settings.probe_batch_size,
                                "probe_micro_batch_size": recipe.micro_batch_size,
                                "gradient_accumulation_steps": (
                                    settings.probe_batch_size
                                    // recipe.micro_batch_size
                                ),
                                "gradient_checkpointing": (
                                    recipe.gradient_checkpointing
                                ),
                                "target_only_training_loss": (
                                    settings.target_only_training_loss
                                ),
                                "probe_learning_rates": list(
                                    settings.probe_learning_rates
                                ),
                                "probe_warmup_steps": settings.probe_warmup_steps,
                                "evaluation_batch_size": (
                                    settings.evaluation_batch_size
                                ),
                                "confidence_level": settings.confidence_level,
                                "expected_wall_time_seconds": (
                                    settings.expected_wall_time_seconds
                                ),
                                "max_new_tokens": settings.max_new_tokens,
                                "scoring_stride_tokens": settings.scoring_stride_tokens,
                                "deterministic_algorithms": (
                                    settings.deterministic_algorithms
                                ),
                                "deterministic_warn_only": (
                                    settings.deterministic_warn_only
                                ),
                            },
                        }
        except Exception as error:
            result = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": manifest_sha256,
                "correlation_id": correlation_id,
                "config_fingerprint": fingerprint,
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "reference_only": candidate.reference_only,
                "partial_measurements": partial,
                "failure": {
                    "phase": phase,
                    "type": type(error).__name__,
                    "message": str(error),
                    "out_of_memory": _is_out_of_memory(error),
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
    memory = sampler.record()
    if result["status"] == "completed":
        cast(dict[str, object], result["cost"])["memory"] = memory
    else:
        result["memory"] = memory
    if equivalence_gate_owner_override is not None:
        result["equivalence_gate_owner_override"] = dict(
            equivalence_gate_owner_override
        )
    if snapshot_directory is not None:
        snapshot_directory.cleanup()
    model = None
    cleanup_device(device)
    return result


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a JSON record so interruption cannot leave partial evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"record must contain a JSON object: {path}")
    return cast(dict[str, object], value)


def evaluate_all_candidates(
    settings: BaseModelEvaluationSettings,
    output_dir: Path,
    *,
    resume: bool,
    loader: CandidateLoader = load_candidate,
    manifest_sha256: str = "test-manifest",
    correlation_id: str = "test-correlation-id",
) -> list[dict[str, object]]:
    """Evaluate candidates independently and persist each result before continuing."""
    records: list[dict[str, object]] = []
    for candidate in settings.candidates:
        record_path = _candidate_record_path(output_dir, candidate.candidate_id)
        fingerprint = _config_fingerprint(candidate, settings.measurement, manifest_sha256)
        if resume and record_path.is_file():
            existing = _read_json(record_path)
            if (
                existing.get("status")
                in {"completed", "hard_constraint_rejected", "reference_only"}
                and existing.get("schema_version") == EVALUATION_SCHEMA_VERSION
                and existing.get("manifest_sha256") == manifest_sha256
                and existing.get("config_fingerprint") == fingerprint
            ):
                records.append(existing)
                continue
        record = evaluate_candidate(
            candidate,
            settings.measurement,
            loader=loader,
            decision_settings=settings.decision,
            manifest_sha256=manifest_sha256,
            correlation_id=correlation_id,
            equivalence_gate_owner_override=(
                settings.equivalence_gate_owner_override
            ),
            progress_path=record_path,
        )
        write_json(record_path, record)
        records.append(record)
    return records


def _nested(record: Mapping[str, object], *keys: str) -> object:
    current: object = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = cast(dict[str, object], current)[key]
    return current


def _metric(record: Mapping[str, object], *keys: str) -> float:
    value = _nested(record, *keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {'.'.join(keys)} is not numeric")
    return float(value)


def _optional_nested(record: Mapping[str, object], *keys: str) -> object | None:
    try:
        return _nested(record, *keys)
    except KeyError:
        return None


def _optional_metric(record: Mapping[str, object], *keys: str) -> float | None:
    try:
        return _metric(record, *keys)
    except (KeyError, TypeError):
        return None


def _hard_constraint_reasons(
    record: Mapping[str, object], settings: DecisionSettings
) -> list[str]:
    if record.get("status") == "hard_constraint_rejected":
        embedded_reasons = record.get("hard_constraint_reasons")
        if isinstance(embedded_reasons, list) and all(
            isinstance(item, str) for item in embedded_reasons
        ):
            return cast(list[str], embedded_reasons)
        return ["malformed hard-constraint rejection record"]
    if record.get("status") != "completed":
        failure = record.get("failure")
        return [f"measurement failed: {failure}"]
    reasons: list[str] = []
    try:
        license_name = _nested(record, "identity", "license")
        allowed = {_canonical_license(item) for item in settings.permissive_licenses}
        if not isinstance(license_name, str) or _canonical_license(license_name) not in allowed:
            reasons.append(f"licence {license_name!r} is not in the permissive allowlist")
        if _nested(record, "identity", "trust_remote_code") is not False:
            reasons.append("loading requires remote code")
        maximum_peak = settings.target_memory_bytes * (
            1.0 - settings.minimum_memory_headroom_fraction
        )
        if _metric(record, "cost", "memory", "hard_constraint_peak_bytes") > maximum_peak:
            reasons.append(
                "peak memory does not leave the configured minimum target-machine headroom"
            )
        tokenizer = _nested(record, "tokenizer_fitness")
        if not isinstance(tokenizer, dict) or not _tokenizer_contract_satisfied(tokenizer):
            reasons.append(
                "tokenizer failed NFC-normalized or exact task-stream round-trip"
            )
        if (
            _nested(record, "tokenizer_fitness", "eos_usable_as_answer_terminator")
            is not True
        ):
            reasons.append("tokenizer/model EOS is not usable as an answer terminator")
        if (
            _nested(record, "structural_fitness", "all_blocks_expose_feed_forward")
            is not True
        ):
            reasons.append(
                "not every transformer block exposes an attachable feed-forward sublayer"
            )
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"malformed completed record: {error}")
    return reasons


def _family_clustered_interval(
    record: Mapping[str, object], metric: str, confidence_level: float
) -> dict[str, float | int | str]:
    return student_t_interval(
        list(_family_metric_values(record, metric).values()), confidence_level
    )


def _record_confidence_level(record: Mapping[str, object]) -> float:
    value = _nested(
        record,
        "task_stream_readiness",
        "mean_learned",
        "wilson_interval",
        "confidence_level",
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("record confidence level must be numeric")
    confidence_level = float(value)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("record confidence level must be between zero and one")
    return confidence_level


def _learnability_tied(
    candidate: Mapping[str, object],
    leader: Mapping[str, object],
    settings: DecisionSettings,
    *,
    family_metric: str,
) -> bool:
    candidate_confidence = _record_confidence_level(candidate)
    leader_confidence = _record_confidence_level(leader)
    if candidate_confidence != leader_confidence:
        raise ValueError("candidate confidence levels differ")
    comparison = paired_family_comparison(
        candidate,
        leader,
        family_metric,
        confidence_level=candidate_confidence,
    )
    return _comparison_is_tied(comparison, settings)


def _comparison_is_tied(
    comparison: Mapping[str, object], settings: DecisionSettings
) -> bool:
    interval = cast(
        dict[str, float | int | str], comparison["mean_difference_interval"]
    )
    difference = abs(cast(float, interval["mean"]))
    if difference <= settings.learnability_tie_margin:
        return True
    if not settings.clustered_confidence_interval_intersects_tie_region:
        return False
    lower = cast(float, interval["lower"])
    upper = cast(float, interval["upper"])
    return lower <= settings.learnability_tie_margin and upper >= (
        -settings.learnability_tie_margin
    )


def _legacy_pooled_interval_bounds(
    record: Mapping[str, object], metric_name: str
) -> tuple[float, float]:
    interval = _nested(record, "task_stream_readiness", *metric_name.split("."))
    if not isinstance(interval, dict):
        raise TypeError(f"{metric_name} must be an interval mapping")
    lower = interval.get("lower")
    upper = interval.get("upper")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (lower, upper)
    ):
        raise TypeError(f"{metric_name} bounds must be numeric")
    return float(cast(float, lower)), float(cast(float, upper))


def _legacy_pooled_tied(
    candidate: Mapping[str, object],
    leader: Mapping[str, object],
    settings: DecisionSettings,
    *,
    point_metric: str,
    interval_metric: str,
) -> bool:
    difference = abs(
        _metric(candidate, "task_stream_readiness", point_metric)
        - _metric(leader, "task_stream_readiness", point_metric)
    )
    if difference <= settings.learnability_tie_margin:
        return True
    if not settings.clustered_confidence_interval_intersects_tie_region:
        return False
    candidate_lower, candidate_upper = _legacy_pooled_interval_bounds(
        candidate, interval_metric
    )
    leader_lower, leader_upper = _legacy_pooled_interval_bounds(
        leader, interval_metric
    )
    return max(candidate_lower, leader_lower) <= min(candidate_upper, leader_upper)


def _ordered_with_legacy_pooled_uncertainty(
    records: Sequence[Mapping[str, object]], settings: DecisionSettings
) -> list[Mapping[str, object]]:
    remaining = list(records)
    ranked: list[Mapping[str, object]] = []
    while remaining:
        learned_leader = max(
            remaining,
            key=lambda item: _metric(
                item, "task_stream_readiness", "mean_learned_accuracy"
            ),
        )
        contenders = [
            item
            for item in remaining
            if _legacy_pooled_tied(
                item,
                learned_leader,
                settings,
                point_metric="mean_learned_accuracy",
                interval_metric="mean_learned.wilson_interval",
            )
        ]
        gain_leader = max(
            contenders,
            key=lambda item: _metric(
                item, "task_stream_readiness", "mean_accuracy_gain"
            ),
        )
        finalists = [
            item
            for item in contenders
            if _legacy_pooled_tied(
                item,
                gain_leader,
                settings,
                point_metric="mean_accuracy_gain",
                interval_metric="mean_accuracy_gain_interval",
            )
        ]
        winner = min(
            finalists,
            key=lambda item: (
                _metric(item, "language_modeling", "combined", "bits_per_byte"),
                _throughput_tie_break_value(item),
                _metric(item, "identity", "unique_parameters"),
                str(item.get("candidate_id")),
            ),
        )
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


def _ordered_with_uncertainty(
    records: Sequence[Mapping[str, object]], settings: DecisionSettings
) -> list[Mapping[str, object]]:
    """Rank by narrowing uncertainty-tied contender sets at each criterion."""
    remaining = list(records)
    ranked: list[Mapping[str, object]] = []
    while remaining:
        learned_leader = max(
            remaining,
            key=lambda item: (
                _metric(item, "task_stream_readiness", "mean_learned_accuracy"),
                str(item.get("candidate_id")),
            ),
        )
        contenders = [
            item
            for item in remaining
            if _learnability_tied(
                item,
                learned_leader,
                settings,
                family_metric="learned_accuracy",
            )
        ]
        gain_leader = max(
            contenders,
            key=lambda item: (
                _metric(item, "task_stream_readiness", "mean_accuracy_gain"),
                str(item.get("candidate_id")),
            ),
        )
        finalists = [
            item
            for item in contenders
            if _learnability_tied(
                item,
                gain_leader,
                settings,
                family_metric="accuracy_gain",
            )
        ]
        winner = min(
            finalists,
            key=lambda item: (
                _metric(item, "language_modeling", "combined", "bits_per_byte"),
                _throughput_tie_break_value(item),
                _metric(item, "identity", "unique_parameters"),
                str(item.get("candidate_id")),
            ),
        )
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


def _throughput_tie_break_value(record: Mapping[str, object]) -> float:
    """Sort valid throughput first; swapped runs have no throughput measurement."""
    if record.get("throughput_compromised") is True:
        return math.inf
    return -_metric(record, "cost", "median_probe_training_tokens_per_second")


def _saturated_family_ids(
    records: Sequence[Mapping[str, object]],
) -> list[str]:
    if not records:
        return []
    learned_by_record: list[dict[str, tuple[int, int]]] = []
    for record in records:
        families = _nested(record, "task_stream_readiness", "families")
        if not isinstance(families, list):
            raise TypeError("task_stream_readiness.families must be a list")
        learned: dict[str, tuple[int, int]] = {}
        for row in families:
            if not isinstance(row, dict) or not isinstance(row.get("family_id"), str):
                raise TypeError("family row is malformed")
            family_id = cast(str, row["family_id"])
            if family_id in learned:
                raise ValueError(f"duplicate family row: {family_id}")
            result = row.get("learned")
            if not isinstance(result, dict):
                raise TypeError("family learned result is malformed")
            successes = result.get("successes")
            total = result.get("total")
            if (
                isinstance(successes, bool)
                or not isinstance(successes, int)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total <= 0
                or not 0 <= successes <= total
            ):
                raise ValueError("family learned counts are invalid")
            learned[family_id] = (successes, total)
        learned_by_record.append(learned)
    family_ids = set(learned_by_record[0])
    if any(set(values) != family_ids for values in learned_by_record[1:]):
        raise ValueError("candidate family IDs differ")
    return sorted(
        family_id
        for family_id in family_ids
        if all(values[family_id][0] == values[family_id][1] for values in learned_by_record)
    )


def _candidate_aggregate_analysis(
    record: Mapping[str, object],
    saturated_family_ids: Sequence[str],
    confidence_level: float,
) -> dict[str, object]:
    saturated = set(saturated_family_ids)
    families = _nested(record, "task_stream_readiness", "families")
    if not isinstance(families, list):
        raise TypeError("task_stream_readiness.families must be a list")

    def pooled(rows: Sequence[dict[str, object]]) -> dict[str, object]:
        successes = 0
        total = 0
        for row in rows:
            learned = row.get("learned")
            if not isinstance(learned, dict):
                raise TypeError("family learned result is malformed")
            successes += cast(int, learned["successes"])
            total += cast(int, learned["total"])
        result = _accuracy_record(successes, total, confidence_level)
        result["precision_note"] = (
            "optimistic: Wilson interval treats items clustered within task families "
            "as independent Bernoulli trials"
        )
        return result

    typed_families = [cast(dict[str, object], row) for row in families]
    retained = [
        row for row in typed_families if row.get("family_id") not in saturated
    ]
    clustered = _family_clustered_interval(
        record, "learned_accuracy", confidence_level
    )
    retained_values = [cast(float, row["learned_accuracy"]) for row in retained]
    return {
        "candidate_id": str(record.get("candidate_id")),
        "all_families": {
            "pooled_item_analysis": pooled(typed_families),
            "family_clustered_analysis": clustered,
        },
        "excluding_saturated_families": {
            "excluded_family_ids": list(saturated_family_ids),
            "pooled_item_analysis": pooled(retained) if retained else None,
            "family_clustered_analysis": (
                student_t_interval(retained_values, confidence_level)
                if len(retained_values) >= 2
                else None
            ),
            "no_signal_bearing_families_remain": not retained,
        },
    }


def apply_decision_rule(
    records: Sequence[Mapping[str, object]], settings: DecisionSettings
) -> dict[str, object]:
    """Reject hard/threshold failures, then apply the precommitted tie-break order."""
    rejected: list[dict[str, object]] = []
    eligible: list[Mapping[str, object]] = []
    references: list[dict[str, str]] = []
    incomplete: list[dict[str, str]] = []
    for record in records:
        candidate_id = str(record.get("candidate_id"))
        if record.get("reference_only") is True:
            references.append(
                {"candidate_id": candidate_id, "reason": "reference-only; excluded from selection"}
            )
            continue
        status = str(record.get("status"))
        footprint_rejection = (
            status == "hard_constraint_rejected"
            and record.get("rejection_category") == "candidate_footprint"
        )
        if (
            settings.require_complete_candidate_measurements
            and status != "completed"
            and not footprint_rejection
        ):
            incomplete.append({"candidate_id": candidate_id, "status": status})
        reasons = _hard_constraint_reasons(record, settings)
        if record.get("status") == "completed":
            try:
                confidence_level = _record_confidence_level(record)
                learned_clustered = _family_clustered_interval(
                    record, "learned_accuracy", confidence_level
                )
                gain_clustered = _family_clustered_interval(
                    record, "accuracy_gain", confidence_level
                )
                if cast(float, learned_clustered["upper"]) < (
                    settings.minimum_mean_learned_accuracy
                ):
                    reasons.append(
                        "family-clustered mean learned accuracy upper confidence bound "
                        "is below the "
                        "configured threshold"
                    )
                if cast(float, gain_clustered["upper"]) < (
                    settings.minimum_mean_accuracy_gain
                ):
                    reasons.append(
                        "family-clustered mean learnability gain upper confidence bound "
                        "is below the "
                        "configured threshold"
                    )
            except (KeyError, TypeError, ValueError) as error:
                malformed = f"malformed ranking metrics: {error}"
                if malformed not in reasons:
                    reasons.append(malformed)
        if reasons:
            rejected.append({"candidate_id": candidate_id, "reasons": reasons})
        else:
            eligible.append(record)
    ranked = _ordered_with_uncertainty(eligible, settings)
    ranking = [
        {
            "rank": index + 1,
            "candidate_id": str(record.get("candidate_id")),
            "mean_learned_accuracy": _metric(
                record, "task_stream_readiness", "mean_learned_accuracy"
            ),
            "mean_learned_accuracy_wilson_interval": _nested(
                record,
                "task_stream_readiness",
                "mean_learned",
                "wilson_interval",
            ),
            "pooled_interval_precision_note": (
                "optimistic: treats items clustered within task families as "
                "independent Bernoulli trials"
            ),
            "family_clustered_mean_learned_accuracy_interval": (
                _family_clustered_interval(
                    record, "learned_accuracy", _record_confidence_level(record)
                )
            ),
            "mean_accuracy_gain": _metric(
                record, "task_stream_readiness", "mean_accuracy_gain"
            ),
            "mean_accuracy_gain_interval": _nested(
                record, "task_stream_readiness", "mean_accuracy_gain_interval"
            ),
            "family_clustered_mean_accuracy_gain_interval": (
                _family_clustered_interval(
                    record, "accuracy_gain", _record_confidence_level(record)
                )
            ),
            "combined_heldout_bits_per_byte": _metric(
                record, "language_modeling", "combined", "bits_per_byte"
            ),
            "training_tokens_per_second": (
                None
                if record.get("throughput_compromised") is True
                else _metric(record, "cost", "median_probe_training_tokens_per_second")
            ),
            "throughput_compromised": record.get("throughput_compromised") is True,
            "throughput_tie_breaker_status": (
                "excluded_swap_compromised"
                if record.get("throughput_compromised") is True
                else "measured"
            ),
            "unique_parameters": int(
                _metric(record, "identity", "unique_parameters")
            ),
            "micro_batch_size": _optional_metric(
                record, "measurement_protocol", "probe_micro_batch_size"
            ),
            "gradient_checkpointing": _optional_nested(
                record, "measurement_protocol", "gradient_checkpointing"
            ),
            "peak_memory_bytes": _optional_metric(
                record, "cost", "memory", "hard_constraint_peak_bytes"
            ),
            "system_swap_before_bytes": _optional_metric(
                record, "cost", "memory", "system_swap_before_bytes"
            ),
            "system_swap_after_bytes": _optional_metric(
                record, "cost", "memory", "system_swap_after_bytes"
            ),
            "system_swap_peak_growth_bytes": _optional_metric(
                record, "cost", "memory", "system_swap_peak_growth_bytes"
            ),
        }
        for index, record in enumerate(ranked)
    ]
    pairwise_comparisons = [
        paired_family_comparison(
            first,
            second,
            metric,
            _record_confidence_level(first),
        )
        for first_index, first in enumerate(ranked)
        for second in ranked[first_index + 1 :]
        for metric in ("learned_accuracy", "accuracy_gain")
    ]
    saturated_family_ids = _saturated_family_ids(eligible)
    aggregate_analysis = [
        _candidate_aggregate_analysis(
            record,
            saturated_family_ids,
            _record_confidence_level(record),
        )
        for record in eligible
    ]
    return {
        "schema_version": RANKING_SCHEMA_VERSION,
        "decision_status": (
            "measurement_pending"
            if not records
            else "measurement_incomplete"
            if incomplete
            else "ranked_no_adr_decision"
            if ranking
            else "no_eligible_candidate"
        ),
        "selected_candidate": (
            ranking[0]["candidate_id"] if ranking and not incomplete else None
        ),
        "missing_candidates": incomplete,
        "ranking_order": [
            (
                "mean learned task accuracy (descending; ties when absolute difference "
                f"<= {settings.learnability_tie_margin} or the paired family-clustered "
                "difference interval intersects the configured practical tie region)"
            ),
            (
                "mean task accuracy gain (descending; same numeric/interval tie rule)"
            ),
            "combined held-out bits per byte (ascending)",
            "median family probe training tokens per second (descending)",
            "unique parameters (ascending)",
        ],
        "ranked_candidates": ranking,
        "pairwise_family_clustered_comparisons": pairwise_comparisons,
        "saturated_family_analysis": {
            "criterion": (
                "learned successes equal total for every eligible probed candidate"
            ),
            "family_ids": saturated_family_ids,
            "candidate_aggregates": aggregate_analysis,
        },
        "rejected_candidates": rejected,
        "reference_records": references,
    }


def run_evaluation(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    resume: bool = True,
    loader: CandidateLoader = load_candidate,
    candidate_ids: set[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run resumable evaluation and write the aggregate ranking record."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(
        config_path, destination, settings
    )
    manifest_rows = {
        str(row.get("candidate_id")): row
        for row in cast(list[dict[str, object]], manifest["candidates"])
    }
    resolved_candidates: list[CandidateSpec] = []
    preflight_failures: list[dict[str, object]] = []
    correlation_id = str(manifest["correlation_id"])
    for candidate in settings.candidates:
        if candidate_ids is not None and candidate.candidate_id not in candidate_ids:
            continue
        row = manifest_rows[candidate.candidate_id]
        failure = row.get("resolution_failure")
        if failure is not None:
            record = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": manifest_sha256,
                "correlation_id": correlation_id,
                "config_fingerprint": _config_fingerprint(
                    candidate, settings.measurement, manifest_sha256
                ),
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "reference_only": candidate.reference_only,
                "equivalence_gate_owner_override": (
                    settings.equivalence_gate_owner_override
                ),
                "failure": {
                    "phase": "preflight_resolution",
                    **cast(dict[str, object], failure),
                    "out_of_memory": False,
                },
            }
            write_json(_candidate_record_path(destination, candidate.candidate_id), record)
            preflight_failures.append(record)
            continue
        resolved_candidates.append(
            CandidateSpec(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                revision=cast(str, row["resolved_revision"]),
                license=cast(str, row["license"]),
                expected_architecture=candidate.expected_architecture,
                reference_only=candidate.reference_only,
                loader=candidate.loader,
            )
        )
    resolved_settings = BaseModelEvaluationSettings(
        candidates=tuple(resolved_candidates),
        measurement=settings.measurement,
        decision=settings.decision,
        execution=settings.execution,
        equivalence_gate_owner_override=settings.equivalence_gate_owner_override,
    )
    records = evaluate_all_candidates(
        resolved_settings,
        destination,
        resume=resume,
        loader=loader,
        manifest_sha256=manifest_sha256,
        correlation_id=correlation_id,
    )
    records.extend(preflight_failures)
    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = manifest_sha256
    ranking["correlation_id"] = correlation_id
    write_json(destination / "ranking.json", ranking)
    return records, ranking


def run_seed_variance_evaluation(
    config_path: Path | str,
    output_dir: Path | str,
    candidate_id: str,
    seed: int,
    *,
    resume: bool = True,
    loader: CandidateLoader = load_candidate,
    primary_output_dir: Path | str | None = None,
) -> dict[str, object]:
    """Rerun a first-pass leader at its validation-selected LR for another seed."""
    settings = load_base_model_evaluation_settings(config_path)
    if seed == settings.measurement.seed or seed not in settings.measurement.variance_seeds:
        raise ValueError("variance seed must be a configured non-primary seed")
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    primary_source = (
        destination if primary_output_dir is None else Path(primary_output_dir)
    )
    primary_path = _candidate_record_path(primary_source, candidate_id)
    if not primary_path.is_file():
        raise ValueError(f"primary candidate record is missing for {candidate_id}")
    primary = _read_json(primary_path)
    if primary.get("status") != "completed":
        raise ValueError(f"primary candidate record is not completed for {candidate_id}")
    learning_rate = _metric(
        primary,
        "task_stream_readiness",
        "learning_rate_selection",
        "selected_learning_rate",
    )
    candidate = next(
        (item for item in settings.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"unknown candidate id: {candidate_id}")
    manifest_rows = {
        str(row.get("candidate_id")): row
        for row in cast(list[dict[str, object]], manifest["candidates"])
    }
    row = manifest_rows[candidate_id]
    failure = row.get("resolution_failure")
    if failure is not None:
        raise ValueError(f"candidate identity resolution failed: {failure}")
    resolved = replace(
        candidate,
        revision=cast(str, row["resolved_revision"]),
        license=cast(str, row["license"]),
    )
    seed_settings = replace(
        settings.measurement,
        seed=seed,
        probe_learning_rates=(learning_rate,),
    )
    record_path = _seed_record_path(destination, candidate_id, seed)
    fingerprint = _config_fingerprint(resolved, seed_settings, manifest_sha256)
    if resume and record_path.is_file():
        existing = _read_json(record_path)
        if (
            existing.get("status") == "completed"
            and existing.get("config_fingerprint") == fingerprint
            and existing.get("manifest_sha256") == manifest_sha256
        ):
            return existing
    record = evaluate_candidate(
        resolved,
        seed_settings,
        loader=loader,
        decision_settings=settings.decision,
        manifest_sha256=manifest_sha256,
        correlation_id=str(manifest["correlation_id"]),
        equivalence_gate_owner_override=settings.equivalence_gate_owner_override,
        progress_path=record_path,
    )
    record["variance_role"] = "top-candidate seed-to-seed estimate"
    record["primary_selected_learning_rate"] = learning_rate
    record["primary_record_source"] = str(primary_path)
    write_json(record_path, record)
    return record


def _seed_variance_summary(
    candidate_id: str,
    primary: Mapping[str, object],
    destination: Path,
    settings: MeasurementSettings,
    manifest_sha256: str,
    *,
    seed_destination: Path | None = None,
    seed_manifest_sha256: str | None = None,
) -> dict[str, object]:
    repeat_destination = seed_destination or destination
    repeat_manifest_sha256 = seed_manifest_sha256 or manifest_sha256
    rows: list[dict[str, object]] = [
        {
            "seed": settings.seed,
            "mean_learned_accuracy": _metric(
                primary, "task_stream_readiness", "mean_learned_accuracy"
            ),
            "mean_accuracy_gain": _metric(
                primary, "task_stream_readiness", "mean_accuracy_gain"
            ),
            "record": str(
                _candidate_record_path(destination, candidate_id).relative_to(destination)
            ),
        }
    ]
    missing: list[int] = []
    for seed in settings.variance_seeds[1:]:
        path = _seed_record_path(repeat_destination, candidate_id, seed)
        if not path.is_file():
            missing.append(seed)
            continue
        record = _read_json(path)
        if (
            record.get("status") != "completed"
            or record.get("manifest_sha256") != repeat_manifest_sha256
        ):
            missing.append(seed)
            continue
        rows.append(
            {
                "seed": seed,
                "mean_learned_accuracy": _metric(
                    record, "task_stream_readiness", "mean_learned_accuracy"
                ),
                "mean_accuracy_gain": _metric(
                    record, "task_stream_readiness", "mean_accuracy_gain"
                ),
                "record": str(path.relative_to(repeat_destination)),
            }
        )
    learned = [cast(float, row["mean_learned_accuracy"]) for row in rows]
    gains = [cast(float, row["mean_accuracy_gain"]) for row in rows]
    return {
        "candidate_id": candidate_id,
        "configured_seeds": list(settings.variance_seeds),
        "completed_seed_count": len(rows),
        "missing_seeds": missing,
        "runs": rows,
        "mean_learned_accuracy_across_seeds": statistics.mean(learned),
        "sample_standard_deviation_learned_accuracy": (
            statistics.stdev(learned) if len(learned) >= 2 else None
        ),
        "mean_accuracy_gain_across_seeds": statistics.mean(gains),
        "sample_standard_deviation_accuracy_gain": (
            statistics.stdev(gains) if len(gains) >= 2 else None
        ),
    }


def _top_two_seed_repeat_agreement(
    summaries: Sequence[Mapping[str, object]], seeds: Sequence[int]
) -> dict[str, object]:
    """Report whether every seed preserves the primary top-two order."""
    candidate_ids = [str(summary["candidate_id"]) for summary in summaries]
    if len(candidate_ids) != 2:
        return {
            "candidate_ids": candidate_ids,
            "configured_seeds": list(seeds),
            "complete": False,
            "agrees": None,
            "reason": "ranking did not produce exactly two repeat-seed leaders",
            "runs": [],
        }
    metrics_by_seed: dict[int, dict[str, tuple[float, float]]] = {}
    missing_by_candidate: dict[str, list[int]] = {}
    for summary in summaries:
        candidate_id = str(summary["candidate_id"])
        missing_by_candidate[candidate_id] = [
            int(seed) for seed in cast(Sequence[int], summary["missing_seeds"])
        ]
        for row in cast(Sequence[Mapping[str, object]], summary["runs"]):
            seed = int(cast(int, row["seed"]))
            metrics_by_seed.setdefault(seed, {})[candidate_id] = (
                float(cast(float, row["mean_learned_accuracy"])),
                float(cast(float, row["mean_accuracy_gain"])),
            )
    complete = not any(missing_by_candidate.values()) and all(
        len(metrics_by_seed.get(seed, {})) == 2 for seed in seeds
    )
    runs: list[dict[str, object]] = []
    for seed in seeds:
        metrics = metrics_by_seed.get(seed, {})
        order = sorted(
            metrics,
            key=lambda candidate_id: (
                -metrics[candidate_id][0],
                -metrics[candidate_id][1],
                candidate_id,
            ),
        )
        runs.append(
            {
                "seed": seed,
                "candidate_order": order,
                "metrics": {
                    candidate_id: {
                        "mean_learned_accuracy": values[0],
                        "mean_accuracy_gain": values[1],
                    }
                    for candidate_id, values in metrics.items()
                },
            }
        )
    primary_order = candidate_ids
    agrees = (
        all(row["candidate_order"] == primary_order for row in runs)
        if complete
        else None
    )
    return {
        "candidate_ids": candidate_ids,
        "configured_seeds": list(seeds),
        "comparison": (
            "descending mean learned accuracy, then descending mean accuracy gain, "
            "then candidate ID"
        ),
        "complete": complete,
        "agrees": agrees,
        "missing_seeds_by_candidate": missing_by_candidate,
        "runs": runs,
    }


def _full_rule_seed_repeat_agreement(
    records_by_seed: Mapping[int, Sequence[Mapping[str, object]] | None],
    seeds: Sequence[int],
    settings: DecisionSettings,
) -> dict[str, object]:
    """Apply the complete decision rule per seed and fail closed on disagreement."""
    runs: list[dict[str, object]] = []
    missing_seeds: list[int] = []
    winners: list[str] = []
    candidate_ids: list[str] = []
    for seed in seeds:
        records = records_by_seed.get(seed)
        if records is None or len(records) != 2:
            missing_seeds.append(seed)
            runs.append(
                {
                    "seed": seed,
                    "decision_status": "measurement_incomplete",
                    "selected_candidate": None,
                    "reason": "both top-two records are required for this seed",
                }
            )
            continue
        ids = [str(record.get("candidate_id")) for record in records]
        if len(set(ids)) != 2:
            raise ValueError("seed repeat records must contain two distinct candidates")
        if not candidate_ids:
            candidate_ids = ids
        elif set(ids) != set(candidate_ids):
            raise ValueError("seed repeat candidate set differs across seeds")
        decision = apply_decision_rule(records, settings)
        winner = decision.get("selected_candidate")
        if not isinstance(winner, str):
            missing_seeds.append(seed)
        else:
            winners.append(winner)
        runs.append(
            {
                "seed": seed,
                "decision_status": decision["decision_status"],
                "selected_candidate": winner,
                "missing_candidates": decision["missing_candidates"],
                "ranked_candidates": decision["ranked_candidates"],
                "rejected_candidates": decision["rejected_candidates"],
                "pairwise_family_clustered_comparisons": decision[
                    "pairwise_family_clustered_comparisons"
                ],
            }
        )
    complete = not missing_seeds and len(winners) == len(seeds)
    agrees = len(set(winners)) == 1 if complete else None
    return {
        "candidate_ids": candidate_ids,
        "configured_seeds": list(seeds),
        "comparison": "complete pre-registered decision rule applied independently per seed",
        "complete": complete,
        "agrees": agrees,
        "selection_stable_across_seeds": agrees,
        "missing_seeds": missing_seeds,
        "seed_winners": winners,
        "runs": runs,
    }


def _finalize_amendment_4g_selection(
    ranking: dict[str, object], agreement: Mapping[str, object]
) -> None:
    """Set the 4g verdict, selecting only a complete unanimous seed winner."""
    ranking["selected_candidate"] = None
    if agreement.get("complete") is not True:
        ranking["decision_status"] = "measurement_incomplete"
        return
    if agreement.get("agrees") is not True:
        ranking["decision_status"] = "selection_unstable"
        return
    seed_winners = agreement.get("seed_winners")
    if (
        not isinstance(seed_winners, list)
        or not seed_winners
        or not all(isinstance(item, str) for item in seed_winners)
    ):
        raise ValueError("complete seed agreement has no valid winners")
    ranking["decision_status"] = "selected"
    ranking["selected_candidate"] = seed_winners[0]


def derive_existing_records_analysis(
    config_path: Path | str, output_dir: Path | str
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive Amendment 2 evidence without mutating or rerunning measurements."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest_path = destination / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("frozen powered run manifest is missing")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("frozen powered run manifest has an unsupported schema")
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    correlation_id = manifest.get("correlation_id")
    if not isinstance(correlation_id, str):
        raise TypeError("frozen powered run manifest has no correlation_id")
    powered_config_path = destination / "powered-sweep-config.yaml"
    if not powered_config_path.is_file():
        raise ValueError("frozen powered sweep config is missing")
    powered_config_sha256 = sha256_file(powered_config_path)
    if powered_config_sha256 != manifest.get("config_sha256"):
        raise ValueError("frozen powered sweep config does not match the manifest")

    records: list[dict[str, object]] = []
    source_records: list[dict[str, str]] = []
    manifest_candidate_ids = {
        str(row.get("candidate_id"))
        for row in cast(list[dict[str, object]], manifest["candidates"])
    }
    for candidate in settings.candidates:
        if candidate.candidate_id not in manifest_candidate_ids:
            continue
        path = _candidate_record_path(destination, candidate.candidate_id)
        if not path.is_file():
            raise ValueError(f"candidate record is missing: {candidate.candidate_id}")
        record = _read_json(path)
        if record.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"candidate record manifest mismatch: {candidate.candidate_id}"
            )
        records.append(record)
        source_records.append(
            {
                "candidate_id": candidate.candidate_id,
                "path": str(path.relative_to(destination)),
                "sha256": sha256_file(path),
                "schema_version": str(record.get("schema_version")),
            }
        )
    if {str(record.get("candidate_id")) for record in records} != manifest_candidate_ids:
        raise ValueError("configured candidates do not cover the frozen manifest")

    old_ranking_path = destination / "ranking.json"
    old_ranking = _read_json(old_ranking_path) if old_ranking_path.is_file() else {}
    analysis_path = destination / "amendment-2-analysis.json"
    existing_analysis = _read_json(analysis_path) if analysis_path.is_file() else {}
    existing_source_ranking = existing_analysis.get("source_ranking")
    if isinstance(existing_source_ranking, dict):
        source_ranking = existing_source_ranking
    else:
        source_ranking = {
            "path": "ranking.json",
            "sha256_before_amendment": (
                sha256_file(old_ranking_path) if old_ranking_path.is_file() else None
            ),
            "schema_version_before_amendment": old_ranking.get("schema_version"),
        }
    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = manifest_sha256
    ranking["correlation_id"] = correlation_id
    record_by_id = {str(record.get("candidate_id")): record for record in records}
    ranked_rows = cast(list[dict[str, object]], ranking["ranked_candidates"])
    ranking["seed_variance"] = [
        _seed_variance_summary(
            cast(str, row["candidate_id"]),
            record_by_id[cast(str, row["candidate_id"])],
            destination,
            settings.measurement,
            manifest_sha256,
        )
        for row in ranked_rows[: settings.measurement.variance_top_candidates]
    ]
    if "orchestration" in old_ranking:
        ranking["orchestration"] = old_ranking["orchestration"]

    stream = ContinualTaskStream(
        num_train_per_task=settings.measurement.train_items_per_family,
        num_val_per_task=settings.measurement.validation_items_per_family,
        num_test_per_task=settings.measurement.test_items_per_family,
        seed=cast(int, manifest["seed"]),
        max_generation_attempts=settings.measurement.max_generation_attempts,
    )
    seed_source_records = [
        {
            "path": str(path.relative_to(destination)),
            "sha256": sha256_file(path),
            "schema_version": str(_read_json(path).get("schema_version")),
        }
        for path in sorted((destination / "seed_runs").glob("*.json"))
    ]
    archive_reference_path = _candidate_record_path(
        destination / "archive" / "first_underpowered_sweep",
        "aptivra-byte-reference",
    )
    archive_reference: dict[str, object] | None = None
    if archive_reference_path.is_file():
        archived = _read_json(archive_reference_path)
        archive_reference = {
            "path": str(archive_reference_path.relative_to(destination)),
            "sha256": sha256_file(archive_reference_path),
            "combined_bits_per_byte": _metric(
                archived, "language_modeling", "combined", "bits_per_byte"
            ),
            "provenance": (
                "archived first underpowered sweep; not measured by the powered protocol"
            ),
        }
    eligible_records = [record_by_id[cast(str, row["candidate_id"])] for row in ranked_rows]
    legacy_order = _ordered_with_legacy_pooled_uncertainty(
        eligible_records, settings.decision
    )
    selected = eligible_records[0]
    runner_up = eligible_records[1] if len(eligible_records) > 1 else None
    learned_comparison = (
        paired_family_comparison(
            selected,
            runner_up,
            "learned_accuracy",
            _record_confidence_level(selected),
        )
        if runner_up is not None
        else None
    )
    gain_comparison = (
        paired_family_comparison(
            selected,
            runner_up,
            "accuracy_gain",
            _record_confidence_level(selected),
        )
        if runner_up is not None
        else None
    )
    learned_tied = (
        _comparison_is_tied(learned_comparison, settings.decision)
        if learned_comparison is not None
        else False
    )
    gain_tied = (
        _comparison_is_tied(gain_comparison, settings.decision)
        if gain_comparison is not None
        else False
    )
    if runner_up is None:
        deciding_criterion = "only eligible candidate"
    elif not learned_tied:
        deciding_criterion = "paired family-clustered mean learned accuracy"
    elif not gain_tied:
        deciding_criterion = "paired family-clustered mean accuracy gain"
    else:
        deciding_criterion = "combined held-out bits per byte"
    amendment_effect = {
        "pre_amendment_selected_candidate": (
            str(legacy_order[0].get("candidate_id")) if legacy_order else None
        ),
        "amended_selected_candidate": ranking["selected_candidate"],
        "selection_changed": bool(legacy_order)
        and str(legacy_order[0].get("candidate_id"))
        != ranking["selected_candidate"],
        "selected_vs_runner_up": {
            "runner_up_candidate": (
                str(runner_up.get("candidate_id")) if runner_up is not None else None
            ),
            "learned_accuracy_tied": learned_tied,
            "accuracy_gain_tied": gain_tied,
            "deciding_criterion": deciding_criterion,
        },
        "rule_timing": (
            "Amendment 2 was introduced after independent review of the powered result"
        ),
    }
    ranking["amendment_2_decision_audit"] = amendment_effect
    analysis: dict[str, object] = {
        "schema_version": DERIVED_ANALYSIS_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": correlation_id,
        "derivation": (
            "analysis-only computation from frozen measurement records; no model was "
            "loaded, trained, or evaluated"
        ),
        "source_records": source_records,
        "source_config": {
            "path": str(powered_config_path.relative_to(destination)),
            "sha256": powered_config_sha256,
        },
        "source_seed_records": seed_source_records,
        "source_ranking": source_ranking,
        "task_stream_properties": compute_task_stream_properties(stream),
        "paired_family_clustered_comparisons": ranking[
            "pairwise_family_clustered_comparisons"
        ],
        "saturated_family_analysis": ranking["saturated_family_analysis"],
        "seed_variance": ranking["seed_variance"],
        "amended_ranking": {
            "schema_version": ranking["schema_version"],
            "selected_candidate": ranking["selected_candidate"],
            "ranking_order": ranking["ranking_order"],
            "ranked_candidates": ranking["ranked_candidates"],
        },
        "amendment_2_decision_audit": amendment_effect,
        "archived_byte_reference": archive_reference,
    }
    write_json(analysis_path, analysis)
    write_json(old_ranking_path, ranking)
    return analysis, ranking


def rank_existing_candidate_records(
    config_path: Path | str, output_dir: Path | str
) -> dict[str, object]:
    """Rank already-written isolated candidate records against the frozen manifest."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    records: list[dict[str, object]] = []
    for candidate in settings.candidates:
        path = _candidate_record_path(destination, candidate.candidate_id)
        if path.is_file():
            record = _read_json(path)
            if record.get("manifest_sha256") == manifest_sha256:
                records.append(record)
                continue
        records.append(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": manifest_sha256,
                "correlation_id": str(manifest["correlation_id"]),
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "reference_only": candidate.reference_only,
                "failure": {
                    "phase": "isolated_process",
                    "type": "MissingCandidateRecord",
                    "message": "isolated candidate process did not produce a valid record",
                    "out_of_memory": False,
                },
            }
        )
    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = manifest_sha256
    ranking["correlation_id"] = str(manifest["correlation_id"])
    record_by_id = {str(record.get("candidate_id")): record for record in records}
    ranked_rows = cast(list[dict[str, object]], ranking["ranked_candidates"])
    leaders = ranked_rows[: settings.measurement.variance_top_candidates]
    ranking["seed_variance"] = [
        _seed_variance_summary(
            cast(str, row["candidate_id"]),
            record_by_id[cast(str, row["candidate_id"])],
            destination,
            settings.measurement,
            manifest_sha256,
        )
        for row in leaders
    ]
    write_json(destination / "ranking.json", ranking)
    return ranking


_AMENDMENT_4E_IGNORED_COMPARISON_KEYS = {
    "config_fingerprint",
    "correlation_id",
    "elapsed_seconds",
    "environment_attempts",
    "equivalence_gate_owner_override",
    "full_finetuning_memory_feasibility",
    "generation_benchmark",
    "load_attempts",
    "manifest_sha256",
    "memory",
    "pristine_model_placement",
    "schema_version",
    "seconds",
    "training_bytes_per_second",
    "training_memory_recipe",
    "training_tokens_per_second",
}


def _amendment_4e_comparison_value(
    value: object, *, include_item_outcomes: bool
) -> object:
    """Remove runtime-only evidence while retaining every decision input."""
    if isinstance(value, dict):
        return {
            key: _amendment_4e_comparison_value(
                item, include_item_outcomes=include_item_outcomes
            )
            for key, item in value.items()
            if key not in _AMENDMENT_4E_IGNORED_COMPARISON_KEYS
            and (include_item_outcomes or key != "item_outcomes")
            and key not in {"initial_loss", "final_loss"}
        }
    if isinstance(value, list):
        return [
            _amendment_4e_comparison_value(
                item, include_item_outcomes=include_item_outcomes
            )
            for item in value
        ]
    return value


def _probe_losses(record: Mapping[str, object]) -> list[float]:
    rows = cast(
        list[dict[str, object]],
        _nested(record, "task_stream_readiness", "families"),
    )
    return [
        _metric(row, key)
        for row in rows
        for key in ("initial_loss", "final_loss")
    ]


def compare_amendment_4e_records(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    include_item_outcomes: bool,
    loss_absolute_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare deterministic decision evidence and separately bound MPS losses."""
    statuses = {
        "reference": reference.get("status"),
        "candidate": candidate.get("status"),
    }
    if any(status != "completed" for status in statuses.values()):
        return {
            "passed": False,
            "exact_decision_and_outcome_evidence_match": False,
            "item_outcomes_compared": include_item_outcomes,
            "loss_absolute_tolerance": loss_absolute_tolerance,
            "maximum_probe_loss_absolute_difference": None,
            "runtime_fields_excluded": sorted(_AMENDMENT_4E_IGNORED_COMPARISON_KEYS),
            "failure": "comparison requires two completed candidate records",
            "record_statuses": statuses,
        }
    reference_value = _amendment_4e_comparison_value(
        dict(reference), include_item_outcomes=include_item_outcomes
    )
    candidate_value = _amendment_4e_comparison_value(
        dict(candidate), include_item_outcomes=include_item_outcomes
    )
    exact_evidence_match = reference_value == candidate_value
    reference_losses = _probe_losses(reference)
    candidate_losses = _probe_losses(candidate)
    if len(reference_losses) != len(candidate_losses):
        maximum_loss_difference = math.inf
    else:
        maximum_loss_difference = max(
            (
                abs(left - right)
                for left, right in zip(
                    reference_losses, candidate_losses, strict=True
                )
            ),
            default=0.0,
        )
    return {
        "passed": (
            exact_evidence_match
            and maximum_loss_difference <= loss_absolute_tolerance
        ),
        "exact_decision_and_outcome_evidence_match": exact_evidence_match,
        "item_outcomes_compared": include_item_outcomes,
        "loss_absolute_tolerance": loss_absolute_tolerance,
        "maximum_probe_loss_absolute_difference": maximum_loss_difference,
        "runtime_fields_excluded": sorted(_AMENDMENT_4E_IGNORED_COMPARISON_KEYS),
    }


def run_amendment_4e_candidate_check(
    config_path: Path | str,
    output_dir: Path | str,
    prior_output_dir: Path | str,
    candidate_id: str,
    *,
    placement_equivalence: bool,
    loader: CandidateLoader = load_candidate,
) -> dict[str, object]:
    """Run a foreground 4e placement or 4c-consistency candidate check."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    configured = next(
        (item for item in settings.candidates if item.candidate_id == candidate_id), None
    )
    if configured is None:
        raise ValueError(f"unknown candidate id: {candidate_id}")
    manifest_row = next(
        row
        for row in cast(list[dict[str, object]], manifest["candidates"])
        if row.get("candidate_id") == candidate_id
    )
    if "resolution_failure" in manifest_row:
        raise RuntimeError(f"candidate resolution failed: {manifest_row['resolution_failure']}")
    candidate = replace(
        configured,
        revision=cast(str, manifest_row["resolved_revision"]),
        license=cast(str, manifest_row["license"]),
    )
    prior_path = _candidate_record_path(Path(prior_output_dir), candidate_id)
    if not prior_path.is_file():
        raise FileNotFoundError(f"Amendment 4c record is missing: {prior_path}")
    prior = _read_json(prior_path)
    check_dir = destination / "checks"
    released_path = check_dir / f"{candidate_id}-released.json"
    released_settings = replace(
        settings.measurement, release_pristine_before_training=True
    )
    released = evaluate_candidate(
        candidate,
        released_settings,
        loader=loader,
        decision_settings=settings.decision,
        manifest_sha256=manifest_sha256,
        correlation_id=str(manifest["correlation_id"]),
        equivalence_gate_owner_override=settings.equivalence_gate_owner_override,
        progress_path=released_path,
    )
    write_json(released_path, released)
    comparisons: dict[str, object] = {
        "released_vs_amendment_4c_aggregates": compare_amendment_4e_records(
            prior, released, include_item_outcomes=False
        )
    }
    if placement_equivalence:
        legacy_path = check_dir / f"{candidate_id}-legacy-live-pristine.json"
        legacy_settings = replace(
            settings.measurement, release_pristine_before_training=False
        )
        legacy = evaluate_candidate(
            candidate,
            legacy_settings,
            loader=loader,
            decision_settings=settings.decision,
            manifest_sha256=manifest_sha256,
            correlation_id=str(manifest["correlation_id"]),
            equivalence_gate_owner_override=settings.equivalence_gate_owner_override,
            progress_path=legacy_path,
        )
        write_json(legacy_path, legacy)
        comparisons["released_vs_current_legacy_per_item"] = (
            compare_amendment_4e_records(
                legacy, released, include_item_outcomes=True
            )
        )
    passed = all(
        isinstance(value, dict) and value.get("passed") is True
        for value in comparisons.values()
    )
    record: dict[str, object] = {
        "schema_version": "pretrained-base-amendment-4e-check-v1",
        "manifest_sha256": manifest_sha256,
        "correlation_id": str(manifest["correlation_id"]),
        "candidate_id": candidate_id,
        "check": "placement_equivalence" if placement_equivalence else "4c_consistency",
        "status": "passed" if passed else "failed",
        "comparisons": comparisons,
        "frozen_4c_item_evidence_limitation": (
            "Amendment 4c retained family aggregates but not per-item outcomes; exact "
            "per-item comparison is therefore performed between the two current-code "
            "placement orders, while the frozen record comparison uses all retained "
            "decision metrics and family counts."
        ),
    }
    write_json(
        check_dir
        / ("placement-equivalence.json" if placement_equivalence else "gpt2-consistency.json"),
        record,
    )
    return record


def rank_amendment_4e_records(
    config_path: Path | str,
    output_dir: Path | str,
    prior_output_dir: Path | str,
    *,
    rerun_completed_4c_candidates: bool,
) -> dict[str, object]:
    """Combine validated 4c and 4e records under the unchanged decision rule."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    prior = Path(prior_output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    prior_manifest = _read_json(prior / "run_manifest.json")
    prior_manifest_sha256 = hashlib.sha256(
        json.dumps(prior_manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    fresh_ids = {"qwen2.5-0.5b", "qwen2.5-1.5b", "smollm2-1.7b"}
    if rerun_completed_4c_candidates:
        fresh_ids.update({"gpt2", "smollm2-135m", "pythia-160m", "smollm2-360m"})
    records: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for candidate in settings.candidates:
        source_root = destination if candidate.candidate_id in fresh_ids else prior
        expected_manifest = (
            manifest_sha256 if source_root == destination else prior_manifest_sha256
        )
        path = _candidate_record_path(source_root, candidate.candidate_id)
        if not path.is_file():
            record = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": expected_manifest,
                "correlation_id": str(manifest["correlation_id"]),
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "reference_only": candidate.reference_only,
                "failure": {
                    "phase": "mixed_provenance_ranking",
                    "type": "MissingCandidateRecord",
                    "message": f"candidate record is missing: {path}",
                    "out_of_memory": False,
                },
            }
        else:
            record = _read_json(path)
            if record.get("candidate_id") != candidate.candidate_id:
                raise ValueError(f"candidate record identity mismatch: {path}")
            if record.get("manifest_sha256") != expected_manifest:
                raise ValueError(f"candidate record manifest mismatch: {path}")
            provenance.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "protocol": "amendment_4e" if source_root == destination else "amendment_4c",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "manifest_sha256": expected_manifest,
                }
            )
        records.append(record)
    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = manifest_sha256
    ranking["correlation_id"] = str(manifest["correlation_id"])
    ranking["record_provenance"] = provenance
    ranking["prior_manifest_sha256"] = prior_manifest_sha256
    ranking["consistency_outcome"] = (
        "reran_four_completed_candidates"
        if rerun_completed_4c_candidates
        else "reused_four_reproduced_amendment_4c_records"
    )
    record_by_id = {str(record.get("candidate_id")): record for record in records}
    ranked_rows = cast(list[dict[str, object]], ranking["ranked_candidates"])
    top_rows = ranked_rows[: settings.measurement.variance_top_candidates]
    seed_variance: list[dict[str, object]] = []
    for row in top_rows:
        candidate_id = cast(str, row["candidate_id"])
        primary_source = destination if candidate_id in fresh_ids else prior
        primary_manifest = (
            manifest_sha256
            if primary_source == destination
            else prior_manifest_sha256
        )
        seed_variance.append(
            _seed_variance_summary(
                candidate_id,
                record_by_id[candidate_id],
                primary_source,
                settings.measurement,
                primary_manifest,
                seed_destination=destination,
                seed_manifest_sha256=manifest_sha256,
            )
        )
    ranking["seed_variance"] = seed_variance
    ranking["top_two_seed_repeat_agreement"] = _top_two_seed_repeat_agreement(
        seed_variance, settings.measurement.variance_seeds
    )
    write_json(destination / "ranking.json", ranking)
    return ranking


def rank_amendment_4g_records(
    config_path: Path | str,
    output_dir: Path | str,
    amendment_4c_output_dir: Path | str,
    amendment_4f_output_dir: Path | str,
) -> dict[str, object]:
    """Rank the five measured candidates and decide only on unanimous seed repeats."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    source_4c = Path(amendment_4c_output_dir)
    source_4f = Path(amendment_4f_output_dir)
    expected_candidate_ids = {
        "gpt2",
        "smollm2-135m",
        "pythia-160m",
        "smollm2-360m",
        "qwen2.5-0.5b",
    }
    configured_candidate_ids = {
        candidate.candidate_id for candidate in settings.candidates
    }
    if configured_candidate_ids != expected_candidate_ids:
        raise ValueError("Amendment 4g config must contain exactly the five measured candidates")
    if settings.measurement.variance_top_candidates != 2:
        raise ValueError("Amendment 4g requires exactly two seed-repeat leaders")
    if settings.measurement.variance_seeds != (20261014, 20261015, 20261016):
        raise ValueError("Amendment 4g requires the three pre-registered seeds")
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)

    def manifest_identity(directory: Path) -> str:
        value = _read_json(directory / "run_manifest.json")
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()

    source_manifests = {
        source_4c: manifest_identity(source_4c),
        source_4f: manifest_identity(source_4f),
    }
    source_by_candidate = {
        "gpt2": source_4c,
        "smollm2-135m": source_4c,
        "pythia-160m": source_4c,
        "smollm2-360m": source_4c,
        "qwen2.5-0.5b": source_4f,
    }
    records: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    record_by_id: dict[str, dict[str, object]] = {}
    for candidate_id, source_root in source_by_candidate.items():
        path = _candidate_record_path(source_root, candidate_id)
        if not path.is_file():
            raise FileNotFoundError(f"Amendment 4g source record is missing: {path}")
        record = _read_json(path)
        if record.get("candidate_id") != candidate_id:
            raise ValueError(f"candidate record identity mismatch: {path}")
        expected_manifest = source_manifests[source_root]
        if record.get("manifest_sha256") != expected_manifest:
            raise ValueError(f"candidate record manifest mismatch: {path}")
        records.append(record)
        record_by_id[candidate_id] = record
        provenance.append(
            {
                "candidate_id": candidate_id,
                "protocol": source_root.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "manifest_sha256": expected_manifest,
            }
        )

    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = manifest_sha256
    ranking["correlation_id"] = str(manifest["correlation_id"])
    ranking["record_provenance"] = provenance
    ranking["candidate_set"] = list(source_by_candidate)
    owner_dispositions: list[dict[str, object]] = []
    for candidate_id in ("qwen2.5-1.5b", "smollm2-1.7b"):
        monitor_path = (
            source_4f / "process_monitors" / f"{candidate_id}-attempt-1.json"
        )
        monitor = _read_json(monitor_path)
        prestart = monitor.get("prestart_check")
        reasons = monitor.get("environment_invalid_reasons")
        prestart_samples = (
            prestart.get("samples") if isinstance(prestart, dict) else None
        )
        if (
            not isinstance(prestart, dict)
            or prestart.get("normal") is not True
            or not isinstance(prestart_samples, list)
            or not prestart_samples
            or any(
                not isinstance(sample, dict)
                or sample.get("other_processes_above_rss_threshold") != []
                for sample in prestart_samples
            )
            or reasons
            != ["system free-memory percentage fell below the critical floor"]
        ):
            raise ValueError(f"owner disposition evidence is malformed: {monitor_path}")
        owner_dispositions.append(
            {
                "candidate_id": candidate_id,
                "verdict": "infeasible_available_hardware",
                "evidence": {
                    "process_monitor": str(monitor_path),
                    "sha256": sha256_file(monitor_path),
                    "elapsed_seconds": monitor["elapsed_seconds"],
                    "minimum_free_memory_percent": monitor[
                        "system_free_memory_minimum_percent"
                    ],
                    "prestart_check": prestart,
                },
            }
        )
    ranking["owner_dispositions"] = owner_dispositions

    ranked_rows = cast(list[dict[str, object]], ranking["ranked_candidates"])
    leaders = [str(row["candidate_id"]) for row in ranked_rows[:2]]
    records_by_seed: dict[int, Sequence[Mapping[str, object]] | None] = {}
    missing_repeat_records: list[dict[str, object]] = []
    for seed in settings.measurement.variance_seeds:
        seed_records: list[dict[str, object]] = []
        for candidate_id in leaders:
            if seed == settings.measurement.seed:
                seed_records.append(record_by_id[candidate_id])
                continue
            path = _seed_record_path(destination, candidate_id, seed)
            if not path.is_file():
                missing_repeat_records.append(
                    {"candidate_id": candidate_id, "seed": seed, "reason": "missing"}
                )
                continue
            record = _read_json(path)
            if record.get("candidate_id") != candidate_id:
                raise ValueError(f"seed record identity mismatch: {path}")
            if record.get("manifest_sha256") != manifest_sha256:
                raise ValueError(f"seed record manifest mismatch: {path}")
            seed_records.append(record)
        records_by_seed[seed] = seed_records if len(seed_records) == 2 else None

    agreement = _full_rule_seed_repeat_agreement(
        records_by_seed,
        settings.measurement.variance_seeds,
        settings.decision,
    )
    ranking["top_two_seed_repeat_agreement"] = agreement
    ranking["selection_stable_across_seeds"] = agreement[
        "selection_stable_across_seeds"
    ]
    ranking["missing_seed_repeat_records"] = missing_repeat_records
    _finalize_amendment_4g_selection(ranking, agreement)

    rank_by_id = {
        str(row["candidate_id"]): int(cast(int, row["rank"]))
        for row in ranked_rows
    }
    rejected_by_id = {
        str(row["candidate_id"]): cast(list[str], row["reasons"])
        for row in cast(list[dict[str, object]], ranking["rejected_candidates"])
    }
    candidate_verdicts: list[dict[str, object]] = []
    for candidate_id in source_by_candidate:
        if candidate_id in rank_by_id:
            verdict = "eligible"
            detail: object = {"rank": rank_by_id[candidate_id]}
        else:
            verdict = "ineligible"
            detail = {"reasons": rejected_by_id.get(candidate_id, [])}
        candidate_verdicts.append(
            {"candidate_id": candidate_id, "verdict": verdict, "detail": detail}
        )
    candidate_verdicts.extend(cast(list[dict[str, object]], ranking["owner_dispositions"]))
    ranking["candidate_verdicts"] = candidate_verdicts

    selected = ranking["selected_candidate"]
    if isinstance(selected, str):
        identity = cast(dict[str, object], record_by_id[selected]["identity"])
        ranking["adr_status_instruction"] = {
            "status": "Accepted",
            "candidate_id": selected,
            "repository": identity["repo_id_or_path"],
            "revision": identity["resolved_revision"],
            "tokenizer": identity["tokenizer_type"],
            "review_required": True,
        }
    else:
        ranking["adr_status_instruction"] = {
            "status": "Proposed",
            "reason": ranking["decision_status"],
            "review_required": True,
        }
    write_json(destination / "ranking.json", ranking)
    return ranking


def derive_amendment_3_analysis(
    config_path: Path | str,
    output_dir: Path | str,
    prior_output_dir: Path | str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Combine immutable prior records with Amendment 3 records and re-rank them."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    prior = Path(prior_output_dir)

    def manifest_identity(directory: Path) -> tuple[dict[str, object], str]:
        path = directory / "run_manifest.json"
        if not path.is_file():
            raise ValueError(f"evidence manifest is missing: {path}")
        value = _read_json(path)
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        return value, hashlib.sha256(payload).hexdigest()

    current_manifest, current_manifest_sha256 = manifest_identity(destination)
    prior_manifest, prior_manifest_sha256 = manifest_identity(prior)
    records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    sources_by_id: dict[str, tuple[Path, str]] = {}
    for candidate in settings.candidates:
        relative = _candidate_record_path(Path(), candidate.candidate_id)
        current_path = destination / relative
        prior_path = prior / relative
        if current_path.is_file():
            path = current_path
            source_root = destination
            expected_manifest = current_manifest_sha256
            protocol = "amendment_3"
        elif prior_path.is_file():
            path = prior_path
            source_root = prior
            expected_manifest = prior_manifest_sha256
            protocol = "prior_powered"
        else:
            raise ValueError(f"candidate record is missing: {candidate.candidate_id}")
        record = _read_json(path)
        if record.get("candidate_id") != candidate.candidate_id:
            raise ValueError(f"candidate record identity mismatch: {path}")
        if record.get("manifest_sha256") != expected_manifest:
            raise ValueError(f"candidate record manifest mismatch: {path}")
        records.append(record)
        sources_by_id[candidate.candidate_id] = (source_root, expected_manifest)
        source_records.append(
            {
                "candidate_id": candidate.candidate_id,
                "protocol": protocol,
                "path": str(path),
                "sha256": sha256_file(path),
                "schema_version": str(record.get("schema_version")),
                "manifest_sha256": expected_manifest,
            }
        )

    ranking = apply_decision_rule(records, settings.decision)
    ranking["manifest_sha256"] = current_manifest_sha256
    ranking["correlation_id"] = str(current_manifest.get("correlation_id"))
    ranking["record_provenance"] = source_records
    record_by_id = {str(record.get("candidate_id")): record for record in records}
    prior_repeat_ids = {
        "smollm2-135m",
        "smollm2-360m",
    }
    current_repeat_ids = {
        path.name.rsplit("-seed-", 1)[0]
        for path in (destination / "seed_runs").glob("*.json")
    }
    variance_ids: list[str] = []
    for row in cast(list[dict[str, object]], ranking["ranked_candidates"]):
        candidate_id = cast(str, row["candidate_id"])
        primary_stem = _candidate_record_path(Path(), candidate_id).stem
        if candidate_id in prior_repeat_ids or primary_stem in current_repeat_ids:
            variance_ids.append(candidate_id)
    ranking["seed_variance"] = [
        _seed_variance_summary(
            candidate_id,
            record_by_id[candidate_id],
            sources_by_id[candidate_id][0],
            settings.measurement,
            sources_by_id[candidate_id][1],
        )
        for candidate_id in variance_ids
    ]
    analysis: dict[str, object] = {
        "schema_version": AMENDMENT_3_ANALYSIS_SCHEMA_VERSION,
        "derivation": (
            "analysis-only combination of immutable prior powered records and Amendment 3 "
            "records; no model was loaded, trained, or evaluated"
        ),
        "source_manifests": {
            "prior_powered": {
                "path": str(prior / "run_manifest.json"),
                "sha256": prior_manifest_sha256,
                "correlation_id": prior_manifest.get("correlation_id"),
            },
            "amendment_3": {
                "path": str(destination / "run_manifest.json"),
                "sha256": current_manifest_sha256,
                "correlation_id": current_manifest.get("correlation_id"),
            },
        },
        "source_records": source_records,
        "amended_ranking": ranking,
    }
    write_json(destination / "amendment-3-analysis.json", analysis)
    write_json(destination / "ranking.json", ranking)
    return analysis, ranking


def write_isolated_process_failure(
    config_path: Path | str,
    output_dir: Path | str,
    candidate_id: str,
    *,
    returncode: int,
    message: str,
    variance_seed: int | None = None,
    parent_process_memory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist a failure when an isolated child exits before Python can record it."""
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    candidate = next(
        item for item in settings.candidates if item.candidate_id == candidate_id
    )
    lowered = message.lower()
    record: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": str(manifest["correlation_id"]),
        "status": "failed",
        "candidate_id": candidate_id,
        "reference_only": candidate.reference_only,
        "equivalence_gate_owner_override": settings.equivalence_gate_owner_override,
        "variance_seed": variance_seed,
        "failure": {
            "phase": "isolated_process",
            "type": "ChildProcessFailure",
            "message": message,
            "returncode": returncode,
            "out_of_memory": (
                "out of memory" in lowered or returncode in {-9, 137}
            ),
        },
    }
    if parent_process_memory is not None:
        record["parent_process_memory"] = dict(parent_process_memory)
    record_path = (
        _candidate_record_path(destination, candidate_id)
        if variance_seed is None
        else _seed_record_path(destination, candidate_id, variance_seed)
    )
    write_json(record_path, record)
    return record


def write_environment_status(
    config_path: Path | str,
    output_dir: Path | str,
    candidate_id: str,
    *,
    status: str,
    attempts: Sequence[Mapping[str, object]],
    variance_seed: int | None = None,
) -> dict[str, object]:
    """Persist an environment-only attempt outcome without rejecting the candidate."""
    if status not in {"environment_invalid", "not_measured_environment"}:
        raise ValueError(f"unsupported environment status: {status}")
    settings = load_base_model_evaluation_settings(config_path)
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    candidate = next(
        item for item in settings.candidates if item.candidate_id == candidate_id
    )
    record: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "correlation_id": str(manifest["correlation_id"]),
        "config_fingerprint": _config_fingerprint(
            candidate, settings.measurement, manifest_sha256
        ),
        "status": status,
        "candidate_id": candidate_id,
        "reference_only": candidate.reference_only,
        "variance_seed": variance_seed,
        "equivalence_gate_owner_override": settings.equivalence_gate_owner_override,
        "environment_attempts": [dict(attempt) for attempt in attempts],
        "failure": {
            "phase": "environment_health",
            "type": "EnvironmentInvalid",
            "message": (
                "all configured attempts were environment-invalid"
                if status == "not_measured_environment"
                else "the latest attempt was environment-invalid and is retryable"
            ),
            "out_of_memory": False,
        },
    }
    record_path = (
        _candidate_record_path(destination, candidate_id)
        if variance_seed is None
        else _seed_record_path(destination, candidate_id, variance_seed)
    )
    write_json(record_path, record)
    return record


def attach_environment_attempts(
    output_dir: Path | str,
    candidate_id: str,
    attempts: Sequence[Mapping[str, object]],
    *,
    variance_seed: int | None = None,
) -> None:
    """Attach parent-observed pre-start/runtime evidence to a child record."""
    destination = Path(output_dir)
    record_path = (
        _candidate_record_path(destination, candidate_id)
        if variance_seed is None
        else _seed_record_path(destination, candidate_id, variance_seed)
    )
    if not record_path.is_file():
        raise FileNotFoundError(f"candidate record is missing: {record_path}")
    record = _read_json(record_path)
    record["environment_attempts"] = [dict(attempt) for attempt in attempts]
    compromised = any(
        attempt.get("throughput_compromised") is True for attempt in attempts
    )
    record["throughput_compromised"] = compromised
    record["throughput_tie_breaker_status"] = (
        "excluded_swap_compromised" if compromised else "measured"
    )
    write_json(record_path, record)
