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
from aptivra.export.hf_conversion import convert_legacy_export_to_huggingface
from aptivra.huggingface.configuration_aptivra import AptivraConfig
from aptivra.huggingface.modeling_aptivra import AptivraForCausalLM
from aptivra.huggingface.tokenization_aptivra import AptivraByteTokenizer
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.stream import ContinualTaskStream

EVALUATION_SCHEMA_VERSION = "pretrained-base-evaluation-v4"
SEED_EVALUATION_SCHEMA_VERSION = "pretrained-base-seed-evaluation-v1"
RANKING_SCHEMA_VERSION = "pretrained-base-ranking-v4"
MANIFEST_SCHEMA_VERSION = "pretrained-base-manifest-v2"
DERIVED_ANALYSIS_SCHEMA_VERSION = "pretrained-base-derived-analysis-v1"
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


@dataclass(frozen=True)
class ExecutionSettings:
    """Operational persistence controls for the evidence sweep."""

    output_dir: Path
    resume: bool


@dataclass(frozen=True)
class BaseModelEvaluationSettings:
    """Complete candidate evaluation configuration."""

    candidates: tuple[CandidateSpec, ...]
    measurement: MeasurementSettings
    decision: DecisionSettings
    execution: ExecutionSettings | None = None


@dataclass(frozen=True)
class LoadedCandidate:
    """A loaded model/tokenizer pair and its resolved source identity."""

    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    resolved_revision: str


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


ROOT_KEYS = {"candidates", "measurement", "decision", "execution"}
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
}
EXECUTION_KEYS = {"output_dir", "resume"}


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
    licenses = tuple(
        cast(str, _string(item, "decision.permissive_licenses[]"))
        for item in _sequence(decision.get("permissive_licenses"), "permissive_licenses")
    )
    if not licenses:
        raise ValueError("decision.permissive_licenses must not be empty")

    return BaseModelEvaluationSettings(
        candidates=tuple(candidates),
        measurement=MeasurementSettings(
            device=cast(str, _string(measurement.get("device"), "measurement.device")),
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
                2,
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
            probe_batch_size=_integer(
                measurement.get("probe_batch_size"),
                "measurement.probe_batch_size",
                1,
                256,
            ),
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
        ),
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
    static: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_file),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "orchestrator_sha256": sha256_file(
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "evaluate_base_model_candidates.py"
        ),
        "heldout_text_sha256": sha256_file(settings.measurement.heldout_text_path),
        "heldout_prompts_sha256": sha256_file(settings.measurement.heldout_prompts_path),
        "candidates": candidates,
        "seed": settings.measurement.seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "mps_available": torch.backends.mps.is_available(),
            "configured_device": settings.measurement.device,
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
    typed_model.to(resolve_device(settings.device))
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
    """Check lossless UTF-8 round trips and token density on the task stream."""
    task_texts: list[str] = []
    for family in stream.families:
        split = stream.get_split(family.family_id)
        for item in (*split.train, *split.val, *split.test):
            task_texts.extend((item.prompt, item.target_solution, item.training_target()))
    samples = [*configured_samples, *task_texts]
    failures: list[dict[str, object]] = []
    for sample in samples:
        token_ids = _encode(tokenizer, sample)
        decoded = _decode(tokenizer, token_ids)
        if decoded != sample:
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
            failures.append(
                {
                    "input": sample,
                    "decoded": decoded,
                    "input_code_points": [f"U+{ord(character):04X}" for character in sample],
                    "decoded_code_points": [
                        f"U+{ord(character):04X}" for character in decoded
                    ],
                    "token_ids": token_ids,
                    "add_special_tokens": False,
                    "decode_skip_special_tokens": False,
                    "decoded_with_cleanup_enabled": cleaned,
                    "decoded_with_cleanup_disabled": decoded,
                    "backend_normalizer": str(normalizer) if normalizer is not None else None,
                    "diagnosed_cause": cause,
                }
            )
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
        "exact_utf8_roundtrip": not failures,
        "roundtrip_samples": len(samples),
        "roundtrip_failures": failures[:10],
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
    return _accuracy_record(successes, len(instances), confidence_level)


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
) -> dict[str, object]:
    """Train an isolated family probe, optionally reserving test for the final LR."""
    seed_everything(settings.seed + family_index)
    family = stream.families[family_index]
    split = stream.get_split(family.family_id)
    selected_learning_rate = (
        settings.probe_learning_rates[0] if learning_rate is None else learning_rate
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
    probe = copy.deepcopy(base_model).to(device)
    probe.train()
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
        input_ids, labels, attention_mask, byte_count = _training_batch(
            tokenizer, batch, context_length, device
        )
        optimizer.zero_grad(set_to_none=True)
        output = probe(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
        )
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("causal model did not return a training loss")
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().item()))
        processed_tokens += int(attention_mask.sum().item())
        supervised_tokens += int((labels != -100).sum().item())
        trained_bytes += byte_count
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
    base_predictor = TransformersPredictor(
        model,
        tokenizer,
        device=device,
        context_length=context_length,
        max_new_tokens=settings.max_new_tokens,
    )
    zero_shot = [
        evaluate_batched_accuracy(
            base_predictor,
            family,
            stream.get_split(family.family_id).test,
            batch_size=settings.evaluation_batch_size,
            confidence_level=settings.confidence_level,
        )
        for family in stream.families
    ]
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
        )
        for index in range(len(stream.families))
    ]
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


def _rss_high_water_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class MemoryPeakSampler(AbstractContextManager["MemoryPeakSampler"]):
    """Sample per-process RSS and accelerator allocation high-water values."""

    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_baseline = _rss_high_water_bytes()
        self.rss_peak = self.rss_baseline
        self.mps_baseline: int | None = self._mps_bytes()
        self.mps_peak: int | None = self.mps_baseline

    def _mps_bytes(self) -> int | None:
        if self.device.type != "mps" or not hasattr(torch, "mps"):
            return None
        try:
            return int(torch.mps.driver_allocated_memory())
        except RuntimeError:
            return None

    def _sample(self) -> None:
        self.rss_peak = max(self.rss_peak, _rss_high_water_bytes())
        mps_value = self._mps_bytes()
        if mps_value is not None:
            self.mps_peak = max(self.mps_peak or 0, mps_value)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> MemoryPeakSampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5))
        self._sample()

    def record(self) -> dict[str, object]:
        """Return raw and derived values without double-counting unified memory."""
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
            "hard_constraint_peak_bytes": max(candidates),
            "hard_constraint_metric": (
                "max(absolute process RSS peak, absolute MPS driver allocation peak)"
                if self.mps_peak is not None
                else "absolute process RSS peak"
            ),
            "sample_interval_seconds": self.interval_seconds,
        }


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
) -> dict[str, object]:
    """Evaluate one candidate, converting every failure into a durable record."""
    fingerprint = _config_fingerprint(candidate, settings, manifest_sha256)
    phase = "load"
    started = time.perf_counter()
    device = resolve_device(settings.device)
    sampler = MemoryPeakSampler(device, settings.memory_sample_interval_seconds)
    partial: dict[str, object] = {}
    model: CausalLM | None = None
    with sampler:
        try:
            seed_everything(settings.seed)
            configure_determinism(settings)
            loaded, load_attempts = _load_with_retries(candidate, settings, loader)
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
            if token_fitness["exact_utf8_roundtrip"] is not True:
                cheap_reasons.append("tokenizer failed exact UTF-8 round-trip")
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
                language = evaluate_language_quality(
                    model, tokenizer, settings, context_length, device
                )
                partial["language_modeling"] = language
                phase = "task_readiness"
                readiness = evaluate_task_readiness(
                    model, tokenizer, stream, settings, context_length, device
                )
                partial["task_stream_readiness"] = readiness
                phase = "cost"
                generation = benchmark_generation(
                    model, tokenizer, stream, settings, context_length, device
                )
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
                        "validation_items_per_family": settings.validation_items_per_family,
                        "test_items_per_family": settings.test_items_per_family,
                        "probe_steps": settings.probe_steps,
                        "probe_batch_size": settings.probe_batch_size,
                        "probe_learning_rates": list(settings.probe_learning_rates),
                        "probe_warmup_steps": settings.probe_warmup_steps,
                        "evaluation_batch_size": settings.evaluation_batch_size,
                        "confidence_level": settings.confidence_level,
                        "expected_wall_time_seconds": settings.expected_wall_time_seconds,
                        "max_new_tokens": settings.max_new_tokens,
                        "scoring_stride_tokens": settings.scoring_stride_tokens,
                        "deterministic_algorithms": settings.deterministic_algorithms,
                        "deterministic_warn_only": settings.deterministic_warn_only,
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
        if _nested(record, "tokenizer_fitness", "exact_utf8_roundtrip") is not True:
            reasons.append("tokenizer failed exact UTF-8 round-trip")
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
                -_metric(item, "cost", "median_probe_training_tokens_per_second"),
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
                -_metric(item, "cost", "median_probe_training_tokens_per_second"),
                _metric(item, "identity", "unique_parameters"),
                str(item.get("candidate_id")),
            ),
        )
        ranked.append(winner)
        remaining.remove(winner)
    return ranked


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
    for record in records:
        candidate_id = str(record.get("candidate_id"))
        if record.get("reference_only") is True:
            references.append(
                {"candidate_id": candidate_id, "reason": "reference-only; excluded from selection"}
            )
            continue
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
            "training_tokens_per_second": _metric(
                record, "cost", "median_probe_training_tokens_per_second"
            ),
            "unique_parameters": int(
                _metric(record, "identity", "unique_parameters")
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
            else "ranked_no_adr_decision"
            if ranking
            else "no_eligible_candidate"
        ),
        "selected_candidate": ranking[0]["candidate_id"] if ranking else None,
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
) -> dict[str, object]:
    """Rerun a first-pass leader at its validation-selected LR for another seed."""
    settings = load_base_model_evaluation_settings(config_path)
    if seed == settings.measurement.seed or seed not in settings.measurement.variance_seeds:
        raise ValueError("variance seed must be a configured non-primary seed")
    destination = Path(output_dir)
    manifest, manifest_sha256 = prepare_run_manifest(config_path, destination, settings)
    primary_path = _candidate_record_path(destination, candidate_id)
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
    )
    record["variance_role"] = "top-candidate seed-to-seed estimate"
    record["primary_selected_learning_rate"] = learning_rate
    write_json(record_path, record)
    return record


def _seed_variance_summary(
    candidate_id: str,
    primary: Mapping[str, object],
    destination: Path,
    settings: MeasurementSettings,
    manifest_sha256: str,
) -> dict[str, object]:
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
        path = _seed_record_path(destination, candidate_id, seed)
        if not path.is_file():
            missing.append(seed)
            continue
        record = _read_json(path)
        if (
            record.get("status") != "completed"
            or record.get("manifest_sha256") != manifest_sha256
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
                "record": str(path.relative_to(destination)),
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
    for candidate in settings.candidates:
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
    analysis = {
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


def write_isolated_process_failure(
    config_path: Path | str,
    output_dir: Path | str,
    candidate_id: str,
    *,
    returncode: int,
    message: str,
    variance_seed: int | None = None,
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
        "variance_seed": variance_seed,
        "failure": {
            "phase": "isolated_process",
            "type": "ChildProcessFailure",
            "message": message,
            "returncode": returncode,
            "out_of_memory": "out of memory" in lowered or returncode in {-9, 137},
        },
    }
    record_path = (
        _candidate_record_path(destination, candidate_id)
        if variance_seed is None
        else _seed_record_path(destination, candidate_id, variance_seed)
    )
    write_json(record_path, record)
    return record
