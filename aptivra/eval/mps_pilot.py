"""Bounded, resumable ADR-0011 lane-8a MPS calibration and noise pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

import numpy as np
import torch
import yaml
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aptivra.baselines.standard import (
    ManualLoRATrainer,
    load_standard_baseline_settings,
    target_only_loss_from_hidden,
)
from aptivra.eval.base_model_evaluation import (
    MemoryPeakSampler,
    system_free_memory_percentage,
    system_swap_used_bytes,
)
from aptivra.eval.config import load_experiment_settings, seed_experiment
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.eval.metrics import (
    compute_average_accuracy,
    compute_backward_transfer,
    compute_max_forgetting,
)
from aptivra.eval.task_stream_audit import load_audit_plan, write_or_check_plan
from aptivra.experts.router import SequenceTop1Router, masked_mean_frozen_embeddings
from aptivra.experts.standard import PeftExpertBank, load_standard_expert_settings
from aptivra.gate.promotion_gate import PromotionGate
from aptivra.inference.batched_generation import generate_task_instance_answers
from aptivra.inference.standard_base import (
    CausalBatch,
    StandardBase,
    collate_task_instances,
    generate_task_instance_answer,
    load_standard_base,
    load_standard_base_settings,
)
from aptivra.resources import aptivra_resource
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.github_issues import (
    GitHubIssuesContinualStream,
    load_github_issue_builder_config,
)
from aptivra.tasks.stream import TaskStreamSplit

DEFAULT_MPS_PILOT_CONFIG = aptivra_resource("configs", "h3_mps_pilot.yaml")
PILOT_SCHEMA = "aptivra.h3_mps_pilot_evidence.v1"
AMENDMENT_SCHEMA = "aptivra.h3_mps_pilot_evidence.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METHODS = (
    "frozen_base",
    "naive_sequential",
    "replay",
    "online_ewc",
    "manual_lora",
    "aptivra_full",
)
PilotMethod = Literal[
    "frozen_base",
    "naive_sequential",
    "replay",
    "online_ewc",
    "manual_lora",
    "aptivra_full",
]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FeatureSettings(_StrictModel):
    enabled: bool


class SyntheticStreamSettings(_StrictModel):
    config: Path
    audit_plan: Path
    audit_record: Path
    config_sha256: str
    audit_plan_sha256: str
    audit_sha256: str
    pair_task_indices: tuple[NonNegativeInt, NonNegativeInt]

    @field_validator("config", "audit_plan", "audit_record", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("pair_task_indices", mode="before")
    @classmethod
    def normalize_pair(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("config_sha256", "audit_plan_sha256", "audit_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("stream hashes must be lowercase SHA-256 values")
        return value


class IssueStreamSettings(_StrictModel):
    config: Path
    raw_data: Path
    manifest: Path
    audit_record: Path
    raw_sha256: str
    config_sha256: str
    manifest_sha256: str
    audit_sha256: str
    pair_task_indices: tuple[NonNegativeInt, NonNegativeInt]

    @field_validator("config", "raw_data", "manifest", "audit_record", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("pair_task_indices", mode="before")
    @classmethod
    def normalize_pair(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("raw_sha256", "config_sha256", "manifest_sha256", "audit_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("stream hashes must be lowercase SHA-256 values")
        return value


class StreamSettings(_StrictModel):
    synthetic_v2: SyntheticStreamSettings
    chronological_issues: IssueStreamSettings


class QuietMachineSettings(_StrictModel):
    minimum_free_memory_percent: float = Field(strict=True, ge=1.0, le=100.0)
    maximum_foreign_process_rss_bytes: PositiveInt
    maximum_foreign_process_cpu_percent: float = Field(strict=True, ge=0.1)
    timeout_seconds: float = Field(strict=True, ge=1.0, le=86_400.0)
    initial_backoff_seconds: PositiveFloat
    maximum_backoff_seconds: PositiveFloat

    @model_validator(mode="after")
    def validate_backoff(self) -> QuietMachineSettings:
        if self.initial_backoff_seconds > self.maximum_backoff_seconds:
            raise ValueError("quiet-machine initial backoff cannot exceed its maximum")
        if self.maximum_backoff_seconds > self.timeout_seconds:
            raise ValueError("quiet-machine maximum backoff cannot exceed its timeout")
        logical_cpu_ceiling = (os.cpu_count() or 1) * 100.0
        if self.maximum_foreign_process_cpu_percent > logical_cpu_ceiling:
            raise ValueError("quiet-machine CPU threshold exceeds logical-CPU capacity")
        return self


class CalibrationAnomalyCell(_StrictModel):
    method: Literal["online_ewc", "manual_lora", "aptivra_full"]
    stream: Literal["chronological_issues"]
    micro_batch: Literal[8]


class CalibrationSettings(_StrictModel):
    historical_evidence: Path
    historical_evidence_sha256: str
    seed: NonNegativeInt
    frozen_base_mode: Literal["forward_only"]
    frozen_generation_path: Literal["max_answer_length"]
    worst_case_example_policy: Literal["longest_distinct_pad_to_stream_max"]
    gradient_checkpointing_fallback: Literal["micro_batch_1_only"]
    resident_adapter_count: Literal[12]
    methods: list[PilotMethod] = Field(min_length=6, max_length=6)
    micro_batches: list[Literal[8, 4, 2, 1]] = Field(min_length=4, max_length=4)
    effective_batch_size: Literal[8]
    warmup_steps: Literal[1]
    measured_steps: PositiveInt = Field(ge=3)
    memory_cap_bytes: Literal[28_991_029_248]
    memory_sample_interval_seconds: float = Field(strict=True, ge=0.001, le=1.0)
    swap_throughput_reporting_threshold_bytes: NonNegativeInt
    anomaly_cells: list[CalibrationAnomalyCell] = Field(min_length=3, max_length=3)

    @field_validator("historical_evidence", mode="before")
    @classmethod
    def normalize_historical_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("historical_evidence_sha256")
    @classmethod
    def validate_historical_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("historical calibration hash must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_protocol(self) -> CalibrationSettings:
        if tuple(self.methods) != _METHODS:
            raise ValueError("calibration methods must use the registered six-method order")
        if self.micro_batches != [8, 4, 2, 1]:
            raise ValueError("calibration micro-batches must be [8, 4, 2, 1]")
        expected = {
            ("online_ewc", "chronological_issues", 8),
            ("manual_lora", "chronological_issues", 8),
            ("aptivra_full", "chronological_issues", 8),
        }
        actual = {(cell.method, cell.stream, cell.micro_batch) for cell in self.anomaly_cells}
        if actual != expected:
            raise ValueError("calibration anomaly cells must cover the three lane-8b issue cells")
        return self
class RepeatabilityMethodMicroBatches(_StrictModel):
    naive_sequential: Literal[8, 4, 2, 1]
    aptivra_full: Literal[8, 4, 2, 1]


class RepeatabilityMicroBatches(_StrictModel):
    synthetic_v2: RepeatabilityMethodMicroBatches
    chronological_issues: RepeatabilityMethodMicroBatches


class RepeatabilityEvaluationBatchSizes(_StrictModel):
    synthetic_v2: PositiveInt = Field(le=16)
    chronological_issues: PositiveInt = Field(le=16)


class RepeatabilitySettings(_StrictModel):
    methods: tuple[Literal["naive_sequential", "aptivra_full"], ...]
    seeds: tuple[
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
    ]
    training_example_policy: Literal["one_epoch_up_to_cycle_caps"]
    evaluation_examples_per_task: PositiveInt
    template_ood_examples_per_task: PositiveInt
    effective_batch_size: Literal[8]
    micro_batches: RepeatabilityMicroBatches
    evaluation_batch_sizes: RepeatabilityEvaluationBatchSizes
    profile_examples: PositiveInt = Field(ge=2, le=32)
    max_new_tokens: PositiveInt
    checkpoint_after_each_run: Literal[True]

    @field_validator("methods", "seeds", mode="before")
    @classmethod
    def normalize_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_repeats(self) -> RepeatabilitySettings:
        if self.methods != ("naive_sequential", "aptivra_full"):
            raise ValueError("repeatability requires one full-FT method and Aptivra full")
        if self.seeds[0] != self.seeds[1] or len(set(self.seeds)) < 4:
            raise ValueError(
                "repeatability requires one duplicated seed and three additional seeds"
            )
        for stream in (self.micro_batches.synthetic_v2, self.micro_batches.chronological_issues):
            for micro_batch in (stream.naive_sequential, stream.aptivra_full):
                if self.effective_batch_size % micro_batch:
                    raise ValueError("repeatability micro-batches must divide effective batch 8")
        return self


class AblationSettings(_StrictModel):
    name: Literal[
        "lora_rank_8",
        "lora_rank_32",
        "mlp_only_projections",
        "task_id_oracle_routing",
        "router_without_replay",
        "no_replay",
        "no_expert_freezing",
        "fixed_one_adapter_capacity",
        "no_promotion_gate",
        "no_verifier_gated_memory",
        "no_growth_trigger",
    ]
    seeds: int = Field(strict=True, ge=1, le=5)


class EstimateSettings(_StrictModel):
    main_method_count: Literal[6]
    main_seed_count: int = Field(strict=True, ge=5)
    claim_stream_count: Literal[2]
    v1_config: Path
    v1_regression_method_count: PositiveInt
    v1_regression_seed_count: PositiveInt
    ablation_stream_count: Literal[2]
    ablations: list[AblationSettings] = Field(min_length=11, max_length=11)

    @field_validator("v1_config", mode="before")
    @classmethod
    def normalize_v1_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_ablations(self) -> EstimateSettings:
        if len({item.name for item in self.ablations}) != len(self.ablations):
            raise ValueError("H3 ablation cells must be uniquely named")
        return self


class CycleCapSettings(_StrictModel):
    adapter_optimizer_steps: PositiveInt
    adapter_target_tokens: PositiveInt
    router_optimizer_steps: PositiveInt
    router_replay_examples_per_expert: PositiveInt
    evaluation_examples_per_seen_task: PositiveInt


class ExecutionSettings(_StrictModel):
    device: Literal["mps"]
    offline: Literal[True]
    wall_time_limit_seconds: PositiveFloat = Field(le=28_800.0)


class MpsPilotSettings(_StrictModel):
    schema_version: Literal["aptivra.h3_mps_pilot_config.v2"]
    feature: FeatureSettings
    output_dir: Path
    base_config: Path
    baseline_config: Path
    expert_config: Path
    streams: StreamSettings
    quiet_machine: QuietMachineSettings
    calibration: CalibrationSettings
    repeatability: RepeatabilitySettings
    h3_estimate: EstimateSettings
    cycle_caps: CycleCapSettings
    execution: ExecutionSettings

    @field_validator("output_dir", "base_config", "baseline_config", "expert_config", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_output_location(self) -> MpsPilotSettings:
        if self.output_dir.is_absolute():
            raise ValueError("pilot output_dir must be worktree-relative")
        worktree = Path.cwd().resolve()
        try:
            (worktree / self.output_dir).resolve().relative_to(worktree)
        except ValueError as error:
            raise ValueError("pilot output_dir must resolve inside the worktree") from error
        if (
            self.repeatability.evaluation_examples_per_task
            < self.cycle_caps.evaluation_examples_per_seen_task
        ):
            raise ValueError("repeatability evaluation must meet the H3 per-task count")
        return self


def load_mps_pilot_settings(
    path: Path | str | Traversable = DEFAULT_MPS_PILOT_CONFIG,
) -> MpsPilotSettings:
    """Load the strict disabled-by-default pilot configuration."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"h3_mps_pilot"}:
        raise ValueError("pilot config root must contain only 'h3_mps_pilot'")
    return MpsPilotSettings.model_validate(raw["h3_mps_pilot"])


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required pilot input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_source_tree_sha256(root: Path = Path("aptivra")) -> str:
    """Bind every importable Aptivra Python source file to measured evidence."""
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no Python sources found below {root}")
    for path in paths:
        relative = path.relative_to(root.parent).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, description: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _json_mapping(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must contain an object: {path}")
    return cast(dict[str, Any], value)


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _accept_resume_amendment(
    progress: dict[str, Any],
    current_manifest: Mapping[str, Any],
    amendment_path: Path,
) -> None:
    """Record an explicit manifest transition and charge interrupted wall time."""
    raw: object = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "reason",
        "from_manifest_sha256",
        "to_manifest_sha256",
        "interrupted_wall_seconds",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("resume amendment must contain exactly the registered fields")
    amendment = cast(dict[str, Any], raw)
    if amendment["schema_version"] != "aptivra.h3_mps_pilot_resume_amendment.v1":
        raise ValueError("unsupported resume amendment schema")
    reason = amendment["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resume amendment reason must be non-empty")
    previous_manifest = progress.get("run_manifest")
    if not isinstance(previous_manifest, dict):
        raise ValueError("existing pilot checkpoint has no valid run manifest")
    expected_from = amendment["from_manifest_sha256"]
    expected_to = amendment["to_manifest_sha256"]
    if not isinstance(expected_from, str) or _SHA256.fullmatch(expected_from) is None:
        raise ValueError("resume amendment from-manifest digest must be SHA-256")
    if not isinstance(expected_to, str) or _SHA256.fullmatch(expected_to) is None:
        raise ValueError("resume amendment to-manifest digest must be SHA-256")
    if _mapping_sha256(previous_manifest) != expected_from:
        raise ValueError("resume amendment does not match the checkpoint manifest")
    if _mapping_sha256(current_manifest) != expected_to:
        raise ValueError("resume amendment does not match the current manifest")
    interrupted = amendment["interrupted_wall_seconds"]
    if not isinstance(interrupted, (int, float)) or isinstance(interrupted, bool):
        raise ValueError("resume amendment interrupted wall time must be numeric")
    interrupted_seconds = float(interrupted)
    if not math.isfinite(interrupted_seconds) or interrupted_seconds <= 0.0:
        raise ValueError("resume amendment interrupted wall time must be positive and finite")
    raw_prior_elapsed = progress.get("elapsed_wall_time_seconds", 0.0)
    if isinstance(raw_prior_elapsed, bool):
        raise ValueError("existing pilot checkpoint elapsed wall time must be numeric")
    prior_elapsed = float(raw_prior_elapsed)
    history = progress.setdefault("run_manifest_history", [])
    amendments = progress.setdefault("resume_amendments", [])
    if not isinstance(history, list) or not isinstance(amendments, list):
        raise ValueError("existing pilot checkpoint has invalid amendment history")
    history.append(
        {
            "run_manifest": previous_manifest,
            "completed_repeatability_records": len(progress.get("repeatability", [])),
            "elapsed_wall_time_seconds": prior_elapsed,
        }
    )
    amendments.append(
        {
            **amendment,
            "artifact": amendment_path.as_posix(),
            "artifact_sha256": _sha256(amendment_path),
        }
    )
    progress["run_manifest"] = dict(current_manifest)
    progress["elapsed_wall_time_seconds"] = prior_elapsed + interrupted_seconds


def validate_claim_streams(settings: MpsPilotSettings) -> dict[str, Any]:
    """Fail closed on both claim-bearing streams before model allocation."""
    synthetic = settings.streams.synthetic_v2
    synthetic_hashes = {
        "config_sha256": _require_hash(
            synthetic.config, synthetic.config_sha256, "synthetic stream config"
        ),
        "audit_plan_sha256": _require_hash(
            synthetic.audit_plan,
            synthetic.audit_plan_sha256,
            "synthetic stream audit plan",
        ),
        "audit_sha256": _require_hash(
            synthetic.audit_record, synthetic.audit_sha256, "synthetic stream audit"
        ),
    }
    audit_plan = load_audit_plan(synthetic.audit_plan)
    matching_entries = [
        entry
        for entry in audit_plan.streams
        if entry.stream_config == synthetic.config and entry.output == synthetic.audit_record
    ]
    if len(matching_entries) != 1:
        raise ValueError(
            "synthetic audit plan must contain exactly one matching v2 config/output entry"
        )
    write_or_check_plan(audit_plan, check=True)
    synthetic_record = _json_mapping(synthetic.audit_record)
    if synthetic_record.get("passed") is not True or synthetic_record.get("violations") != []:
        raise RuntimeError("claim-bearing synthetic v2 audit did not pass")

    issues = settings.streams.chronological_issues
    if not issues.raw_data.is_file():
        raise FileNotFoundError(f"claim-bearing stream raw data is missing: {issues.raw_data}")
    issue_hashes = {
        "raw_sha256": _require_hash(issues.raw_data, issues.raw_sha256, "issue raw data"),
        "config_sha256": _require_hash(issues.config, issues.config_sha256, "issue stream config"),
        "manifest_sha256": _require_hash(
            issues.manifest, issues.manifest_sha256, "issue stream manifest"
        ),
        "audit_sha256": _require_hash(
            issues.audit_record, issues.audit_sha256, "issue stream audit"
        ),
    }
    issue_config = load_github_issue_builder_config(issues.config)
    if issue_config.raw_dataset_path != issues.raw_data:
        raise ValueError("pilot issue raw-data path does not match the stream config")
    if issue_config.manifest_path != issues.manifest:
        raise ValueError("pilot issue manifest path does not match the stream config")
    if issue_config.audit_path != issues.audit_record:
        raise ValueError("pilot issue audit path does not match the stream config")
    issue_stream = GitHubIssuesContinualStream(issues.manifest, issues.raw_data)
    manifest = _json_mapping(issues.manifest)
    audit = _json_mapping(issues.audit_record)
    if manifest.get("schema_version") != 3:
        raise ValueError("issue manifest schema_version must be 3")
    if manifest.get("audit") != audit:
        raise ValueError("issue manifest audit does not match the committed audit record")
    summaries = issue_stream.audit_summary()
    if not summaries:
        raise RuntimeError("issue stream audit summary is empty")
    for family_id, summary in summaries.items():
        coverage = summary.get("coverage")
        overlap = summary.get("overlap")
        if not isinstance(coverage, dict) or coverage.get("meets_threshold") is not True:
            raise RuntimeError(f"issue stream coverage gate failed for {family_id}")
        if overlap != {"train_test": 0, "train_val": 0, "val_test": 0}:
            raise RuntimeError(f"issue stream overlap gate failed for {family_id}")
    split_counts = {
        family.family_id: {
            "train": len(issue_stream.get_split(family.family_id).train),
            "val": len(issue_stream.get_split(family.family_id).val),
            "test": len(issue_stream.get_split(family.family_id).test),
        }
        for family in issue_stream.families
    }
    record_count = sum(sum(counts.values()) for counts in split_counts.values())
    if record_count != manifest.get("record_count"):
        raise ValueError("issue split counts do not equal manifest record_count")
    return {
        "synthetic_v2": {
            "passed": True,
            **synthetic_hashes,
            "stream_version": synthetic_record.get("stream_version"),
            "family_count": len(cast(list[object], synthetic_record.get("families", []))),
        },
        "chronological_issues": {
            "passed": True,
            **issue_hashes,
            "record_count": record_count,
            "manifest_shards": len(cast(list[object], manifest.get("record_shards", []))),
            "split_counts": split_counts,
        },
    }


@dataclass(frozen=True)
class ProcessSample:
    pid: int
    parent_pid: int
    cpu_percent: float
    rss_bytes: int
    command: str


class QuietMachineTimeoutError(TimeoutError):
    """A bounded quiet wait that retains every foreign-load sample."""

    def __init__(self, wait_seconds: float, samples: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            "quiet-machine gate timed out without killing or renicing foreign processes"
        )
        self.wait_seconds = wait_seconds
        self.samples = [dict(sample) for sample in samples]


def _process_samples() -> tuple[ProcessSample, ...]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,%cpu=,rss=,comm="],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    samples: list[ProcessSample] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=4)
        if len(fields) != 5:
            continue
        try:
            pid = int(fields[0])
            parent_pid = int(fields[1])
            cpu_percent = float(fields[2])
            rss_bytes = int(fields[3]) * 1024
        except ValueError:
            continue
        samples.append(
            ProcessSample(
                pid=pid,
                parent_pid=parent_pid,
                cpu_percent=cpu_percent,
                rss_bytes=rss_bytes,
                command=fields[4],
            )
        )
    return tuple(samples)


def _own_process_tree(samples: Sequence[ProcessSample]) -> set[int]:
    own = {os.getpid()}
    changed = True
    while changed:
        changed = False
        for sample in samples:
            if sample.parent_pid in own and sample.pid not in own:
                own.add(sample.pid)
                changed = True
    return own


def quiet_machine_sample(settings: QuietMachineSettings) -> dict[str, Any]:
    """Record host pressure and every foreign process crossing a configured bound."""
    processes = _process_samples()
    own = _own_process_tree(processes)
    foreign = [sample for sample in processes if sample.pid not in own]
    heavy_rss = [
        asdict(sample)
        for sample in foreign
        if sample.rss_bytes > settings.maximum_foreign_process_rss_bytes
    ]
    heavy_cpu = [
        asdict(sample)
        for sample in foreign
        if sample.cpu_percent > settings.maximum_foreign_process_cpu_percent
    ]
    free = system_free_memory_percentage()
    sample = {
        "timestamp_unix": time.time(),
        "system_free_memory_percent": free,
        "system_swap_used_bytes": system_swap_used_bytes(),
        "foreign_process_count": len(foreign),
        "other_processes_above_rss_threshold": heavy_rss,
        "other_processes_above_cpu_threshold": heavy_cpu,
        "top_foreign_processes_by_rss": [
            asdict(item)
            for item in sorted(foreign, key=lambda row: row.rss_bytes, reverse=True)[:20]
        ],
    }
    sample["quiet"] = (
        free >= settings.minimum_free_memory_percent and not heavy_rss and not heavy_cpu
    )
    return sample


def wait_for_quiet_machine(
    settings: QuietMachineSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    sample: Callable[[QuietMachineSettings], dict[str, Any]] = quiet_machine_sample,
) -> dict[str, Any]:
    """Wait with bounded exponential backoff; never mutate foreign processes."""
    started = monotonic()
    delay = settings.initial_backoff_seconds
    records: list[dict[str, Any]] = []
    while True:
        record = sample(settings)
        records.append(record)
        elapsed = monotonic() - started
        if elapsed >= settings.timeout_seconds:
            raise QuietMachineTimeoutError(elapsed, records)
        if record.get("quiet") is True:
            return {"quiet": True, "wait_seconds": elapsed, "samples": records}
        sleep(min(delay, settings.timeout_seconds - elapsed))
        delay = min(settings.maximum_backoff_seconds, delay * 2.0)


def _validate_quiet_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reject direct worker invocations that are not bound to a real quiet gate."""
    if set(record) != {"quiet", "wait_seconds", "samples"}:
        raise ValueError("worker quiet record has an invalid schema")
    if record.get("quiet") is not True:
        raise ValueError("worker quiet record must contain boolean quiet=true")
    wait_seconds = record.get("wait_seconds")
    samples = record.get("samples")
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, int | float)
        or not math.isfinite(float(wait_seconds))
        or float(wait_seconds) < 0.0
    ):
        raise ValueError("worker quiet record wait_seconds must be finite and non-negative")
    if not isinstance(samples, list) or not samples:
        raise ValueError("worker quiet record must retain at least one process sample")
    if not isinstance(samples[-1], dict) or samples[-1].get("quiet") is not True:
        raise ValueError("worker quiet record must end with the accepted quiet sample")
    return dict(record)


def _redacted_failure(error: Exception, *, out_of_memory: bool = False) -> dict[str, Any]:
    """Record failure class without persisting arbitrary dataset-bearing exception text."""
    return {
        "type": type(error).__name__,
        "message": "exception details redacted from portable evidence",
        "out_of_memory": out_of_memory,
    }


class _TaskStream(Protocol):
    families: list[TaskFamily]

    def get_split(self, family_id: str) -> TaskStreamSplit: ...


def _make_streams(settings: MpsPilotSettings) -> dict[str, _TaskStream]:
    synthetic = load_experiment_settings(settings.streams.synthetic_v2.config).make_stream()
    issues = GitHubIssuesContinualStream(
        settings.streams.chronological_issues.manifest,
        settings.streams.chronological_issues.raw_data,
    )
    return {"synthetic_v2": synthetic, "chronological_issues": issues}


def _all_stream_instances(stream: _TaskStream) -> list[TaskInstance]:
    instances: list[TaskInstance] = []
    get_ood = getattr(stream, "get_template_ood_split", None)
    for family in stream.families:
        split = stream.get_split(family.family_id)
        instances.extend((*split.train, *split.val, *split.test))
        if callable(get_ood):
            instances.extend(get_ood(family.family_id))
    ids = [instance.task_id for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("claim-bearing stream has duplicate stable task IDs")
    return instances


def _token_lengths(base: StandardBase, instances: Sequence[TaskInstance]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    eos = base.tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError("pilot tokenizer must expose an integer EOS ID")
    for instance in instances:
        prompt = cast(list[int], base.tokenizer.encode(instance.prompt, add_special_tokens=False))
        target = cast(
            list[int], base.tokenizer.encode(instance.training_target(), add_special_tokens=False)
        )
        if not target or target[-1] != eos:
            target.append(eos)
        lengths[instance.task_id] = len(prompt) + len(target)
    return lengths


def longest_distinct_examples(
    base: StandardBase, stream: _TaskStream, count: int
) -> tuple[list[TaskInstance], list[int]]:
    """Choose real examples by token length with a stable-ID tie break."""
    if isinstance(count, bool) or count <= 0:
        raise ValueError("longest-example count must be positive")
    instances = _all_stream_instances(stream)
    if len(instances) < count:
        raise ValueError(f"stream has fewer than {count} accepted distinct examples")
    lengths = _token_lengths(base, instances)
    selected = sorted(instances, key=lambda item: (-lengths[item.task_id], item.task_id))[:count]
    return selected, [lengths[item.task_id] for item in selected]


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _mps_values() -> tuple[int, int]:
    return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())


def _base_for_pilot(settings: MpsPilotSettings) -> StandardBase:
    base_settings = load_standard_base_settings(settings.base_config)
    enabled = base_settings.model_copy(
        update={"feature": base_settings.feature.model_copy(update={"enabled": True})}
    )
    cache_name = "models--" + enabled.source.repository.replace("/", "--")
    snapshot = Path(HF_HUB_CACHE) / cache_name / "snapshots" / enabled.source.revision
    if not snapshot.is_dir():
        raise FileNotFoundError(
            "pinned offline model snapshot is missing: "
            f"{snapshot}; downloading is forbidden for this pilot"
        )
    return load_standard_base(
        enabled,
        snapshot_path=snapshot,
        device=settings.execution.device,
    )


def _decoder_loss(base: StandardBase, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = base.model.model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    return target_only_loss_from_hidden(base.model, hidden, batch.labels), hidden


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _diagonal_fisher(
    base: StandardBase,
    instances: Sequence[TaskInstance],
) -> tuple[dict[str, torch.Tensor], float]:
    """Estimate EWC Fisher as the mean of per-example squared gradients."""
    if not instances:
        raise ValueError("online EWC Fisher requires at least one example")
    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in base.model.named_parameters()
        if parameter.requires_grad
    }
    loss_sum = 0.0
    cast(torch.nn.Module, base.model).eval()
    for instance in instances:
        base.model.zero_grad(set_to_none=True)
        loss, _ = _decoder_loss(base, collate_task_instances(base, [instance]))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("online EWC Fisher loss is non-finite")
        torch.autograd.backward(loss)
        loss_sum += float(loss.detach().cpu().item())
        for name, parameter in base.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                fisher[name].add_(parameter.grad.detach().square())
    for value in fisher.values():
        value.div_(len(instances))
    base.model.zero_grad(set_to_none=True)
    if any(not bool(torch.isfinite(value).all().item()) for value in fisher.values()):
        raise FloatingPointError("online EWC Fisher contains a non-finite value")
    if not any(bool(torch.any(value > 0).item()) for value in fisher.values()):
        raise RuntimeError("online EWC Fisher is identically zero")
    return fisher, loss_sum / len(instances)


def _ewc_penalty(
    base: StandardBase,
    anchor: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    penalty_lambda: float,
) -> torch.Tensor:
    """Return the conventional half-scaled finite online-EWC penalty."""
    penalty = torch.zeros((), device=base.device, dtype=torch.float32)
    for name, parameter in base.model.named_parameters():
        if parameter.requires_grad:
            penalty = penalty + (fisher[name] * (parameter - anchor[name]).square()).sum()
    weighted = 0.5 * penalty_lambda * penalty
    if not bool(torch.isfinite(weighted).item()):
        raise FloatingPointError("online EWC penalty is non-finite")
    return weighted


def _configure_calibration_method(
    method: PilotMethod,
    base: StandardBase,
    settings: MpsPilotSettings,
) -> tuple[torch.optim.Optimizer | None, torch.nn.Module | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if method == "frozen_base":
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        return None, None, metadata
    if method in {"naive_sequential", "replay", "online_ewc"}:
        for parameter in base.model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(base.model.parameters(), lr=1e-4)
        if method == "replay":
            metadata["replay_capacity_records"] = 500
        if method == "online_ewc":
            baseline = load_standard_baseline_settings(settings.baseline_config)
            anchor = {
                name: parameter.detach().clone()
                for name, parameter in base.model.named_parameters()
                if parameter.requires_grad
            }
            fisher = {
                name: torch.zeros_like(parameter)
                for name, parameter in base.model.named_parameters()
                if parameter.requires_grad
            }
            metadata["ewc_state"] = (anchor, fisher)
            metadata["ewc_penalty_lambda"] = baseline.online_ewc.penalty_lambda
            metadata["ewc_state_bytes"] = sum(
                value.numel() * value.element_size()
                for value in (*anchor.values(), *fisher.values())
            )
        return optimizer, None, metadata
    baseline = load_standard_baseline_settings(settings.baseline_config)
    if method == "manual_lora":
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        layers = ManualLoRATrainer._inject(base.model, baseline.manual_lora)
        names = [
            f"resident_{index:04d}" for index in range(settings.calibration.resident_adapter_count)
        ]
        for layer in layers:
            for name in names:
                layer.add_adapter(name)
            layer.set_active(names[-1])
            for name, adapter in layer.adapters.items():
                for parameter in adapter.parameters():
                    parameter.requires_grad_(name == names[-1])
        optimizer = torch.optim.AdamW(_trainable_parameters(base.model), lr=1e-4)
        metadata["manual_lora_layer_count"] = len(layers)
        metadata["resident_adapters"] = len(names)
        return optimizer, None, metadata
    experts = load_standard_expert_settings(settings.expert_config)
    enabled_experts = experts.model_copy(
        update={"feature": experts.feature.model_copy(update={"enabled": True})}
    )
    bank = PeftExpertBank.start_candidate(base, enabled_experts, "resident_0000")
    router_row = SequenceTop1Router(
        base.settings.architecture.hidden_size,
        enabled_experts.router.confidence_threshold,
    ).to(base.device)
    router_row.add_candidate("resident_0000", "historical_0000")
    for index in range(1, settings.calibration.resident_adapter_count):
        previous = cast(str, bank.trainable_adapter)
        bank.trainable_adapter = None
        bank.mature_adapters.add(previous)
        bank.activate(previous)
        router_row.promote_candidate()
        name = f"resident_{index:04d}"
        bank.add_candidate(name)
        router_row.add_candidate(name, f"historical_{index:04d}")
        router_row.to(base.device)
    active = cast(str, bank.trainable_adapter)
    parameters = [*bank.adapter_parameters(active), *router_row.candidate_parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    metadata["bank"] = bank
    metadata["trainable_adapter_parameters"] = bank.actual_parameter_count(active)
    metadata["resident_adapters"] = len(bank.adapter_names)
    return optimizer, router_row, metadata


def _cleanup_base(base: StandardBase | None) -> None:
    if base is not None:
        del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_calibration_attempt(
    method: PilotMethod,
    stream_name: str,
    stream: _TaskStream,
    micro_batch: int,
    settings: MpsPilotSettings,
    quiet_record: dict[str, Any],
    *,
    checkpointing: bool = False,
) -> dict[str, Any]:
    """Run one isolated calibration attempt and retain failures as evidence."""
    accumulation: int | Literal["not_applicable"] = (
        "not_applicable"
        if method == "frozen_base"
        else settings.calibration.effective_batch_size // micro_batch
    )
    record: dict[str, Any] = {
        "method": method,
        "stream": stream_name,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "gradient_checkpointing": ("not_applicable" if method == "frozen_base" else checkpointing),
        "backward": "not_applicable" if method == "frozen_base" else True,
        "optimizer": "not_applicable" if method == "frozen_base" else "AdamW",
        "quiet_machine": quiet_record,
        "status": "running",
    }
    base: StandardBase | None = None
    started = time.perf_counter()
    try:
        seed_experiment(settings.calibration.seed)
        base = _base_for_pilot(settings)
        if base.device.type != "mps":
            raise RuntimeError("pilot calibration requires MPS and forbids CPU fallback")
        if checkpointing:
            if method == "frozen_base" or micro_batch != 1:
                raise ValueError("gradient checkpointing is only valid for training micro-batch 1")
            base.model.gradient_checkpointing_enable()
            base.model.config.use_cache = False
        count = micro_batch if method == "frozen_base" else 8
        selected, actual_lengths = longest_distinct_examples(base, stream, count)
        padded_length = max(actual_lengths)
        batches = [
            collate_task_instances(base, selected[offset : offset + micro_batch])
            for offset in range(0, len(selected), micro_batch)
        ]
        if any(batch.input_ids.shape[1] != padded_length for batch in batches):
            eos = cast(int, base.tokenizer.eos_token_id)
            padded: list[CausalBatch] = []
            for batch in batches:
                extra = padded_length - batch.input_ids.shape[1]
                padded.append(
                    CausalBatch(
                        input_ids=torch.nn.functional.pad(batch.input_ids, (0, extra), value=eos),
                        attention_mask=torch.nn.functional.pad(
                            batch.attention_mask, (0, extra), value=0
                        ),
                        labels=torch.nn.functional.pad(batch.labels, (0, extra), value=-100),
                        task_ids=batch.task_ids,
                    )
                )
            batches = padded
        record["actual_sequence_lengths"] = actual_lengths
        record["padded_sequence_length"] = padded_length
        optimizer, router_row, method_metadata = _configure_calibration_method(
            method, base, settings
        )
        record["trainable_parameters"] = sum(
            parameter.numel() for parameter in base.model.parameters() if parameter.requires_grad
        ) + (0 if router_row is None else sum(p.numel() for p in router_row.parameters()))
        record["method_state"] = {
            key: value for key, value in method_metadata.items() if key not in {"bank", "ewc_state"}
        }
        losses: list[float] = []
        step_times: list[float] = []
        current_samples: list[int] = []
        driver_samples: list[int] = []
        before_pressure = system_free_memory_percentage()
        before_swap = system_swap_used_bytes()
        sampler = MemoryPeakSampler(
            base.device,
            settings.calibration.memory_sample_interval_seconds,
            swap_sample_interval_seconds=0.5,
        )
        total_steps = settings.calibration.warmup_steps + settings.calibration.measured_steps
        with sampler:
            if method == "frozen_base":
                cast(torch.nn.Module, base.model).eval()
                with torch.no_grad():
                    for index in range(total_steps):
                        _synchronize(base.device)
                        step_started = time.perf_counter()
                        loss, _ = _decoder_loss(base, batches[0])
                        _synchronize(base.device)
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("frozen-base calibration loss is non-finite")
                        if index >= settings.calibration.warmup_steps:
                            losses.append(float(loss.detach().cpu().item()))
                            step_times.append(time.perf_counter() - step_started)
                        current, driver = _mps_values()
                        current_samples.append(current)
                        driver_samples.append(driver)
                    max_answer_tokens = max(
                        len(
                            cast(
                                list[int],
                                base.tokenizer.encode(
                                    item.training_target(), add_special_tokens=False
                                ),
                            )
                        )
                        for item in _all_stream_instances(stream)
                    )
                    generation_started = time.perf_counter()
                    generation = generate_task_instance_answer(
                        base, selected[0], max_new_tokens=max_answer_tokens
                    )
                    _synchronize(base.device)
                    record["generation"] = {
                        "maximum_answer_tokens": max_answer_tokens,
                        "generated_tokens": len(generation.generated_token_ids),
                        "stop_reason": generation.stop_reason,
                        "wall_time_seconds": time.perf_counter() - generation_started,
                    }
            else:
                if optimizer is None:
                    raise RuntimeError("training calibration did not construct an optimizer")
                cast(torch.nn.Module, base.model).train()
                if method == "online_ewc":
                    anchor, fisher = cast(
                        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
                        method_metadata["ewc_state"],
                    )
                    measured_fisher, fisher_loss = _diagonal_fisher(base, selected)
                    for name, value in measured_fisher.items():
                        fisher[name].copy_(value)
                    optimizer.zero_grad(set_to_none=True)
                    record["method_state"]["fisher_initialization_loss"] = fisher_loss
                    record["method_state"]["fisher_examples"] = len(selected)
                    record["method_state"]["fisher_estimator"] = (
                        "mean_per_example_squared_gradient"
                    )
                for index in range(total_steps):
                    optimizer.zero_grad(set_to_none=True)
                    _synchronize(base.device)
                    step_started = time.perf_counter()
                    loss_sum = 0.0
                    for batch in batches:
                        loss, hidden = _decoder_loss(base, batch)
                        if router_row is not None:
                            pooled = hidden.detach().mean(dim=1)
                            router = cast(SequenceTop1Router, router_row)
                            labels = torch.full(
                                (pooled.shape[0],),
                                router.num_experts - 1,
                                dtype=torch.long,
                                device=base.device,
                            )
                            router_loss = torch.nn.functional.cross_entropy(
                                router.classification_logits(pooled), labels + 1
                            )
                            loss = loss + router_loss
                        if method == "online_ewc":
                            anchor, fisher = cast(
                                tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
                                method_metadata["ewc_state"],
                            )
                            loss = loss + _ewc_penalty(
                                base,
                                anchor,
                                fisher,
                                cast(float, method_metadata["ewc_penalty_lambda"]),
                            )
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError(f"{method} calibration loss is non-finite")
                        torch.autograd.backward(loss / len(batches))
                        loss_sum += float(loss.detach().cpu().item())
                    optimizer.step()
                    _synchronize(base.device)
                    if index >= settings.calibration.warmup_steps:
                        losses.append(loss_sum / len(batches))
                        step_times.append(time.perf_counter() - step_started)
                    current, driver = _mps_values()
                    current_samples.append(current)
                    driver_samples.append(driver)
        memory = sampler.record()
        if not current_samples or not driver_samples:
            raise RuntimeError("calibration produced no MPS allocation samples")
        memory["mps_current_peak_bytes"] = max(current_samples)
        memory["mps_driver_step_peak_bytes"] = max(driver_samples)
        hard_peak = max(
            cast(int, memory["process_rss_peak_bytes"]),
            cast(int, memory.get("mps_driver_peak_bytes") or 0),
        )
        tokens_per_step = padded_length * count
        median_step = float(np.median(step_times))
        swap_growth = max(0, system_swap_used_bytes() - before_swap)
        throughput_compromised = (
            swap_growth > settings.calibration.swap_throughput_reporting_threshold_bytes
        )
        record.update(
            {
                "status": "completed",
                "accepted": hard_peak <= settings.calibration.memory_cap_bytes,
                "losses": losses,
                "finite_losses": all(math.isfinite(value) for value in losses),
                "step_times_seconds": step_times,
                "median_step_time_seconds": median_step,
                "tokens_per_second": (
                    None if throughput_compromised else tokens_per_step / median_step
                ),
                "throughput_compromised": throughput_compromised,
                "system_free_memory_before_percent": before_pressure,
                "system_free_memory_after_percent": system_free_memory_percentage(),
                "system_swap_growth_bytes": swap_growth,
                "memory": memory,
                "no_cpu_fallback": all(
                    parameter.device.type == "mps" for parameter in base.model.parameters()
                ),
            }
        )
        if not record["no_cpu_fallback"]:
            raise RuntimeError("calibration detected CPU parameter fallback")
    except Exception as error:
        out_of_memory = "out of memory" in str(error).lower()
        record.update(
            {
                "status": "failed",
                "accepted": False,
                "failure": _redacted_failure(error, out_of_memory=out_of_memory),
            }
        )
    finally:
        record["wall_time_seconds"] = time.perf_counter() - started
        _cleanup_base(base)
    return record


class _BoundedPairStream:
    def __init__(
        self,
        source: _TaskStream,
        indices: tuple[int, int],
        eval_count: int,
        template_ood_count: int,
    ) -> None:
        if indices[0] == indices[1] or max(indices) >= len(source.families):
            raise ValueError("pilot task-pair indices must name two distinct real tasks")
        self.families = [source.families[index] for index in indices]
        self.num_train_per_task = min(
            len(source.get_split(family.family_id).train) for family in self.families
        )
        self.num_val_per_task = eval_count
        self.num_test_per_task = eval_count
        self._splits: dict[str, TaskStreamSplit] = {}
        self._template_ood_splits: dict[str, list[TaskInstance]] = {}
        for family in self.families:
            split = source.get_split(family.family_id)
            if not split.train or len(split.test) < eval_count:
                raise ValueError(f"pilot task {family.family_id} lacks configured real examples")
            self._splits[family.family_id] = TaskStreamSplit(
                train=list(split.train),
                val=list(split.val[:eval_count]),
                test=list(split.test[:eval_count]),
            )
            get_ood = getattr(source, "get_template_ood_split", None)
            ood = list(get_ood(family.family_id)) if callable(get_ood) else []
            if ood and len(ood) < template_ood_count:
                raise ValueError(
                    f"pilot task {family.family_id} lacks configured template-OOD examples"
                )
            self._template_ood_splits[family.family_id] = ood[:template_ood_count]

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]

    def get_template_ood_split(self, family_id: str) -> list[TaskInstance]:
        return self._template_ood_splits[family_id]


def _evaluate_pair(
    stream: _BoundedPairStream,
    predict_many: Callable[[Sequence[TaskInstance]], Sequence[str]],
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    for family in stream.families:
        instances = stream.get_split(family.family_id).test
        passed = _evaluate_instances(family, instances, predict_many, batch_size)
        values.append(passed / len(instances))
    return values


def _evaluate_instances(
    family: TaskFamily,
    instances: Sequence[TaskInstance],
    predict_many: Callable[[Sequence[TaskInstance]], Sequence[str]],
    batch_size: int,
) -> int:
    passed = 0
    for offset in range(0, len(instances), batch_size):
        selected = instances[offset : offset + batch_size]
        predictions = predict_many(selected)
        if len(predictions) != len(selected):
            raise RuntimeError("batched predictor returned the wrong number of outputs")
        for instance, raw in zip(selected, predictions, strict=True):
            outcome = ContinualEvalHarness.verify_generated_output(family, instance, raw)
            passed += int(outcome.passed)
    return passed


def _evaluate_template_ood_pair(
    stream: _BoundedPairStream,
    predict_many: Callable[[Sequence[TaskInstance]], Sequence[str]],
    batch_size: int,
) -> list[float] | None:
    values: list[float] = []
    for family in stream.families:
        instances = stream.get_template_ood_split(family.family_id)
        if not instances:
            return None
        passed = _evaluate_instances(family, instances, predict_many, batch_size)
        values.append(passed / len(instances))
    return values


def _evaluate_one_task(
    stream: _BoundedPairStream,
    task_idx: int,
    predict_many: Callable[[Sequence[TaskInstance]], Sequence[str]],
    batch_size: int,
) -> float:
    family = stream.families[task_idx]
    instances = stream.get_split(family.family_id).test
    passed = _evaluate_instances(family, instances, predict_many, batch_size)
    return passed / len(instances)


def _pair_metrics(
    matrix: list[list[float]],
    frozen: list[float],
    *,
    template_ood_matrix: list[list[float]] | None = None,
    template_ood_frozen: list[float] | None = None,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    forgetting = compute_max_forgetting(values, ["pair_task_0", "pair_task_1"])
    finite_forgetting = [value for value in forgetting.values() if value is not None]
    result: dict[str, Any] = {
        "accuracy_matrix": matrix,
        "frozen_base_accuracy": frozen,
        "per_task_accuracy_after_training": matrix[-1],
        "new_task_acquisition": float(
            np.mean([matrix[index][index] for index in range(len(matrix))])
        ),
        "backward_transfer": compute_backward_transfer(values),
        "forgetting": forgetting,
        "aggregate_forgetting": max(finite_forgetting, default=0.0),
        "final_average": compute_average_accuracy(values),
    }
    if template_ood_matrix is not None and template_ood_frozen is not None:
        result["template_ood_accuracy_matrix"] = template_ood_matrix
        result["template_ood_frozen_base_accuracy"] = template_ood_frozen
        result["template_ood_per_task_accuracy_after_training"] = template_ood_matrix[-1]
        result["template_ood_new_task_acquisition"] = float(
            np.mean(
                [
                    template_ood_matrix[index][index]
                    for index in range(len(template_ood_matrix))
                ]
            )
        )
        result["template_ood_final_average"] = float(
            np.mean(template_ood_matrix[-1])
        )
    return result


def _base_predict_many(
    base: StandardBase,
    instances: Sequence[TaskInstance],
    max_new_tokens: int,
) -> list[str]:
    return [
        result.text
        for result in generate_task_instance_answers(
            base,
            instances,
            max_new_tokens=max_new_tokens,
        )
    ]


def _losses(base: StandardBase, instances: Sequence[TaskInstance]) -> np.ndarray:
    values: list[float] = []
    cast(torch.nn.Module, base.model).eval()
    with torch.no_grad():
        for instance in instances:
            batch = collate_task_instances(base, [instance])
            loss, _ = _decoder_loss(base, batch)
            values.append(float(loss.detach().cpu().item()))
    return np.asarray(values, dtype=np.float64)


def _prompt_features(base: StandardBase, instances: Sequence[TaskInstance]) -> torch.Tensor:
    encoded = [
        cast(list[int], base.tokenizer.encode(item.prompt, add_special_tokens=False))
        for item in instances
    ]
    maximum = max(map(len, encoded))
    eos = cast(int, base.tokenizer.eos_token_id)
    ids = torch.tensor(
        [row + [eos] * (maximum - len(row)) for row in encoded],
        dtype=torch.long,
        device=base.device,
    )
    mask = torch.tensor(
        [[1] * len(row) + [0] * (maximum - len(row)) for row in encoded],
        dtype=torch.long,
        device=base.device,
    )
    return masked_mean_frozen_embeddings(base.model, ids, mask)


def _train_adapter_steps(
    base: StandardBase,
    bank: PeftExpertBank,
    instances: Sequence[TaskInstance],
    max_steps: int,
    max_target_tokens: int,
    micro_batch: int,
    effective_batch: int,
    learning_rate: float,
) -> dict[str, Any]:
    adapter_name = cast(str, bank.trainable_adapter)
    bank.activate(adapter_name)
    return _train_capped_steps(
        base,
        bank.adapter_parameters(adapter_name),
        instances,
        max_steps=max_steps,
        max_target_tokens=max_target_tokens,
        micro_batch=micro_batch,
        effective_batch=effective_batch,
        learning_rate=learning_rate,
        failure_label="Aptivra pilot adapter",
    )


def _train_capped_steps(
    base: StandardBase,
    parameters: Sequence[torch.nn.Parameter],
    instances: Sequence[TaskInstance],
    *,
    max_steps: int,
    max_target_tokens: int,
    micro_batch: int,
    effective_batch: int,
    learning_rate: float,
    failure_label: str,
) -> dict[str, Any]:
    if not instances:
        raise ValueError("capped training requires at least one example")
    if effective_batch % micro_batch:
        raise ValueError("micro-batch must divide the effective batch")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    losses: list[float] = []
    target_tokens = 0
    optimizer_steps = 0
    consumed_examples = 0
    accumulation = effective_batch // micro_batch
    micro_batches = [
        instances[offset : offset + micro_batch]
        for offset in range(0, len(instances), micro_batch)
    ]
    stopped_by = "one_epoch"
    for group_start in range(0, len(micro_batches), accumulation):
        if optimizer_steps >= max_steps:
            stopped_by = "optimizer_step_cap"
            break
        group_instances = micro_batches[group_start : group_start + accumulation]
        batches = [collate_task_instances(base, list(selected)) for selected in group_instances]
        group_target_tokens = sum(
            int((batch.labels[:, 1:] != -100).sum().item()) for batch in batches
        )
        if target_tokens + group_target_tokens > max_target_tokens:
            stopped_by = "target_token_cap"
            break
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for batch in batches:
            loss, _ = _decoder_loss(base, batch)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"{failure_label} loss is non-finite")
            torch.autograd.backward(loss / len(batches))
            step_loss += float(loss.detach().cpu().item())
        optimizer.step()
        losses.append(step_loss / len(batches))
        target_tokens += group_target_tokens
        consumed_examples += sum(len(selected) for selected in group_instances)
        optimizer_steps += 1
    return {
        "losses": losses,
        "target_tokens": target_tokens,
        "optimizer_steps": optimizer_steps,
        "available_examples": len(instances),
        "consumed_examples": consumed_examples,
        "stopped_by": stopped_by,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
    }


def _seeded_task_order(
    instances: Sequence[TaskInstance], seed: int, task_idx: int
) -> list[TaskInstance]:
    ordered = list(instances)
    derived = int.from_bytes(
        hashlib.sha256(f"{seed}:{task_idx}".encode()).digest()[:8], "big"
    )
    random.Random(derived).shuffle(ordered)
    return ordered


def _remove_candidate(bank: PeftExpertBank, router: SequenceTop1Router) -> None:
    candidate = bank.trainable_adapter
    if candidate is None:
        raise RuntimeError("rollback requested without a trainable adapter")
    bank.model.delete_adapter(candidate)
    bank.trainable_adapter = None
    router.weights = torch.nn.ParameterList(tuple(router.weights[:-1]))
    router.biases = torch.nn.ParameterList(tuple(router.biases[:-1]))
    router.expert_ids.pop()
    router.task_families.pop()
    router.frozen_rows.pop()
    if bank.mature_adapters:
        bank.activate(sorted(bank.mature_adapters)[-1])
    else:
        bank.deactivate()


def _add_router_candidate_on_device(
    router: SequenceTop1Router,
    expert_id: str,
    task_family: str,
    device: torch.device,
) -> None:
    """Add a dynamically allocated row, then place every row on the run device."""
    router.add_candidate(expert_id, task_family)
    router.to(device)


def _router_replay_batch(
    base: StandardBase,
    replay: Sequence[TaskInstance],
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a correctly shaped empty batch when no promoted replay exists yet."""
    examples = replay[:limit]
    if not examples:
        return (
            torch.empty((0, base.settings.architecture.hidden_size), device=base.device),
            torch.empty((0,), dtype=torch.long, device=base.device),
        )
    return (
        _prompt_features(base, examples),
        torch.zeros(len(examples), dtype=torch.long, device=base.device),
    )


def _routing_decision_rates(
    routed_expert_ids: Sequence[str | None],
    expected_expert_ids: Sequence[str | None],
) -> dict[str, float | int]:
    """Calculate routing denominators without treating fallback as coverage."""
    if len(routed_expert_ids) != len(expected_expert_ids) or not routed_expert_ids:
        raise ValueError("routing decisions require equal non-empty vectors")
    eligible = [index for index, expected in enumerate(expected_expert_ids) if expected is not None]
    fallback = [index for index, expected in enumerate(expected_expert_ids) if expected is None]
    routed_eligible = [index for index in eligible if routed_expert_ids[index] is not None]
    eligible_misroutes = [
        index for index in routed_eligible if routed_expert_ids[index] != expected_expert_ids[index]
    ]
    routed = [index for index, routed_id in enumerate(routed_expert_ids) if routed_id is not None]
    misroutes = [
        index for index in routed if routed_expert_ids[index] != expected_expert_ids[index]
    ]
    false_routes = [index for index in fallback if routed_expert_ids[index] is not None]
    return {
        "total_prompts": len(routed_expert_ids),
        "eligible_prompts": len(eligible),
        "fallback_prompts": len(fallback),
        "routed_eligible_prompts": len(routed_eligible),
        "coverage": 0.0 if not eligible else len(routed_eligible) / len(eligible),
        "misroute_rate": (0.0 if not routed else len(misroutes) / len(routed)),
        "eligible_misroute_rate": (
            0.0 if not routed_eligible else len(eligible_misroutes) / len(routed_eligible)
        ),
        "fallback_false_route_rate": (0.0 if not fallback else len(false_routes) / len(fallback)),
    }


def _routing_probe(
    base: StandardBase,
    bank: PeftExpertBank,
    router: SequenceTop1Router,
    stream: _BoundedPairStream,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    """Measure learned routing against task-ID oracle on real held-out prompts."""
    expected_by_family = dict(zip(router.task_families, router.expert_ids, strict=True))
    instances = [
        instance
        for family in stream.families
        for instance in stream.get_split(family.family_id).test
    ]
    features = _prompt_features(base, instances)
    decisions = router.route(features)
    routed_ids = [decision.expert_id for decision in decisions]
    expected_ids = [expected_by_family.get(instance.task_family) for instance in instances]
    rates = _routing_decision_rates(routed_ids, expected_ids)
    routed_outputs = _predict_with_adapter_assignments(
        base,
        bank,
        instances,
        routed_ids,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    oracle_outputs = list(routed_outputs)
    oracle_indices = [
        index
        for index, (routed_id, expected_id) in enumerate(
            zip(routed_ids, expected_ids, strict=True)
        )
        if expected_id is not None and routed_id != expected_id
    ]
    if oracle_indices:
        generated = _predict_with_adapter_assignments(
            base,
            bank,
            [instances[index] for index in oracle_indices],
            [cast(str, expected_ids[index]) for index in oracle_indices],
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        for index, output in zip(oracle_indices, generated, strict=True):
            oracle_outputs[index] = output
    routed_passes = 0
    oracle_passes = 0
    eligible_count = 0
    families = {family.family_id: family for family in stream.families}
    for instance, routed, oracle, expected_id in zip(
        instances, routed_outputs, oracle_outputs, expected_ids, strict=True
    ):
        if expected_id is None:
            continue
        family = families[instance.task_family]
        eligible_count += 1
        routed_passes += int(
            ContinualEvalHarness.verify_generated_output(family, instance, routed).passed
        )
        oracle_passes += int(
            ContinualEvalHarness.verify_generated_output(family, instance, oracle).passed
        )
    if eligible_count == 0:
        raise RuntimeError("routing probe has no task-ID-eligible prompts")
    routed_accuracy = routed_passes / eligible_count
    oracle_accuracy = oracle_passes / eligible_count
    return {
        **rates,
        "routed_accuracy": routed_accuracy,
        "task_id_oracle_accuracy": oracle_accuracy,
        "task_id_oracle_gap": oracle_accuracy - routed_accuracy,
        "confidence_threshold": router.confidence_threshold,
    }


def _predict_with_adapter_assignments(
    base: StandardBase,
    bank: PeftExpertBank,
    instances: Sequence[TaskInstance],
    adapter_ids: Sequence[str | None],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    if len(instances) != len(adapter_ids):
        raise ValueError("adapter assignments must align with prediction instances")
    grouped: dict[str | None, list[tuple[int, TaskInstance]]] = {}
    for index, (instance, adapter_id) in enumerate(zip(instances, adapter_ids, strict=True)):
        grouped.setdefault(adapter_id, []).append((index, instance))
    results = [""] * len(instances)
    for adapter_id, members in grouped.items():
        if adapter_id is None:
            bank.deactivate()
        else:
            bank.activate(adapter_id)
        for offset in range(0, len(members), batch_size):
            selected = members[offset : offset + batch_size]
            outputs = _base_predict_many(
                base,
                [instance for _, instance in selected],
                max_new_tokens,
            )
            for (index, _), output in zip(selected, outputs, strict=True):
                results[index] = output
    return results


def _run_aptivra_pair(
    base: StandardBase,
    stream: _BoundedPairStream,
    settings: MpsPilotSettings,
    *,
    stream_name: str,
    seed: int,
) -> dict[str, Any]:
    expert = load_standard_expert_settings(settings.expert_config)
    expert = expert.model_copy(
        update={"feature": expert.feature.model_copy(update={"enabled": True})}
    )
    repeat = settings.repeatability
    first_family = stream.families[0].family_id
    bank = PeftExpertBank.start_candidate(base, expert, "pilot_0000")
    router = SequenceTop1Router(
        base.settings.architecture.hidden_size, expert.router.confidence_threshold
    )
    _add_router_candidate_on_device(router, "pilot_0000", first_family, base.device)
    replay: list[TaskInstance] = []
    verified_memory_ids: set[str] = set()
    promotions: list[dict[str, Any]] = []

    batch_size = getattr(repeat.evaluation_batch_sizes, stream_name)
    micro_batch = getattr(repeat.micro_batches, stream_name).aptivra_full

    def routed_predict_many(instances: Sequence[TaskInstance]) -> list[str]:
        if not instances:
            return []
        if router.num_experts == 0:
            bank.deactivate()
            return _base_predict_many(base, instances, repeat.max_new_tokens)
        decisions = router.route(_prompt_features(base, instances))
        expert_ids = [
            (
                decision.expert_id
                if decision.route_status == "routed" and decision.expert_id is not None
                else None
            )
            for decision in decisions
        ]
        return _predict_with_adapter_assignments(
            base,
            bank,
            instances,
            expert_ids,
            max_new_tokens=repeat.max_new_tokens,
            batch_size=batch_size,
        )

    bank.deactivate()
    frozen = _evaluate_pair(
        stream,
        lambda instances: _base_predict_many(base, instances, repeat.max_new_tokens),
        batch_size,
    )
    frozen_ood = _evaluate_template_ood_pair(
        stream,
        lambda instances: _base_predict_many(base, instances, repeat.max_new_tokens),
        batch_size,
    )
    matrix: list[list[float]] = []
    template_ood_matrix: list[list[float]] | None = [] if frozen_ood is not None else None
    training: list[dict[str, Any]] = []
    for task_idx, family in enumerate(stream.families):
        adapter_name = f"pilot_{task_idx:04d}"
        if task_idx > 0:
            bank.add_candidate(adapter_name)
            _add_router_candidate_on_device(router, adapter_name, family.family_id, base.device)
        train = _seeded_task_order(
            stream.get_split(family.family_id).train, seed, task_idx
        )
        combined = [*train]
        if replay:
            combined.extend(replay[: len(train) // 2])
        validation = stream.get_split(family.family_id).val
        bank.deactivate()
        current_acc = _evaluate_one_task(
            stream, task_idx, routed_predict_many, batch_size
        )
        regression = [
            item
            for seen in stream.families[: task_idx + 1]
            for item in stream.get_split(seen.family_id).val
        ][: max(5, repeat.evaluation_examples_per_task)]
        current_losses = _losses(base, regression)
        train_result = _train_adapter_steps(
            base,
            bank,
            combined,
            settings.cycle_caps.adapter_optimizer_steps,
            settings.cycle_caps.adapter_target_tokens,
            micro_batch,
            repeat.effective_batch_size,
            expert.consolidation.adapter_learning_rate,
        )
        training.append(train_result)
        current_features = _prompt_features(base, train[:8])
        replay_features, replay_labels = _router_replay_batch(
            base, replay, expert.router.replay_examples_per_expert
        )
        router_result, _ = router.fit_candidate(
            current_features,
            replay_features,
            replay_labels,
            learning_rate=expert.router.learning_rate,
            weight_decay=expert.router.weight_decay,
            adam_beta1=expert.router.adam_beta1,
            adam_beta2=expert.router.adam_beta2,
            adam_epsilon=expert.router.adam_epsilon,
            max_steps=min(
                expert.router.max_optimizer_steps,
                settings.cycle_caps.router_optimizer_steps,
            ),
        )
        routing_probe = _routing_probe(
            base,
            bank,
            router,
            stream,
            repeat.max_new_tokens,
            batch_size,
        )
        candidate_acc = _evaluate_one_task(
            stream, task_idx, routed_predict_many, batch_size
        )
        candidate_losses = _losses(base, regression)
        promotion = PromotionGate(
            expert.consolidation.promotion_alpha,
            minimum_regression_samples=expert.consolidation.minimum_regression_samples,
            minimum_validation_gain=expert.consolidation.minimum_validation_gain,
            loss_noninferiority_margin=expert.consolidation.loss_noninferiority_margin,
        ).evaluate_measurements(current_acc, candidate_acc, current_losses, candidate_losses)
        promotions.append(
            {
                **asdict(promotion),
                "adapter_training": train_result,
                "router": asdict(router_result),
                "routing_probe": routing_probe,
            }
        )
        if promotion.promoted:
            candidate = cast(str, bank.trainable_adapter)
            bank.trainable_adapter = None
            bank.mature_adapters.add(candidate)
            bank.activate(candidate)
            router.promote_candidate()
            replay.extend(train)
            for offset in range(0, len(validation), batch_size):
                selected = validation[offset : offset + batch_size]
                predictions = routed_predict_many(selected)
                for instance, prediction in zip(selected, predictions, strict=True):
                    outcome = ContinualEvalHarness.verify_generated_output(
                        family, instance, prediction
                    )
                    if outcome.passed:
                        verified_memory_ids.add(instance.task_id)
        else:
            _remove_candidate(bank, router)
        matrix.append(_evaluate_pair(stream, routed_predict_many, batch_size))
        if template_ood_matrix is not None:
            template_values = _evaluate_template_ood_pair(
                stream, routed_predict_many, batch_size
            )
            if template_values is None:
                raise RuntimeError("template-OOD split disappeared during a pilot cell")
            template_ood_matrix.append(template_values)
    return {
        **_pair_metrics(
            matrix,
            frozen,
            template_ood_matrix=template_ood_matrix,
            template_ood_frozen=frozen_ood,
        ),
        "optimizer_steps": sum(int(item["optimizer_steps"]) for item in training),
        "target_tokens": sum(int(item["target_tokens"]) for item in training),
        "training": training,
        "promotions": promotions,
        "verified_memory_count": len(verified_memory_ids),
        "replay_records": len(replay),
        "router_experts": router.num_experts,
        "evaluation_examples_per_task": repeat.evaluation_examples_per_task,
        "template_ood_examples_per_task": (
            repeat.template_ood_examples_per_task if frozen_ood is not None else 0
        ),
    }


def run_repeatability_cell(
    method: Literal["naive_sequential", "aptivra_full"],
    stream_name: str,
    source: _TaskStream,
    pair_indices: tuple[int, int],
    seed: int,
    repeat_index: int,
    settings: MpsPilotSettings,
    quiet_record: dict[str, Any],
) -> dict[str, Any]:
    """Run one bounded real-model pair cell from pristine cached weights."""
    started = time.perf_counter()
    record: dict[str, Any] = {
        "method": method,
        "stream": stream_name,
        "seed": seed,
        "repeat_index": repeat_index,
        "quiet_machine": quiet_record,
        "status": "running",
    }
    base: StandardBase | None = None
    try:
        seed_experiment(seed)
        pair = _BoundedPairStream(
            source,
            pair_indices,
            settings.repeatability.evaluation_examples_per_task,
            settings.repeatability.template_ood_examples_per_task,
        )
        base = _base_for_pilot(settings)
        sampler = MemoryPeakSampler(
            base.device,
            settings.calibration.memory_sample_interval_seconds,
            swap_sample_interval_seconds=0.5,
        )
        with sampler:
            if method == "aptivra_full":
                metrics = _run_aptivra_pair(
                    base,
                    pair,
                    settings,
                    stream_name=stream_name,
                    seed=seed,
                )
            else:
                for parameter in base.model.parameters():
                    parameter.requires_grad_(True)
                repeat = settings.repeatability
                batch_size = getattr(repeat.evaluation_batch_sizes, stream_name)
                def predict_many(instances: Sequence[TaskInstance]) -> list[str]:
                    return _base_predict_many(base, instances, repeat.max_new_tokens)
                frozen = _evaluate_pair(
                    pair,
                    predict_many,
                    batch_size,
                )
                frozen_ood = _evaluate_template_ood_pair(
                    pair,
                    predict_many,
                    batch_size,
                )
                matrix: list[list[float]] = []
                template_ood_matrix: list[list[float]] | None = (
                    [] if frozen_ood is not None else None
                )
                training: list[dict[str, Any]] = []
                micro_batch = getattr(
                    repeat.micro_batches, stream_name
                ).naive_sequential
                for task_idx, family in enumerate(pair.families):
                    train = _seeded_task_order(
                        pair.get_split(family.family_id).train, seed, task_idx
                    )
                    training.append(
                        _train_capped_steps(
                            base,
                            tuple(base.model.parameters()),
                            train,
                            max_steps=settings.cycle_caps.adapter_optimizer_steps,
                            max_target_tokens=settings.cycle_caps.adapter_target_tokens,
                            micro_batch=micro_batch,
                            effective_batch=repeat.effective_batch_size,
                            learning_rate=1e-4,
                            failure_label="naive pilot",
                        )
                    )
                    matrix.append(
                        _evaluate_pair(
                            pair,
                            predict_many,
                            batch_size,
                        )
                    )
                    if template_ood_matrix is not None:
                        template_values = _evaluate_template_ood_pair(
                            pair,
                            predict_many,
                            batch_size,
                        )
                        if template_values is None:
                            raise RuntimeError(
                                "template-OOD split disappeared during a pilot cell"
                            )
                        template_ood_matrix.append(template_values)
                metrics = {
                    **_pair_metrics(
                        matrix,
                        frozen,
                        template_ood_matrix=template_ood_matrix,
                        template_ood_frozen=frozen_ood,
                    ),
                    "optimizer_steps": sum(
                        int(item["optimizer_steps"]) for item in training
                    ),
                    "target_tokens": sum(int(item["target_tokens"]) for item in training),
                    "training": training,
                    "evaluation_examples_per_task": repeat.evaluation_examples_per_task,
                    "template_ood_examples_per_task": (
                        repeat.template_ood_examples_per_task
                        if frozen_ood is not None
                        else 0
                    ),
                }
            learned = [
                float(cast(list[list[float]], metrics["accuracy_matrix"])[index][index])
                > float(cast(list[float], metrics["frozen_base_accuracy"])[index])
                for index in range(len(pair.families))
            ]
            metrics["trained_task_above_frozen_base"] = learned
            metrics["demonstrably_learned"] = all(learned)
        record.update(
            {
                "status": "completed",
                "metrics": metrics,
                "memory": sampler.record(),
            }
        )
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "failure": _redacted_failure(error),
            }
        )
    finally:
        record["wall_time_seconds"] = time.perf_counter() - started
        _cleanup_base(base)
    return record


def run_evaluation_profile_cell(
    stream_name: str,
    source: _TaskStream,
    pair_indices: tuple[int, int],
    settings: MpsPilotSettings,
    quiet_record: dict[str, Any],
) -> dict[str, Any]:
    """Prove scalar/batched identity and measure the fixed-set evaluation speedup."""
    started = time.perf_counter()
    record: dict[str, Any] = {
        "stream": stream_name,
        "quiet_machine": quiet_record,
        "status": "running",
    }
    base: StandardBase | None = None
    try:
        seed_experiment(settings.calibration.seed)
        pair = _BoundedPairStream(
            source,
            pair_indices,
            settings.repeatability.evaluation_examples_per_task,
            settings.repeatability.template_ood_examples_per_task,
        )
        base = _base_for_pilot(settings)
        cast(torch.nn.Module, base.model).eval()
        per_family = max(1, settings.repeatability.profile_examples // len(pair.families))
        instances = [
            instance
            for family in pair.families
            for instance in pair.get_split(family.family_id).test[:per_family]
        ][: settings.repeatability.profile_examples]
        _synchronize(base.device)
        scalar_started = time.perf_counter()
        scalar = tuple(
            generate_task_instance_answer(
                base,
                instance,
                max_new_tokens=settings.repeatability.max_new_tokens,
            )
            for instance in instances
        )
        _synchronize(base.device)
        scalar_seconds = time.perf_counter() - scalar_started
        _synchronize(base.device)
        batched_started = time.perf_counter()
        configured_batch_size = getattr(
            settings.repeatability.evaluation_batch_sizes, stream_name
        )
        batched = tuple(
            result
            for offset in range(0, len(instances), configured_batch_size)
            for result in generate_task_instance_answers(
                base,
                instances[offset : offset + configured_batch_size],
                max_new_tokens=settings.repeatability.max_new_tokens,
            )
        )
        _synchronize(base.device)
        batched_seconds = time.perf_counter() - batched_started
        if scalar != batched:
            mismatch = next(
                index
                for index, (left, right) in enumerate(zip(scalar, batched, strict=True))
                if left != right
            )
            raise RuntimeError(
                f"batched generation changed the fixed-set output at index {mismatch}"
            )
        digest = hashlib.sha256()
        for result in batched:
            digest.update(
                json.dumps(
                    {
                        "ids": result.generated_token_ids,
                        "stop_reason": result.stop_reason,
                        "matched_terminator": result.matched_terminator,
                        "text_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
        record.update(
            {
                "status": "completed",
                "examples": len(instances),
                "configured_batch_size": configured_batch_size,
                "max_new_tokens": settings.repeatability.max_new_tokens,
                "scalar_seconds": scalar_seconds,
                "batched_seconds": batched_seconds,
                "scalar_seconds_per_example": scalar_seconds / len(instances),
                "batched_seconds_per_example": batched_seconds / len(instances),
                "speedup": scalar_seconds / batched_seconds,
                "output_identity": True,
                "output_sha256": digest.hexdigest(),
            }
        )
    except Exception as error:
        record.update({"status": "failed", "failure": _redacted_failure(error)})
    finally:
        record["wall_time_seconds"] = time.perf_counter() - started
        _cleanup_base(base)
    return record


def noise_floor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize repeatability and seed SD only for demonstrably learning runs."""
    complete = [record for record in records if record.get("status") == "completed"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in complete:
        grouped.setdefault((str(record["method"]), str(record["stream"])), []).append(record)
    result: dict[str, Any] = {}
    for (method, stream), cells in grouped.items():
        ordered = sorted(cells, key=lambda item: int(item["repeat_index"]))
        if len(ordered) != 5:
            raise RuntimeError(f"noise floor is incomplete for {method}/{stream}")
        repeat_indices = [int(cell["repeat_index"]) for cell in ordered]
        if repeat_indices != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"noise floor repeat indices are invalid for {method}/{stream}")
        seeds = [int(cell["seed"]) for cell in ordered]
        if seeds[0] != seeds[1] or len({seeds[0], seeds[2], seeds[3], seeds[4]}) != 4:
            raise RuntimeError(f"noise floor seed design is invalid for {method}/{stream}")
        metric_names = (
            "backward_transfer",
            "aggregate_forgetting",
            "final_average",
            "new_task_acquisition",
        )
        same: dict[str, float] = {}
        for name in metric_names:
            values = [float(cast(Mapping[str, Any], cell["metrics"])[name]) for cell in ordered]
            same[name] = abs(values[1] - values[0])
        task_values = [
            cast(
                list[float],
                cast(Mapping[str, Any], cell["metrics"])["per_task_accuracy_after_training"],
            )
            for cell in ordered
        ]
        same["per_task_accuracy"] = max(
            abs(right - left) for left, right in zip(task_values[0], task_values[1], strict=True)
        )
        learning = [
            bool(cast(Mapping[str, Any], cell["metrics"])["demonstrably_learned"])
            for cell in ordered
        ]
        summary: dict[str, Any] = {
            "status": "completed" if all(learning) else "non_learning",
            "learning_by_run": learning,
            "same_seed_absolute_variation": same,
        }
        if all(learning):
            unique_seed_cells = [ordered[0], *ordered[2:]]
            standard_deviation: dict[str, Any] = {
                name: float(
                    np.std(
                        [
                            float(cast(Mapping[str, Any], cell["metrics"])[name])
                            for cell in unique_seed_cells
                        ],
                        ddof=1,
                    )
                )
                for name in metric_names
            }
            standard_deviation["per_task_accuracy"] = [
                float(
                    np.std(
                        [
                            cast(
                                list[float],
                                cast(Mapping[str, Any], cell["metrics"])[
                                    "per_task_accuracy_after_training"
                                ],
                            )[task_idx]
                            for cell in unique_seed_cells
                        ],
                        ddof=1,
                    )
                )
                for task_idx in range(len(task_values[0]))
            ]
            first_metrics = cast(Mapping[str, Any], ordered[0]["metrics"])
            if "template_ood_final_average" in first_metrics:
                ood_names = (
                    "template_ood_final_average",
                    "template_ood_new_task_acquisition",
                )
                for name in ood_names:
                    same[name] = abs(
                        float(cast(Mapping[str, Any], ordered[1]["metrics"])[name])
                        - float(first_metrics[name])
                    )
                    standard_deviation[name] = float(
                        np.std(
                            [
                                float(cast(Mapping[str, Any], cell["metrics"])[name])
                                for cell in unique_seed_cells
                            ],
                            ddof=1,
                        )
                    )
            summary["seed_to_seed_standard_deviation"] = standard_deviation
            summary["seed_count"] = len(unique_seed_cells)
        result[f"{method}:{stream}"] = summary
    for stream in sorted({key[1] for key in grouped}):
        naive = sorted(
            grouped.get(("naive_sequential", stream), []),
            key=lambda item: int(item["repeat_index"]),
        )
        aptivra = sorted(
            grouped.get(("aptivra_full", stream), []),
            key=lambda item: int(item["repeat_index"]),
        )
        if len(naive) != 5 or len(aptivra) != 5:
            continue
        if [int(cell["seed"]) for cell in naive] != [
            int(cell["seed"]) for cell in aptivra
        ]:
            raise RuntimeError(f"paired noise-floor seeds do not align for {stream}")
        if not all(
            bool(cast(Mapping[str, Any], cell["metrics"])["demonstrably_learned"])
            for cell in [*naive, *aptivra]
        ):
            result[f"paired_difference:{stream}"] = {"status": "non_learning"}
            continue
        unique_indices = (0, 2, 3, 4)
        paired_metric_names = [
            "backward_transfer",
            "aggregate_forgetting",
            "final_average",
            "new_task_acquisition",
        ]
        if "template_ood_final_average" in cast(
            Mapping[str, Any], naive[0]["metrics"]
        ):
            paired_metric_names.extend(
                ("template_ood_final_average", "template_ood_new_task_acquisition")
            )
        paired_sd: dict[str, Any] = {
            name: float(
                np.std(
                    [
                        float(cast(Mapping[str, Any], aptivra[index]["metrics"])[name])
                        - float(cast(Mapping[str, Any], naive[index]["metrics"])[name])
                        for index in unique_indices
                    ],
                    ddof=1,
                )
            )
            for name in paired_metric_names
        }
        first_tasks = cast(
            list[float],
            cast(Mapping[str, Any], naive[0]["metrics"])[
                "per_task_accuracy_after_training"
            ],
        )
        paired_sd["per_task_accuracy"] = [
            float(
                np.std(
                    [
                        cast(
                            list[float],
                            cast(Mapping[str, Any], aptivra[index]["metrics"])[
                                "per_task_accuracy_after_training"
                            ],
                        )[task_index]
                        - cast(
                            list[float],
                            cast(Mapping[str, Any], naive[index]["metrics"])[
                                "per_task_accuracy_after_training"
                            ],
                        )[task_index]
                        for index in unique_indices
                    ],
                    ddof=1,
                )
            )
            for task_index in range(len(first_tasks))
        ]
        result[f"paired_difference:{stream}"] = {
            "status": "completed",
            "seed_count": len(unique_indices),
            "aptivra_minus_naive_standard_deviation": paired_sd,
        }
    return result


def compute_estimate(
    calibration: Sequence[Mapping[str, Any]],
    evaluation_profiles: Sequence[Mapping[str, Any]],
    streams: Mapping[str, _TaskStream],
    base: StandardBase,
    settings: MpsPilotSettings,
) -> dict[str, Any]:
    """Scale measured method step times by exact claim-stream target-token counts."""

    def token_counts(stream: _TaskStream) -> tuple[int, int]:
        instances = [
            instance
            for family in stream.families
            for instance in stream.get_split(family.family_id).train
        ]
        eos = cast(int, base.tokenizer.eos_token_id)
        total_sequence = 0
        total_target = 0
        for instance in instances:
            prompt = cast(
                list[int], base.tokenizer.encode(instance.prompt, add_special_tokens=False)
            )
            target = cast(
                list[int],
                base.tokenizer.encode(instance.training_target(), add_special_tokens=False),
            )
            target_count = len(target) + int(not target or target[-1] != eos)
            total_target += target_count
            total_sequence += len(prompt) + target_count
        return total_sequence, total_target

    sequence_token_counts: dict[str, int] = {}
    target_token_counts: dict[str, int] = {}
    for stream_name, stream in streams.items():
        sequence, target = token_counts(stream)
        sequence_token_counts[stream_name] = sequence
        target_token_counts[stream_name] = target
    v1_stream = load_experiment_settings(settings.h3_estimate.v1_config).make_stream()
    v1_sequence_tokens, v1_target_tokens = token_counts(v1_stream)
    seconds_per_token: dict[str, float] = {}
    missing_measurements: list[str] = []
    for method in _METHODS:
        for stream_key in streams:
            rows = [
                row
                for row in calibration
                if row.get("method") == method
                and row.get("stream") == stream_key
                and row.get("status") == "completed"
                and row.get("tokens_per_second") is not None
            ]
            key = f"{method}:{stream_key}"
            rates = [float(row["tokens_per_second"]) for row in rows]
            if not rates or any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
                missing_measurements.append(key)
                continue
            seconds_per_token[key] = 1.0 / float(np.median(rates))
    generation_samples: dict[str, list[float]] = {
        stream_key: [
            float(row["batched_seconds_per_example"])
            for row in evaluation_profiles
            if row.get("stream") == stream_key and row.get("status") == "completed"
        ]
        for stream_key in streams
    }
    for stream_key, values in generation_samples.items():
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            missing_measurements.append(f"batched_generation:{stream_key}")
    token_evidence = {
        "exact_training_sequence_tokens": {
            **sequence_token_counts,
            "v1_regression": v1_sequence_tokens,
        },
        "exact_training_target_tokens": {
            **target_token_counts,
            "v1_regression": v1_target_tokens,
        },
    }
    if missing_measurements:
        return {
            "status": "incomplete",
            **token_evidence,
            "missing_measurements": sorted(missing_measurements),
            "reason": "failed or invalid measured cells remain in the evidence table",
        }
    per_cell_seconds = {
        f"{method}:{stream_key}": (
            sequence_token_counts[stream_key] * seconds_per_token[f"{method}:{stream_key}"]
        )
        for method in _METHODS
        for stream_key in streams
    }
    generation_seconds = {
        stream_key: float(np.median(values)) for stream_key, values in generation_samples.items()
    }
    evaluation_generation_caps = {
        name: settings.cycle_caps.evaluation_examples_per_seen_task
        * len(stream.families)
        * (len(stream.families) + 1)
        // 2
        for name, stream in streams.items()
    }
    evaluation_seconds_per_method_seed = sum(
        evaluation_generation_caps[name] * generation_seconds[name] for name in streams
    )
    main_one_seed = sum(per_cell_seconds.values())
    estimate = settings.h3_estimate
    main_training = main_one_seed * estimate.main_seed_count
    main_evaluation = (
        estimate.main_method_count * estimate.main_seed_count * evaluation_seconds_per_method_seed
    )
    method_seconds_per_token = {
        method: float(
            np.mean([seconds_per_token[f"{method}:{stream_name}"] for stream_name in streams])
        )
        for method in _METHODS
    }
    v1_training = (
        sum(v1_sequence_tokens * value for value in method_seconds_per_token.values())
        * estimate.v1_regression_seed_count
    )
    mean_generation_seconds = float(np.mean(list(generation_seconds.values())))
    v1_evaluation_cap = (
        settings.cycle_caps.evaluation_examples_per_seen_task
        * len(v1_stream.families)
        * (len(v1_stream.families) + 1)
        // 2
    )
    v1_evaluation = (
        estimate.v1_regression_method_count
        * estimate.v1_regression_seed_count
        * v1_evaluation_cap
        * mean_generation_seconds
    )
    ablation_seed_stream_cells = (
        sum(item.seeds for item in estimate.ablations) * estimate.ablation_stream_count
    )
    mean_training_stream_cell = float(np.mean(list(per_cell_seconds.values())))
    mean_evaluation_stream_cell = float(
        np.mean([evaluation_generation_caps[name] * generation_seconds[name] for name in streams])
    )
    ablation_training = ablation_seed_stream_cells * mean_training_stream_cell
    ablation_evaluation = ablation_seed_stream_cells * mean_evaluation_stream_cell
    total = (
        main_training
        + main_evaluation
        + v1_training
        + v1_evaluation
        + ablation_training
        + ablation_evaluation
    )
    return {
        "status": "completed",
        **token_evidence,
        "measured_seconds_per_sequence_token_by_method_stream": seconds_per_token,
        "measured_batched_generation_seconds_per_example_by_stream": generation_seconds,
        "evaluation_generation_caps_by_stream": evaluation_generation_caps,
        "per_method_stream_cell_seconds": per_cell_seconds,
        "arithmetic": {
            "main_cells": (
                estimate.main_method_count * estimate.main_seed_count * estimate.claim_stream_count
            ),
            "main_training_seconds": main_training,
            "main_evaluation_seconds_at_cap": main_evaluation,
            "v1_regression_cells": (
                estimate.v1_regression_method_count * estimate.v1_regression_seed_count
            ),
            "v1_regression_training_seconds": v1_training,
            "v1_regression_evaluation_seconds_at_cap": v1_evaluation,
            "ablation_seed_stream_cells": ablation_seed_stream_cells,
            "ablation_design": [item.model_dump() for item in estimate.ablations],
            "ablation_training_seconds": ablation_training,
            "ablation_evaluation_seconds_at_cap": ablation_evaluation,
            "total_seconds": total,
            "total_hours": total / 3600.0,
        },
        "pre_registrable_reduction": (
            "Keep all 60 main cells. Run the 11 ablations on both streams at three "
            "declared seeds; "
            "only expand an ablation to five seeds if it changes the H3 conclusion."
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "mps_available": torch.backends.mps.is_available(),
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }


def pilot_run_manifest(
    settings: MpsPilotSettings,
    config_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind resumable evidence to every code/config/model/stream input."""
    base = load_standard_base_settings(settings.base_config)
    return {
        "schema_version": "aptivra.h3_mps_pilot_manifest.v1",
        "pilot_config_sha256": _sha256(config_path),
        "pilot_module_sha256": _sha256(Path(__file__)),
        "aptivra_python_source_tree_sha256": _python_source_tree_sha256(),
        "pilot_runner_sha256": _sha256(Path("scripts/run_smollm2_mps_pilot.py")),
        "base_config_sha256": _sha256(settings.base_config),
        "baseline_config_sha256": _sha256(settings.baseline_config),
        "expert_config_sha256": _sha256(settings.expert_config),
        "historical_calibration_evidence_sha256": _require_hash(
            settings.calibration.historical_evidence,
            settings.calibration.historical_evidence_sha256,
            "lane-8a historical calibration evidence",
        ),
        "model": {
            "repository": base.source.repository,
            "revision": base.source.revision,
            "config_sha256": base.source.config.sha256,
            "weights_sha256": base.source.weights.sha256,
            "unique_parameter_count": base.architecture.unique_parameter_count,
        },
        "claim_stream_preflight": dict(preflight),
    }


def run_calibration_worker(
    settings: MpsPilotSettings,
    method: PilotMethod,
    stream_name: str,
    micro_batch: int,
    quiet_record: dict[str, Any],
    *,
    checkpointing: bool,
) -> dict[str, Any]:
    """Fresh-process calibration entry point used by the resumable orchestrator."""
    quiet_record = _validate_quiet_record(quiet_record)
    stream = _make_streams(settings).get(stream_name)
    if stream is None:
        raise ValueError(f"unknown pilot stream: {stream_name}")
    return run_calibration_attempt(
        method,
        stream_name,
        stream,
        micro_batch,
        settings,
        quiet_record,
        checkpointing=checkpointing,
    )


def run_repeatability_worker(
    settings: MpsPilotSettings,
    method: Literal["naive_sequential", "aptivra_full"],
    stream_name: str,
    seed: int,
    repeat_index: int,
    quiet_record: dict[str, Any],
) -> dict[str, Any]:
    """Fresh-process repeatability entry point used by the orchestrator."""
    quiet_record = _validate_quiet_record(quiet_record)
    streams = _make_streams(settings)
    stream = streams.get(stream_name)
    if stream is None:
        raise ValueError(f"unknown pilot stream: {stream_name}")
    pairs = {
        "synthetic_v2": settings.streams.synthetic_v2.pair_task_indices,
        "chronological_issues": settings.streams.chronological_issues.pair_task_indices,
    }
    return run_repeatability_cell(
        method,
        stream_name,
        stream,
        pairs[stream_name],
        seed,
        repeat_index,
        settings,
        quiet_record,
    )


def run_evaluation_profile_worker(
    settings: MpsPilotSettings,
    stream_name: str,
    quiet_record: dict[str, Any],
) -> dict[str, Any]:
    """Fresh-process scalar/batched evaluation profile entry point."""
    quiet_record = _validate_quiet_record(quiet_record)
    streams = _make_streams(settings)
    stream = streams.get(stream_name)
    if stream is None:
        raise ValueError(f"unknown pilot stream: {stream_name}")
    pairs = {
        "synthetic_v2": settings.streams.synthetic_v2.pair_task_indices,
        "chronological_issues": settings.streams.chronological_issues.pair_task_indices,
    }
    return run_evaluation_profile_cell(
        stream_name,
        stream,
        pairs[stream_name],
        settings,
        quiet_record,
    )


def _spawn_worker(
    config_path: Path,
    arguments: list[str],
    quiet_record: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one memory-isolated worker and return its required JSON record."""
    quiet = _validate_quiet_record(quiet_record)
    expected: dict[str, Any]
    if arguments[:1] == ["--calibration-worker"] and (
        len(arguments) == 4
        or (len(arguments) == 5 and arguments[4] == "--checkpointing")
    ):
        method, stream, micro_batch = arguments[1:4]
        checkpointing = arguments[4:] == ["--checkpointing"]
        expected = {
            "method": method,
            "stream": stream,
            "micro_batch": int(micro_batch),
            "gradient_checkpointing": (
                "not_applicable" if method == "frozen_base" else checkpointing
            ),
        }
    elif arguments[:1] == ["--repeatability-worker"] and len(arguments) == 5:
        method, stream, seed, repeat_index = arguments[1:5]
        expected = {
            "method": method,
            "stream": stream,
            "seed": int(seed),
            "repeat_index": int(repeat_index),
        }
    elif arguments[:1] == ["--evaluation-profile-worker"] and len(arguments) == 2:
        expected = {"stream": arguments[1]}
    else:
        raise ValueError("worker arguments do not identify one registered pilot cell")

    def failed(failure_type: str, **details: Any) -> dict[str, Any]:
        return {
            **expected,
            "status": "failed",
            "accepted": False,
            "failure": {"type": failure_type, **details},
            "quiet_machine": quiet,
        }

    with tempfile.TemporaryDirectory(prefix="aptivra-h3-pilot-") as directory:
        temporary = Path(directory)
        quiet_path = temporary / "quiet.json"
        output_path = temporary / "result.json"
        _write_json_atomic(quiet_path, quiet)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_smollm2_mps_pilot",
                    "--config",
                    str(config_path),
                    "--worker-output",
                    str(output_path),
                    "--quiet-record",
                    str(quiet_path),
                    *arguments,
                ],
                cwd=Path.cwd(),
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path.cwd()),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return failed(
                "WorkerTimeout",
                timeout_seconds=timeout_seconds,
                stdout_bytes=len(error.stdout or b""),
                stderr_bytes=len(error.stderr or b""),
            )
        if completed.returncode != 0 or not output_path.is_file():
            return failed(
                "WorkerProcessError",
                returncode=completed.returncode,
                stdout_bytes=len(completed.stdout.encode("utf-8")),
                stderr_bytes=len(completed.stderr.encode("utf-8")),
            )
        result = _json_mapping(output_path)
        lowered_stderr = completed.stderr.lower()
        metal_out_of_memory = (
            "kiogpucommandbuffercallbackerroroutofmemory" in lowered_stderr
            or "insufficient memory" in lowered_stderr
        )
        if result.get("status") == "failed" and metal_out_of_memory:
            failure = result.get("failure")
            if isinstance(failure, dict):
                failure["out_of_memory"] = True
                failure["accelerator_signal"] = "metal_driver_out_of_memory"
        if result.get("status") not in {"completed", "failed"}:
            return failed("WorkerEvidenceError", reason="invalid worker status")
        if result.get("quiet_machine") != quiet:
            return failed("WorkerEvidenceError", reason="quiet-record mismatch")
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            return failed(
                "WorkerEvidenceError",
                reason="worker cell coordinates mismatch",
                mismatched_fields=mismatches,
            )
        return result


def _quiet_before_measurement(
    settings: MpsPilotSettings,
    progress: dict[str, Any],
    progress_path: Path,
    pilot_started: float,
    elapsed_before_run: float = 0.0,
) -> dict[str, Any]:
    """Persist expired quiet windows and keep waiting within the global pilot cap."""
    events = cast(list[dict[str, Any]], progress["quiet_gate_events"])
    while True:
        try:
            return wait_for_quiet_machine(settings.quiet_machine)
        except QuietMachineTimeoutError as error:
            events.append(
                {
                    "status": "timed_out",
                    "wait_seconds": error.wait_seconds,
                    "samples": error.samples,
                }
            )
            progress["elapsed_wall_time_seconds"] = (
                elapsed_before_run + time.perf_counter() - pilot_started
            )
            _write_json_atomic(progress_path, progress)
            if (
                elapsed_before_run + time.perf_counter() - pilot_started
                >= settings.execution.wall_time_limit_seconds
            ):
                raise TimeoutError(
                    "bounded pilot wall-time cap reached while waiting for a quiet machine"
                ) from error


def _remaining_wall_time(
    settings: MpsPilotSettings,
    pilot_started: float,
    elapsed_before_run: float,
    phase: str,
) -> float:
    """Return the remaining global pilot budget or halt before spawning work."""
    elapsed = elapsed_before_run + time.perf_counter() - pilot_started
    remaining = settings.execution.wall_time_limit_seconds - elapsed
    if remaining <= 0.0:
        raise TimeoutError(f"bounded pilot wall-time cap reached during {phase}")
    return remaining


def run_pilot(
    settings: MpsPilotSettings,
    *,
    authorized: bool,
    config_path: Path = Path("configs/h3_mps_pilot.yaml"),
    resume_amendment_path: Path | None = None,
) -> dict[str, Any]:
    """Execute only the bounded lane-8a pilot with resumable JSON checkpoints."""
    started = time.perf_counter()
    if settings.feature.enabled:
        raise RuntimeError("pilot config must remain disabled pending owner acceptance")
    if not authorized:
        raise RuntimeError("disabled pilot requires the explicit local --authorize-pilot switch")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("pilot requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    preflight = validate_claim_streams(settings)
    _require_hash(
        settings.calibration.historical_evidence,
        settings.calibration.historical_evidence_sha256,
        "lane-8a historical calibration evidence",
    )
    historical_evidence = _json_mapping(settings.calibration.historical_evidence)
    if (
        historical_evidence.get("schema_version") != PILOT_SCHEMA
        or historical_evidence.get("status") != "completed"
        or not isinstance(historical_evidence.get("calibration"), list)
    ):
        raise ValueError("lane-8a historical calibration evidence is not completed v1 evidence")
    historical_calibration = cast(list[dict[str, Any]], historical_evidence["calibration"])
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    output = settings.output_dir
    progress_path = output / "lane8b-pilot-progress.json"
    final_path = output / "lane8b-pilot-evidence.raw.json"
    manifest = pilot_run_manifest(settings, config_path, preflight)
    if progress_path.is_file():
        progress = _json_mapping(progress_path)
        if progress.get("run_manifest") != manifest:
            if resume_amendment_path is None:
                raise ValueError(
                    "existing pilot checkpoint does not match the current "
                    "code/config/input manifest"
                )
            _accept_resume_amendment(progress, manifest, resume_amendment_path)
    else:
        progress = {
            "schema_version": AMENDMENT_SCHEMA,
            "correlation_id": str(uuid.uuid4()),
            "status": "running",
            "run_manifest": manifest,
            "preflight": preflight,
            "environment": _environment(),
            "historical_calibration_basis": {
                "path": settings.calibration.historical_evidence.as_posix(),
                "sha256": settings.calibration.historical_evidence_sha256,
                "records": len(historical_calibration),
                "reuse_reason": "lane-8a calibration retained; lane-8b reruns anomaly cells only",
            },
            "quiet_gate_events": [],
            "evaluation_profiles": [],
            "calibration": [],
            "repeatability": [],
            "elapsed_wall_time_seconds": 0.0,
        }
    raw_elapsed_before_run = progress.get("elapsed_wall_time_seconds", 0.0)
    if isinstance(raw_elapsed_before_run, bool):
        raise ValueError("existing pilot checkpoint has invalid elapsed wall time")
    elapsed_before_run = float(raw_elapsed_before_run)
    if not math.isfinite(elapsed_before_run) or elapsed_before_run < 0.0:
        raise ValueError("existing pilot checkpoint has invalid elapsed wall time")
    if elapsed_before_run >= settings.execution.wall_time_limit_seconds:
        raise TimeoutError("bounded pilot wall-time cap was exhausted before resume")
    progress["status"] = "running"
    progress.pop("failure", None)
    if final_path.is_file():
        final_path.unlink()

    def checkpoint() -> None:
        progress["elapsed_wall_time_seconds"] = elapsed_before_run + time.perf_counter() - started
        _write_json_atomic(progress_path, progress)

    def remaining_budget(phase: str) -> float:
        return _remaining_wall_time(settings, started, elapsed_before_run, phase)

    checkpoint()
    streams = _make_streams(settings)
    profiles = cast(list[dict[str, Any]], progress["evaluation_profiles"])
    completed_profiles = {
        row.get("stream") for row in profiles if row.get("status") == "completed"
    }
    for stream_name in streams:
        if stream_name in completed_profiles:
            continue
        quiet = _quiet_before_measurement(
            settings,
            progress,
            progress_path,
            started,
            elapsed_before_run,
        )
        profile = _spawn_worker(
            config_path,
            ["--evaluation-profile-worker", stream_name],
            quiet,
            timeout_seconds=remaining_budget("evaluation profiles"),
        )
        profiles.append(profile)
        checkpoint()
        remaining_budget("evaluation profiles")
    calibration = cast(list[dict[str, Any]], progress["calibration"])
    completed_attempts = {
        (
            row.get("method"),
            row.get("stream"),
            row.get("micro_batch"),
            row.get("gradient_checkpointing"),
        )
        for row in calibration
    }
    for cell in settings.calibration.anomaly_cells:
        calibration_key = (cell.method, cell.stream, cell.micro_batch, False)
        if calibration_key in completed_attempts:
            continue
        quiet = _quiet_before_measurement(
            settings,
            progress,
            progress_path,
            started,
            elapsed_before_run,
        )
        record = _spawn_worker(
            config_path,
            ["--calibration-worker", cell.method, cell.stream, str(cell.micro_batch)],
            quiet,
            timeout_seconds=remaining_budget("anomaly calibration"),
        )
        calibration.append(record)
        checkpoint()
        remaining_budget("anomaly calibration")
    repeats = cast(list[dict[str, Any]], progress["repeatability"])
    completed_repeats = {
        (row["method"], row["stream"], row["repeat_index"])
        for row in repeats
        if row.get("status") == "completed"
    }
    for stream_name in streams:
        for method in settings.repeatability.methods:
            for index, seed in enumerate(settings.repeatability.seeds):
                repeat_key = (method, stream_name, index)
                if repeat_key in completed_repeats:
                    continue
                quiet = _quiet_before_measurement(
                    settings,
                    progress,
                    progress_path,
                    started,
                    elapsed_before_run,
                )
                record = _spawn_worker(
                    config_path,
                    [
                        "--repeatability-worker",
                        method,
                        stream_name,
                        str(seed),
                        str(index),
                    ],
                    quiet,
                    timeout_seconds=remaining_budget("repeats"),
                )
                repeats.append(record)
                checkpoint()
                remaining_budget("repeats")
    try:
        noise = noise_floor(repeats)
        tokenizer_base = _base_for_pilot(settings)
        try:
            estimate = compute_estimate(
                [*historical_calibration, *calibration],
                profiles,
                streams,
                tokenizer_base,
                settings,
            )
        finally:
            _cleanup_base(tokenizer_base)
        if estimate.get("status") != "completed":
            raise RuntimeError("compute estimate is incomplete; failed cells remain recorded")
        total_elapsed = elapsed_before_run + time.perf_counter() - started
        progress.update(
            {
                "status": "completed",
                "noise_floor": noise,
                "compute_estimate": estimate,
                "elapsed_wall_time_seconds": total_elapsed,
                "pilot_wall_time_seconds": total_elapsed,
            }
        )
    except Exception as error:
        total_elapsed = elapsed_before_run + time.perf_counter() - started
        progress.update(
            {
                "status": "failed",
                "failure": _redacted_failure(error),
                "elapsed_wall_time_seconds": total_elapsed,
                "pilot_wall_time_seconds": total_elapsed,
            }
        )
        _write_json_atomic(progress_path, progress)
        _write_json_atomic(final_path, progress)
        raise
    _write_json_atomic(progress_path, progress)
    _write_json_atomic(final_path, progress)
    return progress
