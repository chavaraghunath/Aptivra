"""Behavioral mutation proofs for pilot bookkeeping and orchestration."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import subprocess
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

import aptivra.eval.mps_pilot as pilot
from aptivra.inference.standard_base import CausalBatch, GenerationResult
from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, exact_match_result
from aptivra.tasks.stream import TaskStreamSplit


class _Verifier(MechanicalVerifier):
    def verify(self, instance: TaskInstance, model_output: str) -> Any:
        return exact_match_result(instance, model_output)


class _Family(TaskFamily):
    def __init__(self, family_id: str) -> None:
        super().__init__(family_id, family_id, "orchestration mutation fixture")
        self._verifier = _Verifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        return [
            TaskInstance(
                f"{self.family_id}-{seed}-{index}",
                self.family_id,
                f"prompt {index}",
                "ok",
            )
            for index in range(count)
        ]


class _Stream:
    def __init__(self, count: int = 5) -> None:
        self.families = [_Family("a"), _Family("b")]
        self._splits = {
            family.family_id: TaskStreamSplit(
                train=family.generate_instances(count, 1),
                val=family.generate_instances(count, 2),
                test=family.generate_instances(count, 3),
            )
            for family in self.families
        }

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]


def _settings() -> pilot.MpsPilotSettings:
    return pilot.load_mps_pilot_settings("configs/h3_mps_pilot.yaml")


def _raw_settings() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/h3_mps_pilot.yaml").read_text(encoding="utf-8")),
    )


def _write_settings(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "pilot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _quiet() -> dict[str, Any]:
    return {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": True}]}


def test_all_pilot_runtime_type_contracts_resolve() -> None:
    resolved = {
        name: typing.get_type_hints(value, include_extras=True)
        for name, value in vars(pilot).items()
        if callable(value)
        and getattr(value, "__module__", None) == pilot.__name__
        and getattr(value, "__annotations__", None)
    }
    assert resolved["load_mps_pilot_settings"]["return"] is pilot.MpsPilotSettings
    assert resolved["_routing_decision_rates"]["return"] == dict[str, float | int]
    assert resolved["run_pilot"]["return"] == dict[str, Any]
    assert (
        inspect.signature(pilot.run_calibration_attempt)
        .parameters["checkpointing"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(pilot.run_calibration_worker)
        .parameters["checkpointing"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                initial_backoff_seconds=10.0, maximum_backoff_seconds=9.0
            ),
            "initial backoff",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_backoff_seconds=901.0
            ),
            "maximum backoff",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                anomaly_cells=[
                    {
                        "method": "online_ewc",
                        "stream": "chronological_issues",
                        "micro_batch": 8,
                    }
                ]
                * 3
            ),
            "three lane-8b issue cells",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                methods=["aptivra_full", "naive_sequential"]
            ),
            "one full-FT method",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                seeds=[1, 1, 1, 2, 3]
            ),
            "three additional seeds",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                evaluation_examples_per_task=63
            ),
            "meet the H3 per-task count",
        ),
    ),
)
def test_config_cross_field_invariants_are_independent(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    raw = _raw_settings()
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        pilot.load_mps_pilot_settings(_write_settings(tmp_path, raw))


def test_config_rejects_absolute_output_and_non_mapping_roots(tmp_path: Path) -> None:
    raw = _raw_settings()
    raw["h3_mps_pilot"]["output_dir"] = str(tmp_path / "absolute")
    with pytest.raises(ValidationError, match="worktree-relative"):
        pilot.load_mps_pilot_settings(_write_settings(tmp_path, raw))

    for index, value in enumerate(([], "scalar", {"h3_mps_pilot": {}, "extra": {}})):
        path = tmp_path / f"root-{index}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with pytest.raises(ValueError, match="contain only"):
            pilot.load_mps_pilot_settings(path)


def test_hash_validators_reject_noncanonical_values(tmp_path: Path) -> None:
    fields = (
        ("streams", "synthetic_v2", "config_sha256"),
        ("streams", "chronological_issues", "raw_sha256"),
        ("calibration", "historical_evidence_sha256"),
    )
    for index, field in enumerate(fields):
        raw = _raw_settings()
        target: dict[str, Any] = raw["h3_mps_pilot"]
        for part in field[:-1]:
            target = target[part]
        target[field[-1]] = "A" * 64
        with pytest.raises(ValidationError, match="SHA-256"):
            pilot.load_mps_pilot_settings(_write_settings(tmp_path / str(index), raw))


@pytest.mark.parametrize(
    ("mutate", "valid"),
    (
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_foreign_process_rss_bytes=0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                timeout_seconds=86_400.0
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                minimum_free_memory_percent=100.0
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                minimum_free_memory_percent=1.0
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                minimum_free_memory_percent=100.5
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                minimum_free_memory_percent="50"
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_foreign_process_cpu_percent="1"
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_foreign_process_cpu_percent=0.0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_foreign_process_cpu_percent=0.1
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                initial_backoff_seconds=0.0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                initial_backoff_seconds=1.0,
                maximum_backoff_seconds=1.0,
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                maximum_backoff_seconds=30.0,
                timeout_seconds=30.0,
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                initial_backoff_seconds=1.0,
                maximum_backoff_seconds=1.0,
                timeout_seconds=1.0,
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(
                timeout_seconds=86_401.0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(measured_steps=2),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                memory_sample_interval_seconds=0.0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                memory_sample_interval_seconds="0.02"
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"]["micro_batches"][
                "synthetic_v2"
            ].update(aptivra_full=0),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"]["micro_batches"][
                "synthetic_v2"
            ].update(aptivra_full=3),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                micro_batches=[8, 4, 2]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                micro_batches=[8, 4, 2, 1, 1]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                micro_batches=[8, 4, 2, 2]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                methods=[
                    "naive_sequential",
                    "frozen_base",
                    "replay",
                    "online_ewc",
                    "manual_lora",
                    "aptivra_full",
                ]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                anomaly_cells=raw["h3_mps_pilot"]["calibration"]["anomaly_cells"][:2]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"].update(
                anomaly_cells=[
                    *raw["h3_mps_pilot"]["calibration"]["anomaly_cells"],
                    raw["h3_mps_pilot"]["calibration"]["anomaly_cells"][0],
                ]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                profile_examples=33
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(profile_examples=2),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(profile_examples=32),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(profile_examples=1),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                methods=["naive_sequential", "naive_sequential"]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                seeds=[2, 1, 3, 4, 5]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"][
                "evaluation_batch_sizes"
            ].update(synthetic_v2=16, chronological_issues=16),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"][
                "evaluation_batch_sizes"
            ].update(synthetic_v2=17),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["repeatability"].update(
                evaluation_examples_per_task=65
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0].update(
                seeds=1
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"].update(main_seed_count=5.0),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0].update(
                seeds=1.0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0].update(
                seeds=0
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0].update(
                seeds=5
            ),
            True,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0].update(
                seeds=6
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"].update(main_seed_count=4),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"].update(
                ablations=raw["h3_mps_pilot"]["h3_estimate"]["ablations"][:10]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"].update(
                ablations=[
                    *raw["h3_mps_pilot"]["h3_estimate"]["ablations"],
                    raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0],
                ]
            ),
            False,
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["execution"].update(
                wall_time_limit_seconds=28_801.0
            ),
            False,
        ),
    ),
)
def test_config_accepts_and_rejects_exact_numeric_boundaries(
    tmp_path: Path,
    mutate: Any,
    valid: bool,
) -> None:
    raw = _raw_settings()
    mutate(raw)
    path = _write_settings(tmp_path, raw)
    if valid:
        pilot.load_mps_pilot_settings(path)
    else:
        with pytest.raises(ValidationError):
            pilot.load_mps_pilot_settings(path)


@pytest.mark.parametrize(
    ("cpu_percent", "valid"), ((99.5, True), (100.0, True), (101.0, False))
)
def test_quiet_cpu_ceiling_uses_the_exact_logical_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_percent: float,
    valid: bool,
) -> None:
    raw = _raw_settings()
    raw["h3_mps_pilot"]["quiet_machine"]["maximum_foreign_process_cpu_percent"] = (
        cpu_percent
    )
    monkeypatch.setattr(pilot.os, "cpu_count", lambda: 1)
    path = _write_settings(tmp_path, raw)
    if valid:
        pilot.load_mps_pilot_settings(path)
    else:
        with pytest.raises(ValidationError, match="logical-CPU capacity"):
            pilot.load_mps_pilot_settings(path)


def test_settings_are_frozen_and_strict() -> None:
    settings = _settings()
    with pytest.raises(ValidationError, match="frozen"):
        settings.feature.enabled = True


def test_base_loader_requires_and_uses_the_exact_offline_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    base_settings = pilot.load_standard_base_settings(settings.base_config)
    repository = base_settings.source.repository.replace("/", "--")
    expected = (
        tmp_path
        / f"models--{repository}"
        / "snapshots"
        / base_settings.source.revision
    )
    monkeypatch.setattr(pilot, "HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="downloading is forbidden"):
        pilot._base_for_pilot(settings)

    expected.mkdir(parents=True)
    captured: dict[str, Any] = {}

    def load(enabled: Any, *, snapshot_path: Path, device: str) -> str:
        captured.update(enabled=enabled, snapshot_path=snapshot_path, device=device)
        return "loaded"

    monkeypatch.setattr(pilot, "load_standard_base", load)
    assert pilot._base_for_pilot(settings) == "loaded"
    assert captured["enabled"].feature.enabled is True
    assert captured["snapshot_path"] == expected
    assert captured["device"] == "mps"


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("no_synthetic_match", "exactly one matching"),
        ("two_synthetic_matches", "exactly one matching"),
        ("partial_synthetic_match", "exactly one matching"),
        ("partial_synthetic_match_high", "exactly one matching"),
        ("partial_synthetic_config_low", "exactly one matching"),
        ("partial_synthetic_config_high", "exactly one matching"),
        ("synthetic_not_passed", "synthetic v2 audit did not pass"),
        ("synthetic_integer_passed", "synthetic v2 audit did not pass"),
        ("synthetic_violations", "synthetic v2 audit did not pass"),
        ("raw_path", "raw-data path"),
        ("raw_path_low", "raw-data path"),
        ("manifest_path", "manifest path"),
        ("manifest_path_low", "manifest path"),
        ("audit_path", "audit path"),
        ("audit_path_low", "audit path"),
        ("manifest_schema", "schema_version"),
        ("manifest_schema_high", "schema_version"),
        ("manifest_audit", "committed audit record"),
        ("empty_summary", "summary is empty"),
        ("coverage_type", "coverage gate failed"),
        ("coverage_integer", "coverage gate failed"),
        ("coverage_failed", "coverage gate failed"),
        ("overlap", "overlap gate failed"),
        ("record_count", "record_count"),
        ("record_count_low", "record_count"),
    ),
)
def test_claim_stream_preflight_rejects_each_independent_gate(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    settings = _settings()
    synthetic = settings.streams.synthetic_v2
    issues = settings.streams.chronological_issues
    stream = _Stream(count=1)
    audit = {"passed": True}
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "audit": audit,
        "record_count": 6,
        "record_shards": ["one"],
    }
    synthetic_record: dict[str, Any] = {
        "passed": True,
        "violations": [],
        "families": ["a", "b"],
        "stream_version": "v2",
    }
    matching = SimpleNamespace(
        stream_config=synthetic.config,
        output=synthetic.audit_record,
    )
    entries = [matching]
    if case == "no_synthetic_match":
        entries = []
    elif case == "two_synthetic_matches":
        entries = [matching, matching]
    elif case == "partial_synthetic_match":
        entries = [
            SimpleNamespace(
                stream_config=synthetic.config,
                output=Path("0-partial-output"),
            )
        ]
    elif case == "partial_synthetic_match_high":
        entries = [
            SimpleNamespace(
                stream_config=synthetic.config,
                output=Path("zzzz-partial-output"),
            )
        ]
    elif case == "partial_synthetic_config_low":
        entries = [
            SimpleNamespace(
                stream_config=Path("0-partial-config"),
                output=synthetic.audit_record,
            )
        ]
    elif case == "partial_synthetic_config_high":
        entries = [
            SimpleNamespace(
                stream_config=Path("zzzz-partial-config"),
                output=synthetic.audit_record,
            )
        ]
    elif case == "synthetic_not_passed":
        synthetic_record["passed"] = False
    elif case == "synthetic_integer_passed":
        synthetic_record["passed"] = 1
    elif case == "synthetic_violations":
        synthetic_record["violations"] = ["failure"]
    elif case == "manifest_schema":
        manifest["schema_version"] = 2
    elif case == "manifest_schema_high":
        manifest["schema_version"] = 4
    elif case == "manifest_audit":
        manifest["audit"] = {"different": True}
    elif case == "record_count":
        manifest["record_count"] = 7
    elif case == "record_count_low":
        manifest["record_count"] = 5

    monkeypatch.setattr(pilot, "_require_hash", lambda _path, expected, _description: expected)
    monkeypatch.setattr(
        pilot, "load_audit_plan", lambda _: SimpleNamespace(streams=tuple(entries))
    )
    monkeypatch.setattr(pilot, "write_or_check_plan", lambda *args, **kwargs: None)

    def mapping(path: Path) -> dict[str, Any]:
        if path == synthetic.audit_record:
            return synthetic_record
        if path == issues.manifest:
            return manifest
        if path == issues.audit_record:
            return audit
        raise AssertionError(path)

    monkeypatch.setattr(pilot, "_json_mapping", mapping)
    issue_config = SimpleNamespace(
        raw_dataset_path=issues.raw_data,
        manifest_path=issues.manifest,
        audit_path=issues.audit_record,
    )
    if case == "raw_path":
        issue_config.raw_dataset_path = Path("wrong-raw")
    elif case == "raw_path_low":
        issue_config.raw_dataset_path = Path("0")
    elif case == "manifest_path":
        issue_config.manifest_path = Path("wrong-manifest")
    elif case == "manifest_path_low":
        issue_config.manifest_path = Path("0")
    elif case == "audit_path":
        issue_config.audit_path = Path("wrong-audit")
    elif case == "audit_path_low":
        issue_config.audit_path = Path("0")
    monkeypatch.setattr(pilot, "load_github_issue_builder_config", lambda _: issue_config)
    summaries: dict[str, Any] = {
        family.family_id: {
            "coverage": {"meets_threshold": True},
            "overlap": {"train_test": 0, "train_val": 0, "val_test": 0},
        }
        for family in stream.families
    }
    if case == "empty_summary":
        summaries = {}
    elif case == "coverage_type":
        summaries["a"]["coverage"] = []
    elif case == "coverage_integer":
        summaries["a"]["coverage"] = {"meets_threshold": 1}
    elif case == "coverage_failed":
        summaries["a"]["coverage"] = {"meets_threshold": False}
    elif case == "overlap":
        summaries["a"]["overlap"] = {"train_test": 1, "train_val": 0, "val_test": 0}
    monkeypatch.setattr(
        pilot,
        "GitHubIssuesContinualStream",
        lambda *_: SimpleNamespace(
            families=stream.families,
            get_split=stream.get_split,
            audit_summary=lambda: summaries,
        ),
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        pilot.validate_claim_streams(settings)


def _amendment_payload(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "aptivra.h3_mps_pilot_resume_amendment.v1",
        "reason": "measured worker correction",
        "from_manifest_sha256": pilot._mapping_sha256(previous),
        "to_manifest_sha256": pilot._mapping_sha256(current),
        "interrupted_wall_seconds": 2.5,
    }


def test_mapping_digest_is_canonical() -> None:
    left = {"b": [2, 1], "a": {"value": True}}
    right = {"a": {"value": True}, "b": [2, 1]}
    expected = hashlib.sha256(
        b'{"a":{"value":true},"b":[2,1]}'
    ).hexdigest()
    assert pilot._mapping_sha256(left) == pilot._mapping_sha256(right) == expected


def test_required_hash_rejects_lexically_higher_expected_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")
    actual = pilot._sha256(artifact)
    assert actual < "f" * 64
    with pytest.raises(ValueError, match=f"expected {'f' * 64}, got {actual}"):
        pilot._require_hash(artifact, "f" * 64, "fixture")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.clear(), "exactly"),
        (lambda payload: payload.update(extra="x"), "exactly"),
        (lambda payload: payload.update(schema_version="v0"), "unsupported"),
        (lambda payload: payload.update(reason="   "), "non-empty"),
        (lambda payload: payload.update(reason=7), "non-empty"),
        (lambda payload: payload.update(from_manifest_sha256=7), "from-manifest"),
        (lambda payload: payload.update(from_manifest_sha256="0" * 63), "from-manifest"),
        (lambda payload: payload.update(from_manifest_sha256="0" * 64), "checkpoint"),
        (lambda payload: payload.update(from_manifest_sha256="f" * 64), "checkpoint"),
        (lambda payload: payload.update(to_manifest_sha256=7), "to-manifest"),
        (lambda payload: payload.update(to_manifest_sha256="0" * 63), "to-manifest"),
        (lambda payload: payload.update(to_manifest_sha256="0" * 64), "current manifest"),
        (lambda payload: payload.update(to_manifest_sha256="f" * 64), "current manifest"),
        (lambda payload: payload.update(interrupted_wall_seconds=True), "numeric"),
        (lambda payload: payload.update(interrupted_wall_seconds="2"), "numeric"),
        (lambda payload: payload.update(interrupted_wall_seconds=0), "positive and finite"),
        (lambda payload: payload.update(interrupted_wall_seconds=-1), "positive and finite"),
        (lambda payload: payload.update(interrupted_wall_seconds=math.nan), "positive and finite"),
        (lambda payload: payload.update(interrupted_wall_seconds=math.inf), "positive and finite"),
    ),
)
def test_resume_amendment_rejects_each_invalid_field(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    previous = {"identity": "before"}
    current = {"identity": "after"}
    progress: dict[str, Any] = {"run_manifest": previous, "elapsed_wall_time_seconds": 4.0}
    before = copy.deepcopy(progress)
    payload = _amendment_payload(previous, current)
    mutation(payload)
    path = tmp_path / "amendment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pilot._accept_resume_amendment(progress, current, path)
    assert progress == before


@pytest.mark.parametrize(
    ("progress_update", "message"),
    (
        ({"run_manifest": None}, "no valid run manifest"),
        ({"elapsed_wall_time_seconds": True}, "must be numeric"),
        ({"run_manifest_history": {}}, "invalid amendment history"),
        ({"resume_amendments": {}}, "invalid amendment history"),
    ),
)
def test_resume_amendment_rejects_invalid_checkpoint_state(
    tmp_path: Path,
    progress_update: dict[str, Any],
    message: str,
) -> None:
    previous = {"identity": "before"}
    current = {"identity": "after"}
    progress: dict[str, Any] = {"run_manifest": previous}
    progress.update(progress_update)
    path = tmp_path / "amendment.yaml"
    path.write_text(yaml.safe_dump(_amendment_payload(previous, current)), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pilot._accept_resume_amendment(progress, current, path)


def test_resume_amendment_appends_defaults_and_exact_artifact(tmp_path: Path) -> None:
    previous = {"identity": "before"}
    current = {"identity": "after"}
    progress: dict[str, Any] = {"run_manifest": previous}
    path = tmp_path / "amendment.yaml"
    path.write_text(yaml.safe_dump(_amendment_payload(previous, current)), encoding="utf-8")
    pilot._accept_resume_amendment(progress, current, path)
    assert progress["elapsed_wall_time_seconds"] == 2.5
    assert progress["run_manifest_history"] == [
        {
            "run_manifest": previous,
            "completed_repeatability_records": 0,
            "elapsed_wall_time_seconds": 0.0,
        }
    ]
    assert progress["resume_amendments"][0]["artifact"] == path.as_posix()
    assert progress["resume_amendments"][0]["artifact_sha256"] == pilot._sha256(path)


def test_resume_amendment_accepts_positive_subsecond_interruption(tmp_path: Path) -> None:
    previous = {"identity": "before"}
    current = {"identity": "after"}
    progress: dict[str, Any] = {"run_manifest": previous}
    payload = _amendment_payload(previous, current)
    payload["interrupted_wall_seconds"] = 0.5
    path = tmp_path / "subsecond.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    pilot._accept_resume_amendment(progress, current, path)
    assert progress["elapsed_wall_time_seconds"] == 0.5


def test_own_process_tree_requires_recursive_fixed_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot.os, "getpid", lambda: 10)
    samples = (
        pilot.ProcessSample(30, 20, 0.0, 1, "grandchild"),
        pilot.ProcessSample(99, 1, 0.0, 1, "unrelated"),
        pilot.ProcessSample(20, 10, 0.0, 1, "child"),
        pilot.ProcessSample(40, 30, 0.0, 1, "great-grandchild"),
    )
    assert pilot._own_process_tree(samples) == {10, 20, 30, 40}


def test_process_samples_skips_malformed_rows_and_preserves_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        stdout="malformed\n7 1 12.5 4 command with spaces\n8 1 nope 5 bad\n"
    )
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: completed)
    assert pilot._process_samples() == (
        pilot.ProcessSample(7, 1, 12.5, 4096, "command with spaces"),
    )


@pytest.mark.parametrize(
    "record",
    (
        {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": True}], "extra": 1},
        {"quiet": True, "wait_seconds": 0.0},
        {"quiet": False, "wait_seconds": 0.0, "samples": [{"quiet": True}]},
        {"quiet": 1, "wait_seconds": 0.0, "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": True, "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": "0", "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": math.nan, "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": math.inf, "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": -1.0, "samples": [{"quiet": True}]},
        {"quiet": True, "wait_seconds": 0.0, "samples": []},
        {"quiet": True, "wait_seconds": 0.0, "samples": "bad"},
        {"quiet": True, "wait_seconds": 0.0, "samples": ["bad"]},
        {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": False}]},
        {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": 1}]},
    ),
)
def test_quiet_record_validation_rejects_every_forgery(record: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="quiet record"):
        pilot._validate_quiet_record(record)


def test_quiet_record_validation_returns_a_plain_copy() -> None:
    record = _quiet()
    result = pilot._validate_quiet_record(record)
    assert result == record
    assert result is not record


def test_quiet_record_missing_field_fails_at_schema_boundary() -> None:
    with pytest.raises(ValueError, match="invalid schema"):
        pilot._validate_quiet_record({"quiet": True, "wait_seconds": 0.0})


def test_quiet_sample_thresholds_exclude_own_tree_and_sort_top_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings().quiet_machine.model_copy(
        update={
            "minimum_free_memory_percent": 50.0,
            "maximum_foreign_process_rss_bytes": 100,
            "maximum_foreign_process_cpu_percent": 10.0,
        }
    )
    samples = [
        pilot.ProcessSample(2, 1, 99.0, 999, "own-child"),
        pilot.ProcessSample(3, 2, 99.0, 999, "own-grandchild"),
        pilot.ProcessSample(10, 0, 10.0, 100, "boundary"),
        pilot.ProcessSample(11, 0, 10.1, 101, "above"),
    ]
    samples.extend(
        pilot.ProcessSample(100 + index, 0, 0.0, index, f"p-{index}")
        for index in range(25)
    )
    monkeypatch.setattr(pilot, "_process_samples", lambda: tuple(samples))
    monkeypatch.setattr(pilot.os, "getpid", lambda: 1)
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 50.0)
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: 7)
    result = pilot.quiet_machine_sample(settings)
    assert result["quiet"] is False
    assert result["foreign_process_count"] == 27
    assert [row["pid"] for row in result["other_processes_above_cpu_threshold"]] == [11]
    assert [row["pid"] for row in result["other_processes_above_rss_threshold"]] == [11]
    top = result["top_foreign_processes_by_rss"]
    assert len(top) == 20
    assert [row["rss_bytes"] for row in top] == sorted(
        [row["rss_bytes"] for row in top], reverse=True
    )
    assert result["system_swap_used_bytes"] == 7


def test_quiet_sample_free_memory_boundary_is_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings().quiet_machine.model_copy(
        update={"minimum_free_memory_percent": 50.0}
    )
    monkeypatch.setattr(pilot, "_process_samples", lambda: ())
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 50.0)
    assert pilot.quiet_machine_sample(settings)["quiet"] is True
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 49.999)
    assert pilot.quiet_machine_sample(settings)["quiet"] is False


def test_wait_for_quiet_machine_clips_final_sleep_and_times_out_at_boundary() -> None:
    settings = _settings().quiet_machine.model_copy(
        update={
            "timeout_seconds": 3.0,
            "initial_backoff_seconds": 2.0,
            "maximum_backoff_seconds": 2.0,
        }
    )
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    with pytest.raises(pilot.QuietMachineTimeoutError) as caught:
        pilot.wait_for_quiet_machine(
            settings,
            sleep=sleep,
            monotonic=lambda: clock[0],
            sample=lambda _: {"quiet": False, "at": clock[0]},
        )
    assert sleeps == [2.0, 1.0]
    assert caught.value.wait_seconds == 3.0
    assert [sample["at"] for sample in caught.value.samples] == [0.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "arguments",
    (
        [],
        ["--unknown"],
        ["--calibration-worker", "online_ewc", "chronological_issues"],
        ["--calibration-worker", "online_ewc", "chronological_issues", "8", "--junk"],
        ["--repeatability-worker", "naive_sequential", "synthetic_v2", "1"],
        ["--evaluation-profile-worker"],
        ["--evaluation-profile-worker", "synthetic_v2", "extra"],
    ),
)
def test_spawn_worker_rejects_unregistered_argument_shapes(arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="registered pilot cell"):
        pilot._spawn_worker(Path("config.yaml"), arguments, _quiet(), timeout_seconds=1.0)


@pytest.mark.parametrize(
    ("arguments", "coordinates"),
    (
        (
            [
                "--calibration-worker",
                "online_ewc",
                "chronological_issues",
                "8",
                "--checkpointing",
            ],
            {
                "method": "online_ewc",
                "stream": "chronological_issues",
                "micro_batch": 8,
                "gradient_checkpointing": True,
            },
        ),
        (
            ["--repeatability-worker", "naive_sequential", "synthetic_v2", "7", "2"],
            {
                "method": "naive_sequential",
                "stream": "synthetic_v2",
                "seed": 7,
                "repeat_index": 2,
            },
        ),
        (
            ["--evaluation-profile-worker", "chronological_issues"],
            {"stream": "chronological_issues"},
        ),
    ),
)
def test_spawn_worker_uses_exact_process_contract(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    coordinates: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--worker-output") + 1])
        quiet_path = Path(command[command.index("--quiet-record") + 1])
        result = {
            **coordinates,
            "status": "completed",
            "quiet_machine": pilot._json_mapping(quiet_path),
        }
        pilot._write_json_atomic(output, result)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pilot.subprocess, "run", run)
    result = pilot._spawn_worker(
        Path("config.yaml"), arguments, _quiet(), timeout_seconds=12.5
    )
    assert result["status"] == "completed"
    command = captured["command"]
    assert command[:3] == [
        pilot.sys.executable,
        "-m",
        "scripts.run_smollm2_mps_pilot",
    ]
    assert command[-len(arguments) :] == arguments
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == Path.cwd()
    assert kwargs["env"]["PYTHONPATH"] == str(Path.cwd())
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 12.5
    assert kwargs["check"] is False


@pytest.mark.parametrize(("returncode", "write_output"), ((7, True), (0, False)))
def test_spawn_worker_process_failure_truth_table(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    write_output: bool,
) -> None:
    def run(command: list[str], **_kwargs: Any) -> Any:
        if write_output:
            output = Path(command[command.index("--worker-output") + 1])
            pilot._write_json_atomic(output, {"status": "completed"})
        return SimpleNamespace(returncode=returncode, stdout="out", stderr="err")

    monkeypatch.setattr(pilot.subprocess, "run", run)
    result = pilot._spawn_worker(
        Path("config.yaml"),
        ["--evaluation-profile-worker", "synthetic_v2"],
        _quiet(),
        timeout_seconds=1.0,
    )
    assert result["status"] == "failed"
    assert result["accepted"] is False
    assert result["failure"] == {
        "type": "WorkerProcessError",
        "returncode": returncode,
        "stdout_bytes": 3,
        "stderr_bytes": 3,
    }


def test_spawn_worker_timeout_with_absent_streams_records_zero_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("worker", 3.0, output=None, stderr=None)

    monkeypatch.setattr(pilot.subprocess, "run", run)
    result = pilot._spawn_worker(
        Path("config.yaml"),
        ["--evaluation-profile-worker", "synthetic_v2"],
        _quiet(),
        timeout_seconds=3.0,
    )
    assert result["failure"] == {
        "type": "WorkerTimeout",
        "timeout_seconds": 3.0,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
    }


def test_evaluation_helpers_preserve_batching_counts_and_fractions() -> None:
    stream = pilot._BoundedPairStream(cast(Any, _Stream()), (0, 1), 5, 0)
    calls: list[list[str]] = []

    def predict(instances: Any) -> list[str]:
        selected = list(instances)
        calls.append([instance.task_id for instance in selected])
        return [
            "ok" if int(instance.task_id.rsplit("-", 1)[1]) % 2 == 0 else "bad"
            for instance in selected
        ]

    assert pilot._evaluate_one_task(stream, 0, predict, 2) == 3 / 5
    assert [len(call) for call in calls] == [2, 2, 1]
    calls.clear()
    assert pilot._evaluate_pair(stream, predict, 3) == [3 / 5, 3 / 5]
    assert [len(call) for call in calls] == [3, 2, 3, 2]
    assert pilot._evaluate_template_ood_pair(stream, predict, 2) is None

    with pytest.raises(RuntimeError, match="wrong number"):
        pilot._evaluate_instances(stream.families[0], stream.get_split("a").test, lambda _: [], 2)


def test_template_ood_evaluation_uses_real_splits_and_rejects_short_splits() -> None:
    class OodStream(_Stream):
        def get_template_ood_split(self, family_id: str) -> list[TaskInstance]:
            return self._splits[family_id].test

    source = OodStream(count=3)
    stream = pilot._BoundedPairStream(cast(Any, source), (0, 1), 2, 2)
    values = pilot._evaluate_template_ood_pair(
        stream,
        lambda instances: ["ok", "bad"][: len(instances)],
        2,
    )
    assert values == [0.5, 0.5]

    with pytest.raises(ValueError, match="lacks configured template-OOD"):
        pilot._BoundedPairStream(cast(Any, source), (0, 1), 2, 4)


def test_synchronize_and_cleanup_touch_mps_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[bool] = []
    emptied: list[bool] = []
    monkeypatch.setattr(pilot.torch.mps, "synchronize", lambda: synchronized.append(True))
    monkeypatch.setattr(pilot.torch.mps, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: True)
    pilot._synchronize(torch.device("mps"))
    pilot._synchronize(torch.device("cpu"))
    pilot._cleanup_base(cast(Any, object()))
    assert synchronized == [True]
    assert emptied == [True]

    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: False)
    pilot._cleanup_base(None)
    assert emptied == [True]


def test_remove_candidate_restores_last_mature_adapter_or_base() -> None:
    actions: list[tuple[str, str | None]] = []
    bank = SimpleNamespace(
        trainable_adapter="candidate",
        mature_adapters={"mature-a", "mature-m", "mature-z"},
        model=SimpleNamespace(delete_adapter=lambda name: actions.append(("delete", name))),
        activate=lambda name: actions.append(("activate", name)),
        deactivate=lambda: actions.append(("deactivate", None)),
    )
    router = SimpleNamespace(
        weights=torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
        ),
        biases=torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
        ),
        expert_ids=["mature", "candidate"],
        task_families=["old", "new"],
        frozen_rows=[True, False],
    )
    pilot._remove_candidate(cast(Any, bank), cast(Any, router))
    assert bank.trainable_adapter is None
    assert len(router.weights) == len(router.biases) == 1
    assert router.expert_ids == ["mature"]
    assert router.task_families == ["old"]
    assert router.frozen_rows == [True]
    assert actions == [("delete", "candidate"), ("activate", "mature-z")]

    bank.trainable_adapter = "candidate-two"
    bank.mature_adapters = set()
    router.weights.append(torch.nn.Parameter(torch.ones(1)))
    router.biases.append(torch.nn.Parameter(torch.ones(1)))
    router.expert_ids.append("candidate-two")
    router.task_families.append("new")
    router.frozen_rows.append(False)
    pilot._remove_candidate(cast(Any, bank), cast(Any, router))
    assert actions[-2:] == [("delete", "candidate-two"), ("deactivate", None)]

    bank.trainable_adapter = None
    with pytest.raises(RuntimeError, match="without a trainable adapter"):
        pilot._remove_candidate(cast(Any, bank), cast(Any, router))


def test_routing_rates_cover_all_zero_denominators() -> None:
    assert pilot._routing_decision_rates([None, None], [None, None]) == {
        "total_prompts": 2,
        "eligible_prompts": 0,
        "fallback_prompts": 2,
        "routed_eligible_prompts": 0,
        "coverage": 0.0,
        "misroute_rate": 0.0,
        "eligible_misroute_rate": 0.0,
        "fallback_false_route_rate": 0.0,
    }
    assert pilot._routing_decision_rates([None, None], ["a", "b"]) == {
        "total_prompts": 2,
        "eligible_prompts": 2,
        "fallback_prompts": 0,
        "routed_eligible_prompts": 0,
        "coverage": 0.0,
        "misroute_rate": 0.0,
        "eligible_misroute_rate": 0.0,
        "fallback_false_route_rate": 0.0,
    }
    assert pilot._routing_decision_rates(["a", "b"], ["a", "b"])["coverage"] == 1.0

    left = "equal-" + "value" * 80
    right = "".join(("equal-", "value" * 80))
    assert left == right and left is not right
    distinct = pilot._routing_decision_rates([left], [right])
    assert distinct["eligible_misroute_rate"] == 0.0

    many = [None] * 300
    assert pilot._routing_decision_rates(many, list(many))["total_prompts"] == 300


def test_routing_probe_separates_abstention_fallback_and_oracle_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = pilot._BoundedPairStream(cast(Any, _Stream(count=2)), (0, 1), 2, 0)
    router = SimpleNamespace(
        task_families=["a"],
        expert_ids=["expert-a"],
        confidence_threshold=0.65,
        route=lambda _features: [
            SimpleNamespace(expert_id="expert-a"),
            SimpleNamespace(expert_id=None),
            SimpleNamespace(expert_id="wrong"),
            SimpleNamespace(expert_id=None),
        ],
    )
    monkeypatch.setattr(pilot, "_prompt_features", lambda _base, rows: torch.ones((len(rows), 1)))
    calls: list[tuple[list[str], list[str | None], int, int]] = []

    def predict(
        _base: Any,
        _bank: Any,
        instances: Any,
        adapter_ids: Any,
        *,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        rows = list(instances)
        ids = list(adapter_ids)
        calls.append(([row.task_id for row in rows], ids, max_new_tokens, batch_size))
        if len(calls) == 1:
            return ["ok", "bad", "bad", "bad"]
        return ["ok"]

    monkeypatch.setattr(pilot, "_predict_with_adapter_assignments", predict)
    result = pilot._routing_probe(
        cast(Any, object()),
        cast(Any, object()),
        cast(Any, router),
        stream,
        max_new_tokens=7,
        batch_size=2,
    )
    assert result == {
        "total_prompts": 4,
        "eligible_prompts": 2,
        "fallback_prompts": 2,
        "routed_eligible_prompts": 1,
        "coverage": 0.5,
        "misroute_rate": 0.5,
        "eligible_misroute_rate": 0.0,
        "fallback_false_route_rate": 0.5,
        "routed_accuracy": 0.5,
        "task_id_oracle_accuracy": 1.0,
        "task_id_oracle_gap": 0.5,
        "confidence_threshold": 0.65,
    }
    assert calls[0][1] == ["expert-a", None, "wrong", None]
    assert calls[1] == (["a-3-1"], ["expert-a"], 7, 2)


def test_routing_probe_rejects_a_stream_with_no_eligible_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = pilot._BoundedPairStream(cast(Any, _Stream(count=1)), (0, 1), 1, 0)
    router = SimpleNamespace(
        task_families=[],
        expert_ids=[],
        confidence_threshold=0.65,
        route=lambda features: [SimpleNamespace(expert_id=None) for _ in range(len(features))],
    )
    monkeypatch.setattr(pilot, "_prompt_features", lambda _base, rows: torch.ones((len(rows), 1)))
    monkeypatch.setattr(
        pilot,
        "_predict_with_adapter_assignments",
        lambda _base, _bank, rows, _ids, **_kwargs: ["bad"] * len(rows),
    )
    with pytest.raises(RuntimeError, match="no task-ID-eligible"):
        pilot._routing_probe(
            cast(Any, object()),
            cast(Any, object()),
            cast(Any, router),
            stream,
            max_new_tokens=7,
            batch_size=1,
        )


def test_capped_training_accounts_for_epoch_steps_tokens_and_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    instances = [TaskInstance(str(index), "a", "p", "ok") for index in range(5)]

    def collate(_base: Any, selected: list[TaskInstance]) -> CausalBatch:
        count = len(selected)
        return CausalBatch(
            input_ids=torch.ones((count, 2), dtype=torch.long),
            attention_mask=torch.ones((count, 2), dtype=torch.long),
            labels=torch.tensor([[-100, 1]] * count),
            task_ids=tuple(item.task_id for item in selected),
        )

    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda _base, _batch: (parameter.square(), torch.empty(0)),
    )
    result = pilot._train_capped_steps(
        cast(Any, object()),
        [parameter],
        instances,
        max_steps=10,
        max_target_tokens=10,
        micro_batch=2,
        effective_batch=4,
        learning_rate=0.01,
        failure_label="fixture",
    )
    assert result == {
        "losses": pytest.approx([1.0, result["losses"][1]]),
        "target_tokens": 5,
        "optimizer_steps": 2,
        "available_examples": 5,
        "consumed_examples": 5,
        "stopped_by": "one_epoch",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 4,
    }
    assert all(math.isfinite(value) for value in result["losses"])


@pytest.mark.parametrize(
    ("max_steps", "max_tokens", "expected"),
    (
        (1, 10, (1, 4, 4, "optimizer_step_cap")),
        (10, 4, (1, 4, 4, "target_token_cap")),
        (10, 3, (0, 0, 0, "target_token_cap")),
    ),
)
def test_capped_training_stops_before_exceeding_each_cap(
    monkeypatch: pytest.MonkeyPatch,
    max_steps: int,
    max_tokens: int,
    expected: tuple[int, int, int, str],
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    instances = [TaskInstance(str(index), "a", "p", "ok") for index in range(5)]

    def collate(_base: Any, selected: list[TaskInstance]) -> CausalBatch:
        return CausalBatch(
            input_ids=torch.ones((len(selected), 2), dtype=torch.long),
            attention_mask=torch.ones((len(selected), 2), dtype=torch.long),
            labels=torch.tensor([[-100, 1]] * len(selected)),
            task_ids=tuple(item.task_id for item in selected),
        )

    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda _base, _batch: (parameter.square(), torch.empty(0)),
    )
    result = pilot._train_capped_steps(
        cast(Any, object()),
        [parameter],
        instances,
        max_steps=max_steps,
        max_target_tokens=max_tokens,
        micro_batch=2,
        effective_batch=4,
        learning_rate=0.01,
        failure_label="fixture",
    )
    assert (
        result["optimizer_steps"],
        result["target_tokens"],
        result["consumed_examples"],
        result["stopped_by"],
    ) == expected


def test_capped_training_rejects_empty_and_nondivisible_batches() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    arguments = {
        "base": cast(Any, object()),
        "parameters": [parameter],
        "max_steps": 1,
        "max_target_tokens": 1,
        "learning_rate": 0.01,
        "failure_label": "fixture",
    }
    with pytest.raises(ValueError, match="at least one"):
        pilot._train_capped_steps(
            **arguments, instances=[], micro_batch=1, effective_batch=1
        )
    with pytest.raises(ValueError, match="must divide"):
        pilot._train_capped_steps(
            **arguments,
            instances=[TaskInstance("one", "a", "p", "ok")],
            micro_batch=2,
            effective_batch=3,
        )


def test_capped_training_scales_accumulated_gradients_by_batch_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    instances = [TaskInstance(str(index), "a", "p", "ok") for index in range(4)]

    def collate(_base: Any, selected: list[TaskInstance]) -> CausalBatch:
        return CausalBatch(
            input_ids=torch.ones((len(selected), 2), dtype=torch.long),
            attention_mask=torch.ones((len(selected), 2), dtype=torch.long),
            labels=torch.tensor([[-100, 1]] * len(selected)),
            task_ids=tuple(item.task_id for item in selected),
        )

    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda _base, _batch: (3.0 * parameter.square(), torch.empty(0)),
    )
    result = pilot._train_capped_steps(
        cast(Any, object()),
        [parameter],
        instances,
        max_steps=1,
        max_target_tokens=10,
        micro_batch=2,
        effective_batch=4,
        learning_rate=0.0,
        failure_label="fixture",
    )
    assert result["optimizer_steps"] == 1
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(6.0)


def test_token_lengths_adds_exactly_one_missing_eos() -> None:
    class Tokenizer:
        eos_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            if text == "prompt":
                return [7, 8]
            if text == "with-eos":
                return [3, 0]
            if "empty" in text:
                return []
            return [4]

    base = cast(Any, SimpleNamespace(tokenizer=Tokenizer()))
    instances = [
        TaskInstance("with", "a", "prompt", "with-eos"),
        TaskInstance("without", "a", "prompt", "without-eos"),
        TaskInstance("empty", "a", "prompt", "empty"),
    ]
    assert pilot._token_lengths(base, instances) == {"with": 4, "without": 4, "empty": 3}
    base.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="integer EOS"):
        pilot._token_lengths(base, instances)


def test_token_lengths_compares_the_last_token_for_eos() -> None:
    class Tokenizer:
        eos_token_id = 5

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [7] if text == "prompt" else [5, 3]

    instance = TaskInstance("one", "a", "prompt", "answer")
    assert pilot._token_lengths(
        cast(Any, SimpleNamespace(tokenizer=Tokenizer())), [instance]
    ) == {"one": 4}


def test_longest_distinct_examples_accepts_one_and_rejects_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = TaskInstance("only", "a", "prompt", "ok")
    stream = SimpleNamespace(families=[], get_split=lambda _family: None)
    monkeypatch.setattr(pilot, "_all_stream_instances", lambda _stream: [instance])
    monkeypatch.setattr(pilot, "_token_lengths", lambda _base, _instances: {"only": 3})
    assert pilot.longest_distinct_examples(cast(Any, object()), cast(Any, stream), 1) == (
        [instance],
        [3],
    )
    with pytest.raises(ValueError, match="must be positive"):
        pilot.longest_distinct_examples(cast(Any, object()), cast(Any, stream), True)


def test_write_json_atomic_creates_nested_parent(tmp_path: Path) -> None:
    destination = tmp_path / "one" / "two" / "record.json"
    pilot._write_json_atomic(destination, {"value": 3})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 3}
    assert destination.read_text(encoding="utf-8") == '{\n  "value": 3\n}\n'
    assert list(destination.parent.glob("*.tmp")) == []


def test_seeded_order_is_deterministic_distinct_and_nonmutating() -> None:
    instances = [TaskInstance(str(index), "a", "p", "ok") for index in range(12)]
    original = [item.task_id for item in instances]
    first = [item.task_id for item in pilot._seeded_task_order(instances, 7, 0)]
    assert first == [item.task_id for item in pilot._seeded_task_order(instances, 7, 0)]
    assert first != [item.task_id for item in pilot._seeded_task_order(instances, 8, 0)]
    assert first != [item.task_id for item in pilot._seeded_task_order(instances, 7, 1)]
    assert [item.task_id for item in instances] == original


def test_pair_metrics_reports_exact_in_domain_and_ood_fields() -> None:
    matrix = [[0.8, 0.2], [0.5, 0.9]]
    ood = [[0.6, 0.1], [0.4, 0.7]]
    result = pilot._pair_metrics(
        matrix,
        [0.1, 0.15],
        template_ood_matrix=ood,
        template_ood_frozen=[0.05, 0.02],
    )
    assert result["accuracy_matrix"] == matrix
    assert result["frozen_base_accuracy"] == [0.1, 0.15]
    assert result["per_task_accuracy_after_training"] == [0.5, 0.9]
    assert result["new_task_acquisition"] == pytest.approx(0.85)
    assert result["backward_transfer"] == pytest.approx(-0.3)
    assert result["aggregate_forgetting"] == pytest.approx(0.3)
    assert result["final_average"] == pytest.approx(0.7)
    assert result["template_ood_accuracy_matrix"] == ood
    assert result["template_ood_frozen_base_accuracy"] == [0.05, 0.02]
    assert result["template_ood_per_task_accuracy_after_training"] == [0.4, 0.7]
    assert result["template_ood_new_task_acquisition"] == pytest.approx(0.65)
    assert result["template_ood_final_average"] == pytest.approx(0.55)


def test_predict_with_assignments_groups_chunks_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = [TaskInstance(str(index), "a", "p", "ok") for index in range(7)]
    assignments = ["a", None, "a", "b", None, "a", None]
    actions: list[str] = []
    chunks: list[list[str]] = []
    bank = SimpleNamespace(
        activate=lambda name: actions.append(f"activate:{name}"),
        deactivate=lambda: actions.append("deactivate"),
    )

    def predict(_base: Any, selected: Any, max_new_tokens: int) -> list[str]:
        assert max_new_tokens == 9
        chunks.append([item.task_id for item in selected])
        return [f"out:{item.task_id}" for item in selected]

    monkeypatch.setattr(pilot, "_base_predict_many", predict)
    assert pilot._predict_with_adapter_assignments(
        cast(Any, object()),
        cast(Any, bank),
        instances,
        assignments,
        max_new_tokens=9,
        batch_size=2,
    ) == [f"out:{index}" for index in range(7)]
    assert actions == ["activate:a", "deactivate", "activate:b"]
    assert chunks == [["0", "2"], ["5"], ["1", "4"], ["6"], ["3"]]
    with pytest.raises(ValueError, match="align"):
        pilot._predict_with_adapter_assignments(
            cast(Any, object()),
            cast(Any, bank),
            instances,
            assignments[:-1],
            max_new_tokens=9,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="align"):
        pilot._predict_with_adapter_assignments(
            cast(Any, object()),
            cast(Any, bank),
            instances,
            [*assignments, "extra"],
            max_new_tokens=9,
            batch_size=2,
        )


def test_evaluation_profile_records_exact_identity_timing_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    source = _Stream(count=64)
    base = SimpleNamespace(model=torch.nn.Linear(1, 1), device=torch.device("cpu"))
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: base)
    cleaned: list[Any] = []
    monkeypatch.setattr(pilot, "_cleanup_base", cleaned.append)
    synchronizations: list[torch.device] = []
    monkeypatch.setattr(pilot, "_synchronize", synchronizations.append)
    clock = iter((0.0, 1.0, 5.0, 6.0, 8.0, 9.0))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))

    def result(instance: TaskInstance) -> GenerationResult:
        index = int(instance.task_id.rsplit("-", 1)[1])
        return GenerationResult(f"text-{index}", (index, index + 1), "length")

    monkeypatch.setattr(
        pilot,
        "generate_task_instance_answer",
        lambda _base, instance, *, max_new_tokens: result(instance),
    )
    batch_sizes: list[int] = []

    def generate_many(
        _base: Any,
        instances: Any,
        *,
        max_new_tokens: int,
    ) -> tuple[GenerationResult, ...]:
        selected = list(instances)
        batch_sizes.append(len(selected))
        return tuple(result(instance) for instance in selected)

    monkeypatch.setattr(pilot, "generate_task_instance_answers", generate_many)
    record = pilot.run_evaluation_profile_cell(
        "synthetic_v2",
        cast(Any, source),
        (0, 1),
        settings,
        _quiet(),
    )
    assert record["status"] == "completed"
    assert record["examples"] == settings.repeatability.profile_examples
    assert record["configured_batch_size"] == 8
    assert record["scalar_seconds"] == 4.0
    assert record["batched_seconds"] == 2.0
    assert record["scalar_seconds_per_example"] == 0.5
    assert record["batched_seconds_per_example"] == 0.25
    assert record["speedup"] == 2.0
    assert record["output_identity"] is True
    digest = hashlib.sha256()
    for index in (*range(4), *range(4)):
        generated = GenerationResult(f"text-{index}", (index, index + 1), "length")
        digest.update(
            json.dumps(
                {
                    "ids": generated.generated_token_ids,
                    "stop_reason": generated.stop_reason,
                    "matched_terminator": generated.matched_terminator,
                    "text_sha256": hashlib.sha256(generated.text.encode()).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    assert record["output_sha256"] == digest.hexdigest()
    assert record["wall_time_seconds"] == 9.0
    assert batch_sizes == [8]
    assert synchronizations == [base.device] * 4
    assert cleaned == [base]


def test_evaluation_profile_fails_closed_on_changed_batched_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    source = _Stream(count=64)
    base = SimpleNamespace(model=torch.nn.Linear(1, 1), device=torch.device("cpu"))
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: base)
    cleaned: list[Any] = []
    monkeypatch.setattr(pilot, "_cleanup_base", cleaned.append)
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    clock = iter(float(value) for value in range(20))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))
    scalar = GenerationResult("same", (1,), "length")
    changed = GenerationResult("changed", (2,), "length")
    monkeypatch.setattr(pilot, "generate_task_instance_answer", lambda *args, **kwargs: scalar)
    monkeypatch.setattr(
        pilot,
        "generate_task_instance_answers",
        lambda _base, instances, **_kwargs: tuple(changed for _ in instances),
    )
    record = pilot.run_evaluation_profile_cell(
        "synthetic_v2",
        cast(Any, source),
        (0, 1),
        settings,
        _quiet(),
    )
    assert record["status"] == "failed"
    assert record["failure"]["type"] == "RuntimeError"
    assert cleaned == [base]


def _noise_records() -> list[dict[str, Any]]:
    seeds = (7, 7, 11, 13, 17)
    records: list[dict[str, Any]] = []
    method_values = {
        "naive_sequential": {
            "backward_transfer": (0.10, 0.15, 0.20, 0.40, 0.80),
            "aggregate_forgetting": (0.80, 0.70, 0.60, 0.30, 0.10),
            "final_average": (0.20, 0.25, 0.45, 0.55, 0.75),
            "new_task_acquisition": (0.30, 0.35, 0.50, 0.70, 0.90),
            "template_ood_final_average": (0.05, 0.10, 0.15, 0.35, 0.45),
            "template_ood_new_task_acquisition": (0.10, 0.20, 0.40, 0.50, 0.70),
        },
        "aptivra_full": {
            "backward_transfer": (0.20, 0.22, 0.35, 0.60, 0.95),
            "aggregate_forgetting": (0.70, 0.65, 0.40, 0.25, 0.05),
            "final_average": (0.30, 0.32, 0.60, 0.80, 0.90),
            "new_task_acquisition": (0.35, 0.38, 0.65, 0.85, 0.95),
            "template_ood_final_average": (0.10, 0.12, 0.30, 0.55, 0.65),
            "template_ood_new_task_acquisition": (0.20, 0.25, 0.55, 0.75, 0.90),
        },
    }
    for method, values in method_values.items():
        for index, seed in enumerate(seeds):
            records.append(
                {
                    "status": "completed",
                    "method": method,
                    "stream": "synthetic_v2",
                    "seed": seed,
                    "repeat_index": index,
                    "metrics": {
                        **{name: row[index] for name, row in values.items()},
                        "per_task_accuracy_after_training": [
                            values["final_average"][index],
                            values["new_task_acquisition"][index],
                        ],
                        "demonstrably_learned": True,
                    },
                }
            )
    return records


def test_noise_floor_uses_registered_seed_subset_and_distinct_metrics() -> None:
    records = list(reversed(_noise_records()))
    ignored = copy.deepcopy(records[0])
    ignored["status"] = "abandoned"
    records.append(ignored)
    result = pilot.noise_floor(records)
    naive = result["naive_sequential:synthetic_v2"]
    assert naive["status"] == "completed"
    assert naive["learning_by_run"] == [True] * 5
    assert naive["seed_count"] == 4
    assert naive["same_seed_absolute_variation"] == pytest.approx(
        {
            "backward_transfer": 0.05,
            "aggregate_forgetting": 0.10,
            "final_average": 0.05,
            "new_task_acquisition": 0.05,
            "per_task_accuracy": 0.05,
            "template_ood_final_average": 0.05,
            "template_ood_new_task_acquisition": 0.10,
        }
    )
    unique = (0, 2, 3, 4)
    expected_final = np.std([0.20, 0.45, 0.55, 0.75], ddof=1)
    assert naive["seed_to_seed_standard_deviation"]["final_average"] == pytest.approx(
        expected_final
    )
    expected_per_task = [
        np.std(values, ddof=1)
        for values in (
            [0.20, 0.45, 0.55, 0.75],
            [0.30, 0.50, 0.70, 0.90],
        )
    ]
    assert naive["seed_to_seed_standard_deviation"]["per_task_accuracy"] == pytest.approx(
        expected_per_task
    )
    paired = result["paired_difference:synthetic_v2"]
    naive_cells = sorted(
        (
            row
            for row in records
            if row.get("status") == "completed"
            and row["method"] == "naive_sequential"
        ),
        key=lambda row: row["repeat_index"],
    )
    aptivra_cells = sorted(
        (
            row
            for row in records
            if row.get("status") == "completed" and row["method"] == "aptivra_full"
        ),
        key=lambda row: row["repeat_index"],
    )
    expected_paired: dict[str, Any] = {}
    for name in (
        "backward_transfer",
        "aggregate_forgetting",
        "final_average",
        "new_task_acquisition",
        "template_ood_final_average",
        "template_ood_new_task_acquisition",
    ):
        differences = [
            aptivra_cells[index]["metrics"][name] - naive_cells[index]["metrics"][name]
            for index in unique
        ]
        expected_paired[name] = np.std(differences, ddof=1)
    expected_paired["per_task_accuracy"] = [
        np.std(
            [
                aptivra_cells[index]["metrics"]["per_task_accuracy_after_training"][task]
                - naive_cells[index]["metrics"]["per_task_accuracy_after_training"][task]
                for index in unique
            ],
            ddof=1,
        )
        for task in range(2)
    ]
    assert paired["status"] == "completed"
    assert paired["aptivra_minus_naive_standard_deviation"] == pytest.approx(
        expected_paired
    )
    assert paired["seed_count"] == 4


def test_noise_floor_continues_after_an_unpaired_stream() -> None:
    synthetic = _noise_records()
    issue_naive = copy.deepcopy(
        [row for row in synthetic if row["method"] == "naive_sequential"]
    )
    for row in issue_naive:
        row["stream"] = "chronological_issues"
    result = pilot.noise_floor([*issue_naive, *synthetic])
    assert "paired_difference:chronological_issues" not in result
    assert result["paired_difference:synthetic_v2"]["status"] == "completed"


@pytest.mark.parametrize(("seed_two", "seed_three"), ((13, 11), (10, 14)))
def test_noise_floor_rejects_paired_seed_order_mismatch(
    seed_two: int, seed_three: int
) -> None:
    records = _noise_records()
    aptivra = [row for row in records if row["method"] == "aptivra_full"]
    aptivra[2]["seed"] = seed_two
    aptivra[3]["seed"] = seed_three
    with pytest.raises(RuntimeError, match="paired noise-floor seeds"):
        pilot.noise_floor(records)


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda records: records[0].update(repeat_index=1),
        lambda records: records[1].update(seed=8),
        lambda records: records[2].update(seed=7),
    ),
)
def test_noise_floor_rejects_corrupt_repeatability_design(corrupt: Any) -> None:
    records = _noise_records()
    corrupt(records)
    with pytest.raises(RuntimeError, match="noise floor (repeat indices|seed design)"):
        pilot.noise_floor(records)


def _prepare_run_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pilot.MpsPilotSettings:
    settings = _settings().model_copy(update={"output_dir": tmp_path})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pilot, "validate_claim_streams", lambda _: {"audited": True})
    monkeypatch.setattr(pilot, "_require_hash", lambda *_: "hash")
    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(pilot, "pilot_run_manifest", lambda *args: {"identity": "same"})
    monkeypatch.setattr(
        pilot,
        "_make_streams",
        lambda _: {"synthetic_v2": object(), "chronological_issues": object()},
    )
    monkeypatch.setattr(pilot, "_quiet_before_measurement", lambda *args: _quiet())
    monkeypatch.setattr(pilot, "noise_floor", lambda _: {"status": "completed"})
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: object())
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(
        pilot, "compute_estimate", lambda *args: {"status": "completed", "hours": 1.0}
    )
    return settings


def _complete_progress(settings: pilot.MpsPilotSettings) -> dict[str, Any]:
    return {
        "run_manifest": {"identity": "same"},
        "status": "running",
        "elapsed_wall_time_seconds": 1.0,
        "quiet_gate_events": [],
        "evaluation_profiles": [
            {"status": "completed", "stream": stream}
            for stream in ("synthetic_v2", "chronological_issues")
        ],
        "calibration": [
            {
                "status": "completed",
                "method": cell.method,
                "stream": cell.stream,
                "micro_batch": cell.micro_batch,
                "gradient_checkpointing": False,
            }
            for cell in settings.calibration.anomaly_cells
        ],
        "repeatability": [
            {
                "status": "completed",
                "stream": stream,
                "method": method,
                "seed": seed,
                "repeat_index": index,
            }
            for stream in ("synthetic_v2", "chronological_issues")
            for method in settings.repeatability.methods
            for index, seed in enumerate(settings.repeatability.seeds)
        ],
    }


def test_run_pilot_checks_each_precondition_before_allocating_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    enabled = settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": True})}
    )
    with pytest.raises(RuntimeError, match="must remain disabled"):
        pilot.run_pilot(enabled, authorized=True)

    for missing in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        monkeypatch.delenv(missing)
        with pytest.raises(RuntimeError, match="requires HF_HUB_OFFLINE"):
            pilot.run_pilot(settings, authorized=True)


@pytest.mark.parametrize(
    "historical",
    (
        {"schema_version": "0", "status": "completed", "calibration": []},
        {"schema_version": pilot.PILOT_SCHEMA, "status": "failed", "calibration": []},
        {
            "schema_version": pilot.PILOT_SCHEMA,
            "status": "completed",
            "calibration": {},
        },
    ),
)
def test_run_pilot_rejects_each_historical_evidence_clause(
    monkeypatch: pytest.MonkeyPatch,
    historical: dict[str, Any],
) -> None:
    settings = _settings()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pilot, "validate_claim_streams", lambda _: {"audited": True})
    monkeypatch.setattr(pilot, "_require_hash", lambda *_: "hash")
    monkeypatch.setattr(pilot, "_json_mapping", lambda _: historical)
    with pytest.raises(ValueError, match="not completed v1"):
        pilot.run_pilot(settings, authorized=True)


def test_run_pilot_rejects_unavailable_mps_before_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pilot, "validate_claim_streams", lambda _: {"audited": True})
    monkeypatch.setattr(pilot, "_require_hash", lambda *_: "hash")
    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: False)
    touched = False

    def manifest(*_args: Any) -> dict[str, Any]:
        nonlocal touched
        touched = True
        return {}

    monkeypatch.setattr(pilot, "pilot_run_manifest", manifest)
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        pilot.run_pilot(settings, authorized=True)
    assert touched is False


def test_run_pilot_persists_redacted_final_failure_from_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare_run_pilot(tmp_path, monkeypatch)
    clock = iter((10.0, 11.0, 13.0))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))
    pilot._write_json_atomic(
        tmp_path / "lane8b-pilot-progress.json", _complete_progress(settings)
    )
    monkeypatch.setattr(
        pilot,
        "noise_floor",
        lambda _: (_ for _ in ()).throw(RuntimeError("private analysis detail")),
    )
    with pytest.raises(RuntimeError, match="private analysis detail"):
        pilot.run_pilot(settings, authorized=True)
    progress = pilot._json_mapping(tmp_path / "lane8b-pilot-progress.json")
    final = pilot._json_mapping(tmp_path / "lane8b-pilot-evidence.raw.json")
    assert final == progress
    assert final["status"] == "failed"
    assert final["failure"] == {
        "type": "RuntimeError",
        "message": "exception details redacted from portable evidence",
        "out_of_memory": False,
    }
    assert final["elapsed_wall_time_seconds"] == final["pilot_wall_time_seconds"]
    assert final["elapsed_wall_time_seconds"] == 4.0
    assert "private analysis detail" not in str(final)


def test_run_pilot_fails_closed_and_cleans_up_an_incomplete_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare_run_pilot(tmp_path, monkeypatch)
    pilot._write_json_atomic(
        tmp_path / "lane8b-pilot-progress.json", _complete_progress(settings)
    )
    monkeypatch.setattr(pilot, "noise_floor", lambda _: {"status": "completed"})
    base = object()
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: base)
    cleaned: list[Any] = []
    monkeypatch.setattr(pilot, "_cleanup_base", cleaned.append)
    monkeypatch.setattr(pilot, "compute_estimate", lambda *args: {"status": "incomplete"})
    with pytest.raises(RuntimeError, match="compute estimate is incomplete"):
        pilot.run_pilot(settings, authorized=True)
    assert cleaned == [base]
    final = pilot._json_mapping(tmp_path / "lane8b-pilot-evidence.raw.json")
    assert final["status"] == "failed"


def test_run_pilot_retries_only_failed_resumable_cells_and_clears_stale_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare_run_pilot(tmp_path, monkeypatch)
    profiles = [
        {"status": "failed", "stream": "synthetic_v2"},
        {"status": "completed", "stream": "chronological_issues"},
    ]
    calibration = [
        {
            "status": "completed",
            "method": cell.method,
            "stream": cell.stream,
            "micro_batch": cell.micro_batch,
            "gradient_checkpointing": False,
        }
        for cell in settings.calibration.anomaly_cells
    ]
    repeats: list[dict[str, Any]] = []
    for stream in ("synthetic_v2", "chronological_issues"):
        for method in settings.repeatability.methods:
            for index, seed in enumerate(settings.repeatability.seeds):
                repeats.append(
                    {
                        "status": (
                            "failed"
                            if (stream, method, index)
                            == ("synthetic_v2", "naive_sequential", 0)
                            else "completed"
                        ),
                        "stream": stream,
                        "method": method,
                        "seed": seed,
                        "repeat_index": index,
                    }
                )
    progress = {
        "run_manifest": {"identity": "same"},
        "status": "failed",
        "failure": {"type": "old"},
        "elapsed_wall_time_seconds": 1.0,
        "quiet_gate_events": [],
        "evaluation_profiles": profiles,
        "calibration": calibration,
        "repeatability": repeats,
    }
    pilot._write_json_atomic(tmp_path / "lane8b-pilot-progress.json", progress)
    pilot._write_json_atomic(tmp_path / "lane8b-pilot-evidence.raw.json", {"stale": True})
    calls: list[list[str]] = []

    def spawn(
        _config: Path,
        arguments: list[str],
        _quiet_record: Any,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert timeout_seconds > 0.0
        calls.append(arguments)
        if arguments[0] == "--evaluation-profile-worker":
            return {"status": "completed", "stream": arguments[1]}
        return {
            "status": "completed",
            "method": arguments[1],
            "stream": arguments[2],
            "seed": int(arguments[3]),
            "repeat_index": int(arguments[4]),
        }

    monkeypatch.setattr(pilot, "_spawn_worker", spawn)
    result = pilot.run_pilot(settings, authorized=True)
    assert calls == [
        ["--evaluation-profile-worker", "synthetic_v2"],
        [
            "--repeatability-worker",
            "naive_sequential",
            "synthetic_v2",
            str(settings.repeatability.seeds[0]),
            "0",
        ],
    ]
    assert result["status"] == "completed"
    assert "failure" not in result
    assert pilot._json_mapping(tmp_path / "lane8b-pilot-evidence.raw.json")["status"] == (
        "completed"
    )


@pytest.mark.parametrize("elapsed", (True, -1.0, math.nan, math.inf, 28_800.0))
def test_run_pilot_rejects_invalid_resume_elapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elapsed: Any,
) -> None:
    settings = _prepare_run_pilot(tmp_path, monkeypatch)
    (tmp_path / "lane8b-pilot-progress.json").write_text(
        json.dumps(
            {"run_manifest": {"identity": "same"}, "elapsed_wall_time_seconds": elapsed}
        ),
        encoding="utf-8",
    )
    error = ValueError if isinstance(elapsed, bool) or elapsed != 28_800.0 else TimeoutError
    with pytest.raises(error):
        pilot.run_pilot(settings, authorized=True)
