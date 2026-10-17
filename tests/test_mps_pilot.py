"""Unit proof for the bounded ADR-0011 lane-8a pilot."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from pydantic import ValidationError

import aptivra.eval.mps_pilot as pilot
from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, exact_match_result
from aptivra.tasks.stream import TaskStreamSplit


class _Verifier(MechanicalVerifier):
    def verify(self, instance: TaskInstance, model_output: str) -> Any:
        return exact_match_result(instance, model_output)


class _Family(TaskFamily):
    def __init__(self, family_id: str) -> None:
        super().__init__(family_id, family_id, "pilot fixture")
        self._verifier = _Verifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        return [
            TaskInstance(
                f"{self.family_id}-{seed}-{index}",
                self.family_id,
                "x " * (index + 1),
                "ok",
            )
            for index in range(count)
        ]


class _Stream:
    def __init__(self, counts: tuple[int, int] = (10, 10)) -> None:
        self.families = [_Family("a"), _Family("b")]
        self._splits = {
            family.family_id: TaskStreamSplit(
                train=family.generate_instances(counts[index], 1),
                val=family.generate_instances(counts[index], 2),
                test=family.generate_instances(counts[index], 3),
            )
            for index, family in enumerate(self.families)
        }

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(max(1, len(text.split()))))


def _fake_base() -> Any:
    return SimpleNamespace(tokenizer=_Tokenizer())


def _settings() -> pilot.MpsPilotSettings:
    return pilot.load_mps_pilot_settings("configs/h3_mps_pilot.yaml")


def _quiet_record() -> dict[str, Any]:
    return {"quiet": True, "wait_seconds": 0.0, "samples": [{"quiet": True}]}


def test_registered_config_is_mirrored_disabled_and_protocol_complete() -> None:
    settings = _settings()
    assert settings.feature.enabled is False
    assert settings.streams.synthetic_v2.pair_task_indices == (0, 2)
    assert settings.calibration.methods == list(pilot._METHODS)
    assert settings.calibration.micro_batches == [8, 4, 2, 1]
    assert settings.calibration.frozen_base_mode == "forward_only"
    assert settings.calibration.resident_adapter_count == 12
    assert settings.h3_estimate.main_seed_count >= 5
    assert settings.h3_estimate.ablation_stream_count == 2
    assert len(settings.h3_estimate.ablations) == 11
    assert settings.cycle_caps.evaluation_examples_per_seen_task == 363
    assert (
        Path("configs/h3_mps_pilot.yaml").read_bytes()
        == Path("aptivra/resources/configs/h3_mps_pilot.yaml").read_bytes()
    )


def test_dynamic_router_candidate_is_placed_on_requested_device() -> None:
    router = pilot.SequenceTop1Router(4, 0.5)
    pilot._add_router_candidate_on_device(router, "expert", "family", torch.device("meta"))
    weight, bias = router.candidate_parameters()
    assert weight.device.type == "meta"
    assert bias.device.type == "meta"


def test_empty_router_replay_batch_has_device_correct_shapes() -> None:
    base = cast(
        Any,
        SimpleNamespace(
            settings=SimpleNamespace(architecture=SimpleNamespace(hidden_size=7)),
            device=torch.device("cpu"),
        ),
    )
    features, labels = pilot._router_replay_batch(base, [], 8)
    assert features.shape == (0, 7)
    assert labels.shape == (0,)
    assert labels.dtype == torch.long


def test_routing_rates_separate_coverage_misroutes_and_fallback() -> None:
    rates = pilot._routing_decision_rates(
        ["a", None, "wrong", "false"],
        ["a", "b", "b", None],
    )
    assert rates == {
        "total_prompts": 4,
        "eligible_prompts": 3,
        "fallback_prompts": 1,
        "routed_eligible_prompts": 2,
        "coverage": 2 / 3,
        "misroute_rate": 2 / 3,
        "eligible_misroute_rate": 0.5,
        "fallback_false_route_rate": 1.0,
    }
    with pytest.raises(ValueError, match="equal non-empty"):
        pilot._routing_decision_rates([], [])
    with pytest.raises(ValueError, match="equal non-empty"):
        pilot._routing_decision_rates(["a"], ["a", "b"])


def test_config_rejects_wrong_root_order_and_repeat_semantics(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contain only"):
        pilot.load_mps_pilot_settings(wrong)

    raw = yaml.safe_load(Path("configs/h3_mps_pilot.yaml").read_text(encoding="utf-8"))
    raw["h3_mps_pilot"]["calibration"]["micro_batches"] = [4, 8, 2, 1]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="micro-batches"):
        pilot.load_mps_pilot_settings(bad)

    raw = yaml.safe_load(Path("configs/h3_mps_pilot.yaml").read_text(encoding="utf-8"))
    raw["h3_mps_pilot"]["repeatability"]["seeds"] = [1, 2, 3, 4]
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="same-seed"):
        pilot.load_mps_pilot_settings(bad)


def test_required_hash_reports_missing_and_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        pilot._require_hash(missing, "0" * 64, "raw data")
    path = tmp_path / "value"
    path.write_text("value", encoding="utf-8")
    with pytest.raises(ValueError, match="raw data SHA-256 mismatch"):
        pilot._require_hash(path, "0" * 64, "raw data")
    assert pilot._require_hash(path, pilot._sha256(path), "raw data") == pilot._sha256(path)
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        pilot._json_mapping(invalid_json)


def test_additional_config_invariants_fail_closed(tmp_path: Path) -> None:
    original = yaml.safe_load(Path("configs/h3_mps_pilot.yaml").read_text(encoding="utf-8"))
    mutations = (
        (
            lambda raw: raw["h3_mps_pilot"]["quiet_machine"].update(initial_backoff_seconds=61.0),
            "initial backoff",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["calibration"]["methods"].reverse(),
            "six-method order",
        ),
        (
            lambda raw: raw["h3_mps_pilot"]["h3_estimate"]["ablations"].__setitem__(
                1, raw["h3_mps_pilot"]["h3_estimate"]["ablations"][0]
            ),
            "uniquely named",
        ),
    )
    for index, (mutate, message) in enumerate(mutations):
        raw = yaml.safe_load(yaml.safe_dump(original))
        mutate(raw)
        path = tmp_path / f"invalid-{index}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValidationError, match=message):
            pilot.load_mps_pilot_settings(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"quiet_machine": {"minimum_free_memory_percent": 0.0}}, "greater than or equal"),
        ({"quiet_machine": {"maximum_backoff_seconds": 901.0}}, "cannot exceed its timeout"),
        ({"calibration": {"warmup_steps": 2}}, "Input should be 1"),
        ({"calibration": {"memory_cap_bytes": 1}}, "Input should be 28991029248"),
        ({"calibration": {"memory_sample_interval_seconds": 2.0}}, "less than or equal"),
        ({"output_dir": "../outside"}, "resolve inside the worktree"),
    ),
)
def test_protocol_bounds_reject_unsafe_values(
    tmp_path: Path, mutation: dict[str, Any], message: str
) -> None:
    raw = yaml.safe_load(Path("configs/h3_mps_pilot.yaml").read_text(encoding="utf-8"))
    for section, values in mutation.items():
        if isinstance(values, dict):
            raw["h3_mps_pilot"][section].update(values)
        else:
            raw["h3_mps_pilot"][section] = values
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match=message):
        pilot.load_mps_pilot_settings(path)


def test_manifest_and_environment_bind_local_inputs() -> None:
    settings = _settings()
    manifest = pilot.pilot_run_manifest(
        settings,
        Path("configs/h3_mps_pilot.yaml"),
        {"audited": True},
    )
    assert manifest["model"]["unique_parameter_count"] == 361_821_120
    assert manifest["claim_stream_preflight"] == {"audited": True}
    environment = pilot._environment()
    assert environment["machine"]
    assert environment["torch"] == torch.__version__


def test_preflight_missing_raw_halts_before_stream_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    issues = settings.streams.chronological_issues.model_copy(
        update={"raw_data": tmp_path / "absent.jsonl"}
    )
    changed = settings.model_copy(
        update={"streams": settings.streams.model_copy(update={"chronological_issues": issues})}
    )
    monkeypatch.setattr(pilot, "write_or_check_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pilot,
        "_require_hash",
        lambda path, expected, description: expected,
    )
    with pytest.raises(FileNotFoundError, match="claim-bearing stream raw data is missing"):
        pilot.validate_claim_streams(changed)


def test_longest_examples_are_distinct_stable_and_pad_bound() -> None:
    stream = _Stream()
    selected, lengths = pilot.longest_distinct_examples(
        cast(Any, _fake_base()), cast(Any, stream), 8
    )
    assert len(selected) == len({item.task_id for item in selected}) == 8
    assert lengths == sorted(lengths, reverse=True)
    tied = [
        item.task_id
        for item, length in zip(selected, lengths, strict=True)
        if length == lengths[-1]
    ]
    assert tied == sorted(tied)
    with pytest.raises(ValueError, match="fewer than 100"):
        pilot.longest_distinct_examples(cast(Any, _fake_base()), cast(Any, stream), 100)
    with pytest.raises(ValueError, match="positive"):
        pilot.longest_distinct_examples(cast(Any, _fake_base()), cast(Any, stream), 0)


def test_duplicate_stable_ids_are_rejected() -> None:
    stream = _Stream()
    first = stream._splits["a"].train[0]
    stream._splits["b"].test[0] = TaskInstance(first.task_id, "b", "duplicate", "ok")
    with pytest.raises(ValueError, match="duplicate stable task IDs"):
        pilot._all_stream_instances(cast(Any, stream))


def test_quiet_wait_uses_backoff_and_never_mutates_processes() -> None:
    settings = _settings().quiet_machine.model_copy(
        update={
            "timeout_seconds": 20.0,
            "initial_backoff_seconds": 2.0,
            "maximum_backoff_seconds": 4.0,
        }
    )
    samples = iter(({"quiet": False}, {"quiet": False}, {"quiet": True}))
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    result = pilot.wait_for_quiet_machine(
        settings,
        sleep=sleep,
        monotonic=lambda: clock[0],
        sample=lambda _: next(samples),
    )
    assert result["quiet"] is True
    assert sleeps == [2.0, 4.0]
    assert result["wait_seconds"] == 6.0


def test_quiet_wait_times_out() -> None:
    settings = _settings().quiet_machine.model_copy(
        update={"timeout_seconds": 1.0, "initial_backoff_seconds": 1.0}
    )
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    with pytest.raises(
        pilot.QuietMachineTimeoutError, match="without killing or renicing"
    ) as caught:
        pilot.wait_for_quiet_machine(
            settings,
            sleep=sleep,
            monotonic=lambda: clock[0],
            sample=lambda _: {"quiet": False},
        )
    assert caught.value.wait_seconds == 1.0
    assert caught.value.samples == [{"quiet": False}, {"quiet": False}]


def test_expired_quiet_window_is_checkpointed_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    progress: dict[str, Any] = {"quiet_gate_events": []}
    attempts = iter(
        (
            pilot.QuietMachineTimeoutError(900.0, [{"quiet": False, "cpu": 150.0}]),
            {"quiet": True},
        )
    )

    def wait(_: Any) -> dict[str, Any]:
        value = next(attempts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(pilot, "wait_for_quiet_machine", wait)
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: 1.0)
    path = tmp_path / "progress.json"
    result = pilot._quiet_before_measurement(settings, progress, path, 0.0)
    assert result == {"quiet": True}
    assert progress["quiet_gate_events"][0]["samples"][0]["cpu"] == 150.0
    assert pilot._json_mapping(path)["quiet_gate_events"][0]["status"] == "timed_out"


def test_process_parsing_and_quiet_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "1 0 0.0 10 init\n2 1 101.0 5 hot\n3 1 0.0 3000000 huge\n"
    completed = SimpleNamespace(stdout=output)
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: completed)
    samples = pilot._process_samples()
    assert samples[1].command == "hot"
    monkeypatch.setattr(pilot.os, "getpid", lambda: 99)
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 80.0)
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: 123)
    result = pilot.quiet_machine_sample(_settings().quiet_machine)
    assert result["quiet"] is False
    assert len(result["other_processes_above_cpu_threshold"]) == 1
    assert len(result["other_processes_above_rss_threshold"]) == 1


def test_bounded_pair_uses_real_prefixes_and_rejects_bad_indices() -> None:
    stream = _Stream()
    pair = pilot._BoundedPairStream(cast(Any, stream), (0, 1), 4, 3)
    assert [family.family_id for family in pair.families] == ["a", "b"]
    assert len(pair.get_split("a").train) == 4
    with pytest.raises(ValueError, match="distinct real tasks"):
        pilot._BoundedPairStream(cast(Any, stream), (0, 0), 4, 3)
    with pytest.raises(ValueError, match="lacks configured real examples"):
        pilot._BoundedPairStream(cast(Any, stream), (0, 1), 50, 3)


def test_pair_metrics_and_noise_floor_are_exact() -> None:
    metrics = pilot._pair_metrics([[0.75, 0.1], [0.5, 0.8]], [0.0, 0.0])
    assert metrics["backward_transfer"] == -0.25
    assert metrics["aggregate_forgetting"] == 0.25
    assert metrics["final_average"] == 0.65
    records: list[dict[str, Any]] = []
    for method in ("naive_sequential", "aptivra_full"):
        for stream in ("synthetic_v2", "chronological_issues"):
            for index, value in enumerate((0.5, 0.625, 0.375, 0.75)):
                records.append(
                    {
                        "status": "completed",
                        "method": method,
                        "stream": stream,
                        "repeat_index": index,
                        "metrics": {
                            "backward_transfer": value,
                            "aggregate_forgetting": value,
                            "final_average": value,
                            "per_task_accuracy_after_training": [value, value],
                        },
                    }
                )
    floor = pilot.noise_floor(records)
    cell = floor["naive_sequential:synthetic_v2"]
    assert cell["same_seed_absolute_variation"]["final_average"] == 0.125
    assert cell["different_seed_max_absolute_variation"]["final_average"] == 0.25
    with pytest.raises(RuntimeError, match="incomplete"):
        pilot.noise_floor(records[:-1])


def test_compute_estimate_shows_exact_arithmetic() -> None:
    settings = _settings()
    calibration = [
        {
            "status": "completed",
            "method": method,
            "stream": stream,
            "tokens_per_second": 10.0,
            **({"generation": {"wall_time_seconds": 0.25}} if method == "frozen_base" else {}),
        }
        for method in pilot._METHODS
        for stream in ("synthetic_v2", "chronological_issues")
    ]
    streams = {
        "synthetic_v2": cast(Any, _Stream((2, 2))),
        "chronological_issues": cast(Any, _Stream((2, 2))),
    }
    result = pilot.compute_estimate(calibration, streams, cast(Any, _fake_base()), settings)
    arithmetic = result["arithmetic"]
    assert arithmetic["main_cells"] == 60
    assert arithmetic["v1_regression_cells"] == 6
    assert arithmetic["ablation_seed_stream_cells"] == 66
    assert arithmetic["total_seconds"] == (
        arithmetic["main_training_seconds"]
        + arithmetic["main_evaluation_seconds_at_cap"]
        + arithmetic["v1_regression_training_seconds"]
        + arithmetic["v1_regression_evaluation_seconds_at_cap"]
        + arithmetic["ablation_training_seconds"]
        + arithmetic["ablation_evaluation_seconds_at_cap"]
    )


def test_atomic_json_is_finite_and_replaces(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    pilot._write_json_atomic(path, {"value": 1})
    assert pilot._json_mapping(path) == {"value": 1}
    pilot._write_json_atomic(path, {"value": 2})
    assert pilot._json_mapping(path) == {"value": 2}
    with pytest.raises(ValueError):
        pilot._write_json_atomic(path, {"bad": float("nan")})


def test_run_pilot_refuses_without_authority_or_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    with pytest.raises(RuntimeError, match="explicit local"):
        pilot.run_pilot(settings, authorized=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="requires HF_HUB_OFFLINE"):
        pilot.run_pilot(settings, authorized=True)


def test_worker_spawn_records_success_and_process_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def success(command: list[str], **_: Any) -> Any:
        output = Path(command[command.index("--worker-output") + 1])
        pilot._write_json_atomic(
            output,
            {
                "status": "completed",
                "value": 7,
                "method": "frozen_base",
                "stream": "synthetic_v2",
                "micro_batch": 1,
                "gradient_checkpointing": "not_applicable",
                "quiet_machine": _quiet_record(),
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pilot.subprocess, "run", success)
    assert (
        pilot._spawn_worker(
            Path("configs/h3_mps_pilot.yaml"),
            ["--calibration-worker", "frozen_base", "synthetic_v2", "1"],
            _quiet_record(),
            timeout_seconds=5.0,
        )["value"]
        == 7
    )

    monkeypatch.setattr(
        pilot.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="worker-out", stderr="worker-error"
        ),
    )
    failed = pilot._spawn_worker(
        Path("configs/h3_mps_pilot.yaml"),
        ["--calibration-worker", "frozen_base", "synthetic_v2", "1"],
        _quiet_record(),
        timeout_seconds=5.0,
    )
    assert failed["status"] == "failed"
    assert failed["failure"]["returncode"] == 9
    assert failed["quiet_machine"] == _quiet_record()
    assert "stdout_tail" not in failed["failure"]
    assert failed["failure"]["stdout_bytes"] == len("worker-out")


def test_worker_spawn_timeout_is_redacted_and_retains_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise pilot.subprocess.TimeoutExpired(
            ["worker"], 5.0, output=b"private prompt", stderr=b"private target"
        )

    monkeypatch.setattr(pilot.subprocess, "run", timeout)
    result = pilot._spawn_worker(
        Path("configs/h3_mps_pilot.yaml"),
        ["--repeatability-worker", "naive_sequential", "synthetic_v2", "7", "2"],
        _quiet_record(),
        timeout_seconds=5.0,
    )
    assert result["status"] == "failed"
    assert result["method"] == "naive_sequential"
    assert result["seed"] == 7
    assert result["failure"] == {
        "type": "WorkerTimeout",
        "timeout_seconds": 5.0,
        "stdout_bytes": len(b"private prompt"),
        "stderr_bytes": len(b"private target"),
    }
    assert "private" not in str(result)


@pytest.mark.parametrize(
    "record",
    (
        {"quiet": False, "wait_seconds": 0.0, "samples": [{"quiet": False}]},
        {"quiet": 1, "wait_seconds": 0.0, "samples": [{"quiet": True}]},
        {"quiet": True},
    ),
)
def test_worker_rejects_forged_quiet_record_before_stream_load(
    record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    touched = False

    def streams(_: Any) -> dict[str, Any]:
        nonlocal touched
        touched = True
        return {}

    monkeypatch.setattr(pilot, "_make_streams", streams)
    with pytest.raises(ValueError, match="quiet record"):
        pilot.run_calibration_worker(
            _settings(), "frozen_base", "synthetic_v2", 1, record, checkpointing=False
        )
    assert touched is False


def test_worker_dispatch_rejects_unknown_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    monkeypatch.setattr(pilot, "_make_streams", lambda _: {"known": cast(Any, _Stream())})
    with pytest.raises(ValueError, match="unknown pilot stream"):
        pilot.run_calibration_worker(
            settings, "frozen_base", "missing", 1, _quiet_record(), checkpointing=False
        )
    with pytest.raises(ValueError, match="unknown pilot stream"):
        pilot.run_repeatability_worker(
            settings, "naive_sequential", "missing", 1, 0, _quiet_record()
        )


def test_run_pilot_orchestrates_all_cells_and_oom_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings().model_copy(update={"output_dir": tmp_path / "evidence"})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pilot, "validate_claim_streams", lambda _: {"audited": True})
    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(pilot, "pilot_run_manifest", lambda *args: {"identity": "v1"})
    streams = {
        "synthetic_v2": cast(Any, _Stream()),
        "chronological_issues": cast(Any, _Stream()),
    }
    monkeypatch.setattr(pilot, "_make_streams", lambda _: streams)
    monkeypatch.setattr(pilot, "wait_for_quiet_machine", lambda _: {"quiet": True})
    calls: list[list[str]] = []

    def spawn(
        _config: Path,
        arguments: list[str],
        _quiet: Any,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert timeout_seconds == settings.execution.wall_time_limit_seconds
        calls.append(arguments)
        if (
            arguments[:4]
            == [
                "--calibration-worker",
                "naive_sequential",
                "synthetic_v2",
                "1",
            ]
            and "--checkpointing" not in arguments
        ):
            return {
                "status": "failed",
                "method": arguments[1],
                "stream": arguments[2],
                "micro_batch": int(arguments[3]),
                "gradient_checkpointing": False,
                "accepted": False,
                "failure": {"out_of_memory": True},
            }
        if arguments[0] == "--repeatability-worker":
            return {
                "status": "completed",
                "method": arguments[1],
                "stream": arguments[2],
                "seed": int(arguments[3]),
                "repeat_index": int(arguments[4]),
                "metrics": {"value": 1},
            }
        return {
            "status": "completed",
            "method": arguments[1],
            "stream": arguments[2],
            "micro_batch": int(arguments[3]),
            "gradient_checkpointing": (
                "not_applicable"
                if arguments[1] == "frozen_base"
                else "--checkpointing" in arguments
            ),
            "accepted": True,
            "tokens_per_second": 10.0,
        }

    monkeypatch.setattr(pilot, "_spawn_worker", spawn)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, object()))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(
        pilot, "compute_estimate", lambda *args: {"status": "completed", "hours": 1.0}
    )
    monkeypatch.setattr(pilot, "noise_floor", lambda _: {"floor": 0.0})
    result = pilot.run_pilot(
        settings,
        authorized=True,
        config_path=Path("configs/h3_mps_pilot.yaml"),
    )
    assert result["status"] == "completed"
    assert len(result["calibration"]) == 49
    assert len(result["repeatability"]) == 16
    assert any("--checkpointing" in call for call in calls)
    assert pilot._json_mapping(settings.output_dir / "pilot-evidence.json")["status"] == (
        "completed"
    )

    calls.clear()
    resumed = pilot.run_pilot(
        settings,
        authorized=True,
        config_path=Path("configs/h3_mps_pilot.yaml"),
    )
    assert resumed["status"] == "completed"
    assert calls == []


def test_run_pilot_rejects_stale_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings().model_copy(update={"output_dir": tmp_path})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pilot, "validate_claim_streams", lambda _: {"audited": True})
    monkeypatch.setattr(pilot.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(pilot, "pilot_run_manifest", lambda *args: {"identity": "new"})
    pilot._write_json_atomic(
        tmp_path / "pilot-progress.json", {"run_manifest": {"identity": "old"}}
    )
    with pytest.raises(ValueError, match="does not match"):
        pilot.run_pilot(
            settings,
            authorized=True,
            config_path=Path("configs/h3_mps_pilot.yaml"),
        )


def test_calibration_rejects_checkpointing_outside_training_mb1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    fake = SimpleNamespace(device=torch.device("mps"))
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: fake)
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    record = pilot.run_calibration_attempt(
        "frozen_base",
        "synthetic_v2",
        cast(Any, _Stream()),
        1,
        settings,
        {"quiet": True},
        checkpointing=True,
    )
    assert record["status"] == "failed"
    assert record["gradient_checkpointing"] == "not_applicable"
    assert record["failure"]["type"] == "ValueError"
    assert "only valid" not in record["failure"]["message"]


def test_evidence_contains_no_raw_chronological_text() -> None:
    source = Path("aptivra/eval/mps_pilot.py").read_text(encoding="utf-8")
    assert 'prompt": instance.prompt' not in source
    assert 'target_solution": instance.target_solution' not in source
