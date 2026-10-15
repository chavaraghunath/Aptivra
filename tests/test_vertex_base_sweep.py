"""Offline tests for Vertex job rendering and resumable artifact persistence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import aptivra.eval.artifact_store as artifact_store
import aptivra.eval.vertex_base_sweep as vertex
from aptivra.eval.artifact_store import GcsArtifactStore, LocalArtifactStore


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", code, "error", {}, None)


def test_local_artifact_store_is_atomic_and_rejects_unsafe_paths(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    assert store.relative_paths() == []
    store.write_bytes("candidate/a.json", b"one")
    assert store.exists("candidate/a.json") is True
    assert store.read_bytes("candidate/a.json") == b"one"
    with pytest.raises(FileExistsError):
        store.write_bytes("candidate/a.json", b"two")
    store.write_bytes("candidate/a.json", b"two", overwrite=True)
    assert store.relative_paths() == ["candidate/a.json"]
    assert store.read_bytes("candidate/a.json") == b"two"
    with pytest.raises(ValueError, match="safe relative"):
        store.exists("../escape")


def test_gcs_store_retries_reads_and_maps_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[object] = [
        urllib.error.URLError("temporary"),
        _Response(b"payload"),
        _http_error(404),
        _http_error(412),
    ]

    def urlopen(*args: object, **kwargs: object) -> _Response:
        del args, kwargs
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return cast(_Response, response)

    monkeypatch.setattr(artifact_store.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(artifact_store.time, "sleep", lambda _: None)
    store = GcsArtifactStore(
        "gs://bucket/prefix",
        timeout_seconds=1.0,
        retries=2,
        retry_backoff_seconds=0.0,
        token_provider=lambda: "token",
    )
    assert store.read_bytes("a.json") == b"payload"
    assert store.exists("missing.json") is False
    with pytest.raises(FileExistsError):
        store.write_bytes("existing.json", b"value")


def test_gcs_store_lists_pages_and_validates_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = iter(
        (
            _Response(
                json.dumps(
                    {
                        "items": [{"name": "prefix/a.json"}],
                        "nextPageToken": "next",
                    }
                ).encode()
            ),
            _Response(json.dumps({"items": [{"name": "prefix/b.json"}]}).encode()),
        )
    )
    monkeypatch.setattr(
        artifact_store.urllib.request,
        "urlopen",
        lambda *args, **kwargs: next(pages),
    )
    store = GcsArtifactStore(
        "gs://bucket/prefix",
        timeout_seconds=1.0,
        retries=1,
        retry_backoff_seconds=0.0,
        token_provider=lambda: "token",
    )
    assert store.relative_paths() == ["a.json", "b.json"]
    with pytest.raises(ValueError, match="gs://"):
        GcsArtifactStore(
            "https://bucket/prefix",
            timeout_seconds=1.0,
            retries=1,
            retry_backoff_seconds=0.0,
        )
    with pytest.raises(ValueError, match="safe limits"):
        GcsArtifactStore(
            "gs://bucket/prefix",
            timeout_seconds=0.0,
            retries=1,
            retry_backoff_seconds=0.0,
        )


def test_gcs_metadata_token_and_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_store.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response(b'{"access_token":"abc"}'),
    )
    store = GcsArtifactStore(
        "gs://bucket",
        timeout_seconds=1.0,
        retries=1,
        retry_backoff_seconds=0.0,
    )
    assert store._metadata_token() == "abc"
    monkeypatch.setattr(
        artifact_store.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )
    with pytest.raises(RuntimeError, match="retry budget"):
        store.read_bytes("a")


def test_download_and_upload_store_round_trip(tmp_path: Path) -> None:
    source = LocalArtifactStore(tmp_path / "source")
    source.write_bytes("nested/a.json", b"a")
    destination = tmp_path / "destination"
    artifact_store.download_store(source, destination)
    assert (destination / "nested" / "a.json").read_bytes() == b"a"
    target = LocalArtifactStore(tmp_path / "target")
    artifact_store.upload_file(target, destination, destination / "nested" / "a.json")
    assert target.read_bytes("nested/a.json") == b"a"


def test_vertex_config_and_specs_are_exact_and_non_mutating(tmp_path: Path) -> None:
    settings = vertex.load_vertex_sweep_settings("configs/vertex_base_sweep.yaml")
    full = vertex.build_custom_job_spec(settings, b"evaluation: full", mode="full")
    pool = cast(list[dict[str, Any]], full["workerPoolSpecs"])[0]
    assert pool["machineSpec"] == {
        "machineType": "a2-highgpu-1g",
        "acceleratorType": "NVIDIA_TESLA_A100",
        "acceleratorCount": 1,
    }
    container = cast(dict[str, Any], pool["containerSpec"])
    assert container["imageUri"] == (
        "docker.io/pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
        "@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56"
    )
    assert "pythonPackageSpec" not in pool
    assert container["command"] == ["/bin/bash", "-ceu"]
    args = cast(list[str], container["args"])
    assert args[2] == "/usr/bin/python3"
    assert args[4:10] == ["3.12", "2.14.1", "12.6", "true", "60", "4"]
    dependency_count = int(args[10])
    dependencies = args[11 : 11 + dependency_count]
    assert "gguf==0.19.0" in dependencies
    assert "transformers==5.18.0" in dependencies
    assert all(not item.startswith("torch==") for item in dependencies)
    assert "--no-deps" in args[0]
    assert '--target "$install_root"' in args[0]
    assert "vertex_runtime_pin_check" in args[0]
    assert 'exec "$python_executable" -m aptivra.eval.vertex_base_sweep' in args[0]
    assert "--worker" in args[11 + dependency_count :]
    assert cast(dict[str, object], full["scheduling"]) == {
        "strategy": "SPOT",
        "timeout": "10800s",
        "restartJobOnWorkerRestart": True,
        "disableRetries": False,
    }
    smoke = vertex.build_custom_job_spec(settings, b"evaluation: smoke", mode="smoke")
    smoke_pool = cast(list[dict[str, Any]], smoke["workerPoolSpecs"])[0]
    assert smoke_pool["machineSpec"]["acceleratorType"] == "NVIDIA_TESLA_T4"
    assert cast(dict[str, object], smoke["scheduling"])["timeout"] == "1200s"
    assert settings.storage.artifact_retention_days == 30
    with pytest.raises(ValueError, match="mode"):
        vertex.build_custom_job_spec(settings, b"x", mode="invalid")

    value = yaml.safe_load(Path("configs/vertex_base_sweep.yaml").read_text())
    value["scheduling_strategy"] = "ON_DEMAND"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="must remain SPOT"):
        vertex.load_vertex_sweep_settings(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("disable_retries", True, "retries"),
        ("executor_image_digest", "latest", "sha256"),
        ("package_uri", "https://example/package", "gs://"),
        ("runtime_dependencies", ["torch==2.14.1"], "must not replace"),
    ],
)
def test_vertex_config_rejects_cost_or_reproducibility_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config = yaml.safe_load(Path("configs/vertex_base_sweep.yaml").read_text())
    config[field] = value
    path = tmp_path / f"{field}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        vertex.load_vertex_sweep_settings(path)


def test_runtime_preflight_passes_exact_runtime_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(vertex.sys, "version_info", SimpleNamespace(major=3, minor=12))
    monkeypatch.setattr(vertex.sys, "version", "3.12.12 container build")
    monkeypatch.setattr(vertex.torch, "__version__", "2.14.1+cu126")
    monkeypatch.setattr(vertex.torch.version, "cuda", "12.6")
    monkeypatch.setattr(vertex.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vertex.torch.cuda, "device_count", lambda: 1)

    result = vertex.runtime_preflight(
        expected_python_version="3.12",
        expected_torch_version="2.14.1",
        expected_torch_cuda_version="12.6",
        expected_cuda_available=True,
    )
    assert result["status"] == "passed"
    assert json.loads(capsys.readouterr().out)["actual"]["cuda_device_count"] == 1

    with pytest.raises(RuntimeError, match="does not match expected"):
        vertex.runtime_preflight(
            expected_python_version="3.11",
            expected_torch_version="2.14.1",
            expected_torch_cuda_version="12.6",
            expected_cuda_available=True,
        )
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert failure["errors"] == ["Python 3.12 does not match expected 3.11"]


def test_target_pin_check_is_scoped_and_fails_on_mismatch(tmp_path: Path) -> None:
    for name, version in (("aptivra", "0.1.0"), ("Example_Package", "1.2.3")):
        metadata = tmp_path / f"{name}-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    passed = subprocess.run(
        [sys.executable, "-c", vertex._PIN_CHECK_CODE, str(tmp_path), "example-package==1.2.3"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["checked_pin_count"] == 1

    failed = subprocess.run(
        [sys.executable, "-c", vertex._PIN_CHECK_CODE, str(tmp_path), "example-package==9.9.9"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 79
    mismatch = json.loads(failed.stdout)["mismatches"]["example-package"]
    assert mismatch == {"actual": "1.2.3", "expected": "9.9.9"}


def test_confirmation_config_freezes_original_recipe(tmp_path: Path) -> None:
    destination = tmp_path / "confirmation.yaml"
    leaders = ["smollm2-360m", "smollm2-135m"]
    vertex._confirmation_config(
        Path("configs/base_model_candidates_amendment_4d.yaml"), destination, leaders
    )
    value = yaml.safe_load(destination.read_text())
    assert [row["id"] for row in value["candidates"]] == leaders[::-1]
    assert value["measurement"]["target_only_training_loss"] is False
    assert value["measurement"]["probe_micro_batch_sizes"] == [8, 4, 2, 1]
    assert value["measurement"]["gradient_checkpointing_fallback"] is False
    loaded = vertex.load_base_model_evaluation_settings(destination)
    assert {item.candidate_id for item in loaded.candidates} == set(leaders)


def test_sync_output_rejects_immutable_mismatch_and_updates_ranking(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "record.json").write_text("one", encoding="utf-8")
    (output / "ranking.json").write_text("rank-one", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    vertex._sync_output(store, output)
    (output / "ranking.json").write_text("rank-two", encoding="utf-8")
    vertex._sync_output(store, output)
    assert store.read_bytes("ranking.json") == b"rank-two"
    (output / "record.json").write_text("two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable artifact mismatch"):
        vertex._sync_output(store, output)


def test_existing_workload_status_is_manifest_and_seed_bound(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "record.json").write_text(
        json.dumps(
            {
                "candidate_id": "gpt2",
                "manifest_sha256": "manifest",
                "variance_seed": None,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    assert vertex._existing_workload_status(tmp_path, "gpt2", "manifest") == "completed"
    assert vertex._existing_workload_status(tmp_path, "gpt2", "other") is None
    assert (
        vertex._existing_workload_status(
            tmp_path, "gpt2", "manifest", variance_seed=20261015
        )
        is None
    )


def test_run_and_publish_persists_logs_before_propagating_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    store = LocalArtifactStore(tmp_path / "store")
    monkeypatch.setattr(
        vertex,
        "_run_candidate_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["worker"], 7, stdout="structured", stderr="failed"
        ),
    )
    with pytest.raises(RuntimeError, match="exit code 7"):
        vertex._run_and_publish(store, tmp_path / "config", output, "gpt2")
    assert store.exists("cloud_events/gpt2.json")


def test_candidate_process_preserves_environment_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Path("configs/base_model_candidates_vertex_smoke.yaml")
    monkeypatch.setattr(
        vertex,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": True, "samples": []},
    )
    monkeypatch.setattr(vertex, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(vertex, "system_free_memory_percentage", lambda: 100.0)
    process = SimpleNamespace(
        poll=lambda: 0,
        communicate=lambda: ("done", ""),
        returncode=0,
        terminate=lambda: None,
        wait=lambda timeout: None,
        kill=lambda: None,
    )
    monkeypatch.setattr(vertex.subprocess, "Popen", lambda *args, **kwargs: process)
    completed = vertex._run_candidate_process(config, tmp_path, "gpt2")
    assert completed.returncode == 0
    monitor = tmp_path / "process_monitors" / "gpt2-attempt-1.json"
    assert json.loads(monitor.read_text())["system_free_memory_minimum_percent"] == 100.0


def test_candidate_process_exhausts_environment_attempts_without_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vertex,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": False, "samples": []},
    )
    statuses: list[str] = []
    monkeypatch.setattr(
        vertex,
        "write_environment_status",
        lambda *args, **kwargs: statuses.append(cast(str, kwargs["status"])) or {},
    )
    completed = vertex._run_candidate_process(
        Path("configs/base_model_candidates_vertex_smoke.yaml"), tmp_path, "gpt2"
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "not_measured_environment"
    assert statuses == ["environment_invalid", "not_measured_environment"]


def test_environment_recovery_waits_for_linux_memory_to_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free = iter((5.0, 15.0))
    clock = iter((0.0, 0.1, 0.2, 0.3))
    monkeypatch.setattr(vertex, "system_swap_used_bytes", lambda: 7)
    monkeypatch.setattr(vertex, "system_free_memory_percentage", lambda: next(free))
    monkeypatch.setattr(vertex.time, "monotonic", lambda: next(clock))
    sleeps: list[float] = []
    monkeypatch.setattr(vertex.time, "sleep", sleeps.append)
    result = vertex._wait_until_environment_normal(
        minimum_free_memory_percent=10.0,
        timeout_seconds=5.0,
        poll_seconds=0.25,
    )
    assert result["normal"] is True
    assert len(cast(list[object], result["samples"])) == 2
    assert sleeps == [0.25]


def test_candidate_timeout_terminates_then_kills_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vertex,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": True, "samples": []},
    )
    monkeypatch.setattr(vertex, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(vertex, "system_free_memory_percentage", lambda: 100.0)
    clock = iter((0.0, 901.0, 902.0))
    monkeypatch.setattr(vertex.time, "monotonic", lambda: next(clock))
    process = SimpleNamespace(returncode=None)
    process.poll = lambda: None
    process.terminate = lambda: setattr(process, "returncode", -15)

    def wait(*, timeout: float) -> None:
        assert timeout == 10.0
        raise subprocess.TimeoutExpired("worker", timeout)

    process.wait = wait
    process.kill = lambda: setattr(process, "returncode", -9)
    process.communicate = lambda: ("", "")
    monkeypatch.setattr(vertex.subprocess, "Popen", lambda *args, **kwargs: process)
    failures: list[str] = []
    monkeypatch.setattr(
        vertex,
        "write_isolated_process_failure",
        lambda *args, **kwargs: failures.append(cast(str, kwargs["message"])) or {},
    )
    completed = vertex._run_candidate_process(
        Path("configs/base_model_candidates_vertex_smoke.yaml"), tmp_path, "gpt2"
    )
    assert completed.returncode == -9
    assert failures == ["candidate process exceeded 900.0 seconds"]
    monitor = json.loads(
        (tmp_path / "process_monitors" / "gpt2-attempt-1.json").read_text()
    )
    assert monitor["forced_termination_reason"] == failures[0]


def test_run_worker_smoke_and_full_orchestration_are_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(vertex, "_store_for_uri", lambda *args, **kwargs: store)
    monkeypatch.setattr(vertex, "_sync_output", lambda *args, **kwargs: None)
    calls: list[tuple[str, int | None, bool]] = []

    def run(
        _store: object,
        _config: Path,
        output: Path,
        candidate_id: str,
        *,
        variance_seed: int | None = None,
        sync_root: Path | None = None,
    ) -> None:
        calls.append((candidate_id, variance_seed, sync_root is not None))
        output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(vertex, "_run_and_publish", run)
    smoke_ranking = {
        "decision_status": "ranked_no_adr_decision",
        "ranked_candidates": [{"candidate_id": "gpt2"}],
    }
    monkeypatch.setattr(vertex, "rank_existing_candidate_records", lambda *args: smoke_ranking)
    vertex.run_worker(
        mode="smoke",
        evaluation_config=Path(
            "configs/base_model_candidates_vertex_smoke.yaml"
        ).read_bytes(),
        artifact_uri=str(tmp_path / "artifacts"),
        storage_timeout_seconds=1.0,
        storage_retries=1,
        storage_retry_backoff_seconds=0.0,
    )
    assert calls == [("gpt2", None, False)]

    calls.clear()
    full_ranking = {
        "decision_status": "ranked_no_adr_decision",
        "ranked_candidates": [
            {"candidate_id": "smollm2-360m"},
            {"candidate_id": "smollm2-135m"},
        ],
        "seed_variance": [
            {"candidate_id": "smollm2-360m", "missing_seeds": []},
            {"candidate_id": "smollm2-135m", "missing_seeds": []},
        ],
    }
    monkeypatch.setattr(vertex, "rank_existing_candidate_records", lambda *args: full_ranking)
    vertex.run_worker(
        mode="full",
        evaluation_config=Path(
            "configs/base_model_candidates_amendment_4d.yaml"
        ).read_bytes(),
        artifact_uri=str(tmp_path / "artifacts"),
        storage_timeout_seconds=1.0,
        storage_retries=1,
        storage_retry_backoff_seconds=0.0,
    )
    assert len([row for row in calls if row[1] is None and not row[2]]) == 7
    assert len([row for row in calls if row[1] is not None]) == 4
    assert len([row for row in calls if row[2]]) == 2


def test_candidate_worker_routes_primary_and_variance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        vertex,
        "run_evaluation",
        lambda *args, **kwargs: ([{"status": "completed"}], {}),
    )
    args = argparse.Namespace(
        evaluation_config=tmp_path / "config",
        output_dir=tmp_path / "output",
        candidate_id="gpt2",
        variance_seed=None,
    )
    vertex._candidate_worker(args)
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(
        vertex,
        "run_seed_variance_evaluation",
        lambda *args, **kwargs: {"status": "completed"},
    )
    args.variance_seed = 2
    vertex._candidate_worker(args)
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_dry_run_main_prints_exact_command_without_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tmp_path / "job.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vertex_base_sweep",
            "--dry-run",
            "--mode=smoke",
            "--vertex-config=configs/vertex_base_sweep.yaml",
            "--evaluation-config=configs/base_model_candidates_vertex_smoke.yaml",
            f"--spec-output={spec}",
        ],
    )
    vertex.main()
    output = capsys.readouterr().out
    assert spec.is_file()
    assert '"strategy": "SPOT"' in output
    assert "gcloud" in output
    assert "custom-jobs" in output


def test_store_factory_and_display_name(tmp_path: Path) -> None:
    local = vertex._store_for_uri(
        str(tmp_path), timeout_seconds=1.0, retries=1, retry_backoff_seconds=0.0
    )
    assert isinstance(local, LocalArtifactStore)
    cloud = vertex._store_for_uri(
        "gs://bucket/prefix",
        timeout_seconds=1.0,
        retries=1,
        retry_backoff_seconds=0.0,
    )
    assert isinstance(cloud, GcsArtifactStore)
    settings = vertex.load_vertex_sweep_settings("configs/vertex_base_sweep.yaml")
    assert vertex._display_name(settings, "full") == "aptivra-base-sweep-4d-full"
