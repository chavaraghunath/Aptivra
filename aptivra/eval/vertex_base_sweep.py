"""Dry-run Vertex specification builder and resumable base-sweep worker."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from aptivra.eval.artifact_store import (
    ArtifactStore,
    GcsArtifactStore,
    LocalArtifactStore,
    download_store,
)
from aptivra.eval.base_model_evaluation import (
    load_base_model_evaluation_settings,
    prepare_run_manifest,
    rank_existing_candidate_records,
    run_evaluation,
    run_seed_variance_evaluation,
    system_free_memory_percentage,
    system_swap_used_bytes,
    write_environment_status,
    write_isolated_process_failure,
    write_json,
)


@dataclass(frozen=True)
class MachineSettings:
    """One Vertex worker-pool machine shape."""

    machine_type: str
    accelerator_type: str
    accelerator_count: int
    timeout_seconds: int


@dataclass(frozen=True)
class StorageSettings:
    """Bounded GCS request policy."""

    timeout_seconds: float
    retries: int
    retry_backoff_seconds: float
    artifact_retention_days: int


@dataclass(frozen=True)
class VertexSweepSettings:
    """Validated CustomJob and artifact settings."""

    project: str
    region: str
    display_name_prefix: str
    package_uri: str
    artifact_uri: str
    executor_image_uri: str
    executor_image_tag: str
    python_executable: str
    expected_python_version: str
    expected_torch_version: str
    image_cuda_version: str
    expected_torch_cuda_version: str
    expected_cuda_available: bool
    pip_timeout_seconds: int
    pip_retries: int
    runtime_dependencies: tuple[str, ...]
    replica_count: int
    boot_disk_type: str
    boot_disk_size_gb: int
    scheduling_strategy: str
    restart_job_on_worker_restart: bool
    disable_retries: bool
    full: MachineSettings
    smoke: MachineSettings
    storage: StorageSettings
    labels: dict[str, str]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _pinned_dependencies(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("runtime_dependencies must be a non-empty list")
    dependencies: list[str] = []
    for index, item in enumerate(value):
        dependency = _string(item, f"runtime_dependencies[{index}]")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[^=\s]+", dependency):
            raise ValueError("runtime_dependencies must use exact name==version pins")
        normalized_name = dependency.split("==", 1)[0].lower().replace("_", "-")
        if normalized_name == "torch" or normalized_name.startswith("nvidia-"):
            raise ValueError("runtime_dependencies must not replace the image Torch/CUDA stack")
        dependencies.append(dependency)
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("runtime_dependencies must not contain duplicates")
    return tuple(dependencies)


def _machine(value: object, name: str) -> MachineSettings:
    row = _mapping(value, name)
    allowed = {"machine_type", "accelerator_type", "accelerator_count", "timeout_seconds"}
    if set(row) - allowed:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(set(row) - allowed))}")
    return MachineSettings(
        machine_type=_string(row.get("machine_type"), f"{name}.machine_type"),
        accelerator_type=_string(
            row.get("accelerator_type"), f"{name}.accelerator_type"
        ),
        accelerator_count=_integer(
            row.get("accelerator_count"), f"{name}.accelerator_count", 1, 8
        ),
        timeout_seconds=_integer(
            row.get("timeout_seconds"), f"{name}.timeout_seconds", 60, 86_400
        ),
    )


def load_vertex_sweep_settings(path: Path | str) -> VertexSweepSettings:
    """Load strict Vertex configuration without importing a cloud SDK."""
    root = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "root")
    allowed = {
        "project",
        "region",
        "display_name_prefix",
        "package_uri",
        "artifact_uri",
        "executor_image_uri",
        "executor_image_tag",
        "python_executable",
        "expected_python_version",
        "expected_torch_version",
        "image_cuda_version",
        "expected_torch_cuda_version",
        "expected_cuda_available",
        "pip_timeout_seconds",
        "pip_retries",
        "runtime_dependencies",
        "replica_count",
        "boot_disk_type",
        "boot_disk_size_gb",
        "scheduling_strategy",
        "restart_job_on_worker_restart",
        "disable_retries",
        "full",
        "smoke",
        "storage",
        "labels",
    }
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"unknown root keys: {', '.join(sorted(unknown))}")
    storage = _mapping(root.get("storage"), "storage")
    storage_allowed = {
        "timeout_seconds",
        "retries",
        "retry_backoff_seconds",
        "artifact_retention_days",
    }
    if set(storage) - storage_allowed:
        raise ValueError(
            f"unknown storage keys: {', '.join(sorted(set(storage) - storage_allowed))}"
        )
    raw_labels = _mapping(root.get("labels", {}), "labels")
    labels = {
        _string(key, "labels key"): _string(value, f"labels.{key}")
        for key, value in raw_labels.items()
    }
    settings = VertexSweepSettings(
        project=_string(root.get("project"), "project"),
        region=_string(root.get("region"), "region"),
        display_name_prefix=_string(
            root.get("display_name_prefix"), "display_name_prefix"
        ),
        package_uri=_string(root.get("package_uri"), "package_uri"),
        artifact_uri=_string(root.get("artifact_uri"), "artifact_uri"),
        executor_image_uri=_string(
            root.get("executor_image_uri"), "executor_image_uri"
        ),
        executor_image_tag=_string(
            root.get("executor_image_tag"), "executor_image_tag"
        ),
        python_executable=_string(
            root.get("python_executable"), "python_executable"
        ),
        expected_python_version=_string(
            root.get("expected_python_version"), "expected_python_version"
        ),
        expected_torch_version=_string(
            root.get("expected_torch_version"), "expected_torch_version"
        ),
        image_cuda_version=_string(
            root.get("image_cuda_version"), "image_cuda_version"
        ),
        expected_torch_cuda_version=_string(
            root.get("expected_torch_cuda_version"),
            "expected_torch_cuda_version",
        ),
        expected_cuda_available=_boolean(
            root.get("expected_cuda_available"), "expected_cuda_available"
        ),
        pip_timeout_seconds=_integer(
            root.get("pip_timeout_seconds"), "pip_timeout_seconds", 1, 300
        ),
        pip_retries=_integer(root.get("pip_retries"), "pip_retries", 0, 10),
        runtime_dependencies=_pinned_dependencies(root.get("runtime_dependencies")),
        replica_count=_integer(root.get("replica_count"), "replica_count", 1, 1),
        boot_disk_type=_string(root.get("boot_disk_type"), "boot_disk_type"),
        boot_disk_size_gb=_integer(
            root.get("boot_disk_size_gb"), "boot_disk_size_gb", 100, 64_000
        ),
        scheduling_strategy=_string(
            root.get("scheduling_strategy"), "scheduling_strategy"
        ),
        restart_job_on_worker_restart=_boolean(
            root.get("restart_job_on_worker_restart"),
            "restart_job_on_worker_restart",
        ),
        disable_retries=_boolean(root.get("disable_retries"), "disable_retries"),
        full=_machine(root.get("full"), "full"),
        smoke=_machine(root.get("smoke"), "smoke"),
        storage=StorageSettings(
            timeout_seconds=_number(
                storage.get("timeout_seconds"), "storage.timeout_seconds", 1.0, 300.0
            ),
            retries=_integer(storage.get("retries"), "storage.retries", 1, 10),
            retry_backoff_seconds=_number(
                storage.get("retry_backoff_seconds"),
                "storage.retry_backoff_seconds",
                0.0,
                60.0,
            ),
            artifact_retention_days=_integer(
                storage.get("artifact_retention_days"),
                "storage.artifact_retention_days",
                1,
                3650,
            ),
        ),
        labels=labels,
    )
    if settings.scheduling_strategy != "SPOT":
        raise ValueError("scheduling_strategy must remain SPOT for the approved cost envelope")
    if not settings.restart_job_on_worker_restart or settings.disable_retries:
        raise ValueError("spot jobs must restart and retries must remain enabled")
    if not settings.package_uri.startswith("gs://") or not settings.artifact_uri.startswith(
        "gs://"
    ):
        raise ValueError("package_uri and artifact_uri must use gs://")
    if not re.fullmatch(
        r"(?:[^/@:]+(?::[0-9]+)?/)*[^/@:]+@sha256:[0-9a-f]{64}",
        settings.executor_image_uri,
    ):
        raise ValueError(
            "executor_image_uri must be a digest-only repository@sha256 URI; "
            "tag+digest URIs are forbidden"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", settings.executor_image_tag):
        raise ValueError("executor_image_tag must be an explicit valid tag")
    if not settings.python_executable.startswith("/"):
        raise ValueError("python_executable must be an absolute image path")
    for name, version in (
        ("expected_python_version", settings.expected_python_version),
        ("expected_torch_version", settings.expected_torch_version),
        ("image_cuda_version", settings.image_cuda_version),
        ("expected_torch_cuda_version", settings.expected_torch_cuda_version),
    ):
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version):
            raise ValueError(f"{name} must be a dotted numeric version")
    if settings.full != MachineSettings(
        "a2-highgpu-1g", "NVIDIA_TESLA_A100", 1, settings.full.timeout_seconds
    ):
        raise ValueError("full worker must be one a2-highgpu-1g with one A100")
    if settings.full.timeout_seconds > 10_800:
        raise ValueError("full timeout cannot exceed the three-hour cost guardrail")
    if settings.smoke.timeout_seconds > 1_200:
        raise ValueError("smoke timeout cannot exceed the twenty-minute cost guardrail")
    return settings


def _artifact_mode_uri(root: str, mode: str) -> str:
    return f"{root.rstrip('/')}/{mode}"


def runtime_preflight(
    *,
    expected_python_version: str,
    expected_torch_version: str,
    expected_torch_cuda_version: str,
    expected_cuda_available: bool,
) -> dict[str, object]:
    """Validate the paid worker runtime before any artifact or model download."""
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    actual_torch = str(torch.__version__).split("+", 1)[0]
    actual_torch_cuda = torch.version.cuda
    actual_cuda_available = torch.cuda.is_available()
    errors: list[str] = []
    if actual_python != expected_python_version:
        errors.append(
            f"Python {actual_python} does not match expected {expected_python_version}"
        )
    if actual_torch != expected_torch_version:
        errors.append(f"Torch {actual_torch} does not match expected {expected_torch_version}")
    if actual_torch_cuda != expected_torch_cuda_version:
        errors.append(
            "Torch CUDA ABI "
            f"{actual_torch_cuda!r} does not match expected {expected_torch_cuda_version!r}"
        )
    if actual_cuda_available is not expected_cuda_available:
        errors.append(
            "CUDA availability "
            f"{actual_cuda_available} does not match expected {expected_cuda_available}"
        )
    result: dict[str, object] = {
        "correlation_id": str(uuid.uuid4()),
        "event": "vertex_runtime_preflight",
        "status": "failed" if errors else "passed",
        "expected": {
            "python_version": expected_python_version,
            "torch_version": expected_torch_version,
            "torch_cuda_version": expected_torch_cuda_version,
            "cuda_available": expected_cuda_available,
        },
        "actual": {
            "python_version": actual_python,
            "python_version_full": sys.version.splitlines()[0],
            "torch_version": actual_torch,
            "torch_cuda_version": actual_torch_cuda,
            "cuda_available": actual_cuda_available,
            "cuda_device_count": torch.cuda.device_count() if actual_cuda_available else 0,
        },
        "errors": errors,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


_BOOTSTRAP_PREFLIGHT_CODE = """
import json
import sys
import uuid

import torch

expected_python, expected_torch, expected_cuda, expected_available = sys.argv[1:]
actual = {
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "torch_version": str(torch.__version__).split("+", 1)[0],
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
expected = {
    "python_version": expected_python,
    "torch_version": expected_torch,
    "torch_cuda_version": expected_cuda,
    "cuda_available": expected_available == "true",
}
errors = [key for key in expected if actual[key] != expected[key]]
print(json.dumps({
    "correlation_id": str(uuid.uuid4()),
    "event": "vertex_bootstrap_preflight",
    "status": "failed" if errors else "passed",
    "expected": expected,
    "actual": actual,
    "mismatched_fields": errors,
}, sort_keys=True), flush=True)
if errors:
    raise SystemExit(78)
""".strip()


_GCS_DOWNLOAD_CODE = """
import json
import pathlib
import sys
import urllib.parse
import urllib.request

source, destination = sys.argv[1:]
if not source.startswith("gs://"):
    raise SystemExit("package URI must use gs://")
bucket, separator, object_name = source[5:].partition("/")
if not bucket or not separator or not object_name:
    raise SystemExit("package URI must identify a GCS object")
token_request = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"},
)
with urllib.request.urlopen(token_request, timeout=10.0) as response:
    token = json.load(response)["access_token"]
object_url = (
    f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
    f"{urllib.parse.quote(object_name, safe='')}?alt=media"
)
object_request = urllib.request.Request(
    object_url, headers={"Authorization": f"Bearer {token}"}
)
with urllib.request.urlopen(object_request, timeout=60.0) as response:
    pathlib.Path(destination).write_bytes(response.read())
""".strip()


_PIN_CHECK_CODE = """
import importlib.metadata
import json
import pathlib
import re
import sys
import uuid

install_root, *pins = sys.argv[1:]
installed = {
    re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(): distribution.version
    for distribution in importlib.metadata.distributions(path=[pathlib.Path(install_root)])
    if distribution.metadata["Name"] is not None
}
expected = {
    re.sub(r"[-_.]+", "-", name).lower(): version
    for name, version in (pin.rsplit("==", 1) for pin in pins)
}
mismatches = {
    name: {"expected": version, "actual": installed.get(name)}
    for name, version in expected.items()
    if installed.get(name) != version
}
if "aptivra" not in installed:
    mismatches["aptivra"] = {"expected": "installed", "actual": None}
print(json.dumps({
    "correlation_id": str(uuid.uuid4()),
    "event": "vertex_runtime_pin_check",
    "status": "failed" if mismatches else "passed",
    "checked_pin_count": len(expected),
    "mismatches": mismatches,
}, sort_keys=True), flush=True)
if mismatches:
    raise SystemExit(79)
""".strip()


def _bootstrap_script() -> str:
    """Return the fail-fast, image-interpreter bootstrap used by custom containers."""
    preflight = shlex.quote(_BOOTSTRAP_PREFLIGHT_CODE)
    downloader = shlex.quote(_GCS_DOWNLOAD_CODE)
    pin_check = shlex.quote(_PIN_CHECK_CODE)
    return f"""python_executable="$1"
package_uri="$2"
expected_python="$3"
expected_torch="$4"
expected_torch_cuda="$5"
expected_cuda_available="$6"
pip_timeout_seconds="$7"
pip_retries="$8"
dependency_count="$9"
shift 9
runtime_dependencies=("${{@:1:dependency_count}}")
shift "$dependency_count"

"$python_executable" -c {preflight} \
  "$expected_python" "$expected_torch" "$expected_torch_cuda" \
  "$expected_cuda_available"
package_path=/tmp/aptivra-package.tar.gz
"$python_executable" -c {downloader} "$package_uri" "$package_path"
install_root=/tmp/aptivra-site
mkdir -p "$install_root"
"$python_executable" -m pip install --disable-pip-version-check --no-cache-dir \
  --timeout "$pip_timeout_seconds" --retries "$pip_retries" \
  --no-deps --target "$install_root" "${{runtime_dependencies[@]}}"
"$python_executable" -m pip install --disable-pip-version-check --no-cache-dir \
  --timeout "$pip_timeout_seconds" --retries "$pip_retries" \
  --no-deps --no-build-isolation --target "$install_root" "$package_path"
export PYTHONPATH="$install_root${{PYTHONPATH:+:$PYTHONPATH}}"
"$python_executable" -c {pin_check} "$install_root" "${{runtime_dependencies[@]}}"
exec "$python_executable" -m aptivra.eval.vertex_base_sweep "$@"
"""


def _worker_args(
    settings: VertexSweepSettings, *, mode: str, evaluation_config: str, artifact_uri: str
) -> list[str]:
    return [
        "--worker",
        f"--mode={mode}",
        f"--evaluation-config-base64={evaluation_config}",
        f"--artifact-uri={artifact_uri}",
        f"--expected-python-version={settings.expected_python_version}",
        f"--expected-torch-version={settings.expected_torch_version}",
        f"--expected-torch-cuda-version={settings.expected_torch_cuda_version}",
        f"--expected-cuda-available={str(settings.expected_cuda_available).lower()}",
        f"--storage-timeout-seconds={settings.storage.timeout_seconds}",
        f"--storage-retries={settings.storage.retries}",
        (
            "--storage-retry-backoff-seconds="
            f"{settings.storage.retry_backoff_seconds}"
        ),
    ]


def build_custom_job_spec(
    settings: VertexSweepSettings,
    evaluation_config: bytes,
    *,
    mode: str,
) -> dict[str, object]:
    """Build the exact gcloud CustomJobSpec; this function never submits it."""
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    machine = settings.full if mode == "full" else settings.smoke
    encoded_config = base64.b64encode(evaluation_config).decode("ascii")
    artifact_uri = _artifact_mode_uri(settings.artifact_uri, mode)
    image = settings.executor_image_uri
    worker_args = _worker_args(
        settings,
        mode=mode,
        evaluation_config=encoded_config,
        artifact_uri=artifact_uri,
    )
    return {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": machine.machine_type,
                    "acceleratorType": machine.accelerator_type,
                    "acceleratorCount": machine.accelerator_count,
                },
                "replicaCount": settings.replica_count,
                "diskSpec": {
                    "bootDiskType": settings.boot_disk_type,
                    "bootDiskSizeGb": settings.boot_disk_size_gb,
                },
                "containerSpec": {
                    "imageUri": image,
                    "command": ["/bin/bash", "-ceu"],
                    "args": [
                        _bootstrap_script(),
                        "aptivra-vertex-bootstrap",
                        settings.python_executable,
                        settings.package_uri,
                        settings.expected_python_version,
                        settings.expected_torch_version,
                        settings.expected_torch_cuda_version,
                        str(settings.expected_cuda_available).lower(),
                        str(settings.pip_timeout_seconds),
                        str(settings.pip_retries),
                        str(len(settings.runtime_dependencies)),
                        *settings.runtime_dependencies,
                        *worker_args,
                    ],
                },
            }
        ],
        "scheduling": {
            "strategy": settings.scheduling_strategy,
            "timeout": f"{machine.timeout_seconds}s",
            "restartJobOnWorkerRestart": settings.restart_job_on_worker_restart,
            "disableRetries": settings.disable_retries,
        },
        "baseOutputDirectory": {"outputUriPrefix": artifact_uri},
    }


def _store_for_uri(
    uri: str, *, timeout_seconds: float, retries: int, retry_backoff_seconds: float
) -> ArtifactStore:
    if uri.startswith("gs://"):
        return GcsArtifactStore(
            uri,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    return LocalArtifactStore(uri)


def _sync_output(store: ArtifactStore, output_dir: Path) -> None:
    """Publish complete files; immutable mismatches fail closed."""
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        payload = path.read_bytes()
        overwrite = relative in {"ranking.json", "confirmation/ranking.json"}
        if store.exists(relative) and not overwrite:
            if store.read_bytes(relative) != payload:
                raise RuntimeError(f"immutable artifact mismatch: {relative}")
            continue
        store.write_bytes(relative, payload, overwrite=overwrite)


def _existing_workload_status(
    output_dir: Path,
    candidate_id: str,
    manifest_sha256: str,
    *,
    variance_seed: int | None = None,
) -> str | None:
    directory = output_dir / ("candidates" if variance_seed is None else "seed_runs")
    for path in directory.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(value, dict)
            and value.get("candidate_id") == candidate_id
            and value.get("manifest_sha256") == manifest_sha256
            and value.get("variance_seed") == variance_seed
            and isinstance(value.get("status"), str)
        ):
            return cast(str, value["status"])
    return None


def _run_candidate_process(
    config_path: Path,
    output_dir: Path,
    candidate_id: str,
    *,
    variance_seed: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "aptivra.eval.vertex_base_sweep",
        "--candidate-worker",
        f"--evaluation-config={config_path}",
        f"--output-dir={output_dir}",
        f"--candidate-id={candidate_id}",
    ]
    if variance_seed is not None:
        command.append(f"--variance-seed={variance_seed}")
    settings = load_base_model_evaluation_settings(config_path)
    attempts: list[dict[str, object]] = []
    measurement = settings.measurement
    for attempt in range(1, measurement.environment_max_attempts + 1):
        recovery = _wait_until_environment_normal(
            minimum_free_memory_percent=(
                measurement.environment_minimum_free_memory_percent
            ),
            timeout_seconds=measurement.environment_recovery_timeout_seconds,
            poll_seconds=measurement.swap_sample_interval_seconds,
        )
        if recovery["normal"] is not True:
            monitor: dict[str, object] = {
                "attempt": attempt,
                "status": "environment_invalid",
                "process_started": False,
                "environment_invalid_reasons": [
                    "system free-memory percentage did not recover before timeout"
                ],
                "recovery": recovery,
            }
            attempts.append(monitor)
            _write_process_monitor(
                output_dir, candidate_id, attempt, monitor, variance_seed
            )
            write_environment_status(
                config_path,
                output_dir,
                candidate_id,
                status="environment_invalid",
                attempts=attempts,
                variance_seed=variance_seed,
            )
            continue

        swap_before = system_swap_used_bytes()
        swap_peak = swap_before
        free_before = system_free_memory_percentage()
        free_minimum = free_before
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        environment_reasons: list[str] = []
        timeout_message: str | None = None
        while process.poll() is None:
            swap_now = system_swap_used_bytes()
            free_now = system_free_memory_percentage()
            swap_peak = max(swap_peak, swap_now)
            free_minimum = min(free_minimum, free_now)
            if swap_peak - swap_before > settings.decision.maximum_swap_growth_bytes:
                environment_reasons.append(
                    "system swap growth exceeded the configured environment limit"
                )
            if free_now < measurement.environment_minimum_free_memory_percent:
                environment_reasons.append(
                    "system free-memory percentage fell below the normal threshold"
                )
            if environment_reasons:
                process.terminate()
                break
            if time.monotonic() - started > measurement.candidate_timeout_seconds:
                timeout_message = (
                    "candidate process exceeded "
                    f"{measurement.candidate_timeout_seconds} seconds"
                )
                process.terminate()
                break
            time.sleep(measurement.swap_sample_interval_seconds)
        if environment_reasons or timeout_message is not None:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
        stdout, stderr = process.communicate()
        swap_after = system_swap_used_bytes()
        free_after = system_free_memory_percentage()
        swap_peak = max(swap_peak, swap_after)
        free_minimum = min(free_minimum, free_after)
        if swap_peak - swap_before > settings.decision.maximum_swap_growth_bytes:
            environment_reasons.append(
                "system swap growth exceeded the configured environment limit"
            )
        if free_after < measurement.environment_minimum_free_memory_percent:
            environment_reasons.append(
                "system free-memory percentage fell below the normal threshold"
            )
        environment_reasons = list(dict.fromkeys(environment_reasons))
        monitor = {
            "attempt": attempt,
            "status": "environment_invalid" if environment_reasons else "completed",
            "process_started": True,
            "system_swap_before_bytes": swap_before,
            "system_swap_peak_bytes": swap_peak,
            "system_swap_after_bytes": swap_after,
            "system_swap_peak_growth_bytes": max(0, swap_peak - swap_before),
            "system_swap_growth_bytes": max(0, swap_after - swap_before),
            "system_free_memory_before_percent": free_before,
            "system_free_memory_minimum_percent": free_minimum,
            "system_free_memory_after_percent": free_after,
            "configured_maximum_swap_growth_bytes": (
                settings.decision.maximum_swap_growth_bytes
            ),
            "configured_minimum_free_memory_percent": (
                measurement.environment_minimum_free_memory_percent
            ),
            "poll_interval_seconds": measurement.swap_sample_interval_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "environment_invalid_reasons": environment_reasons,
            "forced_termination_reason": timeout_message,
            "recovery": recovery,
        }
        attempts.append(monitor)
        _write_process_monitor(
            output_dir, candidate_id, attempt, monitor, variance_seed
        )
        if environment_reasons:
            write_environment_status(
                config_path,
                output_dir,
                candidate_id,
                status="environment_invalid",
                attempts=attempts,
                variance_seed=variance_seed,
            )
            continue
        returncode = process.returncode if process.returncode is not None else -1
        if timeout_message is not None or returncode != 0:
            write_isolated_process_failure(
                config_path,
                output_dir,
                candidate_id,
                returncode=returncode,
                message=timeout_message or (stderr or stdout)[-2000:],
                variance_seed=variance_seed,
                parent_process_memory=monitor,
            )
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    write_environment_status(
        config_path,
        output_dir,
        candidate_id,
        status="not_measured_environment",
        attempts=attempts,
        variance_seed=variance_seed,
    )
    return subprocess.CompletedProcess(
        command,
        0,
        json.dumps({"candidate_id": candidate_id, "status": "not_measured_environment"}),
        "",
    )


def _wait_until_environment_normal(
    *, minimum_free_memory_percent: float, timeout_seconds: float, poll_seconds: float
) -> dict[str, object]:
    started = time.monotonic()
    samples: list[dict[str, float | int]] = []
    while True:
        sample: dict[str, float | int] = {
            "elapsed_seconds": time.monotonic() - started,
            "swap_used_bytes": system_swap_used_bytes(),
            "free_memory_percent": system_free_memory_percentage(),
        }
        samples.append(sample)
        if sample["free_memory_percent"] >= minimum_free_memory_percent:
            return {"normal": True, "samples": samples}
        if time.monotonic() - started >= timeout_seconds:
            return {"normal": False, "samples": samples}
        time.sleep(poll_seconds)


def _write_process_monitor(
    output_dir: Path,
    candidate_id: str,
    attempt: int,
    monitor: dict[str, object],
    variance_seed: int | None,
) -> None:
    monitor_name = candidate_id.replace("/", "_")
    if variance_seed is not None:
        monitor_name += f"-seed-{variance_seed}"
    write_json(
        output_dir
        / "process_monitors"
        / f"{monitor_name}-attempt-{attempt}.json",
        monitor,
    )


def _event_path(output_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return output_dir / "cloud_events" / f"{safe}.json"


def _record_process_event(
    output_dir: Path,
    name: str,
    completed: subprocess.CompletedProcess[str],
) -> None:
    path = _event_path(output_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "event": "candidate_process_finished",
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-20_000:],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_and_publish(
    store: ArtifactStore,
    config_path: Path,
    output_dir: Path,
    candidate_id: str,
    *,
    variance_seed: int | None = None,
    sync_root: Path | None = None,
) -> None:
    label = (
        candidate_id if variance_seed is None else f"{candidate_id}-seed-{variance_seed}"
    )
    completed = _run_candidate_process(
        config_path, output_dir, candidate_id, variance_seed=variance_seed
    )
    _record_process_event(output_dir, label, completed)
    _sync_output(store, sync_root or output_dir)
    if completed.returncode != 0:
        raise RuntimeError(
            f"candidate worker failed for {label} with exit code {completed.returncode}"
        )


def _confirmation_config(
    source: Path, destination: Path, leaders: list[str]
) -> None:
    value = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "evaluation")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("evaluation candidates must be a list")
    value["candidates"] = [
        row
        for row in candidates
        if isinstance(row, dict) and row.get("id") in set(leaders)
    ]
    measurement = _mapping(value.get("measurement"), "measurement")
    measurement["target_only_training_loss"] = False
    measurement["gradient_checkpointing_fallback"] = False
    measurement["variance_top_candidates"] = len(leaders)
    measurement["equivalence_candidate_ids"] = leaders
    execution = _mapping(value.get("execution"), "execution")
    execution["output_dir"] = "../evidence/confirmation"
    execution["resume"] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def run_worker(
    *,
    mode: str,
    evaluation_config: bytes,
    artifact_uri: str,
    storage_timeout_seconds: float,
    storage_retries: int,
    storage_retry_backoff_seconds: float,
) -> None:
    """Run or resume one sweep, publishing after every candidate workload."""
    store = _store_for_uri(
        artifact_uri,
        timeout_seconds=storage_timeout_seconds,
        retries=storage_retries,
        retry_backoff_seconds=storage_retry_backoff_seconds,
    )
    with tempfile.TemporaryDirectory(prefix="aptivra-vertex-sweep-") as temporary:
        work_dir = Path(temporary)
        output_dir = work_dir / "evidence"
        download_store(store, output_dir)
        config_path = work_dir / "configs" / "evaluation.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes(evaluation_config)
        packaged_eval = Path(__file__).resolve().parent
        staged_eval = work_dir / "aptivra" / "eval"
        staged_eval.mkdir(parents=True)
        for name in ("model_card_heldout_text_v1.txt", "model_card_heldout_v1.json"):
            shutil.copyfile(packaged_eval / name, staged_eval / name)
        settings = load_base_model_evaluation_settings(config_path)
        _, manifest_sha256 = prepare_run_manifest(config_path, output_dir, settings)
        _sync_output(store, output_dir)
        selectable = [item for item in settings.candidates if not item.reference_only]
        for candidate in selectable:
            if (
                _existing_workload_status(
                    output_dir, candidate.candidate_id, manifest_sha256
                )
                is not None
            ):
                continue
            _run_and_publish(store, config_path, output_dir, candidate.candidate_id)
        ranking = rank_existing_candidate_records(config_path, output_dir)
        _sync_output(store, output_dir)
        if mode == "smoke":
            if ranking.get("decision_status") == "measurement_incomplete":
                raise RuntimeError("smoke measurement is incomplete")
            return
        ranked = ranking.get("ranked_candidates")
        if (
            not isinstance(ranked, list)
            or ranking.get("decision_status") == "measurement_incomplete"
        ):
            raise RuntimeError("full sweep cannot continue because primary ranking is incomplete")
        leaders = [
            cast(str, row["candidate_id"])
            for row in ranked[: settings.measurement.variance_top_candidates]
            if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
        ]
        if len(leaders) != settings.measurement.variance_top_candidates:
            raise RuntimeError("primary ranking did not yield the configured leaders")
        for candidate_id in leaders:
            for seed in settings.measurement.variance_seeds[1:]:
                if (
                    _existing_workload_status(
                        output_dir,
                        candidate_id,
                        manifest_sha256,
                        variance_seed=seed,
                    )
                    is not None
                ):
                    continue
                _run_and_publish(
                    store,
                    config_path,
                    output_dir,
                    candidate_id,
                    variance_seed=seed,
                )
        repeated_ranking = rank_existing_candidate_records(config_path, output_dir)
        _sync_output(store, output_dir)
        seed_variance = repeated_ranking.get("seed_variance")
        if not isinstance(seed_variance, list) or any(
            not isinstance(row, dict) or row.get("missing_seeds") != []
            for row in seed_variance
        ):
            raise RuntimeError("full sweep cannot continue because repeat seeds are incomplete")
        confirmation_config = work_dir / "configs" / "confirmation.yaml"
        _confirmation_config(config_path, confirmation_config, leaders)
        confirmation_dir = output_dir / "confirmation"
        confirmation_settings = load_base_model_evaluation_settings(confirmation_config)
        _, confirmation_manifest_sha256 = prepare_run_manifest(
            confirmation_config, confirmation_dir, confirmation_settings
        )
        _sync_output(store, output_dir)
        for candidate_id in leaders:
            if (
                _existing_workload_status(
                    confirmation_dir,
                    candidate_id,
                    confirmation_manifest_sha256,
                )
                is not None
            ):
                continue
            _run_and_publish(
                store,
                confirmation_config,
                confirmation_dir,
                candidate_id,
                sync_root=output_dir,
            )
        rank_existing_candidate_records(confirmation_config, confirmation_dir)
        _sync_output(store, output_dir)


def _candidate_worker(args: argparse.Namespace) -> None:
    config = cast(Path, args.evaluation_config)
    output = cast(Path, args.output_dir)
    candidate_id = cast(str, args.candidate_id)
    if args.variance_seed is None:
        records, _ = run_evaluation(
            config, output, resume=True, candidate_ids={candidate_id}
        )
        status = records[0].get("status") if records else "missing"
    else:
        record = run_seed_variance_evaluation(
            config,
            output,
            candidate_id,
            cast(int, args.variance_seed),
            resume=True,
        )
        status = record.get("status")
    print(json.dumps({"candidate_id": candidate_id, "status": status}, sort_keys=True))


def _display_name(settings: VertexSweepSettings, mode: str) -> str:
    return f"{settings.display_name_prefix}-{mode}"


def main() -> None:
    """Render a job spec or execute the installed worker module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex-config", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spec-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluation-config-base64", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-uri", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-id", help=argparse.SUPPRESS)
    parser.add_argument("--variance-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--expected-python-version", help=argparse.SUPPRESS)
    parser.add_argument("--expected-torch-version", help=argparse.SUPPRESS)
    parser.add_argument("--expected-torch-cuda-version", help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-cuda-available",
        choices=("true", "false"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--storage-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--storage-retries", type=int, default=4)
    parser.add_argument("--storage-retry-backoff-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.candidate_worker:
        if args.evaluation_config is None or args.output_dir is None or args.candidate_id is None:
            parser.error("candidate worker requires config, output directory, and candidate id")
        _candidate_worker(args)
        return
    if args.worker:
        required_runtime = (
            args.expected_python_version,
            args.expected_torch_version,
            args.expected_torch_cuda_version,
            args.expected_cuda_available,
        )
        if (
            args.evaluation_config_base64 is None
            or args.artifact_uri is None
            or any(item is None for item in required_runtime)
        ):
            parser.error(
                "worker requires embedded evaluation config, artifact URI, and runtime expectations"
            )
        runtime_preflight(
            expected_python_version=cast(str, args.expected_python_version),
            expected_torch_version=cast(str, args.expected_torch_version),
            expected_torch_cuda_version=cast(str, args.expected_torch_cuda_version),
            expected_cuda_available=args.expected_cuda_available == "true",
        )
        run_worker(
            mode=args.mode,
            evaluation_config=base64.b64decode(
                args.evaluation_config_base64, validate=True
            ),
            artifact_uri=args.artifact_uri,
            storage_timeout_seconds=args.storage_timeout_seconds,
            storage_retries=args.storage_retries,
            storage_retry_backoff_seconds=args.storage_retry_backoff_seconds,
        )
        return
    if not args.dry_run:
        parser.error("submission is intentionally unsupported; use --dry-run and reviewer gcloud")
    if args.vertex_config is None or args.evaluation_config is None:
        parser.error("dry run requires --vertex-config and --evaluation-config")
    vertex = load_vertex_sweep_settings(args.vertex_config)
    spec = build_custom_job_spec(
        vertex, args.evaluation_config.read_bytes(), mode=args.mode
    )
    rendered = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if args.spec_output is not None:
        args.spec_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    spec_path = str(args.spec_output or Path("/tmp/aptivra-custom-job-spec.json"))
    command = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--project={vertex.project}",
        f"--region={vertex.region}",
        f"--display-name={_display_name(vertex, args.mode)}",
        "--labels="
        + ",".join(f"{key}={value}" for key, value in sorted(vertex.labels.items())),
        f"--config={spec_path}",
    ]
    print(json.dumps({"submit_command": command}, sort_keys=True))


if __name__ == "__main__":
    main()
