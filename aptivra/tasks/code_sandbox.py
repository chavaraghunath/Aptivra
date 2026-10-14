"""Contain generated-code hangs and accidents in a resource-limited child.

This runner is an evaluation reliability mechanism, not a security boundary against
hostile code. Its AST policy, reduced environment, resource limits, timeout, and
process-group kill are defense in depth; callers must not treat them as adversarial
code isolation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml

from aptivra.resources import aptivra_resource
from aptivra.tasks.base import ExecutionStatus

DEFAULT_SANDBOX_CONFIG = aptivra_resource("configs", "verifier_sandbox.yaml")
_WORKER_PATH = Path(__file__).with_name("sandbox_worker.py")
_MAX_SOURCE_BYTES = 256 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class SandboxConfig:
    """Validated limits for one generated-code verification item."""

    timeout_seconds: float = 1.0
    memory_limit_mb: int = 512


@dataclass(frozen=True)
class SandboxResult:
    """Structured worker outcome; failures never escape into the evaluator."""

    passed: bool
    actual: str | None = None
    error_message: str | None = None
    value: object | None = None
    ran_in_child_process: bool = False
    execution_status: ExecutionStatus = "completed"


@lru_cache(maxsize=4)
def load_sandbox_config(
    path: Path | Traversable = DEFAULT_SANDBOX_CONFIG,
) -> SandboxConfig:
    """Load verifier resource limits from versioned config."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("verifier sandbox config must be a YAML mapping")
    section = payload.get("verifier_sandbox")
    if not isinstance(section, dict):
        raise ValueError("verifier sandbox config missing verifier_sandbox")
    timeout = section.get("timeout_seconds")
    memory = section.get("memory_limit_mb")
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise ValueError("verifier timeout_seconds must be numeric")
    if not 0.05 <= float(timeout) <= 30.0:
        raise ValueError("verifier timeout_seconds must be between 0.05 and 30")
    if isinstance(memory, bool) or not isinstance(memory, int):
        raise ValueError("verifier memory_limit_mb must be an integer")
    if not 64 <= memory <= 2048:
        raise ValueError("verifier memory_limit_mb must be between 64 and 2048")
    return SandboxConfig(timeout_seconds=float(timeout), memory_limit_mb=memory)


def verify_function_source(
    source: str,
    function_name: str,
    test_cases: list[tuple[Any, Any]],
    *,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Compile one generated function and run all of its cases in one child."""
    cases: list[dict[str, object]] = []
    for test_input, expected in test_cases:
        arguments = list(test_input) if isinstance(test_input, tuple) else [test_input]
        cases.append(
            {
                "args": arguments,
                "expected": expected,
                "display_input": repr(test_input),
            }
        )
    return _run_worker(
        {
            "mode": "function",
            "source": source,
            "function_name": function_name,
            "test_cases": cases,
        },
        config or load_sandbox_config(),
    )


def verify_snippet_source(
    source: str,
    expected: str,
    *,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute one generated snippet and compare its ``res`` value in a child."""
    return _run_worker(
        {
            "mode": "snippet",
            "source": source,
            "expected": expected,
        },
        config or load_sandbox_config(),
    )


def validate_function_source(
    source: str,
    function_name: str,
    *,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Validate and load a generated function entirely inside one child process."""
    return _run_worker(
        {
            "mode": "validate_function",
            "source": source,
            "function_name": function_name,
        },
        config or load_sandbox_config(),
    )


def execute_function_source(
    source: str,
    function_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute a generated function in one child and return a JSON-compatible value."""
    return _run_worker(
        {
            "mode": "execute_function",
            "source": source,
            "function_name": function_name,
            "args": list(args),
            "kwargs": kwargs,
        },
        config or load_sandbox_config(),
    )


def _run_worker(payload: dict[str, object], config: SandboxConfig) -> SandboxResult:
    source = payload.get("source")
    if not isinstance(source, str):
        return SandboxResult(False, error_message="Invalid generated source.")
    if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
        return SandboxResult(False, error_message="Generated source exceeds sandbox limit.")
    payload["timeout_seconds"] = config.timeout_seconds
    payload["memory_limit_mb"] = config.memory_limit_mb
    try:
        serialized_payload = json.dumps(payload)
    except (TypeError, ValueError) as error:
        return SandboxResult(
            False,
            error_message=f"Execution input is not JSON-compatible: {error}",
        )
    safe_env = {
        "LANG": "C.UTF-8",
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        temporary_directory = tempfile.TemporaryDirectory(prefix="aptivra-verifier-")
    except OSError as error:
        return SandboxResult(
            False,
            error_message=f"Execution temporary directory could not be created: {error}",
            execution_status="infrastructure_error",
        )
    with temporary_directory as working_dir:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    (
                        "import runpy,sys;sys.path.insert(0,sys.argv[1]);"
                        "runpy.run_path(sys.argv[2],run_name='__main__')"
                    ),
                    str(_WORKER_PATH.parent),
                    str(_WORKER_PATH),
                ],
                cwd=working_dir,
                env=safe_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            return SandboxResult(
                False,
                error_message=f"Execution process could not start: {error}",
                execution_status="infrastructure_error",
            )
        try:
            stdout, stderr = process.communicate(
                serialized_payload, timeout=config.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return SandboxResult(
                False,
                error_message=(
                    "Execution timed out after "
                    f"{config.timeout_seconds:.3g} seconds."
                ),
                ran_in_child_process=True,
            )
    if process.returncode != 0:
        detail = stderr.strip()[:500] or f"exit status {process.returncode}"
        return SandboxResult(
            False,
            error_message=f"Execution process crashed: {detail}",
            ran_in_child_process=True,
        )
    if len(stdout.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return SandboxResult(
            False,
            error_message="Execution response exceeded sandbox limit.",
            ran_in_child_process=True,
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            False,
            error_message="Execution process returned invalid output.",
            ran_in_child_process=True,
        )
    if not isinstance(response, dict) or not isinstance(response.get("passed"), bool):
        return SandboxResult(
            False,
            error_message="Execution process returned invalid output.",
            ran_in_child_process=True,
        )
    actual = response.get("actual")
    response_error = response.get("error")
    return SandboxResult(
        passed=response["passed"],
        actual=actual if isinstance(actual, str) else None,
        error_message=response_error if isinstance(response_error, str) else None,
        value=response.get("value"),
        ran_in_child_process=True,
    )
