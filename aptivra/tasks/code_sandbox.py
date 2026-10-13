"""Subprocess isolation for mechanical verification of generated Python."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SANDBOX_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "verifier_sandbox.yaml"
)
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


@lru_cache(maxsize=4)
def load_sandbox_config(path: Path = DEFAULT_SANDBOX_CONFIG) -> SandboxConfig:
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
    builtin_names: tuple[str, ...],
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
            "builtins": list(builtin_names),
        },
        config or load_sandbox_config(),
    )


def verify_snippet_source(
    source: str,
    expected: str,
    *,
    builtin_names: tuple[str, ...],
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute one generated snippet and compare its ``res`` value in a child."""
    return _run_worker(
        {
            "mode": "snippet",
            "source": source,
            "expected": expected,
            "builtins": list(builtin_names),
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
    safe_env = {
        "LANG": "C.UTF-8",
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    with tempfile.TemporaryDirectory(prefix="aptivra-verifier-") as working_dir:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(_WORKER_PATH)],
            cwd=working_dir,
            env=safe_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(
                json.dumps(payload), timeout=config.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return SandboxResult(
                False,
                error_message=(
                    "Execution timed out after "
                    f"{config.timeout_seconds:.3g} seconds."
                ),
            )
    if process.returncode != 0:
        detail = stderr.strip()[:500] or f"exit status {process.returncode}"
        return SandboxResult(False, error_message=f"Execution process crashed: {detail}")
    if len(stdout.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return SandboxResult(False, error_message="Execution response exceeded sandbox limit.")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        return SandboxResult(False, error_message="Execution process returned invalid output.")
    if not isinstance(response, dict) or not isinstance(response.get("passed"), bool):
        return SandboxResult(False, error_message="Execution process returned invalid output.")
    actual = response.get("actual")
    error = response.get("error")
    return SandboxResult(
        passed=response["passed"],
        actual=actual if isinstance(actual, str) else None,
        error_message=error if isinstance(error, str) else None,
    )
