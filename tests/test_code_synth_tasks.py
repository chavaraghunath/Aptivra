"""Unit tests for Real-World Code Synthesis Tasks ($T_9$–$T_{12}$)."""

import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from aptivra.tasks import code_sandbox, sandbox_worker
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily, AlgoSynthVerifier
from aptivra.tasks.base import TaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily, BugFixerVerifier
from aptivra.tasks.code_sandbox import (
    SandboxConfig,
    load_sandbox_config,
    verify_snippet_source,
)
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily, DocstringToCodeVerifier
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily, TypeInferrerVerifier


def test_algo_synth_task_family() -> None:
    fam = AlgoSynthTaskFamily()
    verifier = AlgoSynthVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_docstring_code_task_family() -> None:
    fam = DocstringToCodeTaskFamily()
    verifier = DocstringToCodeVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_bug_fixer_task_family() -> None:
    fam = BugFixerTaskFamily()
    verifier = BugFixerVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_type_inferrer_task_family() -> None:
    fam = TypeInferrerTaskFamily()
    verifier = TypeInferrerVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


@pytest.mark.parametrize(
    "family",
    [
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        PythonMicroSynthTaskFamily(),
    ],
)
def test_generated_code_timeout_is_a_bounded_verification_failure(family: TaskFamily) -> None:
    instance = family.generate_instances(1, seed=42)[0]
    started = time.perf_counter()

    result = family.verifier.verify(instance, "while True:\n    pass")

    assert result.passed is False
    assert "timed out" in (result.error_message or "")
    assert time.perf_counter() - started < 3.0


@pytest.mark.parametrize(
    "family",
    [
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        PythonMicroSynthTaskFamily(),
    ],
)
def test_generated_code_crash_and_dunder_escape_do_not_abort_evaluation(
    family: TaskFamily,
) -> None:
    instance = family.generate_instances(1, seed=42)[0]

    crash = family.verifier.verify(instance, "raise SystemExit(17)")
    escape = family.verifier.verify(instance, "res = ().__class__.__bases__")

    assert crash.passed is False
    assert "Execution Error" in (crash.error_message or "")
    assert escape.passed is False
    assert "Security Error" in (escape.error_message or "")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]\n", "must be a YAML mapping"),
        ("verifier_sandbox: []\n", "missing verifier_sandbox"),
        (
            "verifier_sandbox:\n  timeout_seconds: true\n  memory_limit_mb: 512\n",
            "timeout_seconds must be numeric",
        ),
        (
            "verifier_sandbox:\n  timeout_seconds: 0.01\n  memory_limit_mb: 512\n",
            "timeout_seconds must be between",
        ),
        (
            "verifier_sandbox:\n  timeout_seconds: 1.0\n  memory_limit_mb: true\n",
            "memory_limit_mb must be an integer",
        ),
        (
            "verifier_sandbox:\n  timeout_seconds: 1.0\n  memory_limit_mb: 32\n",
            "memory_limit_mb must be between",
        ),
    ],
)
def test_verifier_sandbox_config_rejects_unsafe_limits(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "sandbox.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_sandbox_config(path)


def test_verifier_child_gets_sanitized_environment_and_temporary_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APTIVRA_TEST_SECRET", "must-not-leak")
    real_popen = subprocess.Popen
    captured: dict[str, object] = {}

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    result = verify_snippet_source(
        "res = 1",
        "1",
        builtin_names=("int",),
        config=SandboxConfig(timeout_seconds=1.0, memory_limit_mb=512),
    )

    assert result.passed is True
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "APTIVRA_TEST_SECRET" not in child_env
    child_cwd = captured["cwd"]
    assert isinstance(child_cwd, str)
    assert not os.path.exists(child_cwd)


def test_sandbox_rejects_invalid_or_oversized_source_before_starting_child() -> None:
    config = SandboxConfig(timeout_seconds=1.0, memory_limit_mb=512)

    invalid = code_sandbox._run_worker({"source": None}, config)
    oversized = code_sandbox._run_worker({"source": "x" * (256 * 1024 + 1)}, config)

    assert invalid.passed is False
    assert invalid.error_message == "Invalid generated source."
    assert oversized.passed is False
    assert oversized.error_message == "Generated source exceeds sandbox limit."


def test_sandbox_worker_validates_and_executes_supported_payloads() -> None:
    parsed_function = sandbox_worker._validate_ast("def add(a, b):\n    return a + b")
    passed = sandbox_worker._run_function(
        {
            "builtins": ["int"],
            "function_name": "add",
            "test_cases": [
                {"args": [2, 3], "expected": 5, "display_input": "(2, 3)"}
            ],
        },
        parsed_function,
    )
    failed = sandbox_worker._run_function(
        {
            "builtins": [],
            "function_name": "add",
            "test_cases": [
                {"args": [2, 3], "expected": 6, "display_input": "(2, 3)"}
            ],
        },
        parsed_function,
    )
    missing = sandbox_worker._run_function(
        {"builtins": [], "function_name": "missing", "test_cases": []},
        parsed_function,
    )
    snippet = sandbox_worker._run_snippet(
        {"builtins": [], "expected": "3"},
        sandbox_worker._validate_ast("res = 3"),
    )
    no_result = sandbox_worker._run_snippet(
        {"builtins": [], "expected": "3"},
        sandbox_worker._validate_ast("x = 3"),
    )

    assert passed == {"passed": True, "error": None}
    assert failed["passed"] is False
    assert "Test failed" in failed["error"]
    assert missing["passed"] is False
    assert snippet["passed"] is True
    assert snippet["actual"] == "3"
    assert no_result["passed"] is False
    with pytest.raises(ValueError, match="Forbidden import"):
        sandbox_worker._validate_ast("import os")
    with pytest.raises(ValueError, match="Forbidden attribute"):
        sandbox_worker._validate_ast("res = ().__class__")


def test_sandbox_worker_main_returns_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "mode": "unknown",
        "source": "res = 1",
        "memory_limit_mb": 512,
        "timeout_seconds": 1.0,
    }
    output = io.StringIO()
    monkeypatch.setattr(sandbox_worker.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sandbox_worker.sys, "stdout", output)
    monkeypatch.setattr(sandbox_worker.os, "environ", {})
    monkeypatch.setattr(sandbox_worker, "_apply_limits", lambda *_args: None)

    sandbox_worker.main()

    response = json.loads(output.getvalue())
    assert response["passed"] is False
    assert "unknown sandbox mode" in response["error"]
