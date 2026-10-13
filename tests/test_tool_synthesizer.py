"""Generated-tool execution containment tests."""

import builtins
import os
import signal
import time
from typing import Any

import pytest

from aptivra.agents import tool_synthesizer
from aptivra.agents.tool_synthesizer import ToolSynthesizerEngine
from aptivra.tasks import code_sandbox, sandbox_worker
from aptivra.tasks.code_sandbox import SandboxConfig, SandboxResult


def _test_config() -> SandboxConfig:
    return SandboxConfig(timeout_seconds=0.5, memory_limit_mb=512)


def test_synthesized_tool_never_compiles_or_executes_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ToolSynthesizerEngine(config=_test_config())

    def reject_parent_compile(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generated source compiled in parent process")

    monkeypatch.setattr(builtins, "compile", reject_parent_compile)

    tool = engine.synthesize_tool(
        "multiply",
        "(left, right) -> int",
        "def multiply(left, right):\n    return left * right\n",
    )

    assert tool.is_sandboxed is True
    assert tool.execution_fn(6, right=7) == 42
    assert engine.execute_tool("multiply", 8, 9) == 72


def test_synthesized_tool_timeout_kills_the_child_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ToolSynthesizerEngine(config=_test_config())
    engine.synthesize_tool("hang", "() -> None", "def hang():\n    while True: pass\n")
    real_killpg = os.killpg
    kill_calls: list[tuple[int, int]] = []

    def recording_killpg(process_group: int, sig: int) -> None:
        kill_calls.append((process_group, sig))
        real_killpg(process_group, sig)

    def fail_if_parent_hangs(_signum: int, _frame: object) -> None:
        raise AssertionError("synthesized tool hung the parent process")

    monkeypatch.setattr(code_sandbox.os, "killpg", recording_killpg)
    previous_alarm_handler = signal.signal(signal.SIGALRM, fail_if_parent_hangs)
    signal.setitimer(signal.ITIMER_REAL, 3.0)
    started = time.perf_counter()

    try:
        with pytest.raises(RuntimeError, match="timed out"):
            engine.execute_tool("hang")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)

    assert time.perf_counter() - started < 2.0
    assert len(kill_calls) == 1
    assert kill_calls[0][1] == signal.SIGKILL


def test_synthesized_tool_rejects_escape_and_missing_callable() -> None:
    engine = ToolSynthesizerEngine(config=_test_config())

    with pytest.raises(ValueError, match="Forbidden attribute"):
        engine.synthesize_tool(
            "escape",
            "() -> object",
            "def escape():\n    return ().__class__.__bases__\n",
        )
    with pytest.raises(ValueError, match="not defined"):
        engine.synthesize_tool("missing", "() -> None", "value = 1\n")

    assert len(engine) == 0


def test_synthesized_tool_reports_non_json_result_as_failure() -> None:
    engine = ToolSynthesizerEngine(config=_test_config())
    engine.synthesize_tool("make_set", "() -> set[int]", "def make_set():\n    return {1}\n")

    with pytest.raises(RuntimeError, match="not JSON-compatible"):
        engine.execute_tool("make_set")


def test_worker_validates_and_executes_tool_modes() -> None:
    parsed = sandbox_worker._validate_ast(
        "def affine(value, *, scale):\n    return value * scale + 1\n"
    )
    payload = {"function_name": "affine", "args": [3], "kwargs": {"scale": 4}}

    assert sandbox_worker._validate_function(payload, parsed) == {
        "passed": True,
        "error": None,
    }
    assert sandbox_worker._execute_function(payload, parsed) == {
        "passed": True,
        "value": 13,
        "error": None,
    }


def test_is_sandboxed_is_false_without_child_process_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ToolSynthesizerEngine(config=_test_config())
    monkeypatch.setattr(
        tool_synthesizer,
        "validate_function_source",
        lambda *_args, **_kwargs: SandboxResult(
            passed=True,
            ran_in_child_process=False,
        ),
    )

    tool = engine.synthesize_tool(
        "identity",
        "(value) -> object",
        "def identity(value):\n    return value\n",
    )

    assert tool.is_sandboxed is False


def test_execute_unknown_tool_still_fails_closed() -> None:
    engine = ToolSynthesizerEngine(config=_test_config())

    with pytest.raises(KeyError, match="not registered"):
        engine.execute_tool("missing")
