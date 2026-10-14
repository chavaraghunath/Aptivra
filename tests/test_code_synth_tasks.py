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
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.bug_fixer import BugFixerTaskFamily, BugFixerVerifier
from aptivra.tasks.code_sandbox import (
    SandboxBatchRequest,
    SandboxConfig,
    load_sandbox_config,
    verify_snippet_source,
    verify_source_batch,
)
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily, DocstringToCodeVerifier
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.sandbox_policy import ALLOWED_BUILTINS
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily, TypeInferrerVerifier


def _function_instance(
    family: TaskFamily,
    function_name: str,
    test_cases: list[tuple[Any, Any]],
) -> TaskInstance:
    return TaskInstance(
        task_id=f"builtin_corpus_{family.family_id}",
        task_family=family.family_id,
        prompt=f"Implement {function_name}.",
        target_solution=f"def {function_name}(*args):\n    raise NotImplementedError",
        metadata={"fn_name": function_name, "test_cases": test_cases},
    )


@pytest.mark.parametrize(
    "source",
    [
        "def is_palindrome(s):\n    return s == s[::-1]",
        "def is_palindrome(s):\n    return list(s) == list(reversed(s))",
        (
            "def is_palindrome(s):\n"
            "    for index in range(len(s) // 2):\n"
            "        if s[index] != s[-index - 1]:\n"
            "            return False\n"
            "    return True"
        ),
        (
            "def is_palindrome(s):\n"
            "    return sum(left != right for left, right in zip(s, reversed(s))) == 0"
        ),
    ],
    ids=["slicing", "reversed", "loop", "zip-comprehension"],
)
def test_algo_synth_accepts_distinct_correct_python_idioms(source: str) -> None:
    family = AlgoSynthTaskFamily()
    instance = _function_instance(
        family,
        "is_palindrome",
        [("racecar", True), ("hello", False), ("", True), ("abccba", True)],
    )

    assert family.verifier.verify(instance, source).passed is True


@pytest.mark.parametrize(
    "source",
    [
        (
            "def count_vowels(s):\n"
            "    return sum(1 for character in s.lower() if character in 'aeiou')"
        ),
        (
            "def count_vowels(s):\n"
            "    total = 0\n"
            "    for _, character in enumerate(s.lower()):\n"
            "        if character in 'aeiou':\n"
            "            total += 1\n"
            "    return total"
        ),
        (
            "def count_vowels(s):\n"
            "    def visit(index):\n"
            "        if index == len(s):\n"
            "            return 0\n"
            "        return int(s[index].lower() in 'aeiou') + visit(index + 1)\n"
            "    return visit(0)"
        ),
    ],
    ids=["comprehension", "enumerate-loop", "recursion"],
)
def test_docstring_code_accepts_distinct_correct_python_idioms(source: str) -> None:
    family = DocstringToCodeTaskFamily()
    instance = _function_instance(
        family,
        "count_vowels",
        [("hello", 2), ("xyz", 0), ("AEIOU", 5), ("", 0)],
    )

    assert family.verifier.verify(instance, source).passed is True


@pytest.mark.parametrize(
    "source",
    [
        "def order_values(values):\n    return sorted(values)",
        (
            "def order_values(values):\n"
            "    result = list(values)\n"
            "    for end in range(len(result), 0, -1):\n"
            "        for index in range(1, end):\n"
            "            if result[index - 1] > result[index]:\n"
            "                result[index - 1], result[index] = (\n"
            "                    result[index], result[index - 1]\n"
            "                )\n"
            "    return result"
        ),
        (
            "def order_values(values):\n"
            "    if len(values) < 2:\n"
            "        return list(values)\n"
            "    pivot = values[0]\n"
            "    lower = [value for value in values[1:] if value <= pivot]\n"
            "    upper = [value for value in values[1:] if value > pivot]\n"
            "    return order_values(lower) + [pivot] + order_values(upper)"
        ),
    ],
    ids=["sorted", "nested-loops", "recursive-comprehensions"],
)
def test_bug_fixer_accepts_distinct_correct_python_idioms(source: str) -> None:
    family = BugFixerTaskFamily()
    instance = _function_instance(
        family,
        "order_values",
        [([3, 1, 2], [1, 2, 3]), ([], []), ([2, 1, 2], [1, 2, 2])],
    )

    assert family.verifier.verify(instance, source).passed is True


@pytest.mark.parametrize(
    "source",
    [
        "res = sum(number * number for number in range(1, 6))",
        "res = 0\nfor number in range(1, 6):\n    res += number * number",
        (
            "res = sum(index * value for index, value in "
            "enumerate(range(1, 6), start=1))"
        ),
        (
            "def sum_squares(number):\n"
            "    if number == 0:\n"
            "        return 0\n"
            "    return number * number + sum_squares(number - 1)\n"
            "res = sum_squares(5)"
        ),
    ],
    ids=["comprehension", "loop", "enumerate", "recursion"],
)
def test_python_microsynth_accepts_distinct_correct_python_idioms(source: str) -> None:
    family = PythonMicroSynthTaskFamily()
    instance = TaskInstance(
        task_id="builtin_corpus_microsynth",
        task_family=family.family_id,
        prompt="Set res to the sum of squares from one through five.",
        target_solution="55",
    )

    assert family.verifier.verify(instance, source).passed is True


@pytest.mark.parametrize(
    ("family", "instance", "broken_source"),
    [
        (
            AlgoSynthTaskFamily(),
            _function_instance(
                AlgoSynthTaskFamily(),
                "is_palindrome",
                [("racecar", True), ("hello", False)],
            ),
            "def is_palindrome(s):\n    return False",
        ),
        (
            DocstringToCodeTaskFamily(),
            _function_instance(
                DocstringToCodeTaskFamily(),
                "count_vowels",
                [("hello", 2), ("xyz", 0)],
            ),
            "def count_vowels(s):\n    return len(s)",
        ),
        (
            BugFixerTaskFamily(),
            _function_instance(
                BugFixerTaskFamily(),
                "order_values",
                [([3, 1, 2], [1, 2, 3]), ([], [])],
            ),
            "def order_values(values):\n    return list(values)",
        ),
        (
            PythonMicroSynthTaskFamily(),
            TaskInstance(
                task_id="builtin_corpus_microsynth_failures",
                task_family="T8_python_microsynth",
                prompt="Set res to 55.",
                target_solution="55",
            ),
            "res = 54",
        ),
    ],
    ids=["algo", "docstring", "bug-fixer", "micro-synth"],
)
def test_code_task_families_reject_hostile_and_broken_solutions(
    family: TaskFamily,
    instance: TaskInstance,
    broken_source: str,
) -> None:
    hostile = family.verifier.verify(instance, "import os")
    broken = family.verifier.verify(instance, broken_source)

    assert hostile.passed is False
    assert "Security Error: Forbidden import" in (hostile.error_message or "")
    assert broken.passed is False


def test_generated_code_builtin_policy_is_fixed_and_excludes_dangerous_names() -> None:
    assert {
        "enumerate",
        "reversed",
        "sorted",
        "zip",
    } <= ALLOWED_BUILTINS.keys()
    assert {
        "__import__",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
        "print",
    }.isdisjoint(ALLOWED_BUILTINS)


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
    assert result.execution_status == "completed"
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
        config=SandboxConfig(timeout_seconds=1.0, memory_limit_mb=512),
    )

    assert result.passed is True
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "APTIVRA_TEST_SECRET" not in child_env
    child_cwd = captured["cwd"]
    assert isinstance(child_cwd, str)
    assert not os.path.exists(child_cwd)


def test_sandbox_batch_preserves_tuple_expectations_and_item_failures() -> None:
    results = verify_source_batch(
        [
            SandboxBatchRequest(
                "function",
                "def pair(values):\n    return tuple(values)",
                function_name="pair",
                test_cases=[(["a", "b"], ("a", "b"))],
            ),
            SandboxBatchRequest("snippet", "res = 2", expected="3"),
        ]
    )

    assert [result.passed for result in results] == [True, False]
    assert all(result.ran_in_child_process for result in results)


def test_sandbox_batch_rejects_invalid_request_shapes() -> None:
    with pytest.raises(ValueError, match="between 1 and 128"):
        verify_source_batch([])
    with pytest.raises(ValueError, match="requires name and test cases"):
        verify_source_batch([SandboxBatchRequest("function", "def f(): pass")])
    with pytest.raises(ValueError, match="requires expected result"):
        verify_source_batch([SandboxBatchRequest("snippet", "res = 1")])


def test_verifier_temp_directory_failure_is_structured_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_temp_directory(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("no writable temporary directory")

    monkeypatch.setattr(code_sandbox.tempfile, "TemporaryDirectory", fail_temp_directory)

    result = verify_snippet_source(
        "res = 1",
        "1",
        config=SandboxConfig(timeout_seconds=1.0, memory_limit_mb=512),
    )

    assert result.passed is False
    assert result.execution_status == "infrastructure_error"
    assert result.ran_in_child_process is False
    assert "temporary directory" in (result.error_message or "")


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
