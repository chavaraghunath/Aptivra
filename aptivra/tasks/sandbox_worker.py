"""Child worker that contains generated-code hangs and accidental failures.

The worker is not a security boundary against hostile code. AST checks, restricted
builtins, resource limits, and the parent timeout reduce accidental blast radius but
do not establish adversarial isolation.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from aptivra.tasks.sandbox_policy import allowed_builtins
else:
    from sandbox_policy import allowed_builtins


def _apply_limits(memory_limit_mb: int, timeout_seconds: float) -> None:
    try:
        import resource

        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_seconds = max(1, int(math.ceil(timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (ImportError, OSError, ValueError):
        return


def _validate_ast(source: str) -> ast.Module:
    parsed = ast.parse(source)
    forbidden_names = {"eval", "exec", "open", "compile", "__import__", "input"}
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise ValueError("Security Error: Forbidden import.")
        if isinstance(node, ast.Name) and (
            node.id in forbidden_names or node.id.startswith("__")
        ):
            raise ValueError(f"Security Error: Forbidden name '{node.id}'.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Security Error: Forbidden attribute '{node.attr}'.")
    return parsed


def _run_function(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": allowed_builtins()}
    exec(compile(parsed, filename="<generated>", mode="exec"), namespace)
    function_name = payload["function_name"]
    function = namespace.get(function_name)
    if not callable(function):
        return {"passed": False, "error": f"Function '{function_name}' not defined."}
    for case in payload["test_cases"]:
        actual = function(*case["args"])
        expected = _decode_json_value(case["expected"])
        if actual != expected:
            return {
                "passed": False,
                "error": (
                    f"Test failed for {case['display_input']}: got {actual!r}, "
                    f"want {expected!r}"
                ),
            }
    return {"passed": True, "error": None}


def _decode_json_value(value: object) -> object:
    if isinstance(value, dict) and set(value) == {"__aptivra_tuple__"}:
        items = value["__aptivra_tuple__"]
        if not isinstance(items, list):
            raise ValueError("Invalid encoded tuple.")
        return tuple(_decode_json_value(item) for item in items)
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _run_snippet(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "__builtins__": allowed_builtins(),
        "math": math,
    }
    exec(compile(parsed, filename="<generated>", mode="exec"), namespace)
    if "res" not in namespace:
        return {
            "passed": False,
            "actual": None,
            "error": "Executable output did not define 'res'.",
        }
    actual = str(namespace["res"]).strip()
    expected = payload["expected"]
    return {
        "passed": actual == expected,
        "actual": actual,
        "error": None if actual == expected else f"Expected '{expected}', got '{actual}'",
    }


def _load_function(payload: dict[str, Any], parsed: ast.Module) -> Any:
    namespace: dict[str, Any] = {"__builtins__": allowed_builtins()}
    exec(compile(parsed, filename="<generated>", mode="exec"), namespace)
    function_name = payload["function_name"]
    function = namespace.get(function_name)
    if not callable(function):
        raise ValueError(f"Function '{function_name}' not defined.")
    return function


def _validate_function(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    _load_function(payload, parsed)
    return {"passed": True, "error": None}


def _execute_function(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    function = _load_function(payload, parsed)
    value = function(*payload["args"], **payload["kwargs"])
    return {"passed": True, "value": value, "error": None}


def _run_batch(payload: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for request in payload["requests"]:
        try:
            parsed = _validate_ast(request["source"])
            if request["mode"] == "function":
                result = _run_function(request, parsed)
            elif request["mode"] == "snippet":
                result = _run_snippet(request, parsed)
            else:
                raise ValueError("unknown sandbox batch mode")
        except BaseException as error:
            result = {
                "passed": False,
                "error": f"Execution Error: {type(error).__name__}: {error}",
            }
        results.append(result)
    return {"passed": True, "value": results, "error": None}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    os.environ.clear()
    os.environ.update({"LANG": "C.UTF-8", "PATH": os.defpath, "PYTHONHASHSEED": "0"})
    _apply_limits(int(payload["memory_limit_mb"]), float(payload["timeout_seconds"]))
    try:
        if payload["mode"] == "batch":
            result = _run_batch(payload)
        elif payload["mode"] == "function":
            parsed = _validate_ast(payload["source"])
            result = _run_function(payload, parsed)
        elif payload["mode"] == "snippet":
            parsed = _validate_ast(payload["source"])
            result = _run_snippet(payload, parsed)
        elif payload["mode"] == "validate_function":
            parsed = _validate_ast(payload["source"])
            result = _validate_function(payload, parsed)
        elif payload["mode"] == "execute_function":
            parsed = _validate_ast(payload["source"])
            result = _execute_function(payload, parsed)
        else:
            raise ValueError("unknown sandbox mode")
    except BaseException as error:
        result = {
            "passed": False,
            "error": f"Execution Error: {type(error).__name__}: {error}",
        }
    try:
        output = json.dumps(result, ensure_ascii=True)
    except (TypeError, ValueError) as error:
        output = json.dumps(
            {
                "passed": False,
                "error": f"Execution result is not JSON-compatible: {error}",
            },
            ensure_ascii=True,
        )
    sys.stdout.write(output[:65536])


if __name__ == "__main__":
    main()
