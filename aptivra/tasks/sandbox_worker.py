"""Isolated stdlib-only worker for executing one generated Python item."""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from typing import Any


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


def _builtins(names: list[str]) -> dict[str, object]:
    allowed: dict[str, object] = {
        "abs": abs,
        "bool": bool,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    return {name: allowed[name] for name in names if name in allowed}


def _run_function(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": _builtins(payload["builtins"])}
    exec(compile(parsed, filename="<generated>", mode="exec"), namespace)
    function_name = payload["function_name"]
    function = namespace.get(function_name)
    if not callable(function):
        return {"passed": False, "error": f"Function '{function_name}' not defined."}
    for case in payload["test_cases"]:
        actual = function(*case["args"])
        if actual != case["expected"]:
            return {
                "passed": False,
                "error": (
                    f"Test failed for {case['display_input']}: got {actual!r}, "
                    f"want {case['expected']!r}"
                ),
            }
    return {"passed": True, "error": None}


def _run_snippet(payload: dict[str, Any], parsed: ast.Module) -> dict[str, Any]:
    globals_dict: dict[str, Any] = {
        "__builtins__": _builtins(payload["builtins"]),
        "math": math,
    }
    locals_dict: dict[str, Any] = {}
    exec(compile(parsed, filename="<generated>", mode="exec"), globals_dict, locals_dict)
    if "res" not in locals_dict:
        return {
            "passed": False,
            "actual": None,
            "error": "Executable output did not define 'res'.",
        }
    actual = str(locals_dict["res"]).strip()
    expected = payload["expected"]
    return {
        "passed": actual == expected,
        "actual": actual,
        "error": None if actual == expected else f"Expected '{expected}', got '{actual}'",
    }


def main() -> None:
    payload = json.loads(sys.stdin.read())
    os.environ.clear()
    os.environ.update({"LANG": "C.UTF-8", "PATH": os.defpath, "PYTHONHASHSEED": "0"})
    _apply_limits(int(payload["memory_limit_mb"]), float(payload["timeout_seconds"]))
    try:
        parsed = _validate_ast(payload["source"])
        if payload["mode"] == "function":
            result = _run_function(payload, parsed)
        elif payload["mode"] == "snippet":
            result = _run_snippet(payload, parsed)
        else:
            raise ValueError("unknown sandbox mode")
    except BaseException as error:
        result = {
            "passed": False,
            "error": f"Execution Error: {type(error).__name__}: {error}",
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=True)[:65536])


if __name__ == "__main__":
    main()
