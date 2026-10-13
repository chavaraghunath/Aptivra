"""T9: Algorithmic Python Synthesis Task Family and Mechanical Verifier."""

import ast
import random
from typing import Any, override

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_python_code,
)


class AlgoSynthVerifier(MechanicalVerifier):
    """Mechanical verifier for algorithmic Python synthesis via AST & sandboxed execution."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        extracted = extract_python_code(model_output)
        if extracted is None:
            return VerifierResult(False, expected, model_output.strip(), "Unterminated code fence.")
        cleaned_output = extracted

        try:
            parsed_ast = ast.parse(cleaned_output)
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import | ast.ImportFrom):
                    return VerifierResult(
                        passed=False,
                        expected=expected,
                        actual=cleaned_output,
                        error_message="Security Error: Forbidden import.",
                    )

            sandbox_globals: dict[str, Any] = {
                "__builtins__": {
                    "range": range,
                    "len": len,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "sorted": sorted,
                    "sum": sum,
                    "list": list,
                    "bool": bool,
                }
            }
            exec(compile(parsed_ast, filename="<string>", mode="exec"), sandbox_globals)

            test_cases: list[tuple[Any, Any]] = instance.metadata.get("test_cases", [])
            fn_name: str = instance.metadata.get("fn_name", "")

            if not fn_name or not test_cases:
                return VerifierResult(
                    passed=False,
                    expected=expected,
                    actual=cleaned_output,
                    error_message="Verifier metadata requires a function name and test cases.",
                )

            if fn_name not in sandbox_globals:
                return VerifierResult(
                    passed=False,
                    expected=expected,
                    actual=cleaned_output,
                    error_message=f"Function '{fn_name}' not defined.",
                )

            fn = sandbox_globals[fn_name]
            for test_in, expected_out in test_cases:
                res = fn(*test_in) if isinstance(test_in, tuple) else fn(test_in)
                if res != expected_out:
                    err_msg = f"Test failed for {test_in}: got {res}, want {expected_out}"
                    return VerifierResult(
                        passed=False,
                        expected=expected,
                        actual=cleaned_output,
                        error_message=err_msg,
                    )

            return VerifierResult(passed=True, expected=expected, actual=cleaned_output)
        except Exception as err:
            return VerifierResult(
                passed=False,
                expected=expected,
                actual=cleaned_output,
                error_message=f"Execution Error: {err}",
            )


class AlgoSynthTaskFamily(TaskFamily):
    """Task family generating algorithmic synthesis instances."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T9_algo_synth",
            name="Algorithmic Synthesis",
            description="Synthesize algorithmic Python functions.",
        )
        self._verifier = AlgoSynthVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        templates: list[dict[str, Any]] = [
            {
                "fn_name": "find_max",
                "prompt": "# Write function find_max(arr) to return max in list arr\n",
                "solution": "def find_max(arr):\n    return max(arr)",
                "test_cases": [([1, 5, 3], 5), ([-10, -2, -5], -2), ([42], 42)],
            },
            {
                "fn_name": "is_palindrome",
                "prompt": "# Write function is_palindrome(s) returning True if palindrome\n",
                "solution": "def is_palindrome(s):\n    return s == s[::-1]",
                "test_cases": [("racecar", True), ("hello", False), ("a", True)],
            },
            {
                "fn_name": "sum_even",
                "prompt": "# Write function sum_even(arr) to return sum of even numbers in arr\n",
                "solution": "def sum_even(arr):\n    return sum(x for x in arr if x % 2 == 0)",
                "test_cases": [([1, 2, 3, 4], 6), ([1, 3, 5], 0), ([2, 4, 6], 12)],
            },
        ]

        instances: list[TaskInstance] = []
        for i in range(count):
            tpl = rng.choice(templates)
            base_name = str(tpl["fn_name"])
            fn_name = f"{base_name}_{seed}_{i}"
            prompt = str(tpl["prompt"]).replace(base_name, fn_name)
            solution = str(tpl["solution"]).replace(base_name, fn_name)
            inst = TaskInstance(
                task_id=f"t9_{i:04d}",
                task_family=self.family_id,
                prompt=prompt,
                target_solution=solution,
                metadata={"fn_name": fn_name, "test_cases": tpl["test_cases"]},
            )
            instances.append(inst)
        return instances
