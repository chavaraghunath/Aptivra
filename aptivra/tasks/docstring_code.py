"""T10: Docstring-to-Code Python Synthesis Task Family and Mechanical Verifier."""

import ast
import random
from typing import Any, override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class DocstringToCodeVerifier(MechanicalVerifier):
    """Mechanical verifier for docstring-to-code synthesis via AST & sandboxed execution."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        cleaned_output = model_output.strip()

        if "```python" in cleaned_output:
            cleaned_output = cleaned_output.split("```python")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()

        if cleaned_output == expected:
            return VerifierResult(passed=True, expected=expected, actual=cleaned_output)

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
                    "str": str,
                }
            }
            exec(compile(parsed_ast, filename="<string>", mode="exec"), sandbox_globals)

            test_cases: list[tuple[Any, Any]] = instance.metadata.get("test_cases", [])
            fn_name: str = instance.metadata.get("fn_name", "")

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


class DocstringToCodeTaskFamily(TaskFamily):
    """Task family generating Docstring-to-Code synthesis instances."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T10_docstring_code",
            name="Docstring-to-Code Synthesis",
            description="Synthesize Python functions from docstrings.",
        )
        self._verifier = DocstringToCodeVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        templates: list[dict[str, Any]] = [
            {
                "fn_name": "reverse_words",
                "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse words."""\n',
                "solution": (
                    "def reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])"
                ),
                "test_cases": [("hello world", "world hello"), ("a b c", "c b a")],
            },
            {
                "fn_name": "count_vowels",
                "prompt": 'def count_vowels(s: str) -> int:\n    """Count vowels."""\n',
                "solution": (
                    "def count_vowels(s: str) -> int:\n"
                    "    return sum(1 for c in s.lower() if c in 'aeiou')"
                ),
                "test_cases": [("hello", 2), ("xyz", 0), ("AEIOU", 5)],
            },
        ]

        instances: list[TaskInstance] = []
        for i in range(count):
            tpl = rng.choice(templates)
            inst = TaskInstance(
                task_id=f"t10_{i:04d}",
                task_family=self.family_id,
                prompt=tpl["prompt"],
                target_solution=tpl["solution"],
                metadata={"fn_name": tpl["fn_name"], "test_cases": tpl["test_cases"]},
            )
            instances.append(inst)
        return instances
