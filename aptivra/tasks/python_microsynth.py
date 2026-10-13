"""Restricted Python Micro-Synthesis Task Family and Mechanical Verifier."""

import ast
import math
import random
from typing import Any, override

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_python_code,
    normalize_verifier_text,
)


class PythonMicroSynthVerifier(MechanicalVerifier):
    """Mechanical verifier for restricted Python micro-synthesis tasks via AST & sandboxed exec."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = normalize_verifier_text(instance.target_solution)
        extracted = extract_python_code(model_output)
        if extracted is None:
            return VerifierResult(
                passed=False,
                expected=expected,
                actual=model_output.strip(),
                error_message="Invalid output: unterminated code fence.",
            )
        cleaned_output = extracted

        # Check exact string match first
        if cleaned_output == expected:
            return VerifierResult(passed=True, expected=expected, actual=cleaned_output)

        # Execute in sandboxed restricted globals namespace
        try:
            parsed_ast = ast.parse(cleaned_output)

            # Security guard: Ensure no dangerous imports or calls in AST
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import | ast.ImportFrom):
                    return VerifierResult(
                        passed=False,
                        expected=expected,
                        actual=cleaned_output,
                        error_message="Security Error: Forbidden imports in synthesis.",
                    )
                if isinstance(node, ast.Name) and node.id in {"eval", "exec", "open", "__import__"}:
                    return VerifierResult(
                        passed=False,
                        expected=expected,
                        actual=cleaned_output,
                        error_message=f"Security Error: Forbidden function '{node.id}'",
                    )

            # Sandboxed execution environment
            restricted_globals: dict[str, Any] = {
                "__builtins__": {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "len": len,
                    "range": range,
                    "list": list,
                    "str": str,
                    "int": int,
                    "bool": bool,
                },
                "math": math,
            }
            local_vars: dict[str, Any] = {}

            # Execute code safely
            exec(
                compile(parsed_ast, filename="<string>", mode="exec"),
                restricted_globals,
                local_vars,
            )

            # Result retrieval: check 'res' or return value
            if "res" in local_vars:
                actual_res = str(local_vars["res"]).strip()
                passed = actual_res == expected
                return VerifierResult(
                    passed=passed,
                    expected=expected,
                    actual=actual_res,
                    error_message=None if passed else f"Expected '{expected}', got '{actual_res}'",
                )

            return VerifierResult(
                passed=False,
                expected=expected,
                actual=cleaned_output,
                error_message="Executable output did not define 'res'.",
            )

        except Exception as e:
            return VerifierResult(
                passed=False,
                expected=expected,
                actual=cleaned_output,
                error_message=f"Execution error: {e}",
            )


class PythonMicroSynthTaskFamily(TaskFamily):
    """Generates small program synthesis tasks in a restricted Python subset."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T8_python_microsynth",
            name="Python Micro-Synthesis",
            description="Synthesize Python snippets (e.g. factorial, GCD, vowel count).",
        )
        self._verifier = PythonMicroSynthVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        modes = ["factorial", "gcd", "count_vowels", "sum_squares"]

        for i in range(count):
            mode = rng.choice(modes)

            if mode == "factorial":
                n = rng.randint(3, 8)
                sol_val = math.factorial(n)
                prompt = f"Write Python code to set `res` to factorial of {n} ="
                target_sol = str(sol_val)
            elif mode == "gcd":
                a = rng.randint(12, 100)
                b = rng.randint(12, 100)
                sol_val = math.gcd(a, b)
                prompt = f"Write Python code to set `res` to GCD of {a} and {b} ="
                target_sol = str(sol_val)
            elif mode == "count_vowels":
                chars = "abcdefghijklmnopqrstuvwxyz"
                s = "".join(rng.choices(chars, k=12))
                sol_val = sum(1 for c in s if c in "aeiou")
                prompt = f"Write Python code to set `res` to count of vowels in string '{s}' ="
                target_sol = str(sol_val)
            else:  # sum_squares
                n = rng.randint(3, 8)
                sol_val = sum(k**2 for k in range(1, n + 1))
                prompt = f"Write Python code to set `res` to sum of squares from 1 to {n} ="
                target_sol = str(sol_val)

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=target_sol,
                    metadata={"mode": mode},
                )
            )

        return instances
