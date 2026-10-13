"""Restricted Python Micro-Synthesis Task Family and Mechanical Verifier."""

import math
import random
from typing import override

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_python_code,
    normalize_verifier_text,
)
from aptivra.tasks.code_sandbox import verify_snippet_source

_SNIPPET_BUILTINS = (
    "abs",
    "min",
    "max",
    "sum",
    "len",
    "range",
    "list",
    "str",
    "int",
    "bool",
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

        result = verify_snippet_source(
            cleaned_output,
            expected,
            builtin_names=_SNIPPET_BUILTINS,
        )
        return VerifierResult(
            passed=result.passed,
            expected=expected,
            actual=result.actual if result.actual is not None else cleaned_output,
            error_message=result.error_message,
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
