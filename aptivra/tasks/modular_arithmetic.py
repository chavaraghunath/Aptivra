"""Modular Arithmetic Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class ModularArithmeticVerifier(MechanicalVerifier):
    """Mechanical verifier for modular arithmetic tasks."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected_str = instance.target_solution.strip()
        cleaned_output = model_output.strip().split()[-1] if model_output.strip() else ""

        # Check if the output matches expected integer
        if cleaned_output == expected_str:
            return VerifierResult(
                passed=True,
                expected=expected_str,
                actual=cleaned_output,
            )

        # Try converting both to int to be resilient to formatting
        try:
            expected_val = int(expected_str)
            actual_val = int(cleaned_output)
            passed = expected_val == actual_val
            return VerifierResult(
                passed=passed,
                expected=expected_str,
                actual=cleaned_output,
                error_message=None if passed else f"Expected {expected_val}, got {actual_val}",
            )
        except ValueError:
            return VerifierResult(
                passed=False,
                expected=expected_str,
                actual=model_output.strip(),
                error_message=f"Could not parse integer from '{model_output.strip()}'",
            )


class ModularArithmeticTaskFamily(TaskFamily):
    """Generates modular arithmetic expressions of varying complexity."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T1_modular_arithmetic",
            name="Modular Arithmetic",
            description="Evaluate modular operations (addition, multiplication, exponentiation).",
        )
        self._verifier = ModularArithmeticVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        ops = ["+", "*", "^"]
        for i in range(count):
            modulus = rng.randint(7, 97)
            a = rng.randint(1, 500)
            b = rng.randint(1, 500)
            op = rng.choice(ops)

            if op == "+":
                sol = (a + b) % modulus
                prompt = f"Evaluate: ({a} + {b}) mod {modulus} ="
            elif op == "*":
                sol = (a * b) % modulus
                prompt = f"Evaluate: ({a} * {b}) mod {modulus} ="
            else:  # op == "^"
                exp = rng.randint(1, 20)
                sol = pow(a, exp, modulus)
                prompt = f"Evaluate: ({a} ^ {exp}) mod {modulus} ="

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=str(sol),
                    metadata={"modulus": modulus, "op": op},
                )
            )

        return instances
