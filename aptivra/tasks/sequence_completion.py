"""Sequence Completion Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class SequenceCompletionVerifier(MechanicalVerifier):
    """Mechanical verifier for numerical sequence completion."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        cleaned = model_output.strip().split()[-1] if model_output.strip() else ""

        passed = cleaned == expected
        if not passed:
            try:
                passed = int(cleaned) == int(expected)
            except ValueError:
                passed = False

        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=expected if passed else cleaned,
            error_message=None if passed else f"Expected '{expected}', got '{cleaned}'",
        )


class SequenceCompletionTaskFamily(TaskFamily):
    """Generates sequence completion tasks (Arithmetic, Geometric, Fibonacci, Quadratic)."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T6_sequence_completion",
            name="Sequence Completion",
            description="Predict the next element in numerical sequences.",
        )
        self._verifier = SequenceCompletionVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        types = ["arithmetic", "geometric", "fibonacci", "quadratic"]

        for i in range(count):
            seq_type = rng.choice(types)
            seq_len = rng.randint(4, 6)

            if seq_type == "arithmetic":
                start = rng.randint(-20, 50)
                diff = rng.randint(2, 15)
                seq = [start + j * diff for j in range(seq_len + 1)]
            elif seq_type == "geometric":
                start = rng.randint(1, 10)
                ratio = rng.randint(2, 4)
                seq = [start * (ratio**j) for j in range(seq_len + 1)]
            elif seq_type == "fibonacci":
                a = rng.randint(1, 10)
                b = rng.randint(1, 10)
                seq = [a, b]
                for _ in range(seq_len - 1):
                    seq.append(seq[-1] + seq[-2])
            else:  # quadratic n^2 + c
                c = rng.randint(-5, 15)
                seq = [(j**2) + c for j in range(1, seq_len + 2)]

            input_seq = seq[:-1]
            next_val = seq[-1]
            seq_str = ", ".join(map(str, input_seq))
            prompt = f"Predict next value in sequence: {seq_str}, _ ="

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=str(next_val),
                    metadata={"seq_type": seq_type},
                )
            )

        return instances
