"""List Operations Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class ListOperationsVerifier(MechanicalVerifier):
    """Mechanical verifier for list sorting, filtering, and top-K selection."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        cleaned = model_output.strip().replace("[", "").replace("]", "").replace(",", " ")
        cleaned_str = " ".join(cleaned.split())
        exp_clean = expected.replace("[", "").replace("]", "").replace(",", " ")
        expected_str = " ".join(exp_clean.split())

        passed = cleaned_str == expected_str
        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=cleaned_str,
            error_message=None if passed else f"Expected '{expected}', got '{cleaned_str}'",
        )


class ListOperationsTaskFamily(TaskFamily):
    """Generates list manipulation tasks (Sort, Filter Even/Odd, Top-K)."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T3_list_operations",
            name="List Operations",
            description="Perform list sorting, filtering (even/odd), and top-K extraction.",
        )
        self._verifier = ListOperationsVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        modes = ["sort_asc", "sort_desc", "filter_even", "top_k"]
        for i in range(count):
            mode = rng.choice(modes)
            size = rng.randint(6, 12)
            nums = [rng.randint(-50, 100) for _ in range(size)]

            if mode == "sort_asc":
                sol_list = sorted(nums)
                prompt = f"Sort ascending {nums} ="
            elif mode == "sort_desc":
                sol_list = sorted(nums, reverse=True)
                prompt = f"Sort descending {nums} ="
            elif mode == "filter_even":
                sol_list = [x for x in nums if x % 2 == 0]
                prompt = f"Filter even numbers in {nums} ="
            else:  # top_k
                k = rng.randint(2, min(4, size))
                sol_list = sorted(nums, reverse=True)[:k]
                prompt = f"Find top {k} maximum values in {nums} ="

            target_sol = " ".join(map(str, sol_list))
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
