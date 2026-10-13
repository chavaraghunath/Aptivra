"""Syntax & Bracket Matching Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class SyntaxMatchingVerifier(MechanicalVerifier):
    """Mechanical verifier for bracket validation and completion."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        actual = model_output.strip().split()[-1] if model_output.strip() else ""

        passed = (actual == expected) or (model_output.strip() == expected)
        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=expected if passed else actual,
            error_message=None if passed else f"Expected '{expected}', got '{actual}'",
        )


class SyntaxMatchingTaskFamily(TaskFamily):
    """Generates syntax matching tasks (Balanced bracket check and completion)."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T4_syntax_matching",
            name="Syntax Matching",
            description="Verify bracket balance or predict missing completion brackets.",
        )
        self._verifier = SyntaxMatchingVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @staticmethod
    def _is_balanced(s: str) -> bool:
        stack: list[str] = []
        matching = {")": "(", "]": "[", "}": "{"}
        for char in s:
            if char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or stack[-1] != matching[char]:
                    return False
                stack.pop()
        return len(stack) == 0

    @staticmethod
    def _completion(s: str) -> str:
        stack: list[str] = []
        matching = {"(": ")", "[": "]", "{": "}"}
        for char in s:
            if char in "([{":
                stack.append(char)
            elif char in ")]}" and stack:
                stack.pop()
        return "".join([matching[c] for c in reversed(stack)])

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        modes = ["balance_check", "completion"]
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]

        for i in range(count):
            mode = rng.choice(modes)
            depth = rng.randint(3, 8)

            if mode == "balance_check":
                # 50% chance valid, 50% chance invalid
                if rng.random() > 0.5:
                    # Generate valid
                    val_seq: list[str] = []
                    open_stack: list[str] = []
                    for _ in range(depth):
                        op, cl = rng.choice(pairs)
                        val_seq.append(op)
                        open_stack.append(cl)
                    while open_stack:
                        val_seq.append(open_stack.pop())
                    s = "".join(val_seq)
                    sol = "VALID"
                else:
                    # Generate invalid
                    chars = ["(", ")", "[", "]", "{", "}"]
                    s = "".join(rng.choices(chars, k=depth * 2))
                    sol = "VALID" if self._is_balanced(s) else "INVALID"

                prompt = f"Check if bracket sequence '{s}' is VALID or INVALID ="

            else:  # completion
                comp_seq: list[str] = []
                for _ in range(depth):
                    op, _ = rng.choice(pairs)
                    comp_seq.append(op)
                s = "".join(comp_seq)
                sol = self._completion(s)
                prompt = f"Complete bracket sequence '{s}' ="

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=sol,
                    metadata={"mode": mode},
                )
            )

        return instances
