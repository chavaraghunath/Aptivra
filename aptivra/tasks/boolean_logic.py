"""Boolean Logic Synthesis Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class BooleanLogicVerifier(MechanicalVerifier):
    """Mechanical verifier for boolean logic expressions."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip().upper()
        cleaned = model_output.strip().split()[-1].upper() if model_output.strip() else ""

        passed = (cleaned == expected) or (model_output.strip().upper() == expected)
        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=expected if passed else cleaned,
            error_message=None if passed else f"Expected '{expected}', got '{cleaned}'",
        )


class BooleanLogicTaskFamily(TaskFamily):
    """Generates boolean logic expression evaluation tasks."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T5_boolean_logic",
            name="Boolean Logic",
            description="Evaluate complex boolean logic expressions (AND, OR, NOT, XOR).",
        )
        self._verifier = BooleanLogicVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        ops = ["AND", "OR", "XOR"]

        for i in range(count):
            depth = rng.randint(2, 4)
            vals = [rng.choice([True, False]) for _ in range(depth + 1)]
            selected_ops = [rng.choice(ops) for _ in range(depth)]

            # Build expression string and evaluate
            expr_parts: list[str] = [str(vals[0])]
            curr_val = vals[0]

            for op, val in zip(selected_ops, vals[1:], strict=True):
                expr_parts.append(f"{op} {val}")
                if op == "AND":
                    curr_val = curr_val and val
                elif op == "OR":
                    curr_val = curr_val or val
                else:  # XOR
                    curr_val = curr_val != val

            expr_str = " ".join(expr_parts)
            target_sol = "TRUE" if curr_val else "FALSE"
            prompt = f"Evaluate boolean expression: {expr_str} ="

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=target_sol,
                    metadata={"depth": depth},
                )
            )

        return instances
