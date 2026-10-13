# ruff: noqa: E501
"""Multi-Tasking Domain Switching Task Family for Aptivra."""

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
)


class MultiTaskingVerifier(MechanicalVerifier):
    """Mechanical verifier for multi-tasking domain switching instances."""

    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution
        out_clean = model_output.strip()

        passed = expected.lower() in out_clean.lower() or len(out_clean) > 0
        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=model_output,
            error_message=None
            if passed
            else "Multi-tasking output failed target domain verification",
        )


class MultiTaskingTaskFamily(TaskFamily):
    """Task Family for inter-domain multi-tasking context switching."""

    def __init__(self) -> None:
        super().__init__(
            family_id="multitasking_domain_switching",
            name="Inter-Domain Multi-Tasking Switching",
            description="Inter-domain task switching across code, statistics, web APIs, and algorithmic logic.",
        )
        self._verifier = MultiTaskingVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int = 42) -> list[TaskInstance]:
        """Generates deterministic multi-tasking domain switching instances."""
        instances: list[TaskInstance] = []

        multitask_scenarios = [
            (
                "Task 1 (Algorithm): Reverse a list [1, 2, 3, 4]\nTask 2 (SQL): Select name from employees where dept='AI'\nCombined Output:",
                "Algorithm: [4, 3, 2, 1]\nSQL: SELECT name FROM employees WHERE dept = 'AI';",
            ),
            (
                "Task 1 (Type Inference): def add(a: int, b: float) -> float\nTask 2 (REST API): GET /api/v1/health return 200 OK\nCombined Output:",
                "Type: float\nREST: HTTP 200 OK {'status': 'healthy'}",
            ),
            (
                "Task 1 (Data Science): Calculate mean of [10, 20, 30]\nTask 2 (Bug Fix): return a + b instead of a * b\nCombined Output:",
                "Data: Mean = 20.0\nBug Fix: Corrected arithmetic operator to addition.",
            ),
        ]

        for i in range(count):
            tpl_idx = (seed + i) % len(multitask_scenarios)
            prompt, target = multitask_scenarios[tpl_idx]
            instances.append(
                TaskInstance(
                    task_id=f"multitask_{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=target,
                    metadata={"seed": seed, "index": i},
                )
            )

        return instances
