"""T12: Python Type Annotations Inferrer Task Family and Mechanical Verifier."""

import random
from typing import Any, override

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)


class TypeInferrerVerifier(MechanicalVerifier):
    """Mechanical verifier for Python type annotation inference via AST verification."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class TypeInferrerTaskFamily(TaskFamily):
    """Task family generating type annotation inference instances."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T12_type_inferrer",
            name="Python Type Annotations Inferrer",
            description="Infer typing annotations for Python functions.",
        )
        self._verifier = TypeInferrerVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        templates: list[dict[str, Any]] = [
            {
                "prompt": "# Add type annotations: def multiply(a, b): return a * b\n",
                "solution": "def multiply(a: int, b: int) -> int:\n    return a * b",
            },
            {
                "prompt": "# Add type annotations: def greet(name): return f'Hello {name}'\n",
                "solution": "def greet(name: str) -> str:\n    return f'Hello {name}'",
            },
        ]

        instances: list[TaskInstance] = []
        for i in range(count):
            tpl = rng.choice(templates)
            source = str(tpl["prompt"])
            base_name = source.split("def ", maxsplit=1)[1].split("(", maxsplit=1)[0]
            fn_name = f"{base_name}_{seed}_{i}"
            inst = TaskInstance(
                task_id=f"t12_{i:04d}",
                task_family=self.family_id,
                prompt=source.replace(base_name, fn_name),
                target_solution=str(tpl["solution"]).replace(base_name, fn_name),
            )
            instances.append(inst)
        return instances
