"""T12: Python Type Annotations Inferrer Task Family and Mechanical Verifier."""

import ast
import random
from typing import Any, override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class TypeInferrerVerifier(MechanicalVerifier):
    """Mechanical verifier for Python type annotation inference via AST verification."""

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
            fn_node = None
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.FunctionDef):
                    fn_node = node
                    break

            if fn_node is None:
                return VerifierResult(
                    passed=False,
                    expected=expected,
                    actual=cleaned_output,
                    error_message="No FunctionDef node found in AST.",
                )

            missing_arg_type = any(arg.annotation is None for arg in fn_node.args.args)
            missing_ret_type = fn_node.returns is None

            if missing_arg_type or missing_ret_type:
                return VerifierResult(
                    passed=False,
                    expected=expected,
                    actual=cleaned_output,
                    error_message="Missing type annotations on arguments or return type.",
                )

            return VerifierResult(passed=True, expected=expected, actual=cleaned_output)
        except Exception as err:
            return VerifierResult(
                passed=False,
                expected=expected,
                actual=cleaned_output,
                error_message=f"AST Parse Error: {err}",
            )


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
            inst = TaskInstance(
                task_id=f"t12_{i:04d}",
                task_family=self.family_id,
                prompt=tpl["prompt"],
                target_solution=tpl["solution"],
            )
            instances.append(inst)
        return instances
