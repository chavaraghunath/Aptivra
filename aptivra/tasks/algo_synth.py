"""T9: Algorithmic Python Synthesis Task Family and Mechanical Verifier."""

import random
from typing import Any, override

from aptivra.tasks.base import (
    MULTI_LINE_STOP_CHARACTERS,
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_python_code,
)
from aptivra.tasks.code_sandbox import verify_function_source


class AlgoSynthVerifier(MechanicalVerifier):
    """Verify synthesized Python with AST checks and child-process containment."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        extracted = extract_python_code(model_output)
        if extracted is None:
            return VerifierResult(False, expected, model_output.strip(), "Unterminated code fence.")
        cleaned_output = extracted

        test_cases: list[tuple[Any, Any]] = instance.metadata.get("test_cases", [])
        fn_name: str = instance.metadata.get("fn_name", "")
        if not fn_name or not test_cases:
            return VerifierResult(
                passed=False,
                expected=expected,
                actual=cleaned_output,
                error_message="Verifier metadata requires a function name and test cases.",
            )
        result = verify_function_source(
            cleaned_output,
            fn_name,
            test_cases,
        )
        return VerifierResult(
            passed=result.passed,
            expected=expected,
            actual=cleaned_output,
            error_message=result.error_message,
            execution_status=result.execution_status,
        )


class AlgoSynthTaskFamily(TaskFamily):
    """Task family generating algorithmic synthesis instances."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T9_algo_synth",
            name="Algorithmic Synthesis",
            description="Synthesize algorithmic Python functions.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = AlgoSynthVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        templates: list[dict[str, Any]] = [
            {
                "fn_name": "find_max",
                "prompt": "# Write function find_max(arr) to return max in list arr\n",
                "solution": "def find_max(arr):\n    return max(arr)",
                "test_cases": [([1, 5, 3], 5), ([-10, -2, -5], -2), ([42], 42)],
            },
            {
                "fn_name": "is_palindrome",
                "prompt": "# Write function is_palindrome(s) returning True if palindrome\n",
                "solution": "def is_palindrome(s):\n    return s == s[::-1]",
                "test_cases": [("racecar", True), ("hello", False), ("a", True)],
            },
            {
                "fn_name": "sum_even",
                "prompt": "# Write function sum_even(arr) to return sum of even numbers in arr\n",
                "solution": "def sum_even(arr):\n    return sum(x for x in arr if x % 2 == 0)",
                "test_cases": [([1, 2, 3, 4], 6), ([1, 3, 5], 0), ([2, 4, 6], 12)],
            },
        ]

        instances: list[TaskInstance] = []
        for i in range(count):
            tpl = rng.choice(templates)
            base_name = str(tpl["fn_name"])
            fn_name = f"{base_name}_{seed}_{i}"
            prompt = str(tpl["prompt"]).replace(base_name, fn_name)
            solution = str(tpl["solution"]).replace(base_name, fn_name)
            inst = TaskInstance(
                task_id=f"t9_{i:04d}",
                task_family=self.family_id,
                prompt=prompt,
                target_solution=solution,
                metadata={"fn_name": fn_name, "test_cases": tpl["test_cases"]},
                answer_stop_characters=self.answer_stop_characters,
                training_terminator=self.training_terminator,
            )
            instances.append(inst)
        return instances
