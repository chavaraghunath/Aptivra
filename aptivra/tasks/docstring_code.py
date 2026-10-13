"""T10: Docstring-to-Code Python Synthesis Task Family and Mechanical Verifier."""

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


class DocstringToCodeVerifier(MechanicalVerifier):
    """Verify docstring-to-code output with AST checks and child-process containment."""

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


class DocstringToCodeTaskFamily(TaskFamily):
    """Task family generating Docstring-to-Code synthesis instances."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T10_docstring_code",
            name="Docstring-to-Code Synthesis",
            description="Synthesize Python functions from docstrings.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = DocstringToCodeVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        templates: list[dict[str, Any]] = [
            {
                "fn_name": "reverse_words",
                "prompt": 'def reverse_words(s: str) -> str:\n    """Reverse words."""\n',
                "solution": (
                    "def reverse_words(s: str) -> str:\n    return ' '.join(s.split()[::-1])"
                ),
                "test_cases": [("hello world", "world hello"), ("a b c", "c b a")],
            },
            {
                "fn_name": "count_vowels",
                "prompt": 'def count_vowels(s: str) -> int:\n    """Count vowels."""\n',
                "solution": (
                    "def count_vowels(s: str) -> int:\n"
                    "    return sum(1 for c in s.lower() if c in 'aeiou')"
                ),
                "test_cases": [("hello", 2), ("xyz", 0), ("AEIOU", 5)],
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
                task_id=f"t10_{i:04d}",
                task_family=self.family_id,
                prompt=prompt,
                target_solution=solution,
                metadata={"fn_name": fn_name, "test_cases": tpl["test_cases"]},
                answer_stop_characters=self.answer_stop_characters,
                training_terminator=self.training_terminator,
            )
            instances.append(inst)
        return instances
