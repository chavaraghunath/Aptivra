"""String Transformation Task Family and Mechanical Verifier."""

import random
import string
from typing import override

from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult


class StringTransformVerifier(MechanicalVerifier):
    """Mechanical verifier for string transformation tasks."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        expected = instance.target_solution.strip()
        cleaned_output = model_output.strip().split()[-1] if model_output.strip() else ""

        passed = (cleaned_output == expected) or (model_output.strip() == expected)
        actual = expected if passed else cleaned_output

        return VerifierResult(
            passed=passed,
            expected=expected,
            actual=actual,
            error_message=None if passed else f"Expected '{expected}', got '{cleaned_output}'",
        )


class StringTransformTaskFamily(TaskFamily):
    """Generates string transformation tasks (Reversal, RLE, Caesar Cipher)."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T2_string_transforms",
            name="String Transformations",
            description="Apply string operations: reversal, run-length encoding, Caesar shift.",
        )
        self._verifier = StringTransformVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @staticmethod
    def _run_length_encode(s: str) -> str:
        if not s:
            return ""
        res: list[str] = []
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                res.append(f"{count}{s[i - 1]}")
                count = 1
        res.append(f"{count}{s[-1]}")
        return "".join(res)

    @staticmethod
    def _caesar_shift(s: str, shift: int) -> str:
        res: list[str] = []
        for char in s:
            if char in string.ascii_lowercase:
                idx = (ord(char) - ord("a") + shift) % 26
                res.append(chr(ord("a") + idx))
            else:
                res.append(char)
        return "".join(res)

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        modes = ["reverse", "rle", "caesar"]
        for i in range(count):
            mode = rng.choice(modes)
            length = rng.randint(6, 14)

            if mode == "reverse":
                src = "".join(rng.choices(string.ascii_lowercase, k=length))
                sol = src[::-1]
                prompt = f"Reverse string '{src}' ="
            elif mode == "rle":
                chars: list[str] = []
                for _ in range(rng.randint(3, 6)):
                    ch = rng.choice(string.ascii_lowercase)
                    chars.extend([ch] * rng.randint(1, 4))
                src = "".join(chars)
                sol = self._run_length_encode(src)
                prompt = f"RLE encode '{src}' ="
            else:  # caesar
                shift = rng.randint(1, 10)
                src = "".join(rng.choices(string.ascii_lowercase, k=length))
                sol = self._caesar_shift(src, shift)
                prompt = f"Caesar shift +{shift} on '{src}' ="

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
