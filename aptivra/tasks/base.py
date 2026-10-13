"""Base interfaces and data structures for task families and mechanical verifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


def normalize_verifier_text(value: str) -> str:
    """Normalize verifier text by trimming and collapsing all whitespace.

    Equality remains case- and punctuation-sensitive.  This deliberately avoids
    keyword, substring, last-token, and formatting-deletion shortcuts that can
    turn a near miss into a false positive.
    """
    return " ".join(value.strip().split())


def exact_match_result(instance: "TaskInstance", model_output: str) -> "VerifierResult":
    """Return an exact-match result after the documented text normalization."""
    expected = normalize_verifier_text(instance.target_solution)
    actual = normalize_verifier_text(model_output)
    passed = actual == expected
    return VerifierResult(
        passed=passed,
        expected=expected,
        actual=actual,
        error_message=None if passed else f"Expected '{expected}', got '{actual}'",
    )


def extract_python_code(model_output: str) -> str | None:
    """Extract one complete Markdown code fence, or return plain stripped source.

    An unterminated fence is invalid instead of raising or silently accepting a
    partial snippet.  Both `````python`` and unlabelled fences are supported.
    """
    stripped = model_output.strip()
    fence_start = stripped.find("```")
    if fence_start < 0:
        return stripped
    fence_end = stripped.find("```", fence_start + 3)
    if fence_end < 0:
        return None
    fenced = stripped[fence_start + 3 : fence_end]
    if fenced.startswith("python"):
        fenced = fenced[len("python") :]
    return fenced.strip()


@dataclass(frozen=True)
class TaskInstance:
    """Represents a single problem instance within a task family."""

    task_id: str
    task_family: str
    prompt: str
    target_solution: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    """Outcome of evaluating a model output against a mechanical verifier."""

    passed: bool
    expected: str
    actual: str
    error_message: str | None = None


class MechanicalVerifier(ABC):
    """Abstract base class for deterministic, non-model programmatic verifiers."""

    @abstractmethod
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        """Deterministically evaluates correctness of model_output for the given instance."""
        pass


class TaskFamily(ABC):
    """Abstract base class for a continual learning task family."""

    def __init__(self, family_id: str, name: str, description: str) -> None:
        self.family_id = family_id
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def verifier(self) -> MechanicalVerifier:
        """Returns the mechanical verifier for this task family."""
        pass

    def generate_unique_instances(
        self,
        count: int,
        seed: int,
        *,
        excluded_instances: list[TaskInstance] | None = None,
        max_attempts: int,
    ) -> list[TaskInstance]:
        """Generate distinct prompt/solution pairs or fail after bounded attempts."""
        excluded_keys = {
            (instance.task_family, instance.prompt, instance.target_solution)
            for instance in (excluded_instances or [])
        }
        unique_instances: list[TaskInstance] = []
        unique_keys: set[tuple[str, str, str]] = set()

        for attempt in range(max_attempts):
            remaining = count - len(unique_instances)
            if remaining <= 0:
                return unique_instances
            candidates = self.generate_instances(
                max(remaining * 2, 1), seed=seed + attempt * 10_000
            )
            for instance in candidates:
                key = (instance.task_family, instance.prompt, instance.target_solution)
                if key in excluded_keys or key in unique_keys:
                    continue
                unique_keys.add(key)
                unique_instances.append(instance)
                if len(unique_instances) == count:
                    return unique_instances

        raise ValueError(
            f"Task family '{self.family_id}' could generate only "
            f"{len(unique_instances)} of {count} requested unique instances after "
            f"{max_attempts} attempts. Reduce the split sizes or increase task diversity."
        )

    @abstractmethod
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        """Generates deterministic task instances for given seed."""
        pass
