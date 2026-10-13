"""Base interfaces and data structures for task families and mechanical verifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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

    @abstractmethod
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        """Generates deterministic task instances for given seed."""
        pass
