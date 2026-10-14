"""Validated configuration and reproducibility helpers for evaluation runners."""

import random
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from aptivra.resources import aptivra_resource
from aptivra.tasks.stream import ContinualTaskStream

DEFAULT_TASK_STREAM_CONFIG = aptivra_resource("configs", "task_stream.yaml")


@dataclass(frozen=True)
class TaskStreamSettings:
    """Dataset split sizes and the default experiment seed."""

    num_train_per_task: int
    num_val_per_task: int
    num_test_per_task: int
    seed: int
    max_generation_attempts: int


@dataclass(frozen=True)
class EvaluationSettings:
    """Generation limits that determine evaluation cost."""

    max_new_tokens: int


@dataclass(frozen=True)
class ExperimentSettings:
    """Complete settings used by continual-learning experiment scripts."""

    task_stream: TaskStreamSettings
    evaluation: EvaluationSettings

    def make_stream(self, seed: int | None = None) -> ContinualTaskStream:
        """Construct a task stream using validated split sizes."""
        return ContinualTaskStream(
            num_train_per_task=self.task_stream.num_train_per_task,
            num_val_per_task=self.task_stream.num_val_per_task,
            num_test_per_task=self.task_stream.num_test_per_task,
            seed=self.task_stream.seed if seed is None else seed,
            max_generation_attempts=self.task_stream.max_generation_attempts,
        )


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping.")
    return value


def _bounded_int(
    section: dict[str, Any], key: str, *, minimum: int, maximum: int
) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"'{key}' must be between {minimum} and {maximum}, got {value}.")
    return value


def load_experiment_settings(
    path: Path | str | Traversable = DEFAULT_TASK_STREAM_CONFIG,
) -> ExperimentSettings:
    """Load and validate task-stream and evaluation settings from YAML."""
    config_source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")
    task_stream = _require_mapping(root.get("task_stream"), "task_stream")
    evaluation = _require_mapping(root.get("evaluation"), "evaluation")

    return ExperimentSettings(
        task_stream=TaskStreamSettings(
            num_train_per_task=_bounded_int(
                task_stream, "num_train_per_task", minimum=10, maximum=10_000
            ),
            num_val_per_task=_bounded_int(
                task_stream, "num_val_per_task", minimum=5, maximum=2_000
            ),
            num_test_per_task=_bounded_int(
                task_stream, "num_test_per_task", minimum=5, maximum=2_000
            ),
            seed=_bounded_int(task_stream, "seed", minimum=0, maximum=2**31 - 1),
            max_generation_attempts=_bounded_int(
                task_stream, "max_generation_attempts", minimum=1, maximum=1_000
            ),
        ),
        evaluation=EvaluationSettings(
            max_new_tokens=_bounded_int(
                evaluation, "max_new_tokens", minimum=1, maximum=512
            )
        ),
    )


def seed_experiment(seed: int) -> None:
    """Seed every RNG used by model initialization and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
