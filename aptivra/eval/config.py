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
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamVersion

DEFAULT_TASK_STREAM_CONFIG = aptivra_resource("configs", "task_stream.yaml")


@dataclass(frozen=True)
class TaskStreamSettings:
    """Dataset split sizes and the default experiment seed."""

    num_train_per_task: int
    num_val_per_task: int
    num_test_per_task: int
    num_template_ood_test_per_code_family: int
    seed: int
    max_generation_attempts: int
    version: TaskStreamVersion
    family_roles: dict[str, str]
    headline_roles: frozenset[str]


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
            num_template_ood_test_per_code_family=(
                self.task_stream.num_template_ood_test_per_code_family
            ),
            seed=self.task_stream.seed if seed is None else seed,
            max_generation_attempts=self.task_stream.max_generation_attempts,
            version=self.task_stream.version,
            family_roles=self.task_stream.family_roles or None,
            headline_roles=self.task_stream.headline_roles,
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


def _string_set(value: object, name: str) -> frozenset[str]:
    """Validate a non-empty sequence of unique, non-empty strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty YAML list.")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{name}' must contain only non-empty strings.")
    result = frozenset(value)
    if len(result) != len(value):
        raise ValueError(f"'{name}' must not contain duplicate values.")
    return result


def _family_role_settings(
    task_stream: dict[str, Any], version: TaskStreamVersion
) -> tuple[dict[str, str], frozenset[str]]:
    """Validate configured family roles and headline-role selection."""
    if "family_roles" not in task_stream:
        if version == "v2":
            raise ValueError("'family_roles' is required for task stream v2.")
        return {}, frozenset({"skill"})

    from aptivra.tasks.v2 import v2_families

    allowed_roles = _string_set(task_stream.get("roles"), "roles")
    headline_roles = _string_set(task_stream.get("headline_roles"), "headline_roles")
    unknown_headline_roles = headline_roles - allowed_roles
    if unknown_headline_roles:
        raise ValueError(
            "'headline_roles' contains unknown roles: "
            + ", ".join(sorted(unknown_headline_roles))
        )

    raw_family_roles = _require_mapping(task_stream.get("family_roles"), "family_roles")
    family_roles: dict[str, str] = {}
    for family_id, role in raw_family_roles.items():
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("'family_roles' keys must be non-empty family IDs.")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError(
                f"Family '{family_id}' references unknown role '{role}'."
            )
        family_roles[family_id] = role

    known_family_ids = {family.family_id for family in v2_families()}
    unknown_family_ids = set(family_roles) - known_family_ids
    if unknown_family_ids:
        raise ValueError(
            "'family_roles' contains unknown family IDs: "
            + ", ".join(sorted(unknown_family_ids))
        )
    missing_family_ids = known_family_ids - set(family_roles)
    if missing_family_ids:
        raise ValueError(
            "'family_roles' is missing family IDs: "
            + ", ".join(sorted(missing_family_ids))
        )
    if not any(role in headline_roles for role in family_roles.values()):
        raise ValueError("At least one task family must be headline-eligible.")
    return family_roles, headline_roles


def load_experiment_settings(
    path: Path | str | Traversable = DEFAULT_TASK_STREAM_CONFIG,
) -> ExperimentSettings:
    """Load and validate task-stream and evaluation settings from YAML."""
    config_source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "root")
    task_stream = _require_mapping(root.get("task_stream"), "task_stream")
    evaluation = _require_mapping(root.get("evaluation"), "evaluation")
    version = task_stream.get("version", "v1")
    if version not in ("v1", "v2"):
        raise ValueError("'version' must be either 'v1' or 'v2'.")
    family_roles, headline_roles = _family_role_settings(task_stream, version)

    raw_template_ood_size = task_stream.get(
        "num_template_ood_test_per_code_family", 300 if version == "v2" else 0
    )
    if isinstance(raw_template_ood_size, bool) or not isinstance(
        raw_template_ood_size, int
    ):
        raise ValueError(
            "'num_template_ood_test_per_code_family' must be an integer."
        )
    if not 0 <= raw_template_ood_size <= 2_000:
        raise ValueError(
            "'num_template_ood_test_per_code_family' must be between 0 and 2000, "
            f"got {raw_template_ood_size}."
        )
    if version == "v2" and raw_template_ood_size < 5:
        raise ValueError(
            "'num_template_ood_test_per_code_family' must be between 5 and 2000 "
            "for task stream v2."
        )

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
            num_template_ood_test_per_code_family=raw_template_ood_size,
            seed=_bounded_int(task_stream, "seed", minimum=0, maximum=2**31 - 1),
            max_generation_attempts=_bounded_int(
                task_stream, "max_generation_attempts", minimum=1, maximum=1_000
            ),
            version=version,
            family_roles=family_roles,
            headline_roles=headline_roles,
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
