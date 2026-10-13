"""Tests for task-stream experiment configuration and generator cardinality."""

from pathlib import Path

import pytest

from aptivra.eval.config import load_experiment_settings
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


def test_repository_task_stream_config_has_statistically_useful_test_split() -> None:
    settings = load_experiment_settings(Path("configs/task_stream.yaml"))
    assert settings.task_stream.num_test_per_task >= 300
    assert settings.task_stream.seed == 42
    assert settings.task_stream.max_generation_attempts == 100
    assert settings.evaluation.max_new_tokens == 32


def test_task_stream_config_rejects_invalid_generation_limit(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """task_stream:
  num_train_per_task: 10
  num_val_per_task: 5
  num_test_per_task: 300
  seed: 42
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        load_experiment_settings(config)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("[]\n", "root"),
        (
            """task_stream:
  num_train_per_task: true
  num_val_per_task: 5
  num_test_per_task: 300
  seed: 42
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 32
""",
            "num_train_per_task",
        ),
        (
            """task_stream:
  num_train_per_task: 10
  num_val_per_task: 5
  num_test_per_task: 300
  seed: -1
  max_generation_attempts: 100
evaluation:
  max_new_tokens: 32
""",
            "seed",
        ),
    ],
)
def test_task_stream_config_rejects_malformed_values(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    config = tmp_path / "malformed.yaml"
    config.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_settings(config)


@pytest.mark.parametrize(
    "family",
    [
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        TypeInferrerTaskFamily(),
    ],
)
def test_code_task_generators_have_enough_unique_instances_for_default_splits(
    family: object,
) -> None:
    instances = family.generate_instances(900, seed=42)  # type: ignore[attr-defined]
    prompt_targets = {(instance.prompt, instance.target_solution) for instance in instances}
    assert len(prompt_targets) == 900
