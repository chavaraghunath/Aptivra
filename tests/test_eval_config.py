"""Tests for task-stream experiment configuration and generator cardinality."""

from pathlib import Path

import pytest

from aptivra.eval.config import load_experiment_settings
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import TaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.tasks.structure import structural_hash
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


def test_repository_task_stream_config_has_statistically_useful_test_split() -> None:
    settings = load_experiment_settings(Path("configs/task_stream.yaml"))
    assert settings.task_stream.num_test_per_task >= 300
    assert settings.task_stream.num_template_ood_test_per_code_family == 0
    assert settings.task_stream.seed == 42
    assert settings.task_stream.max_generation_attempts == 100
    assert settings.task_stream.version == "v1"
    assert settings.evaluation.max_new_tokens == 128


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


def test_default_generation_bound_covers_every_current_code_target() -> None:
    settings = load_experiment_settings(Path("configs/task_stream.yaml"))
    families: tuple[TaskFamily, ...] = (
        PythonMicroSynthTaskFamily(),
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        TypeInferrerTaskFamily(),
    )

    for family in families:
        for instance in family.generate_instances(100, seed=42):
            assert len(instance.training_target().encode("utf-8")) <= (
                settings.evaluation.max_new_tokens
            )


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


def test_v2_code_tasks_have_structurally_unique_configured_splits() -> None:
    stream = ContinualTaskStream(500, 100, 300, seed=42, version="v2")

    for family in stream.families[7:]:
        split = stream.get_split(family.family_id)
        train = {structural_hash(item.target_solution) for item in split.train}
        val = {structural_hash(item.target_solution) for item in split.val}
        test = {structural_hash(item.target_solution) for item in split.test}

        assert None not in train | val | test
        assert (len(train), len(val), len(test)) == (500, 100, 300)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

        template_ood = stream.get_template_ood_split(family.family_id)
        template_ood_structures = {
            structural_hash(item.target_solution) for item in template_ood
        }
        assert len(template_ood_structures) == 300
        assert None not in template_ood_structures
        assert train.isdisjoint(template_ood_structures)
        assert val.isdisjoint(template_ood_structures)
        assert test.isdisjoint(template_ood_structures)

    for family in stream.families:
        split = stream.get_split(family.family_id)
        for instance in split.train + split.val + split.test:
            assert len(instance.training_target().encode("utf-8")) <= 128
            assert len(
                (instance.prompt + instance.training_target()).encode("utf-8")
            ) <= 256

        for instance in stream.get_template_ood_split(family.family_id):
            assert len(instance.training_target().encode("utf-8")) <= 128
            assert len(
                (instance.prompt + instance.training_target()).encode("utf-8")
            ) <= 256
