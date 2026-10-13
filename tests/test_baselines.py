"""Unit tests for all 4 continual learning baseline trainers."""

import pytest

from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.model.config import ModelConfig
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        device="cpu",
    )


def test_sequential_baseline(small_config: ModelConfig) -> None:
    trainer = SequentialBaselineTrainer(small_config, epochs_per_task=1, batch_size=4)
    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res = trainer.train_task(task_idx=0, train_instances=instances)
    assert "loss" in res
    assert res["flops"] > 0

    pred = trainer.predict(prompt="Evaluate: (5 + 5) mod 7 =")
    assert isinstance(pred, str)


def test_replay_baseline(small_config: ModelConfig) -> None:
    trainer = ReplayBaselineTrainer(
        small_config, replay_capacity=20, epochs_per_task=1, batch_size=4
    )
    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res1 = trainer.train_task(task_idx=0, train_instances=instances)
    assert res1["flops"] > 0
    assert len(trainer.replay_buffer) > 0

    res2 = trainer.train_task(task_idx=1, train_instances=instances)
    assert res2["flops"] > res1["flops"]


def test_ewc_baseline(small_config: ModelConfig) -> None:
    trainer = EWCBaselineTrainer(small_config, ewc_lambda=10.0, epochs_per_task=1, batch_size=4)
    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res1 = trainer.train_task(task_idx=0, train_instances=instances)
    assert len(trainer.fisher_matrices) == 1

    res2 = trainer.train_task(task_idx=1, train_instances=instances)
    assert len(trainer.fisher_matrices) == 2
    assert res2["flops"] > res1["flops"]


def test_joint_multitask_baseline(small_config: ModelConfig) -> None:
    trainer = JointMultitaskBaselineTrainer(small_config, total_epochs=1, batch_size=4)
    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res = trainer.train_task(task_idx=0, train_instances=instances)
    assert res["flops"] > 0
