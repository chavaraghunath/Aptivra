"""Unit tests for all 4 continual learning baseline trainers."""

import pytest
import torch
import torch.nn as nn

from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.model.config import ModelConfig
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


class _ScriptedNextTokenModel(nn.Module):
    def __init__(self, token_ids: list[int], vocab_size: int) -> None:
        super().__init__()
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        token_id = self.token_ids[self.calls]
        self.calls += 1
        logits = torch.full(
            (tokens.shape[0], tokens.shape[1], self.vocab_size),
            -1.0,
            device=tokens.device,
        )
        logits[:, -1, token_id] = 1.0
        return logits, {}


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


def test_predict_honors_task_supplied_stop_tokens(small_config: ModelConfig) -> None:
    trainer = SequentialBaselineTrainer(small_config, epochs_per_task=1, batch_size=1)
    scripted_tokens = [ord("a"), ord("\n"), ord("b"), 0]
    trainer.model = _ScriptedNextTokenModel(scripted_tokens, small_config.vocab_size)

    multiline = trainer.predict("p", max_new_tokens=8, stop_token_ids=(0,))

    trainer.model = _ScriptedNextTokenModel(scripted_tokens, small_config.vocab_size)
    single_line = trainer.predict(
        "p", max_new_tokens=8, stop_token_ids=(0, ord("\n"))
    )

    assert multiline == "a\nb\x00"
    assert single_line == "a\n"


def test_continual_training_targets_end_with_family_stop_byte() -> None:
    code_families = (
        PythonMicroSynthTaskFamily(),
        AlgoSynthTaskFamily(),
        DocstringToCodeTaskFamily(),
        BugFixerTaskFamily(),
        TypeInferrerTaskFamily(),
    )
    for family in code_families:
        instance = family.generate_instances(1, seed=42)[0]
        encoded = SequentialBaselineTrainer.encode_training_target(instance)
        assert encoded[-1].item() == 0
        assert instance.training_target().endswith("\x00")

    single_line = ModularArithmeticTaskFamily().generate_instances(1, seed=42)[0]
    encoded_single_line = SequentialBaselineTrainer.encode_training_target(single_line)
    assert encoded_single_line[-1].item() == ord("\n")
