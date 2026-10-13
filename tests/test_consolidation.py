"""Unit tests for Consolidation Cycle."""

from pathlib import Path

import torch

from aptivra.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import VerifierResult
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily


def test_consolidation_cycle_execution() -> None:
    model_cfg = ModelConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64, device="cpu"
    )
    cons_cfg = ConsolidationConfig(
        use_t1_memory=True,
        use_expert_growth=True,
        use_replay_buffer=True,
        use_ewc_penalty=True,
        epochs=1,
    )
    trainer = ConsolidationCycle(model_cfg, cons_cfg)

    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res = trainer.run_cycle(task_idx=0, task_family=family.family_id, train_instances=instances)
    assert "loss" in res
    assert res["flops"] > 0


def test_weight_snapshot_restores_backbone_and_expert_topology(tmp_path: Path) -> None:
    model_cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=32,
        device="cpu",
    )
    cycle = ConsolidationCycle(model_cfg, ConsolidationConfig(epochs=1))
    cycle.spawner.config.entropy_threshold = -1.0
    family = ModularArithmeticTaskFamily()
    instance = family.generate_instances(1, seed=42)[0]
    cycle.t1_store.write_record(
        instance.prompt,
        instance.target_solution,
        VerifierResult(True, instance.target_solution, instance.target_solution),
        instance.task_family,
    )
    cycle.t1_store.query_top_k(instance.prompt, top_k=1)
    cycle.run_cycle(0, instance.task_family, [instance])
    assert cycle.router.num_experts == 2
    assert cycle.experts[0].is_mature is True
    expected_state = {
        name: value.detach().clone() for name, value in cycle.model.state_dict().items()
    }
    snapshot = tmp_path / "weights.pt"
    cycle.save_weight_snapshot(snapshot)

    with torch.no_grad():
        for parameter in cycle.model.parameters():
            parameter.add_(10.0)
    cycle.router.spawn_expert_row()

    cycle.restore_weight_snapshot(snapshot)

    assert cycle.router.num_experts == 2
    assert cycle.router.frozen_mask == [True, False]
    assert len(cycle.experts) == 2
    assert cycle.experts[0].is_mature is True
    for name, value in cycle.model.state_dict().items():
        assert torch.equal(value, expected_state[name])
