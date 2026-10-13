"""Unit tests for Consolidation Cycle."""

from aptivra.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.model.config import ModelConfig
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
