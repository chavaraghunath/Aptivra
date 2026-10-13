"""Laptop smoke test verifying full Aptivra pipeline, baselines, and eval harness."""

from pathlib import Path

from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.gate.promotion_gate import PromotionGate
from aptivra.model.config import ModelConfig
from aptivra.tasks.stream import ContinualTaskStream
from scripts.run_aptivra import run_single_aptivra


def test_baselines_end_to_end_smoke() -> None:
    """Verifies baseline trainers run end-to-end on miniature task stream."""
    stream = ContinualTaskStream(
        num_train_per_task=5,
        num_val_per_task=2,
        num_test_per_task=2,
        seed=123,
    )
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)

    seq_trainer = SequentialBaselineTrainer(cfg)
    rep_trainer = ReplayBaselineTrainer(cfg)

    # Train sequential baseline on task 0
    t0_split = stream.get_split(stream.families[0].family_id)
    seq_trainer.train_task(0, t0_split.train)
    assert seq_trainer.total_flops > 0

    # Train replay baseline on task 0
    rep_trainer.train_task(0, t0_split.train)
    assert rep_trainer.total_flops > 0


def test_aptivra_pipeline_end_to_end_smoke(tmp_path: Path) -> None:
    """Verifies full Aptivra pipeline executes end-to-end with metrics generation."""
    stream = ContinualTaskStream(
        num_train_per_task=5,
        num_val_per_task=2,
        num_test_per_task=2,
        seed=42,
    )
    model_cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    cons_cfg = ConsolidationConfig(epochs=1)

    log_file = tmp_path / "smoke_aptivra.jsonl"
    run_single_aptivra(model_cfg, cons_cfg, stream, log_file)

    assert log_file.exists()
    assert log_file.stat().st_size > 0


def test_consolidation_cycle_and_gate_smoke() -> None:
    """Verifies consolidation cycle and promotion gate interaction."""
    model_cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    cons_cfg = ConsolidationConfig(epochs=1)
    cycle = ConsolidationCycle(model_cfg, cons_cfg)
    gate = PromotionGate(alpha=0.05)
    assert gate.alpha == 0.05

    stream = ContinualTaskStream(
        num_train_per_task=5,
        num_val_per_task=2,
        num_test_per_task=2,
        seed=777,
    )
    split0 = stream.get_split(stream.families[0].family_id)
    res = cycle.run_cycle(0, stream.families[0].family_id, split0.train)
    assert res["loss"] >= 0.0

    # Test harness evaluation matrix
    harness = ContinualEvalHarness(stream)
    metrics = harness.evaluate_matrix([cycle.predict] * len(stream.families))
    assert 0.0 <= metrics.average_accuracy <= 1.0
    assert isinstance(metrics.backward_transfer, float)
