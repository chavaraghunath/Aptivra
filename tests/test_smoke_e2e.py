"""Laptop smoke test verifying full Aptivra pipeline, baselines, and eval harness."""

import json
from pathlib import Path

from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.model.config import ModelConfig
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.v0.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.v0.gate.promotion_gate import PromotionGate
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
    run_single_aptivra(model_cfg, cons_cfg, stream, log_file, max_new_tokens=2)

    assert log_file.exists()
    assert log_file.stat().st_size > 0
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    metadata = record["metadata"]
    assert metadata["harness_version"] == "continual-eval-v4"
    assert metadata["rows_evaluated_per_cycle"] is True
    assert metadata["split_sizes"] == {
        "train_per_task": 5,
        "val_per_task": 2,
        "test_per_task": 2,
    }
    assert metadata["max_new_tokens"] == 2


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

    # Test incremental harness evaluation without retaining a live bound method.
    harness = ContinualEvalHarness(stream)
    evaluation = harness.start_run()
    evaluation.measure_untrained(cycle.predict)
    evaluation.measure_post_cycle(0, cycle.predict)
    metrics = evaluation.finalize()
    assert metrics.average_accuracy is None
    assert metrics.backward_transfer is None
