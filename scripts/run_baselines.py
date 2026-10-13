"""Script executing all 4 baselines across the 8-task continual learning stream."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.model.config import ModelConfig
from aptivra.tasks.stream import ContinualTaskStream


def run_single_baseline(
    baseline_type: str,
    model_config: ModelConfig,
    stream: ContinualTaskStream,
    output_log: Path,
) -> None:
    """Runs a single baseline trainer on the given continual task stream."""
    eval_fn_at_cycle: list[Callable[[str], str]] = []
    trainer: Any = None

    if baseline_type == "sequential":
        trainer = SequentialBaselineTrainer(model_config)
        for idx, family in enumerate(stream.families):
            split = stream.get_split(family.family_id)
            trainer.train_task(idx, split.train)
            eval_fn_at_cycle.append(trainer.predict)

    elif baseline_type == "replay":
        trainer = ReplayBaselineTrainer(model_config)
        for idx, family in enumerate(stream.families):
            split = stream.get_split(family.family_id)
            trainer.train_task(idx, split.train)
            eval_fn_at_cycle.append(trainer.predict)

    elif baseline_type == "ewc":
        trainer = EWCBaselineTrainer(model_config)
        for idx, family in enumerate(stream.families):
            split = stream.get_split(family.family_id)
            trainer.train_task(idx, split.train)
            eval_fn_at_cycle.append(trainer.predict)

    elif baseline_type == "joint":
        trainer = JointMultitaskBaselineTrainer(model_config)
        all_train_instances = []
        for family in stream.families:
            split = stream.get_split(family.family_id)
            all_train_instances.extend(split.train)

        trainer.train_task(0, all_train_instances)
        eval_fn_at_cycle = [trainer.predict] * len(stream.families)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    harness = ContinualEvalHarness(stream)
    metrics = harness.evaluate_matrix(eval_fn_at_cycle)
    metrics.total_flops = trainer.total_flops
    metrics.metadata = {
        "baseline_type": baseline_type,
        "seed": stream.seed,
        "d_model": model_config.d_model,
        "n_layers": model_config.n_layers,
    }
    harness.save_metrics_jsonl(metrics, output_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra Phase 1 Baselines")
    parser.add_argument("--config", type=str, default="configs/backbone_50m.yaml")
    parser.add_argument("--output", type=str, default="experiments/baseline_results.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict)

    output_path = Path(args.output)
    baselines = ["sequential", "replay", "ewc", "joint"]

    for seed in args.seeds:
        stream = ContinualTaskStream(
            num_train_per_task=100,
            num_val_per_task=20,
            num_test_per_task=20,
            seed=seed,
        )
        for b_type in baselines:
            print(f"Executing baseline '{b_type}' with seed={seed}...")
            run_single_baseline(b_type, model_cfg, stream, output_path)

    print(f"All baselines completed successfully. Metrics written to {output_path}")


if __name__ == "__main__":
    main()
