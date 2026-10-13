"""Execute all four baselines across the continual-learning task stream."""

import argparse
from pathlib import Path

import yaml

from aptivra.baselines.base import BaselineTrainer
from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.eval.config import load_experiment_settings, seed_experiment
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.model.config import ModelConfig
from aptivra.tasks.stream import ContinualTaskStream


def run_single_baseline(
    baseline_type: str,
    model_config: ModelConfig,
    stream: ContinualTaskStream,
    output_log: Path,
    max_new_tokens: int,
) -> None:
    """Runs a single baseline trainer on the given continual task stream."""
    seed_experiment(stream.seed)
    trainer_types: dict[str, type[BaselineTrainer]] = {
        "sequential": SequentialBaselineTrainer,
        "replay": ReplayBaselineTrainer,
        "ewc": EWCBaselineTrainer,
        "joint": JointMultitaskBaselineTrainer,
    }
    trainer_type = trainer_types.get(baseline_type)
    if trainer_type is None:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    trainer = trainer_type(model_config)

    def predict(prompt: str) -> str:
        return trainer.predict(prompt, max_new_tokens=max_new_tokens)

    harness = ContinualEvalHarness(stream)
    evaluation = harness.start_run()
    evaluation.measure_untrained(predict)
    cumulative_joint_instances = []
    for idx, family in enumerate(stream.families):
        split = stream.get_split(family.family_id)
        train_instances = split.train
        if baseline_type == "joint":
            cumulative_joint_instances.extend(split.train)
            train_instances = cumulative_joint_instances
        trainer.train_task(idx, train_instances)
        evaluation.measure_post_cycle(idx, predict)

    metrics = evaluation.finalize(
        {
            "baseline_type": baseline_type,
            "baseline_protocol": (
                "cumulative_joint_retraining" if baseline_type == "joint" else "sequential"
            ),
            "seed": stream.seed,
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "max_new_tokens": max_new_tokens,
        }
    )
    metrics.total_flops = trainer.total_flops
    harness.save_metrics_jsonl(metrics, output_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra Phase 1 Baselines")
    parser.add_argument("--config", type=str, default="configs/backbone_50m.yaml")
    parser.add_argument("--task-stream-config", type=str, default="configs/task_stream.yaml")
    parser.add_argument("--output", type=str, default="experiments/baseline_results.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict)
    experiment_cfg = load_experiment_settings(args.task_stream_config)

    output_path = Path(args.output)
    baselines = ["sequential", "replay", "ewc", "joint"]

    seeds = args.seeds or [experiment_cfg.task_stream.seed]
    for seed in seeds:
        stream = experiment_cfg.make_stream(seed)
        for b_type in baselines:
            print(f"Executing baseline '{b_type}' with seed={seed}...")
            run_single_baseline(
                b_type,
                model_cfg,
                stream,
                output_path,
                experiment_cfg.evaluation.max_new_tokens,
            )

    print(f"All baselines completed successfully. Metrics written to {output_path}")


if __name__ == "__main__":
    main()
