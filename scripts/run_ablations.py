"""Script executing the 5 M5 ablation configurations across the 8-task stream."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from aptivra.consolidation.cycle import ConsolidationConfig
from aptivra.model.config import ModelConfig
from aptivra.tasks.stream import ContinualTaskStream
from scripts.run_aptivra import run_single_aptivra


def get_ablation_configs() -> dict[str, ConsolidationConfig]:
    """Returns the 5 milestone M5 ablation configurations."""
    return {
        "no_t1": ConsolidationConfig(use_t1_memory=False),
        "no_expert_growth": ConsolidationConfig(use_expert_growth=False),
        "no_replay": ConsolidationConfig(use_replay_buffer=False),
        "no_ewc": ConsolidationConfig(use_ewc_penalty=False),
        "no_gate": ConsolidationConfig(use_promotion_gate=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra Phase 1 Ablations")
    parser.add_argument("--config", type=str, default="configs/backbone_50m.yaml")
    parser.add_argument("--output", type=str, default="experiments/ablation_results.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict)

    output_path = Path(args.output)
    ablations = get_ablation_configs()

    for seed in args.seeds:
        for abl_name, cons_cfg in ablations.items():
            stream = ContinualTaskStream(
                num_train_per_task=100,
                num_val_per_task=20,
                num_test_per_task=20,
                seed=seed,
            )
            print(f"Executing ablation '{abl_name}' with seed={seed}...")
            run_single_aptivra(model_cfg, cons_cfg, stream, output_path)

    print(f"All ablations completed successfully. Metrics written to {output_path}")


if __name__ == "__main__":
    main()
