"""Script executing Aptivra continual learning runner across seeds and logging metrics."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml

from aptivra.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.eval.config import load_experiment_settings, seed_experiment
from aptivra.eval.harness import ContinualEvalHarness, PredictFunction
from aptivra.gate.promotion_gate import PromotionGate
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.stream import ContinualTaskStream


def verifier_eval_wrapper(
    predict_fn: PredictFunction,
    instances: list[TaskInstance],
    harness: ContinualEvalHarness,
) -> float:
    """Helper calculating overall accuracy of predict_fn across task instances."""
    return harness.evaluate_instances(predict_fn, instances)


def loss_eval_wrapper(
    predict_fn: PredictFunction,
    instances: list[TaskInstance],
    cycle: ConsolidationCycle,
    harness: ContinualEvalHarness,
) -> np.ndarray:
    """Helper calculating token cross-entropy loss vector over task instances."""
    losses: list[float] = []
    for inst in instances:
        out = harness.predict_answer(predict_fn, inst)
        s_toks = cycle.encode_text(f"{out}{inst.training_terminator}")
        tgt_toks = cycle.encode_training_target(inst)
        max_len = min(len(s_toks), len(tgt_toks))
        if max_len == 0:
            losses.append(2.0)
            continue
        errs = sum(
            1.0 for a, b in zip(s_toks[:max_len], tgt_toks[:max_len], strict=False) if a != b
        )
        losses.append(errs / max_len)
    return np.array(losses, dtype=np.float64)


def run_single_aptivra(
    model_config: ModelConfig,
    cons_config: ConsolidationConfig,
    stream: ContinualTaskStream,
    output_log: Path,
    max_new_tokens: int,
) -> None:
    """Runs a single Aptivra continual learning cycle across the task stream."""
    seed_experiment(stream.seed)
    cycle = ConsolidationCycle(model_config, cons_config)
    gate = PromotionGate(alpha=0.05)
    held_out_regression_suite: list[TaskInstance] = []
    gate_results: list[dict[str, Any]] = []
    harness = ContinualEvalHarness(stream)
    evaluation = harness.start_run()

    def predict(prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        return cycle.predict(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )

    evaluation.measure_untrained(predict)

    with TemporaryDirectory(prefix="aptivra-promotion-") as snapshot_dir:
        snapshot_path = Path(snapshot_dir) / "promoted_weights.pt"
        for task_idx, family in enumerate(stream.families):
            split = stream.get_split(family.family_id)
            current_val_accuracy = 0.0
            current_losses = np.empty(0, dtype=np.float64)
            if cons_config.use_promotion_gate and task_idx > 0:
                current_val_accuracy = verifier_eval_wrapper(predict, split.val, harness)
                current_losses = loss_eval_wrapper(
                    predict, held_out_regression_suite, cycle, harness
                )
                cycle.save_promotion_snapshot(snapshot_path)

            # 1. Populate T1 memory with verifier-verified examples if enabled
            if cons_config.use_t1_memory:
                for inst in split.train[:20]:
                    res = family.verifier.verify(inst, inst.target_solution)
                    if res.passed:
                        cycle.t1_store.write_record(
                            prompt=inst.prompt,
                            target_solution=inst.target_solution,
                            verifier_result=res,
                            task_family=family.family_id,
                            answer_stop_characters=inst.answer_stop_characters,
                            training_terminator=inst.training_terminator,
                        )

            # 2. Run Consolidation Cycle on current task
            cycle.run_cycle(task_idx, family.family_id, split.train)

            # 3. Compare candidate measurements with the pre-training promoted state.
            promoted = True
            if cons_config.use_promotion_gate and task_idx > 0:
                gate_res = gate.evaluate_candidate_with_rollback(
                    current_val_accuracy=current_val_accuracy,
                    current_losses=current_losses,
                    cand_predict_fn=predict,
                    new_task_val_instances=split.val,
                    held_out_regression_suite=held_out_regression_suite,
                    verifier_eval_fn=lambda pfn, insts: verifier_eval_wrapper(
                        pfn, insts, harness
                    ),
                    loss_eval_fn=lambda pfn, insts: loss_eval_wrapper(
                        pfn, insts, cycle, harness
                    ),
                    rollback_fn=lambda: cycle.restore_promotion_snapshot(snapshot_path),
                )
                promoted = gate_res.promoted
                gate_results.append(
                    {
                        "task_idx": task_idx,
                        "promoted": promoted,
                        "p_value": gate_res.p_value,
                        "val_acc_gain": gate_res.val_acc_gain,
                    }
                )
            if promoted:
                held_out_regression_suite.extend(split.val)
            evaluation.measure_post_cycle(task_idx, predict)

    metrics = evaluation.finalize(
        {
            "model_type": "aptivra",
            "seed": stream.seed,
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "max_new_tokens": max_new_tokens,
            "use_t1_memory": cons_config.use_t1_memory,
            "use_expert_growth": cons_config.use_expert_growth,
            "use_replay_buffer": cons_config.use_replay_buffer,
            "use_ewc_penalty": cons_config.use_ewc_penalty,
            "use_promotion_gate": cons_config.use_promotion_gate,
            "promotion_results": gate_results,
        }
    )
    metrics.total_flops = cycle.total_flops
    metrics.t1_store_size = len(cycle.t1_store)
    harness.save_metrics_jsonl(metrics, output_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra Phase 1 Continual Learning")
    parser.add_argument("--config", type=str, default="configs/backbone_50m.yaml")
    parser.add_argument("--task-stream-config", type=str, default="configs/task_stream.yaml")
    parser.add_argument("--output", type=str, default="experiments/aptivra_results.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict)
    cons_cfg = ConsolidationConfig()
    experiment_cfg = load_experiment_settings(args.task_stream_config)

    output_path = Path(args.output)

    seeds = args.seeds or [experiment_cfg.task_stream.seed]
    for seed in seeds:
        stream = experiment_cfg.make_stream(seed)
        print(f"Executing Aptivra full pipeline with seed={seed}...")
        run_single_aptivra(
            model_cfg,
            cons_cfg,
            stream,
            output_path,
            experiment_cfg.evaluation.max_new_tokens,
        )

    print(f"Aptivra execution completed successfully. Metrics written to {output_path}")


if __name__ == "__main__":
    main()
