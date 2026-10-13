"""Script executing Aptivra continual learning runner across seeds and logging metrics."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aptivra.consolidation.cycle import ConsolidationConfig, ConsolidationCycle
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.gate.promotion_gate import PromotionGate
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.stream import ContinualTaskStream


def verifier_eval_wrapper(
    predict_fn: Callable[[str], str],
    instances: list[TaskInstance],
    stream: ContinualTaskStream,
) -> float:
    """Helper calculating overall accuracy of predict_fn across task instances."""
    if not instances:
        return 1.0

    family_map = {f.family_id: f for f in stream.families}
    passed = 0
    for inst in instances:
        out = predict_fn(inst.prompt)
        fam = family_map[inst.task_family]
        res = fam.verifier.verify(inst, out)
        if res.passed:
            passed += 1
    return float(passed / len(instances))


def loss_eval_wrapper(
    predict_fn: Callable[[str], str],
    instances: list[TaskInstance],
    cycle: ConsolidationCycle,
) -> np.ndarray:
    """Helper calculating token cross-entropy loss vector over task instances."""
    losses: list[float] = []
    for inst in instances:
        out = predict_fn(inst.prompt)
        s_toks = cycle.encode_text(out + "\n")
        tgt_toks = cycle.encode_text(inst.target_solution + "\n")
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
) -> None:
    """Runs a single Aptivra continual learning cycle across the task stream."""
    cycle = ConsolidationCycle(model_config, cons_config)
    gate = PromotionGate(alpha=0.05)
    eval_fn_at_cycle: list[Callable[[str], str]] = []
    held_out_regression_suite: list[TaskInstance] = []

    for task_idx, family in enumerate(stream.families):
        split = stream.get_split(family.family_id)

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
                    )

        # 2. Run Consolidation Cycle on current task
        cycle.run_cycle(task_idx, family.family_id, split.train)

        # 3. Evaluate Promotion Gate if enabled
        curr_predict_fn: Callable[[str], str] = (
            eval_fn_at_cycle[-1] if eval_fn_at_cycle else cycle.predict
        )
        cand_predict_fn: Callable[[str], str] = cycle.predict

        if cons_config.use_promotion_gate and task_idx > 0:
            gate_res = gate.evaluate_candidate(
                curr_predict_fn=curr_predict_fn,
                cand_predict_fn=cand_predict_fn,
                new_task_val_instances=split.val,
                held_out_regression_suite=held_out_regression_suite,
                verifier_eval_fn=lambda pfn, insts: verifier_eval_wrapper(pfn, insts, stream),
                loss_eval_fn=lambda pfn, insts: loss_eval_wrapper(pfn, insts, cycle),
            )
            if gate_res.promoted:
                held_out_regression_suite.extend(split.val)
                active_pred = cand_predict_fn
            else:
                active_pred = curr_predict_fn
        else:
            held_out_regression_suite.extend(split.val)
            active_pred = cand_predict_fn

        eval_fn_at_cycle.append(active_pred)

    # 4. Evaluate harness metrics
    harness = ContinualEvalHarness(stream)
    metrics = harness.evaluate_matrix(eval_fn_at_cycle)
    metrics.total_flops = cycle.total_flops
    metrics.t1_store_size = len(cycle.t1_store)
    metrics.metadata = {
        "model_type": "aptivra",
        "seed": stream.seed,
        "d_model": model_config.d_model,
        "n_layers": model_config.n_layers,
        "use_t1_memory": cons_config.use_t1_memory,
        "use_expert_growth": cons_config.use_expert_growth,
        "use_replay_buffer": cons_config.use_replay_buffer,
        "use_ewc_penalty": cons_config.use_ewc_penalty,
        "use_promotion_gate": cons_config.use_promotion_gate,
    }
    harness.save_metrics_jsonl(metrics, output_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aptivra Phase 1 Continual Learning")
    parser.add_argument("--config", type=str, default="configs/backbone_50m.yaml")
    parser.add_argument("--output", type=str, default="experiments/aptivra_results.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_dict: dict[str, Any] = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict)
    cons_cfg = ConsolidationConfig()

    output_path = Path(args.output)

    for seed in args.seeds:
        stream = ContinualTaskStream(
            num_train_per_task=100,
            num_val_per_task=20,
            num_test_per_task=20,
            seed=seed,
        )
        print(f"Executing Aptivra full pipeline with seed={seed}...")
        run_single_aptivra(model_cfg, cons_cfg, stream, output_path)

    print(f"Aptivra execution completed successfully. Metrics written to {output_path}")


if __name__ == "__main__":
    main()
