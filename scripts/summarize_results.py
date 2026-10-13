"""Script summarizing experiment results across baselines, Aptivra, and ablations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def compute_mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Computes sample mean and 95% confidence interval half-width."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(np.mean(arr)), 0.0
    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    if sem == 0.0 or np.isnan(sem):
        return mean, 0.0
    ci_half = float(stats.t.ppf((1 + confidence) / 2.0, df=len(arr) - 1) * sem)
    return mean, ci_half


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Loads JSONL records from path."""
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def format_optional_metric(values: list[float | None]) -> str:
    """Format a metric only when every record contains a measured value."""
    if not values or any(value is None for value in values):
        return "unmeasured"
    measured = [value for value in values if value is not None]
    mean, ci = compute_mean_ci(measured)
    return f"{mean:.3f} ± {ci:.3f}"


def format_table(records_by_type: dict[str, list[dict[str, Any]]]) -> str:
    """Formats markdown table summarizing average accuracy, BWT, FWT, and total FLOPs."""
    lines: list[str] = [
        "| Configuration | Avg Acc ($A_T$) | BWT | FWT | Total FLOPs | Seeds |",
        "|---|---|---|---|---|---|",
    ]

    for cfg_name, recs in records_by_type.items():
        if not recs:
            continue
        accs = [r["average_accuracy"] for r in recs]
        bwts = [r["backward_transfer"] for r in recs]
        fwts = [r["forward_transfer"] for r in recs]
        flops = [r["total_flops"] for r in recs]

        m_flop = float(np.mean(flops))

        lines.append(
            f"| **{cfg_name}** | {format_optional_metric(accs)} | "
            f"{format_optional_metric(bwts)} | {format_optional_metric(fwts)} | "
            f"{m_flop:.2e} | {len(recs)} |"
        )

    return "\n".join(lines)


def main() -> None:
    base_records = load_jsonl(Path("experiments/baseline_results.jsonl"))
    apt_records = load_jsonl(Path("experiments/aptivra_results.jsonl"))
    abl_records = load_jsonl(Path("experiments/ablation_results.jsonl"))

    grouped: dict[str, list[dict[str, Any]]] = {}

    for r in base_records:
        b_type = r["metadata"]["baseline_type"]
        grouped.setdefault(f"Baseline: {b_type}", []).append(r)

    for r in apt_records:
        grouped.setdefault("Aptivra (Full)", []).append(r)

    for r in abl_records:
        meta = r["metadata"]
        # Identify active ablation
        if not meta.get("use_t1_memory", True):
            name = "Ablation: no_t1"
        elif not meta.get("use_expert_growth", True):
            name = "Ablation: no_expert_growth"
        elif not meta.get("use_replay_buffer", True):
            name = "Ablation: no_replay"
        elif not meta.get("use_ewc_penalty", True):
            name = "Ablation: no_ewc"
        elif not meta.get("use_promotion_gate", True):
            name = "Ablation: no_gate"
        else:
            name = "Ablation: unknown"
        grouped.setdefault(name, []).append(r)

    summary_md = format_table(grouped)
    print("### Experiment Results Summary\n")
    print(summary_md)


if __name__ == "__main__":
    main()
