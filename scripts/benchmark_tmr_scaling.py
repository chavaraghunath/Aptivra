"""Milestone 4.4: Tiered Memory Runtime (TMR) High-Memory Scaling Benchmark."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.multitier import MultitierExporter, MultitierLoader
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def run_tmr_scaling_benchmark(
    num_experts: int = 8,
    d_model: int = 1024,
    d_ff: int = 4096,
    scratch_dir: str = "scratch/tmr_bench",
) -> dict[str, Any]:
    """Benchmarks TMR async NVMe/RAM export throughput and model reloading."""
    export_path = Path(scratch_dir)
    if export_path.exists():
        shutil.rmtree(export_path)

    config = ModelConfig(
        vocab_size=256,
        d_model=d_model,
        n_layers=4,
        n_heads=8,
        d_ff=d_ff,
        device="cpu",
    )

    print(f"Initializing TMR benchmark ({num_experts} experts, d_model={d_model}, d_ff={d_ff})...")
    model = BackboneTransformer(config)
    router = GatingRouter(d_model=d_model, initial_num_experts=num_experts)
    experts = [ExpertModule(expert_id=i, d_model=d_model) for i in range(num_experts)]

    exporter = MultitierExporter(export_dir=export_path)
    loader = MultitierLoader(export_dir=export_path)

    # 1. Measure Shard Export Throughput
    start_export = time.perf_counter()
    exporter.export_model(model=model, router=router, experts=experts, model_config=config)
    export_elapsed = time.perf_counter() - start_export

    total_bytes = sum(f.stat().st_size for f in export_path.glob("**/*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    export_mb_s = total_mb / max(export_elapsed, 1e-6)

    # 2. Measure Multitier Shard Load Performance
    start_load = time.perf_counter()
    _restored_model, _restored_router, _restored_experts = loader.load_model(config)
    load_elapsed = time.perf_counter() - start_load
    load_mb_s = total_mb / max(load_elapsed, 1e-6)

    # 3. Measure Routing Heat Logging Latency
    start_heat = time.perf_counter()
    for i in range(50):
        loader.record_expert_routing_heat(i % num_experts)
    heat_elapsed = time.perf_counter() - start_heat

    results = {
        "num_experts": num_experts,
        "d_model": d_model,
        "d_ff": d_ff,
        "total_shard_mb": round(total_mb, 2),
        "export_elapsed_sec": round(export_elapsed, 4),
        "export_throughput_mb_s": round(export_mb_s, 2),
        "load_elapsed_sec": round(load_elapsed, 4),
        "load_throughput_mb_s": round(load_mb_s, 2),
        "heat_logging_elapsed_sec": round(heat_elapsed, 4),
    }

    if export_path.exists():
        shutil.rmtree(export_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TMR 1B+ Parameter High-Memory Scaling Benchmark")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-ff", type=int, default=4096)
    parser.add_argument("--output", type=str, default="experiments/tmr_benchmark_results.json")
    args = parser.parse_args()

    results = run_tmr_scaling_benchmark(
        num_experts=args.experts,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )

    print("\n--- TMR High-Memory Scaling Benchmark Results ---")
    print(json.dumps(results, indent=2))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
