"""MLX inference path for Aptivra shards on Apple Silicon (option I)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aptivra.export.mlx_export import export_mlx_npz
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.model.config import ModelConfig


def try_mlx_matmul(name: str, path: Path) -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        print("MLX not installed. Use: pip install -e \".[mlx]\"")
        return False

    data = mx.load(str(path))
    keys = list(data.keys())
    print(f"MLX loaded {name}: {len(keys)} tensors (sample keys: {keys[:3]})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Aptivra MLX export / smoke inference")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--mlx-dir", type=Path, default=Path("scratch/mlx_export"))
    parser.add_argument("--prompt", type=str, default="User: ping\nAssistant:")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    export_dir = args.export_dir or Path(cfg_yaml["local_export_dir"])
    mlx_dir = args.mlx_dir

    manifest_path = export_mlx_npz(export_dir, mlx_dir)
    print(f"Wrote MLX NPZ manifest: {manifest_path}")

    mlx_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = try_mlx_matmul("backbone", mlx_dir / mlx_manifest["backbone"])
    if ok:
        try_mlx_matmul("router", mlx_dir / mlx_manifest["router"])

    # Reference PyTorch generation for parity until full MLX graph is wired
    model_config = ModelConfig(
        vocab_size=256,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        device="auto",
    )
    lm = AptivraCausalLM.from_export_dir(export_dir, model_config=model_config)
    text = lm.generate(args.prompt, max_new_tokens=int(cfg_yaml["max_new_tokens"]))
    print(f"PyTorch (MPS/CPU) reference generation:\n{text!r}")


if __name__ == "__main__":
    main()
