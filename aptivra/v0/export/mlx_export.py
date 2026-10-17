"""Export multitier PyTorch shards to NPZ bundles for MLX experiments (option I)."""

import json
from pathlib import Path

import numpy as np
import torch

from aptivra.model.config import ModelConfig
from aptivra.v0.export.multitier import MultitierLoader


def export_mlx_npz(export_dir: Path | str, output_dir: Path | str) -> Path:
    """Writes one NPZ per shard plus manifest for MLX loaders."""
    src = Path(export_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = MultitierLoader(src)
    manifest = loader.load_manifest()
    cfg = ModelConfig(
        vocab_size=int(manifest.metadata.get("vocab_size", 256)),
        d_model=manifest.d_model,
        n_layers=manifest.n_layers,
        n_heads=16,
        d_ff=4096,
    )
    backbone, router, experts = loader.load_model(cfg)

    def save_module(module: torch.nn.Module, name: str) -> str:
        arrays = {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}
        rel = f"{name}.npz"
        np.savez(out / rel, **arrays)
        return rel

    mlx_manifest = {
        "format": "aptivra_mlx_npz_v1",
        "backbone": save_module(backbone, "backbone"),
        "router": save_module(router, "router"),
        "experts": [save_module(exp, f"expert_{i}") for i, exp in enumerate(experts)],
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
        },
    }
    manifest_path = out / "mlx_manifest.json"
    manifest_path.write_text(json.dumps(mlx_manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
