"""Build ModelConfig from on-disk Hub export (config.json + manifest)."""

import json
from pathlib import Path

from aptivra.export.multitier import MultitierLoader
from aptivra.model.config import ModelConfig


def model_config_from_export_dir(
    export_dir: Path | str,
    device: str = "auto",
) -> ModelConfig:
    """Reads `config.json` when present, otherwise multitier manifest metadata."""
    export_path = Path(export_dir)
    config_json = export_path / "config.json"
    if config_json.is_file():
        data = json.loads(config_json.read_text(encoding="utf-8"))
        return ModelConfig(
            vocab_size=int(data.get("vocab_size", 256)),
            d_model=int(data["d_model"]),
            n_layers=int(data["n_layers"]),
            n_heads=int(data["n_heads"]),
            d_ff=int(data["d_ff"]),
            max_seq_len=int(data.get("max_seq_len", 512)),
            dropout=float(data.get("dropout", 0.1)),
            device=device,
        )

    manifest = MultitierLoader(export_path).load_manifest()
    return ModelConfig(
        vocab_size=int(manifest.metadata.get("vocab_size", 256)),
        d_model=manifest.d_model,
        n_layers=manifest.n_layers,
        n_heads=16,
        d_ff=4096,
        max_seq_len=int(manifest.metadata.get("max_seq_len", 512)),
        device=device,
    )
