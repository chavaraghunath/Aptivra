"""Stage Hugging Face Hub files (config, auto_map, custom modeling) for upload."""

import json
import shutil
from pathlib import Path

from aptivra.model.config import ModelConfig

_HF_SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "deploy" / "huggingface_hub"


def stage_huggingface_hub_artifacts(
    export_dir: Path | str,
    model_config: ModelConfig,
    num_experts: int,
) -> None:
    """Writes config.json and copies Transformers auto_map Python modules into export_dir."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    for src in _HF_SOURCE_DIR.glob("*.py"):
        shutil.copy2(src, export_path / src.name)

    config_payload = {
        "model_type": "aptivra",
        "architectures": ["AptivraForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_aptivra.AptivraConfig",
            "AutoModelForCausalLM": "modeling_aptivra.AptivraForCausalLM",
        },
        "vocab_size": model_config.vocab_size,
        "d_model": model_config.d_model,
        "n_layers": model_config.n_layers,
        "n_heads": model_config.n_heads,
        "d_ff": model_config.d_ff,
        "max_seq_len": model_config.max_seq_len,
        "num_experts": num_experts,
        "top_k_experts": 2,
        "dropout": model_config.dropout,
        "execution_platform": "pytorch",
        "torch_dtype": "float32",
        "transformers_version": "4.40.0",
    }
    config_path = export_path / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
