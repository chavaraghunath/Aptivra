"""Create the self-contained Hugging Face Transformers layout for Aptivra."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer

_HF_SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "deploy" / "huggingface_hub"

HUB_LAYOUT_FILES = (
    "config.json",
    "model.safetensors",
    "configuration_aptivra.py",
    "modeling_aptivra.py",
    "tokenization_aptivra.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _parameter_count(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_safetensors(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(state_dict, temporary, metadata={"format": "pt"})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def consolidated_state_dict(
    model: BackboneTransformer,
    router: GatingRouter,
    experts: list[ExpertModule] | nn.ModuleList,
) -> dict[str, torch.Tensor]:
    """Builds a CPU state dict matching both package and remote-code model names."""
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        state[f"backbone.{name}"] = tensor.detach().cpu().contiguous()
    for name, tensor in router.state_dict().items():
        state[f"router.{name}"] = tensor.detach().cpu().contiguous()
    for index, expert in enumerate(experts):
        for name, tensor in expert.state_dict().items():
            state[f"experts.{index}.{name}"] = tensor.detach().cpu().contiguous()
    return state


def stage_huggingface_hub_artifacts(
    export_dir: Path | str,
    model_config: ModelConfig,
    num_experts: int,
    *,
    parameter_count: int | None = None,
    use_moe_at_inference: bool = False,
    top_k_experts: int = 2,
    d_expert: int = 128,
) -> None:
    """Writes config, tokenizer metadata, and self-contained remote-code modules."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    for source_name in (
        "configuration_aptivra.py",
        "modeling_aptivra.py",
        "tokenization_aptivra.py",
    ):
        shutil.copy2(_HF_SOURCE_DIR / source_name, export_path / source_name)

    config_payload: dict[str, object] = {
        "model_type": "aptivra",
        "architectures": ["AptivraForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_aptivra.AptivraConfig",
            "AutoModelForCausalLM": "modeling_aptivra.AptivraForCausalLM",
            "AutoTokenizer": ["tokenization_aptivra.AptivraByteTokenizer", None],
        },
        "vocab_size": model_config.vocab_size,
        "d_model": model_config.d_model,
        "n_layers": model_config.n_layers,
        "n_heads": model_config.n_heads,
        "d_ff": model_config.d_ff,
        "max_seq_len": model_config.max_seq_len,
        "max_position_embeddings": model_config.max_seq_len,
        "num_experts": num_experts,
        "top_k_experts": top_k_experts,
        "d_expert": d_expert,
        "dropout": model_config.dropout,
        "use_moe_at_inference": use_moe_at_inference,
        "execution_platform": "pytorch",
        "torch_dtype": "float32",
        "tie_word_embeddings": True,
        "use_cache": False,
        "is_decoder": True,
        "pad_token_id": 0,
        "eos_token_id": [0, 10],
    }
    if parameter_count is not None:
        config_payload["parameter_count"] = parameter_count
    _atomic_write_json(export_path / "config.json", config_payload)

    tokenizer_payload: dict[str, object] = {
        "tokenizer_class": "AptivraByteTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_aptivra.AptivraByteTokenizer", None]},
        "model_max_length": model_config.max_seq_len,
        "padding_side": "right",
        "truncation_side": "left",
        "clean_up_tokenization_spaces": False,
        "pad_token": "\u0000",
        "eos_token": "\u0000",
        "padding_eos_policy": (
            "Byte 0 is both padding and EOS; attention_mask distinguishes padding. "
            "Byte 10 is also a generation stop ID. No tokens are added to the 256-byte vocabulary."
        ),
    }
    _atomic_write_json(export_path / "tokenizer_config.json", tokenizer_payload)
    _atomic_write_json(
        export_path / "special_tokens_map.json",
        {"pad_token": "\u0000", "eos_token": "\u0000"},
    )


def export_huggingface_hub_layout(
    export_dir: Path | str,
    model: BackboneTransformer,
    router: GatingRouter,
    experts: list[ExpertModule] | nn.ModuleList,
    model_config: ModelConfig,
    *,
    use_moe_at_inference: bool = False,
) -> Path:
    """Exports one safetensors checkpoint plus the complete stock-Transformers layout."""
    export_path = Path(export_dir)
    if any(not isinstance(expert, ExpertModule) for expert in experts):
        raise TypeError("Hub export requires ExpertModule instances")
    expert_widths = {
        expert.d_expert for expert in experts if isinstance(expert, ExpertModule)
    }
    if len(expert_widths) > 1:
        raise ValueError("All experts must use the same d_expert for Hub export")
    d_expert = next(iter(expert_widths), 128)
    modules: list[nn.Module] = [model, router, *list(experts)]
    stage_huggingface_hub_artifacts(
        export_path,
        model_config,
        len(experts),
        parameter_count=_parameter_count(modules),
        use_moe_at_inference=use_moe_at_inference,
        top_k_experts=router.top_k,
        d_expert=d_expert,
    )
    weights_path = export_path / "model.safetensors"
    _atomic_save_safetensors(weights_path, consolidated_state_dict(model, router, experts))
    return weights_path
