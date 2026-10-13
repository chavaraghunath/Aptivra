# ruff: noqa: E501
"""GGUF Binary Exporter for Aptivra Models."""

from pathlib import Path
from typing import Literal

import gguf
import numpy as np
import torch

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def export_model_to_gguf(
    model: BackboneTransformer,
    router: GatingRouter,
    experts: list[ExpertModule],
    model_config: ModelConfig,
    output_path: Path | str,
    quant_type: Literal["fp32", "fp16", "q8_0"] = "fp16",
) -> Path:
    """Exports Aptivra 200M backbone, router, and expert weights to GGUF format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(path=str(out_file), arch="aptivra")

    # 1. Add GGUF General & Aptivra Metadata KV Pairs
    writer.add_name("Aptivra 200M Tiered-Memory")
    writer.add_architecture()
    writer.add_author("raghunath1")
    writer.add_description(
        "Aptivra 200M MoE Continual Learning Model with C-Engine Tiered Memory Runtime"
    )

    ft_map = {"fp32": 0, "fp16": 1, "q8_0": 7}
    writer.add_file_type(ft_map.get(quant_type, 1))

    # Architecture metadata
    writer.add_context_length(2048)
    writer.add_embedding_length(model_config.d_model)
    writer.add_block_count(model_config.n_layers)
    writer.add_head_count(model_config.n_heads)
    writer.add_feed_forward_length(model_config.d_ff)
    writer.add_expert_count(len(experts))
    writer.add_expert_used_count(2)

    # 2. Add Backbone Dense Transformer Tensors
    for name, param in model.state_dict().items():
        arr = param.detach().cpu().to(torch.float32).numpy()
        if quant_type == "fp16":
            arr = arr.astype(np.float16)
        writer.add_tensor(f"backbone.{name}", arr)

    # 3. Add Soft Top-2 Gating Router Tensors
    for name, param in router.state_dict().items():
        arr = param.detach().cpu().to(torch.float32).numpy()
        if quant_type == "fp16":
            arr = arr.astype(np.float16)
        writer.add_tensor(f"router.{name}", arr)

    # 4. Add Expert Shard Tensors
    for expert_idx, expert in enumerate(experts):
        for name, param in expert.state_dict().items():
            arr = param.detach().cpu().to(torch.float32).numpy()
            if quant_type == "fp16":
                arr = arr.astype(np.float16)
            writer.add_tensor(f"experts.{expert_idx}.{name}", arr)

    # 5. Write GGUF Header, KV Metadata, and Tensor Payloads
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully exported GGUF model [{quant_type.upper()}] to: {out_file}")
    return out_file
