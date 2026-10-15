#!/usr/bin/env python3
"""Reproduce ADR-0011 parameter and static-memory arithmetic from a cached snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open

REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
CAP_BYTES = 28_991_029_248


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_inventory(path: Path) -> tuple[int, int, dict[str, int], bool]:
    count = 0
    tensor_count = 0
    by_dtype: dict[str, int] = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        names = set(tensors.keys())
        # ``safe_open`` exposes ``keys`` but intentionally is not a Mapping/iterable.
        for name in names:
            tensor = tensors.get_slice(name)
            parameters = math.prod(tensor.get_shape())
            count += parameters
            tensor_count += 1
            dtype = str(tensor.get_dtype())
            by_dtype[dtype] = by_dtype.get(dtype, 0) + parameters
    return count, tensor_count, by_dtype, "lm_head.weight" in names


def _memory(parameters: int) -> dict[str, int]:
    return {
        "bf16_weight_bytes": parameters * 2,
        "fp32_weight_bytes": parameters * 4,
        "full_fp32_adamw_tensor_state_bytes": parameters * 16,
    }


def calculate(snapshot: Path) -> dict[str, Any]:
    """Calculate exact counts without loading model tensors into accelerator memory."""
    if snapshot.name != REVISION:
        raise ValueError(
            f"expected immutable revision directory {REVISION}, got {snapshot.name}"
        )
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    tokenizer_path = snapshot / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))

    expected_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"unexpected config {key}: {config.get(key)!r}")
    if tokenizer_config.get("tokenizer_class") != "GPT2Tokenizer":
        raise ValueError("expected GPT2Tokenizer")
    if tokenizer_config.get("bos_token") != "<|endoftext|>":
        raise ValueError("unexpected tokenizer BOS token")
    if tokenizer_config.get("eos_token") != "<|endoftext|>":
        raise ValueError("unexpected tokenizer EOS token")
    if tokenizer_config.get("unk_token") != "<|endoftext|>":
        raise ValueError("unexpected tokenizer UNK token")
    if tokenizer_config.get("pad_token") is not None:
        raise ValueError("expected no tokenizer pad token")
    if tokenizer_config.get("chat_template") is not None:
        raise ValueError("expected no tokenizer chat template")
    if (snapshot / "chat_template.jinja").exists():
        raise ValueError("expected no standalone tokenizer chat template")
    required_added_tokens = {
        "0": "<|endoftext|>",
        "1": "<|im_start|>",
        "2": "<|im_end|>",
    }
    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    for token_id, expected_content in required_added_tokens.items():
        if added_tokens.get(token_id, {}).get("content") != expected_content:
            raise ValueError(f"unexpected added token {token_id}")

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    vocab = int(config["vocab_size"])
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim

    embedding = vocab * hidden
    attention_per_layer = (
        hidden * hidden
        + hidden * kv_hidden
        + hidden * kv_hidden
        + hidden * hidden
    )
    mlp_per_layer = 3 * hidden * intermediate
    norms_per_layer = 2 * hidden
    formula_total = (
        embedding
        + layers * (attention_per_layer + mlp_per_layer + norms_per_layer)
        + hidden
    )
    actual_total, tensor_count, by_dtype, lm_head_present = _tensor_inventory(weights_path)
    if formula_total != actual_total:
        raise ValueError(
            f"config formula gives {formula_total}, safetensors contains {actual_total}"
        )
    if lm_head_present:
        raise ValueError("tied model unexpectedly contains a separate lm_head.weight")

    lora_per_rank_attention = layers * (
        (hidden + hidden)
        + (hidden + kv_hidden)
        + (hidden + kv_hidden)
        + (hidden + hidden)
    )
    lora_per_rank_mlp = layers * 3 * (hidden + intermediate)
    lora_per_rank_all = lora_per_rank_attention + lora_per_rank_mlp

    lora: dict[str, Any] = {
        "formula": "rank * sum(in_features + out_features) over q/k/v/o/gate/up/down",
        "parameters_per_rank_attention": lora_per_rank_attention,
        "parameters_per_rank_mlp": lora_per_rank_mlp,
        "parameters_per_rank_all_seven_projections": lora_per_rank_all,
        "ranks": {},
    }
    for rank in (8, 16, 32):
        parameters = lora_per_rank_all * rank
        lora["ranks"][str(rank)] = {
            "parameters_per_expert": parameters,
            "fraction_of_base": parameters / actual_total,
            **_memory(parameters),
        }

    parallel: dict[str, Any] = {
        "formula": (
            "layers * 3 * hidden_size * bottleneck for projection weights only; "
            "excludes router, branch norms, and biases"
        ),
        "bottlenecks": {},
    }
    for bottleneck in (64, 128, 256, 640, 2560):
        parameters = layers * 3 * hidden * bottleneck
        parallel["bottlenecks"][str(bottleneck)] = {
            "parameters_per_expert": parameters,
            "fraction_of_base": parameters / actual_total,
            **_memory(parameters),
        }

    router_per_expert = layers * (hidden + 1)
    upcycled: dict[str, Any] = {
        "full_ffn_copy_parameters": layers * mlp_per_layer,
        "per_layer_router_parameters_per_expert": router_per_expert,
        "formula": (
            "base + (expert_count - 1) * full_ffn_copy + "
            "expert_count * per_layer_router"
        ),
        "expert_counts": {},
    }
    for expert_count in (2, 4, 8, 12):
        parameters = (
            actual_total
            + (expert_count - 1) * layers * mlp_per_layer
            + expert_count * router_per_expert
        )
        upcycled["expert_counts"][str(expert_count)] = {
            "total_parameters": parameters,
            **_memory(parameters),
            "full_adamw_state_exceeds_cap": parameters * 16 > CAP_BYTES,
        }

    rank16 = lora_per_rank_all * 16
    resident_experts = 12
    lora_rank16_twelve_train_one = (
        actual_total * 4
        + resident_experts * rank16 * 4
        + rank16 * 12
    )
    replay_cached_bytes = 500 * 8192 * 2 * 4

    tokenizer_asset_names = (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    tokenizer_assets = {
        name: {
            "bytes": (snapshot / name).stat().st_size,
            "sha256": _sha256(snapshot / name),
        }
        for name in tokenizer_asset_names
    }

    return {
        "schema_version": "adr-0011-smollm2-arithmetic-v1",
        "source": {
            "repository": "HuggingFaceTB/SmolLM2-360M",
            "revision": REVISION,
            "config_sha256": _sha256(config_path),
            "tokenizer_assets": tokenizer_assets,
            "weights_sha256": _sha256(weights_path),
            "weights_file_bytes": weights_path.stat().st_size,
            "tensor_count": tensor_count,
            "parameters_by_dtype": by_dtype,
            "lm_head_weight_present": lm_head_present,
        },
        "cap_bytes": CAP_BYTES,
        "config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "kv_hidden_size": kv_hidden,
            "vocab_size": vocab,
            "max_position_embeddings": int(config["max_position_embeddings"]),
            "architectures": config["architectures"],
            "model_type": config["model_type"],
            "tie_word_embeddings": config["tie_word_embeddings"],
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
        },
        "tokenizer": {
            "tokenizer_class": tokenizer_config["tokenizer_class"],
            "bos_token": tokenizer_config["bos_token"],
            "eos_token": tokenizer_config["eos_token"],
            "unk_token": tokenizer_config["unk_token"],
            "pad_token": tokenizer_config.get("pad_token"),
            "chat_template": tokenizer_config.get("chat_template"),
            "chat_template_file_present": (snapshot / "chat_template.jinja").exists(),
            "required_added_tokens": required_added_tokens,
        },
        "base": {
            "embedding_parameters": embedding,
            "attention_parameters_per_layer": attention_per_layer,
            "mlp_parameters_per_layer": mlp_per_layer,
            "norm_parameters_per_layer": norms_per_layer,
            "final_norm_parameters": hidden,
            "total_parameters": actual_total,
            **_memory(actual_total),
        },
        "lora": lora,
        "parallel_swiglu_adapter": parallel,
        "upcycled_ffn": upcycled,
        "selected_static_scenario": {
            "description": (
                "fp32 frozen base, 12 resident rank-16 LoRA experts, one expert with "
                "fp32 gradients and two AdamW moments; excludes activations and runtime"
            ),
            "bytes": lora_rank16_twelve_train_one,
            "headroom_to_cap_bytes": CAP_BYTES - lora_rank16_twelve_train_one,
        },
        "ewc": {
            "base_fisher_plus_anchor_fp32_bytes": actual_total * 8,
            "rank16_lora_fisher_plus_anchor_fp32_bytes": rank16 * 8,
        },
        "replay": {
            "example_cached_upper_bound": (
                "500 examples * 8192 positions * (input_ids + labels) * 4-byte int32"
            ),
            "cached_upper_bound_bytes": replay_cached_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "snapshot",
        type=Path,
        help="cached immutable SmolLM2-360M snapshot directory",
    )
    args = parser.parse_args()
    print(json.dumps(calculate(args.snapshot.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
