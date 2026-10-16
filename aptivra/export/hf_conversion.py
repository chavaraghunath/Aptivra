"""Transactional conversion from legacy Aptivra shards to the Hub layout."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from aptivra.eval.model_card_eval import sha256_file
from aptivra.export.hf_hub_artifacts import (
    HUB_LAYOUT_FILES,
    export_huggingface_hub_layout,
    write_export_provenance,
)
from aptivra.export.multitier import MultitierLoader
from aptivra.model.config import ModelConfig
from aptivra.v0.inference.causal_lm import AptivraCausalLM
from aptivra.v0.inference.hub_config import model_config_from_export_dir


def _checkpoint_config(
    source_dir: Path,
    model_config: ModelConfig | None,
) -> ModelConfig:
    loader = MultitierLoader(source_dir)
    manifest = loader.load_manifest()
    backbone_path = source_dir / manifest.dense_shard_path
    backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    position_key = "pos_embedding.weight"
    if position_key not in backbone_state:
        raise RuntimeError(f"Legacy backbone is missing required tensor: {position_key}")
    position_weight = backbone_state[position_key]
    if not isinstance(position_weight, torch.Tensor) or position_weight.ndim != 2:
        raise RuntimeError(f"Legacy tensor has invalid shape: {position_key}")
    checkpoint_context = int(position_weight.shape[0])
    config = model_config or model_config_from_export_dir(source_dir, device="cpu")
    if config.max_seq_len != checkpoint_context:
        config = replace(config, max_seq_len=checkpoint_context)
    return replace(config, device="cpu")


def _load_legacy_model(source_dir: Path, config: ModelConfig) -> AptivraCausalLM:
    backbone, router, experts = MultitierLoader(source_dir).load_model(config)
    model = AptivraCausalLM(config, backbone, router, experts)
    for name, tensor in model.state_dict().items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Legacy checkpoint contains non-finite tensor: {name}")
    return model.eval()


def _verify_logits_parity(
    legacy: AptivraCausalLM,
    converted: AptivraCausalLM,
) -> None:
    token_count = min(8, legacy.config.max_seq_len)
    if token_count < 1:
        raise RuntimeError("Checkpoint context length must be at least one token")
    fixed_input = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
    for use_moe in (False, True):
        legacy.use_moe_at_inference = use_moe
        converted.use_moe_at_inference = use_moe
        with torch.no_grad():
            expected = legacy(fixed_input)
            actual = converted(fixed_input)
        try:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        except AssertionError as exc:
            mode = "enabled" if use_moe else "disabled"
            raise RuntimeError(f"Converted logits mismatch with MoE {mode}") from exc


def _install_verified_layout(staging_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "model.safetensors"
    for filename in HUB_LAYOUT_FILES:
        if filename == weights_name:
            continue
        source = staging_dir / filename
        temporary = output_dir / f".{filename}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, output_dir / filename)

    staged_weights = staging_dir / weights_name
    temporary_weights = output_dir / f".{weights_name}.tmp"
    try:
        shutil.copy2(staged_weights, temporary_weights)
        os.replace(temporary_weights, output_dir / weights_name)
    finally:
        temporary_weights.unlink(missing_ok=True)
    return output_dir / weights_name


def convert_legacy_export_to_huggingface(
    source_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    model_config: ModelConfig | None = None,
    source_hub_revision: str | None = None,
) -> Path:
    """Converts legacy shards only after strict two-mode logits parity succeeds."""
    source_path = Path(source_dir)
    destination = Path(output_dir) if output_dir is not None else source_path
    config = _checkpoint_config(source_path, model_config)
    loader = MultitierLoader(source_path)
    manifest = loader.load_manifest()
    legacy = _load_legacy_model(source_path, config)
    source_paths = [
        Path("multitier_manifest.json"),
        Path(manifest.dense_shard_path),
        Path("dense/router.pt"),
        *(Path(path) for path in manifest.expert_shard_paths),
    ]
    source_hashes = {
        path.as_posix(): sha256_file(source_path / path) for path in source_paths
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="aptivra-hf-conversion-",
        dir=destination.parent,
    ) as tmp:
        staging = Path(tmp)
        export_huggingface_hub_layout(
            staging,
            legacy.backbone,
            legacy.router,
            legacy.experts,
            config,
            use_moe_at_inference=False,
        )
        write_export_provenance(
            staging,
            source_layout="legacy_multitier",
            source_manifest_version=manifest.version,
            source_shards_sha256=source_hashes,
            source_hub_revision=source_hub_revision,
        )
        converted = AptivraCausalLM.from_export_dir(staging, model_config=config).eval()
        _verify_logits_parity(legacy, converted)
        return _install_verified_layout(staging, destination)
