"""Tests for Aptivra MoE causal LM inference."""

from pathlib import Path

import torch
import torch.nn as nn

from aptivra.export.multitier import MultitierExporter
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter


def _export_tiny(tmp_path: Path) -> Path:
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=64)
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=32, initial_num_experts=2)
    experts = nn.ModuleList(
        [ExpertModule(expert_id=0, d_model=32), ExpertModule(expert_id=1, d_model=32)]
    )
    export_dir = tmp_path / "export"
    MultitierExporter(export_dir).export_model(model, router, experts, cfg)
    return export_dir


def test_causal_lm_forward_and_generate(tmp_path: Path) -> None:
    export_dir = _export_tiny(tmp_path)
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=64, device="cpu")
    lm = AptivraCausalLM.from_export_dir(export_dir, model_config=cfg)
    lm.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits = lm(ids)
    assert logits.shape == (1, 6, cfg.vocab_size)
    out = lm.generate("hi", max_new_tokens=4)
    assert isinstance(out, str)


def test_hf_hub_artifacts_stage(tmp_path: Path) -> None:
    from aptivra.export.hf_hub_artifacts import stage_huggingface_hub_artifacts

    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    stage_huggingface_hub_artifacts(tmp_path, cfg, num_experts=2)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "configuration_aptivra.py").exists()
    assert (tmp_path / "modeling_aptivra.py").exists()
