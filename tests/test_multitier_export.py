"""Comprehensive unit and integration tests for Tiered Memory Runtime (TMR) exporter & loader."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.multitier import MultitierExporter, MultitierLoader
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def test_multitier_export_and_loader_roundtrip(tmp_path: Path) -> None:
    """Verifies roundtrip export and restoration of model, router, and expert modules."""
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    orig_model = BackboneTransformer(cfg)
    orig_router = GatingRouter(d_model=32, initial_num_experts=2)
    orig_experts = nn.ModuleList(
        [
            ExpertModule(expert_id=0, d_model=32),
            ExpertModule(expert_id=1, d_model=32),
        ]
    )

    export_dir = tmp_path / "tmr_export"
    exporter = MultitierExporter(export_dir)
    manifest_path = exporter.export_model(orig_model, orig_router, orig_experts, cfg)

    assert manifest_path.exists()
    assert (export_dir / "dense" / "backbone.pt").exists()
    assert (export_dir / "dense" / "router.pt").exists()
    assert (export_dir / "experts" / "expert_0.pt").exists()
    assert (export_dir / "experts" / "expert_1.pt").exists()
    assert (export_dir / ".expert_usage").exists()

    # Restore via loader
    loader = MultitierLoader(export_dir)
    rest_model, rest_router, rest_experts = loader.load_model(cfg)

    # Verify numerical equivalence of restored model output
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 8))
    orig_model.eval()
    rest_model.eval()
    with torch.no_grad():
        orig_logits, _ = orig_model(dummy_input)
        rest_logits, _ = rest_model(dummy_input)

    assert torch.allclose(orig_logits, rest_logits, atol=1e-5)
    assert len(rest_experts) == 2


def test_expert_routing_heat_tracking(tmp_path: Path) -> None:
    """Verifies update of .expert_usage heat log for dynamic auto-pinning."""
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=32, initial_num_experts=2)
    experts = nn.ModuleList(
        [
            ExpertModule(expert_id=0, d_model=32),
            ExpertModule(expert_id=1, d_model=32),
        ]
    )

    export_dir = tmp_path / "heat_test"
    exporter = MultitierExporter(export_dir)
    exporter.export_model(model, router, experts, cfg)

    loader = MultitierLoader(export_dir)
    usage = loader.record_expert_routing_heat(0)
    assert usage["expert_0"] == 1

    usage = loader.record_expert_routing_heat(0)
    assert usage["expert_0"] == 2

    usage = loader.record_expert_routing_heat(1)
    assert usage["expert_1"] == 1


def test_loader_missing_manifest_unhappy_path(tmp_path: Path) -> None:
    """Verifies loader raises FileNotFoundError when manifest is missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    loader = MultitierLoader(empty_dir)

    with pytest.raises(FileNotFoundError, match="Execution manifest not found"):
        loader.load_manifest()


def test_loader_missing_shard_unhappy_path(tmp_path: Path) -> None:
    """Verifies loader raises FileNotFoundError when an expert shard is deleted."""
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=32, initial_num_experts=2)
    experts = nn.ModuleList(
        [
            ExpertModule(expert_id=0, d_model=32),
        ]
    )

    export_dir = tmp_path / "corrupt_export"
    exporter = MultitierExporter(export_dir)
    exporter.export_model(model, router, experts, cfg)

    # Delete the expert shard to simulate corruption
    shard = export_dir / "experts" / "expert_0.pt"
    shard.unlink()

    loader = MultitierLoader(export_dir)
    with pytest.raises(FileNotFoundError, match="Expert shard missing"):
        loader.load_manifest()


def test_simulate_lookahead_prefetch(tmp_path: Path) -> None:
    """Verifies simulation of 1-layer lookahead prefetching (PILOT=1) hit metrics."""
    loader = MultitierLoader(tmp_path)
    routed = torch.tensor([[0, 1], [0, 2]])
    predicted = [0, 1, 3]

    res = loader.simulate_lookahead_prefetch(routed, predicted)
    assert res["hits"] == 2.0
    assert res["total_requested"] == 3.0
    assert 0.6 <= res["hit_rate"] <= 0.7
