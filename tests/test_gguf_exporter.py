"""Unit tests for GGUF Exporter."""

from pathlib import Path

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.gguf_exporter import export_model_to_gguf
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def test_export_model_to_gguf(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        device="cpu",
    )
    model = BackboneTransformer(config)
    router = GatingRouter(d_model=config.d_model, initial_num_experts=2)
    experts = [ExpertModule(expert_id=i, d_model=config.d_model) for i in range(2)]

    out_file = tmp_path / "aptivra_test.gguf"
    result_path = export_model_to_gguf(
        model=model,
        router=router,
        experts=experts,
        model_config=config,
        output_path=out_file,
        quant_type="fp16",
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0
