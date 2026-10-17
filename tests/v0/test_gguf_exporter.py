"""Unit tests for GGUF Exporter."""

from pathlib import Path
from typing import Literal, cast

import pytest

from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter
from aptivra.v0.export.gguf_exporter import export_model_to_gguf


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


def test_q8_0_is_rejected_without_creating_a_false_quantized_file(tmp_path: Path) -> None:
    config = ModelConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, device="cpu")
    model = BackboneTransformer(config)
    router = GatingRouter(d_model=config.d_model, initial_num_experts=1)
    experts = [ExpertModule(expert_id=0, d_model=config.d_model)]
    out_file = tmp_path / "false-q8.gguf"

    with pytest.raises(ValueError, match="Unsupported GGUF quantization type"):
        export_model_to_gguf(
            model=model,
            router=router,
            experts=experts,
            model_config=config,
            output_path=out_file,
            quant_type=cast(Literal["fp32", "fp16"], "q8_0"),
        )

    assert not out_file.exists()
