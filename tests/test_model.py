"""Unit tests for backbone transformer and attention interface."""

import torch

from aptivra.model.attention import StandardCausalAttention
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def test_model_config_device() -> None:
    config = ModelConfig(device="cpu")
    dev = config.get_device()
    assert dev.type == "cpu"
    assert config.param_count_estimate > 0


def test_standard_causal_attention_kv_cache() -> None:
    attn = StandardCausalAttention(d_model=64, n_heads=4, dropout=0.0)
    attn.eval()

    x1 = torch.randn(2, 5, 64)  # batch_size=2, seq_len=5
    out1, cache1 = attn(x1)
    assert out1.shape == (2, 5, 64)
    assert cache1 is not None
    assert cache1[0].shape == (2, 4, 5, 16)

    # Next step generation (seq_len=1) with past cache
    x2 = torch.randn(2, 1, 64)
    out2, cache2 = attn(x2, kv_cache=cache1)
    assert out2.shape == (2, 1, 64)
    assert cache2 is not None
    assert cache2[0].shape == (2, 4, 6, 16)


def test_backbone_transformer_forward() -> None:
    config = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, d_ff=128, device="cpu")
    model = BackboneTransformer(config)
    model.eval()

    input_ids = torch.randint(0, 128, (2, 10))
    logits, caches = model(input_ids)
    assert logits.shape == (2, 10, 128)
    assert len(caches) == 2

    # Step generation
    next_input = torch.randint(0, 128, (2, 1))
    next_logits, new_caches = model(next_input, past_kv_caches=caches)
    assert next_logits.shape == (2, 1, 128)
    assert new_caches[0][0].shape[2] == 11
