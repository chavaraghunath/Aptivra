"""Unit tests for PretrainedEmbeddingLayer ($T_0$)."""

import pytest
import torch

from aptivra.model.config import ModelConfig
from aptivra.model.embeddings import PretrainedEmbeddingLayer
from aptivra.model.transformer import BackboneTransformer


def test_pretrained_embedding_layer_shape() -> None:
    vocab_size = 256
    d_embedding = 128
    d_model = 512
    batch_size = 2
    seq_len = 16

    layer = PretrainedEmbeddingLayer(
        vocab_size=vocab_size,
        d_embedding=d_embedding,
        d_model=d_model,
    )
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = layer(input_ids)

    assert output.shape == (batch_size, seq_len, d_model)


def test_pretrained_embedding_layer_copy_weights() -> None:
    vocab_size = 100
    d_embedding = 64
    d_model = 64

    custom_weights = torch.randn(vocab_size, d_embedding)
    layer = PretrainedEmbeddingLayer(
        vocab_size=vocab_size,
        d_embedding=d_embedding,
        d_model=d_model,
        pretrained_weights=custom_weights,
        freeze_embedding=True,
    )

    assert torch.allclose(layer.embedding.weight, custom_weights)
    assert not layer.embedding.weight.requires_grad


def test_pretrained_embedding_layer_invalid_weights_shape() -> None:
    vocab_size = 100
    d_embedding = 64
    d_model = 128
    invalid_weights = torch.randn(50, 32)

    with pytest.raises(ValueError, match="Expected pretrained_weights shape"):
        PretrainedEmbeddingLayer(
            vocab_size=vocab_size,
            d_embedding=d_embedding,
            d_model=d_model,
            pretrained_weights=invalid_weights,
        )


def test_backbone_transformer_with_pretrained_embeddings() -> None:
    config = ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        use_pretrained_embeddings=True,
        pretrained_embedding_dim=64,
        device="cpu",
    )
    model = BackboneTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits, _ = model(input_ids)

    assert logits.shape == (2, 10, config.vocab_size)
    assert isinstance(model.token_embedding, PretrainedEmbeddingLayer)
