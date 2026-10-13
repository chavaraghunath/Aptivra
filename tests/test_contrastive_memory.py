"""Unit tests for ContrastiveMemoryEncoder and InfoNCE memory index optimization."""

import torch

from aptivra.memory.retrieval import ContrastiveMemoryEncoder, VectorEmbeddingIndex


def test_contrastive_encoder_output_shape_and_norm() -> None:
    """Verifies ContrastiveMemoryEncoder produces unit-normalized 128-dim vectors."""
    encoder = ContrastiveMemoryEncoder(in_features=256, embedding_dim=128)
    dummy_input = torch.randn(4, 256)
    output = encoder(dummy_input)

    assert output.shape == (4, 128)
    norms = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_contrastive_training_step() -> None:
    """Verifies contrastive training step decreases loss and separates positive/negative pairs."""
    index = VectorEmbeddingIndex(embedding_dim=128)

    anchors = ["task_0: calculate 5 + 3", "task_1: reverse hello"]
    positives = ["solution: 8", "solution: olleh"]
    negatives = ["solution: 99", "solution: wrong_output"]

    # Initial loss
    loss_0 = index.train_contrastive_step(anchors, positives, negatives)
    assert loss_0 > 0.0

    # Execute multiple optimization steps
    for _ in range(10):
        loss_n = index.train_contrastive_step(anchors, positives, negatives)

    assert loss_n < loss_0


def test_contrastive_retrieval_search() -> None:
    """Verifies vector embedding index top-k cosine similarity search after encoding."""
    index = VectorEmbeddingIndex(embedding_dim=128)

    text_pos = "task_0: modular add 5 mod 7 = 5"
    text_neg = "task_1: reverse string abc"

    emb_pos = index.encode(text_pos)
    emb_neg = index.encode(text_neg)

    index.add_vector("rec_pos", emb_pos)
    index.add_vector("rec_neg", emb_neg)

    query_emb = index.encode(text_pos)
    results = index.search(query_emb, top_k=1)

    assert len(results) == 1
    assert results[0][0] == "rec_pos"
    assert results[0][1] >= 0.99
