"""Pre-trained Token Embedding Alignment Layer ($T_0$)."""

from typing import cast

import torch
import torch.nn as nn


class PretrainedEmbeddingLayer(nn.Module):
    """Token embedding layer with optional pre-aligned weights and linear projection adaptors."""

    def __init__(
        self,
        vocab_size: int,
        d_embedding: int,
        d_model: int,
        pretrained_weights: torch.Tensor | None = None,
        freeze_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_embedding)
        if pretrained_weights is not None:
            if pretrained_weights.shape != (vocab_size, d_embedding):
                raise ValueError(
                    f"Expected pretrained_weights shape ({vocab_size}, {d_embedding}), "
                    f"got {pretrained_weights.shape}"
                )
            self.embedding.weight.data.copy_(pretrained_weights)

        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        if d_embedding != d_model:
            proj = nn.Linear(d_embedding, d_model, bias=False)
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            self.projection: nn.Module = proj
        else:
            self.projection = nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and project to d_model hidden dimension."""
        embedded = self.embedding(input_ids)
        projected = self.projection(embedded)
        return cast(torch.Tensor, projected)
