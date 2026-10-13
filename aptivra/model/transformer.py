"""Backbone Decoder-Only Transformer Architecture."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from aptivra.model.attention import AttentionInterface, StandardCausalAttention
from aptivra.model.config import ModelConfig
from aptivra.model.embeddings import PretrainedEmbeddingLayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return cast(torch.Tensor, res)


class TransformerLayer(nn.Module):
    """Transformer decoder block with pluggable AttentionInterface."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_module: AttentionInterface | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = (
            attention_module
            if attention_module is not None
            else StandardCausalAttention(d_model, n_heads, dropout)
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        norm_x = self.norm1(x)
        attn_out, new_cache = self.attn(
            norm_x, context=context, attention_mask=attention_mask, kv_cache=kv_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class BackboneTransformer(nn.Module):
    """Decoder-only autoregressive transformer backbone (50M-200M params)."""

    def __init__(
        self,
        config: ModelConfig,
        pretrained_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if config.use_pretrained_embeddings or pretrained_weights is not None:
            self.token_embedding: nn.Module = PretrainedEmbeddingLayer(
                vocab_size=config.vocab_size,
                d_embedding=config.pretrained_embedding_dim,
                d_model=config.d_model,
                pretrained_weights=pretrained_weights,
                freeze_embedding=config.freeze_pretrained_embeddings,
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying where dimensions match
        if isinstance(self.token_embedding, nn.Embedding):
            self.lm_head.weight = self.token_embedding.weight
        elif (
            isinstance(self.token_embedding, PretrainedEmbeddingLayer)
            and config.pretrained_embedding_dim == config.d_model
        ):
            self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Runs embeddings and decoder blocks; returns normalized hidden states."""
        device = input_ids.device
        seq_len = input_ids.shape[1]

        past_len = past_kv_caches[0][0].shape[2] if past_kv_caches else 0
        positions = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_caches[i] if past_kv_caches else None
            x, new_cache = layer(x, attention_mask=attention_mask, kv_cache=layer_cache)
            if new_cache is not None:
                new_kv_caches.append(new_cache)

        return self.final_norm(x), new_kv_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden, new_kv_caches = self.forward_blocks(
            input_ids, attention_mask=attention_mask, past_kv_caches=past_kv_caches
        )
        logits = self.lm_head(hidden)
        return cast(torch.Tensor, logits), new_kv_caches
