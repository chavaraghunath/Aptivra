"""AttentionInterface Abstract Base Class and Standard Causal Attention Module."""

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionInterface(nn.Module, ABC):
    """Abstract interface isolating attention mechanics for drop-in delta-rule / KDA layers."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Calculates attention output and returns updated KV cache tuple (k_cache, v_cache)."""
        pass


class StandardCausalAttention(AttentionInterface):
    """Standard multi-head scaled dot-product self-attention with causal masking and KV cache."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    @override
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        ctx = context if context is not None else x

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(ctx)
            .view(batch_size, ctx.shape[1], self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(ctx)
            .view(batch_size, ctx.shape[1], self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v)
        total_seq_len = k.shape[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            scores = scores + attention_mask
        elif seq_len > 1 and context is None and kv_cache is None:
            # Causal mask for autoregressive generation
            causal_mask = torch.triu(
                torch.full((seq_len, total_seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), new_kv_cache
