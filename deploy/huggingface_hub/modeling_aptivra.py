"""Self-contained Transformers model for Aptivra safetensors checkpoints."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_aptivra import AptivraConfig


class RMSNorm(nn.Module):
    """Root mean square normalization used by the package model."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block used by the package model."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalAttention(nn.Module):
    """Package-identical multi-head causal attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    dtype=x.dtype,
                    device=x.device,
                ),
                diagonal=1,
            )[None, None, :, :]
        scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1)
        if self.training:
            weights = F.dropout(weights, p=self.dropout_p)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class TransformerLayer(nn.Module):
    """Aptivra decoder layer."""

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        return x + self.ffn(self.norm2(x))


class Backbone(nn.Module):
    """Dense Aptivra decoder backbone."""

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.final_norm(hidden)


class Router(nn.Module):
    """Soft top-k expert router with package-compatible parameter names."""

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.router_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(config.d_model)) for _ in range(config.num_experts)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(hidden, torch.stack(list(self.router_weights), dim=0).t())
        k = min(self.top_k, self.num_experts)
        top_values, top_indices = torch.topk(logits, k=k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, top_indices, top_values)
        return F.softmax(masked, dim=-1)


class Expert(nn.Module):
    """Aptivra expert adapter with package-compatible parameter names."""

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__()
        self.down_proj = nn.Linear(config.d_model, config.d_expert, bias=False)
        self.up_proj = nn.Linear(config.d_expert, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up_proj(F.silu(self.down_proj(hidden)))


class AptivraForCausalLM(PreTrainedModel, GenerationMixin):
    """Aptivra causal LM whose parameters load directly from model.safetensors."""

    config_class = AptivraConfig
    _tied_weights_keys = {"backbone.lm_head.weight": "backbone.token_embedding.weight"}

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__(config)
        self.backbone = Backbone(config)
        self.router = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.token_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.token_embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Linear):
            raise TypeError("Aptivra output embeddings must be torch.nn.Linear")
        self.backbone.lm_head = value

    def _attention_bias(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None or bool(torch.all(attention_mask == 1)):
            return None
        seq_len = input_ids.shape[1]
        dtype = self.backbone.token_embedding.weight.dtype
        causal = torch.triu(
            torch.full(
                (seq_len, seq_len),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        padding = (1 - attention_mask.to(dtype=dtype))[:, None, None, :]
        return causal[None, None, :, :] + padding * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Input length {input_ids.shape[1]} exceeds context length "
                f"{self.config.max_seq_len}"
            )
        hidden = self.backbone.forward_blocks(
            input_ids,
            self._attention_bias(input_ids, attention_mask),
        )
        if self.config.use_moe_at_inference:
            probs = self.router(hidden)
            mixture = torch.zeros_like(hidden)
            for index, expert in enumerate(self.experts):
                weight = probs[..., index : index + 1]
                if weight.max().item() <= 0.0:
                    continue
                mixture = mixture + weight * expert(hidden)
            hidden = hidden + mixture
        logits = self.backbone.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
