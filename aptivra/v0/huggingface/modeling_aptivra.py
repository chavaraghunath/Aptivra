"""Hugging Face `PreTrainedModel` implementation for Aptivra causal LM."""

from typing import Any, cast

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter
from aptivra.v0.huggingface.configuration_aptivra import AptivraConfig
from aptivra.v0.inference.causal_lm import apply_top2_moe


class AptivraForCausalLM(PreTrainedModel, GenerationMixin):
    """Transformers-compatible Aptivra model with eagerly registered parameters."""

    config_class = AptivraConfig
    _tied_weights_keys = {"backbone.lm_head.weight": "backbone.token_embedding.weight"}

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__(config)
        model_cfg = ModelConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            device="cpu",
        )
        self.backbone = BackboneTransformer(model_cfg)
        self.router = GatingRouter(
            d_model=config.d_model,
            initial_num_experts=config.num_experts,
            top_k=config.top_k_experts,
        )
        self.experts = nn.ModuleList(
            [
                ExpertModule(i, config.d_model, d_expert=config.d_expert)
                for i in range(config.num_experts)
            ]
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Returns the byte-token embedding table used by Transformers helpers."""
        return self.backbone.token_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replaces the byte-token embedding table."""
        self.backbone.token_embedding = value

    def get_output_embeddings(self) -> nn.Module:
        """Returns the language-model projection."""
        return self.backbone.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Replaces the language-model projection."""
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError("Aptivra output embeddings must be torch.nn.Linear")
        self.backbone.lm_head = new_embeddings

    def _attention_bias(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None or bool(torch.all(attention_mask == 1)):
            return None
        seq_len = input_ids.shape[1]
        dtype = self.backbone.pos_embedding.weight.dtype
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
        bias = causal[None, None, :, :] + padding * torch.finfo(dtype).min
        return bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Input length {input_ids.shape[1]} exceeds context length "
                f"{self.config.max_seq_len}"
            )
        attention_bias = self._attention_bias(input_ids, attention_mask)
        hidden, _ = self.backbone.forward_blocks(input_ids, attention_mask=attention_bias)
        if self.config.use_moe_at_inference:
            gating_probs, _, _ = self.router(hidden)
            hidden = apply_top2_moe(hidden, gating_probs, self.experts)
        logits = self.backbone.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        output = CausalLMOutputWithPast(
            loss=cast(torch.FloatTensor | None, loss),
            logits=logits,
        )
        if return_dict is False:
            return cast(CausalLMOutputWithPast, output.to_tuple())
        return output
