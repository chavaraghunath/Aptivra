"""Hub-side Transformers model (requires `pip install aptivra` for shard loader)."""

from pathlib import Path
from typing import Any

import torch
from configuration_aptivra import AptivraConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.model.config import ModelConfig


class AptivraForCausalLM(PreTrainedModel):
    """Transformers-compatible causal LM over Aptivra multitier shard layout."""

    config_class = AptivraConfig

    def __init__(self, config: AptivraConfig) -> None:
        super().__init__(config)
        self.lm: AptivraCausalLM | None = None

    def _ensure_loaded(self) -> AptivraCausalLM:
        if self.lm is not None:
            return self.lm
        model_cfg = ModelConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            device="auto",
        )
        export_dir = Path(str(self.name_or_path))
        self.lm = AptivraCausalLM.from_export_dir(export_dir, model_config=model_cfg)
        return self.lm

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs
        lm = self._ensure_loaded()
        logits = lm(input_ids.to(lm.device))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
