"""Full Top-2 MoE causal LM for Aptivra tiered-memory checkpoints."""

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.multitier import MultitierLoader
from aptivra.inference.hub_config import model_config_from_export_dir
from aptivra.inference.tokenizer import ByteLevelTokenizer
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def apply_top2_moe(
    hidden: torch.Tensor,
    gating_probs: torch.Tensor,
    experts: nn.ModuleList | list[ExpertModule],
) -> torch.Tensor:
    """Applies soft top-2 expert mixture to hidden states."""
    mixture = torch.zeros_like(hidden)
    for expert_idx, expert in enumerate(experts):
        weight = gating_probs[..., expert_idx : expert_idx + 1]
        if weight.max().item() <= 0.0:
            continue
        mixture = mixture + weight * expert(hidden)
    return hidden + mixture


class AptivraCausalLM(nn.Module):
    """Backbone + router + expert shards for autoregressive inference."""

    def __init__(
        self,
        model_config: ModelConfig,
        backbone: BackboneTransformer,
        router: GatingRouter,
        experts: list[ExpertModule],
        *,
        use_moe_at_inference: bool = False,
    ) -> None:
        super().__init__()
        self.config = model_config
        self.backbone = backbone
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.use_moe_at_inference = use_moe_at_inference
        self.tokenizer = ByteLevelTokenizer(max_len=model_config.max_seq_len)
        self.device = model_config.get_device()

    @classmethod
    def from_export_dir(
        cls,
        export_dir: Path | str,
        model_config: ModelConfig | None = None,
    ) -> "AptivraCausalLM":
        export_path = Path(export_dir)
        loader = MultitierLoader(export_path)
        cfg = model_config or model_config_from_export_dir(export_path, device="auto")
        backbone, router, experts = loader.load_model(cfg)
        device = cfg.get_device()
        backbone = backbone.to(device)
        router = router.to(device)
        experts = [exp.to(device) for exp in experts]
        return cls(cfg, backbone, router, experts)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        export_dir: Path | str = "scratch/hf_export",
        revision: str | None = None,
        token: str | None = None,
    ) -> "AptivraCausalLM":
        """Downloads shard layout from Hugging Face Hub then loads locally."""
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(export_dir),
            revision=revision,
            token=token,
        )
        return cls.from_export_dir(local_dir)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.backbone.forward_blocks(input_ids)
        if self.use_moe_at_inference:
            gating_probs, _, _ = self.router(hidden)
            hidden = apply_top2_moe(hidden, gating_probs, self.experts)
        return cast(torch.Tensor, self.backbone.lm_head(hidden))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        *,
        stop_on_single_newline: bool = True,
        stop_on_sentence_end: bool = False,
    ) -> str:
        """Greedy byte-level generation."""
        self.eval()
        prompt_ids = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
        curr = prompt_ids
        prompt_len = prompt_ids.shape[1]
        for _ in range(max_new_tokens):
            if curr.shape[1] >= self.config.max_seq_len:
                break
            logits = self.forward(curr)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = int(next_tok.item())
            curr = torch.cat([curr, next_tok], dim=1)
            if stop_on_single_newline and tok_id in (0, ord("\n")):
                break
            gen_len = curr.shape[1] - prompt_len
            if stop_on_sentence_end and gen_len >= 12 and tok_id in (ord("."), ord("!"), ord("?")):
                break
            gen_so_far = self.tokenizer.decode(curr[0, prompt_len:])
            if "User:" in gen_so_far or "\n\n" in gen_so_far:
                break
        new_tokens = curr[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens)
        if "User:" in text:
            text = text.split("User:", 1)[0]
        return text.strip()
