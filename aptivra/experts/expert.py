"""Expert Module Block Architecture."""

from typing import cast

import torch
import torch.nn as nn


class ExpertModule(nn.Module):
    """Parameter-efficient expert adapter module block."""

    def __init__(self, expert_id: int, d_model: int, d_expert: int = 128) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.d_model = d_model
        self.d_expert = d_expert
        self.is_mature = False

        self.down_proj = nn.Linear(d_model, d_expert, bias=False)
        self.up_proj = nn.Linear(d_expert, d_model, bias=False)
        self.act = nn.SiLU()

        # Initialize small near-zero output
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up_proj.weight)

    def freeze(self) -> None:
        """Freezes all parameters when expert matures."""
        self.is_mature = True
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.up_proj(self.act(self.down_proj(x)))
        return cast(torch.Tensor, res)
