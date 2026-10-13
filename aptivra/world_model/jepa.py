"""Milestone Genesis: Hierarchical Joint-Embedding Predictive World Model (H-JEPA)."""

from typing import cast

import torch
import torch.nn as nn


class LatentWorldModel(nn.Module):
    """Predicts state transitions directly within continuous latent representation manifolds."""

    def __init__(self, latent_dim: int = 512, action_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def predict_next_latent(
        self, current_latent: torch.Tensor, action_latent: torch.Tensor
    ) -> torch.Tensor:
        """Projects current state and action into predicted subsequent latent state."""
        combined = torch.cat([current_latent, action_latent], dim=-1)
        return cast(torch.Tensor, self.transition_net(combined))

    def energy_loss(
        self, predicted_latent: torch.Tensor, actual_latent: torch.Tensor
    ) -> torch.Tensor:
        """Computes energy distance (prediction error) between predicted and actual latents."""
        return torch.mean((predicted_latent - actual_latent) ** 2)
