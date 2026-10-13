"""Milestone Genesis: Sensorimotor Tokenizer for Physical Intelligence."""

from typing import cast

import torch
import torch.nn as nn


class SensorimotorTokenizer(nn.Module):
    """Maps continuous kinematic coordinates and telemetry into latent tokens."""

    def __init__(self, kinematics_dim: int = 12, d_model: int = 512) -> None:
        super().__init__()
        self.kinematics_dim = kinematics_dim
        self.d_model = d_model
        self.proj = nn.Linear(kinematics_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, kinematic_telemetry: torch.Tensor) -> torch.Tensor:
        """Projects joint positions, velocities, and torques into model latent space."""
        return cast(torch.Tensor, self.norm(self.proj(kinematic_telemetry)))
