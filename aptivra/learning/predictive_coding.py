"""Milestone Genesis: Local Predictive Coding for Backprop-Free Edge Adaptation."""

import torch
import torch.nn as nn


class LocalPredictiveLayer(nn.Module):
    """Predictive coding layer performing local error minimization without autograd graphs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting next layer activations."""
        predicted = torch.matmul(x, self.weight.t()) + self.bias
        return predicted, x

    def update_local(self, x: torch.Tensor, target: torch.Tensor, lr: float = 1e-3) -> float:
        """Applies local Hebbian error-driven gradient update directly to layer parameters."""
        with torch.no_grad():
            batch_size = max(1, x.size(0))
            pred = torch.matmul(x, self.weight.t()) + self.bias
            error = target - pred
            # Weight update: correlation between error and inputs
            delta_w = torch.matmul(error.t(), x) / batch_size
            self.weight.data.add_(delta_w, alpha=lr)
            self.bias.data.add_(error.mean(dim=0), alpha=lr)
            return float(error.pow(2).mean().item())
