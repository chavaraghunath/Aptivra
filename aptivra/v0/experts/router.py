"""Module-Level Gating Router with Soft Top-K Routing and Load-Balancing Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingRouter(nn.Module):
    """Module-level soft top-k router with dynamic parameter expansion and load balancing."""

    def __init__(self, d_model: int, initial_num_experts: int = 1, top_k: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = initial_num_experts
        self.top_k = top_k

        # Router weight matrix of shape (num_experts, d_model)
        self.router_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model) * 0.02) for _ in range(initial_num_experts)]
        )
        self.frozen_mask = [False] * initial_num_experts

    def spawn_expert_row(self) -> int:
        """Appends a new trainable router parameter row for a newly spawned expert."""
        new_row = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.router_weights.append(new_row)
        self.frozen_mask.append(False)
        self.num_experts += 1
        return self.num_experts - 1

    def freeze_expert_row(self, expert_idx: int) -> None:
        """Freezes the routing parameter row for a mature expert."""
        if 0 <= expert_idx < self.num_experts:
            self.router_weights[expert_idx].requires_grad = False
            self.frozen_mask[expert_idx] = True

    def compute_load_balancing_loss(
        self, full_probs: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Computes auxiliary load balancing loss encouraging uniform expert utilization."""
        # full_probs shape: (batch_size * seq_len, num_experts)
        flat_probs = full_probs.view(-1, self.num_experts)
        flat_indices = top_k_indices.view(-1, top_k_indices.shape[-1])

        # Calculate fraction of tokens routed to each expert f_i
        mask = torch.zeros_like(flat_probs, dtype=torch.float32)
        mask.scatter_(1, flat_indices, 1.0)
        f_i = mask.mean(dim=0)  # Fraction of tokens per expert

        # Calculate average probability allocated to each expert P_i
        p_i = flat_probs.mean(dim=0)

        # Load balancing loss: N * sum(f_i * P_i)
        lb_loss = self.num_experts * torch.sum(f_i * p_i)
        return lb_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates soft top-k routing probabilities, router entropy, and load-balancing loss.

        Returns (gating_probs, entropy_tensor, load_balancing_loss).
        """
        # Stack router weight vectors -> (num_experts, d_model)
        weight_matrix = torch.stack(list(self.router_weights), dim=0)

        # Compute raw logits: x @ weight_matrix.T -> (batch, seq, num_experts)
        logits = torch.matmul(x, weight_matrix.t())
        full_probs = F.softmax(logits, dim=-1)

        # Compute soft top-k gating mask
        k = min(self.top_k, self.num_experts)
        top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)

        # Construct masked logits with -inf for non-top-k entries
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_k_indices, top_k_values)
        gating_probs = F.softmax(masked_logits, dim=-1)

        # Calculate router entropy over active experts
        log_probs = torch.log(full_probs + 1e-9)
        entropy = -torch.sum(full_probs * log_probs, dim=-1).mean()

        # Calculate load balancing loss
        lb_loss = self.compute_load_balancing_loss(full_probs, top_k_indices)

        return gating_probs, entropy, lb_loss
