"""Unit tests for soft top-k GatingRouter and load-balancing auxiliary loss."""

import torch

from aptivra.experts.router import GatingRouter


def test_soft_gating_router_top_k_probs() -> None:
    """Verifies GatingRouter computes soft top-k probability vectors correctly."""
    router = GatingRouter(d_model=32, initial_num_experts=4, top_k=2)
    dummy_x = torch.randn(2, 5, 32)

    gating_probs, entropy, lb_loss = router(dummy_x)

    assert gating_probs.shape == (2, 5, 4)
    assert entropy.item() >= 0.0
    assert lb_loss.item() >= 0.0

    # Verify that exactly top_k (2) entries per token are non-zero
    non_zero_counts = (gating_probs > 1e-6).sum(dim=-1)
    assert torch.all(non_zero_counts == 2)

    # Verify probabilities sum to 1.0 per token
    prob_sums = gating_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_router_spawning_and_freezing() -> None:
    """Verifies dynamic router row spawning and freezing."""
    router = GatingRouter(d_model=32, initial_num_experts=1, top_k=2)
    assert router.num_experts == 1

    new_idx = router.spawn_expert_row()
    assert new_idx == 1
    assert router.num_experts == 2

    router.freeze_expert_row(0)
    assert router.router_weights[0].requires_grad is False
    assert router.router_weights[1].requires_grad is True
