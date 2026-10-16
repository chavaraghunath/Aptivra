"""Unit tests for Tier 2 Expert Pool, Gating Router, and Spawner."""

import torch

from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter
from aptivra.v0.experts.spawner import ExpertSpawner, SpawnerConfig


def test_expert_module_freezing() -> None:
    expert = ExpertModule(expert_id=0, d_model=64, d_expert=32)
    x = torch.randn(2, 5, 64)
    out = expert(x)
    assert out.shape == (2, 5, 64)
    assert not expert.is_mature

    # Freeze expert
    expert.freeze()
    assert expert.is_mature
    for p in expert.parameters():
        assert not p.requires_grad


def test_gating_router_expansion_and_freezing() -> None:
    router = GatingRouter(d_model=64, initial_num_experts=2)
    x = torch.randn(2, 5, 64)

    probs, entropy, _lb_loss = router(x)
    assert probs.shape == (2, 5, 2)
    assert entropy.item() > 0.0

    # Freeze row 0
    router.freeze_expert_row(0)
    assert not router.router_weights[0].requires_grad
    assert router.router_weights[1].requires_grad

    # Spawn new expert row 2
    new_idx = router.spawn_expert_row()
    assert new_idx == 2
    assert router.num_experts == 3

    probs2, _, _ = router(x)
    assert probs2.shape == (2, 5, 3)


def test_expert_spawner() -> None:
    spawner = ExpertSpawner(SpawnerConfig(entropy_threshold=1.0, verifier_failure_threshold=0.25))

    # Low values -> no spawn
    should_1, reason_1 = spawner.should_spawn(
        current_num_experts=2, entropy_val=0.5, verifier_failure_rate=0.1
    )
    assert not should_1
    assert reason_1 == "NO_SPAWN_TRIGGERED"

    # High entropy -> spawn
    should_2, reason_2 = spawner.should_spawn(
        current_num_experts=2, entropy_val=1.5, verifier_failure_rate=0.1
    )
    assert should_2
    assert "HIGH_ROUTER_ENTROPY" in reason_2

    # High verifier failure -> spawn
    should_3, reason_3 = spawner.should_spawn(
        current_num_experts=2, entropy_val=0.5, verifier_failure_rate=0.4
    )
    assert should_3
    assert "HIGH_VERIFIER_FAILURE" in reason_3
