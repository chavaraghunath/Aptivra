"""Unit tests for Aptivra-Genesis Local Plasticity, World Model & Active Inference."""

import torch

from aptivra.learning.predictive_coding import LocalPredictiveLayer
from aptivra.planning.active_inference import ActiveInferencePlanner
from aptivra.world_model.jepa import LatentWorldModel


def test_local_predictive_layer() -> None:
    layer = LocalPredictiveLayer(in_features=8, out_features=4)
    x = torch.randn(2, 8)
    pred, inp = layer(x)
    assert pred.shape == (2, 4)
    assert torch.equal(inp, x)

    target = torch.randn(2, 4)
    err = layer.update_local(x, target, lr=1e-2)
    assert isinstance(err, float)
    assert err >= 0.0


def test_latent_world_model() -> None:
    model = LatentWorldModel(latent_dim=16, action_dim=4)
    s_t = torch.randn(2, 16)
    a_t = torch.randn(2, 4)

    s_next = model.predict_next_latent(s_t, a_t)
    assert s_next.shape == (2, 16)

    loss = model.energy_loss(s_next, s_t)
    assert loss.dim() == 0
    assert float(loss.item()) >= 0.0


def test_active_inference_planner() -> None:
    model = LatentWorldModel(latent_dim=8, action_dim=4)
    planner = ActiveInferencePlanner(model, horizon=3)

    current_state = torch.randn(1, 8)
    goal_prior = torch.ones(1, 8)

    candidates = [
        torch.zeros(1, 4),
        torch.ones(1, 4),
        torch.randn(1, 4),
    ]

    best_idx, efe = planner.select_best_action(current_state, candidates, goal_prior)
    assert best_idx in [0, 1, 2]
    assert isinstance(efe, float)

    # Empty candidate safety test
    empty_idx, empty_efe = planner.select_best_action(current_state, [], goal_prior)
    assert empty_idx == 0
    assert empty_efe == 0.0
