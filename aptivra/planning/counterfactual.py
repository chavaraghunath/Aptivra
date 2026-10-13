"""Milestone Genesis Phase 2: Multi-Step Counterfactual Rollout Planner."""

from dataclasses import dataclass

import torch

from aptivra.world_model.jepa import LatentWorldModel


@dataclass
class RolloutTrajectory:
    """Hypothetical counterfactual trajectory evaluated in latent space."""

    trajectory_id: str
    action_sequence: list[torch.Tensor]
    predicted_states: list[torch.Tensor]
    total_energy_cost: float
    goal_alignment_score: float
    is_counterfactually_safe: bool


class CounterfactualPlanner:
    """Explores hypothetical action rollouts without physical or external side-effects."""

    def __init__(self, world_model: LatentWorldModel, max_horizon: int = 10) -> None:
        self.world_model = world_model
        self.max_horizon = max_horizon

    def evaluate_what_if_rollout(
        self,
        initial_state: torch.Tensor,
        action_sequence: list[torch.Tensor],
        goal_prior: torch.Tensor,
    ) -> RolloutTrajectory:
        """Simulates sequential state transitions in latent world model space."""
        horizon = min(len(action_sequence), self.max_horizon)
        current_state = initial_state
        predicted_states: list[torch.Tensor] = []
        total_energy = 0.0

        for i in range(horizon):
            act = action_sequence[i]
            next_state = self.world_model.predict_next_latent(current_state, act)
            energy = self.world_model.energy_loss(next_state, current_state)
            total_energy += float(energy.item())
            predicted_states.append(next_state)
            current_state = next_state

        final_dist = float(torch.norm(current_state - goal_prior).item())
        goal_score = max(0.0, 1.0 / (1.0 + final_dist))
        is_safe = total_energy < 50.0 and final_dist < 10.0

        return RolloutTrajectory(
            trajectory_id="cf_traj_001",
            action_sequence=action_sequence[:horizon],
            predicted_states=predicted_states,
            total_energy_cost=total_energy,
            goal_alignment_score=goal_score,
            is_counterfactually_safe=is_safe,
        )
