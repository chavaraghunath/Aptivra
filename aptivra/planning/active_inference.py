"""Milestone Genesis: Active Inference Planning Core."""

import torch

from aptivra.world_model.jepa import LatentWorldModel


class ActiveInferencePlanner:
    """Selects actions minimizing Expected Free Energy (balancing pragmatic and epistemic value)."""

    def __init__(self, world_model: LatentWorldModel, horizon: int = 5) -> None:
        self.world_model = world_model
        self.horizon = horizon

    def select_best_action(
        self,
        current_state: torch.Tensor,
        candidate_actions: list[torch.Tensor],
        goal_prior: torch.Tensor,
    ) -> tuple[int, float]:
        """Selects the action candidate that minimizes Expected Free Energy (EFE)."""
        if not candidate_actions:
            return 0, 0.0

        best_idx = 0
        min_efe = float("inf")

        for idx, act in enumerate(candidate_actions):
            next_state = self.world_model.predict_next_latent(current_state, act)
            # Pragmatic value: distance to target goal
            pragmatic = float(torch.norm(next_state - goal_prior).item())
            # Epistemic value: proxy via representation variance (curiosity / information gain)
            epistemic = -0.05 * float(torch.var(next_state).item())
            efe = pragmatic + epistemic
            if efe < min_efe:
                min_efe = efe
                best_idx = idx

        return best_idx, min_efe
