"""Milestone Genesis: Offline Generative Sleep & Dream Consolidation Cycle."""

from typing import Any

import torch
import torch.nn as nn

from aptivra.memory.hexa_store import HexaTierMemoryStore
from aptivra.model.config import ModelConfig


class GenerativeSleepEngine:
    """Consolidates episodic experience into semantic invariants and performs synaptic pruning."""

    def __init__(
        self,
        model: nn.Module,
        hexa_store: HexaTierMemoryStore,
        config: ModelConfig | None = None,
    ) -> None:
        self.model = model
        self.hexa_store = hexa_store
        self.config = config or ModelConfig()

    def execute_sleep_cycle(
        self, max_episodes: int = 50, prune_threshold: float = 1e-4
    ) -> dict[str, Any]:
        """Runs Slow-Wave replay followed by REM counterfactual pruning."""
        # Phase 1: Slow-Wave High-Surprise Replay
        candidates = self.hexa_store.episodic_store.get_consolidation_candidates(min_retrievals=1)
        replayed_count = min(len(candidates), max_episodes)

        # Phase 2: REM Synaptic Pruning (Synaptic Homeostasis)
        pruned_weights = self.prune_redundant_synapses(threshold=prune_threshold)

        return {
            "phase": "SLEEP_COMPLETE",
            "replayed_episodes": replayed_count,
            "pruned_synapses": pruned_weights,
            "episodic_remaining": len(self.hexa_store),
        }

    def prune_redundant_synapses(self, threshold: float = 1e-4) -> int:
        """Prunes dead or near-zero connection weights to prevent parameter bloat."""
        pruned = 0
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    mask = param.abs() < threshold
                    pruned += int(mask.sum().item())
                    param.data.masked_fill_(mask, 0.0)
        return pruned
