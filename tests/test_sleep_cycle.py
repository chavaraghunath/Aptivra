"""Behavioral proof for the active generative sleep cycle."""

from typing import cast

import torch
import torch.nn as nn

from aptivra.consolidation.sleep_cycle import GenerativeSleepEngine
from aptivra.memory.hexa_store import HexaTierMemoryStore
from aptivra.model.config import ModelConfig


class _EpisodicProbe:
    def __init__(self, candidate_count: int) -> None:
        self.candidate_count = candidate_count
        self.minimum_retrievals: list[int] = []

    def get_consolidation_candidates(self, min_retrievals: int) -> list[object]:
        self.minimum_retrievals.append(min_retrievals)
        return [object() for _ in range(self.candidate_count)]


class _StoreProbe:
    def __init__(self, candidate_count: int, size: int) -> None:
        self.episodic_store = _EpisodicProbe(candidate_count)
        self.size = size

    def __len__(self) -> int:
        return self.size


class _PruningFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor([0.5, 1.0, 2.0]))
        self.frozen = nn.Parameter(torch.tensor([0.25]), requires_grad=False)


def _engine(model: nn.Module, store: _StoreProbe) -> GenerativeSleepEngine:
    return GenerativeSleepEngine(model, cast(HexaTierMemoryStore, store))


def test_sleep_cycle_defaults_cap_replay_and_query_eligible_episodes() -> None:
    model = _PruningFixture()
    with torch.no_grad():
        model.trainable.fill_(2.0)
    store = _StoreProbe(candidate_count=60, size=7)
    engine = _engine(model, store)

    result = engine.execute_sleep_cycle()

    assert isinstance(engine.config, ModelConfig)
    assert store.episodic_store.minimum_retrievals == [1]
    assert result == {
        "phase": "SLEEP_COMPLETE",
        "replayed_episodes": 50,
        "pruned_synapses": 0,
        "episodic_remaining": 7,
    }


def test_sleep_cycle_preserves_explicit_config_and_respects_replay_limit() -> None:
    model = _PruningFixture()
    config = ModelConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32)
    store = _StoreProbe(candidate_count=3, size=3)
    engine = GenerativeSleepEngine(model, cast(HexaTierMemoryStore, store), config)

    result = engine.execute_sleep_cycle(max_episodes=2, prune_threshold=0.0)

    assert engine.config is config
    assert result["replayed_episodes"] == 2
    assert result["pruned_synapses"] == 0


def test_pruning_is_strict_and_skips_frozen_parameters() -> None:
    model = _PruningFixture()
    engine = _engine(model, _StoreProbe(candidate_count=0, size=0))

    pruned = engine.prune_redundant_synapses(threshold=1.0)

    assert pruned == 1
    assert torch.equal(model.trainable, torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(model.frozen, torch.tensor([0.25]))


def test_pruning_default_threshold_is_effective() -> None:
    model = _PruningFixture()
    with torch.no_grad():
        model.trainable.copy_(torch.tensor([0.00005, 0.0001, 0.0002]))
    engine = _engine(model, _StoreProbe(candidate_count=0, size=0))

    pruned = engine.prune_redundant_synapses()

    assert pruned == 1
    assert torch.equal(model.trainable, torch.tensor([0.0, 0.0001, 0.0002]))
