"""Unit and integration tests for Aptivra-Genesis Unified Facade & AID-Bench."""

from aptivra.benchmarks.aid_bench import AIDevelopmentalBenchmark
from aptivra.consolidation.sleep_cycle import GenerativeSleepEngine
from aptivra.genesis import AptivraGenesis
from aptivra.governance.dual_boundary import UserPreferenceProfile
from aptivra.model.config import ModelConfig


def test_generative_sleep_engine() -> None:
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32)
    genesis = AptivraGenesis(config=cfg)

    # Put a small weight to prune
    with torch_no_grad():
        for p in genesis.parameters():
            p.data.masked_fill_(p.abs() < 0.05, 0.00001)

    sleep_engine = GenerativeSleepEngine(genesis, genesis.hexa_memory, cfg)
    res = sleep_engine.execute_sleep_cycle(max_episodes=5, prune_threshold=0.001)
    assert res["phase"] == "SLEEP_COMPLETE"
    assert isinstance(res["pruned_synapses"], int)


def test_aptivra_genesis_lifecycle_and_bench() -> None:
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, d_ff=32)
    profile = UserPreferenceProfile(risk_tolerance=0.5)
    agent = AptivraGenesis(config=cfg, user_profile=profile)

    assert agent.developmental_stage == 0
    new_stage = agent.advance_developmental_stage()
    assert new_stage == 1

    # Test sensory-action step
    res = agent.step("user_requests_algorithm_optimization")
    assert res["status"] == "SUCCESS"
    assert "Aptivra-Genesis" in res["output"]

    # Test AID-Bench
    bench = AIDevelopmentalBenchmark()
    report = bench.evaluate_infant_milestone(agent)
    assert report.sensorimotor_score > 0.8
    assert report.zero_forgetting_retention is None
    assert report.curiosity_entropy is None


def torch_no_grad():
    import torch

    return torch.no_grad()
