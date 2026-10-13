"""Unit tests for HybridMemoryStore ($T_1$-Hybrid)."""

import torch

from aptivra.memory.hybrid_store import HybridMemoryStore
from aptivra.tasks.base import VerifierResult


def test_hybrid_memory_store_admit_and_replay() -> None:
    store = HybridMemoryStore(embedding_dim=64, max_raw_replay_capacity=5)

    passed_res = VerifierResult(passed=True, expected="4", actual="4")
    failed_res = VerifierResult(passed=False, expected="5", actual="6")

    input_ids = torch.tensor([1, 2, 3, 4])
    labels = torch.tensor([1, 2, 3, 4])

    admitted = store.write_sample(
        prompt="calc 2+2",
        target_solution="4",
        verifier_result=passed_res,
        task_family="modular_arithmetic",
        input_ids=input_ids,
        labels=labels,
    )
    assert admitted is True
    assert len(store) == 1
    assert len(store.raw_replay_buffer) == 1

    # Write failed sample -> Should not admit to T1 episodic store
    store.write_sample(
        prompt="calc 2+3",
        target_solution="6",
        verifier_result=failed_res,
        task_family="modular_arithmetic",
        input_ids=input_ids,
        labels=labels,
    )
    assert len(store) == 1
    assert len(store.raw_replay_buffer) == 2


def test_hybrid_memory_store_reservoir_sampling() -> None:
    store = HybridMemoryStore(embedding_dim=64, max_raw_replay_capacity=3)
    res = VerifierResult(passed=True, expected="ok", actual="ok")

    for i in range(10):
        store.write_sample(
            prompt=f"prompt {i}",
            target_solution=f"sol {i}",
            verifier_result=res,
            task_family="modular_arithmetic",
            input_ids=torch.tensor([i]),
            labels=torch.tensor([i]),
        )

    assert len(store.raw_replay_buffer) == 3
    samples = store.sample_raw_replay(2)
    assert len(samples) == 2
