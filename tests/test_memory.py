"""Unit tests for Tier 1 Episodic Memory Store and Vector Retrieval."""

from aptivra.memory.retrieval import VectorEmbeddingIndex
from aptivra.memory.store import EpisodicStore
from aptivra.tasks.base import VerifierResult


def test_vector_embedding_index() -> None:
    index = VectorEmbeddingIndex(embedding_dim=64)
    emb1 = index.encode("Evaluate: (5 + 5) mod 7 =")
    emb2 = index.encode("Evaluate: (10 + 10) mod 7 =")
    emb3 = index.encode("Reverse string 'abcdef' =")

    assert emb1.shape == (64,)
    index.add_vector("rec_1", emb1)
    index.add_vector("rec_2", emb2)
    index.add_vector("rec_3", emb3)

    results = index.search(emb1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "rec_1"


def test_episodic_store_gated_write() -> None:
    store = EpisodicStore(embedding_dim=64)

    # 1. Verified passed result -> admitted
    passed_result = VerifierResult(passed=True, expected="3", actual="3")
    ok = store.write_record(
        prompt="Evaluate: (5 + 5) mod 7 =",
        target_solution="3",
        verifier_result=passed_result,
        task_family="T1_modular_arithmetic",
    )
    assert ok is True
    assert len(store) == 1

    # 2. Verified failed result -> STRICTLY REJECTED
    failed_result = VerifierResult(passed=False, expected="3", actual="99", error_message="Wrong")
    rejected = store.write_record(
        prompt="Evaluate: (5 + 5) mod 7 =",
        target_solution="3",
        verifier_result=failed_result,
        task_family="T1_modular_arithmetic",
    )
    assert rejected is False
    assert len(store) == 1  # Count remains 1


def test_episodic_store_query_and_consolidation() -> None:
    store = EpisodicStore(embedding_dim=64)
    passed_result = VerifierResult(passed=True, expected="sol", actual="sol")

    store.write_record("prompt_1", "sol_1", passed_result, "T1")
    store.write_record("prompt_2", "sol_2", passed_result, "T1")

    # Query top-k
    retrieved = store.query_top_k("prompt_1", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].retrieval_count == 1

    candidates = store.get_consolidation_candidates(min_retrievals=1)
    assert len(candidates) == 1
    assert candidates[0].prompt == "prompt_1"
