"""Unit tests for Aptivra-Genesis Hexa-Tier Memory Components."""

import torch

from aptivra.memory.hexa_store import HexaTierMemoryStore
from aptivra.memory.semantic_graph import CausalEdge, ConceptNode, SemanticGraphCore
from aptivra.memory.somatic_valence import SomaticValenceEngine
from aptivra.memory.working_memory import WorkingMemoryBuffer
from aptivra.tasks.base import VerifierResult


def test_working_memory_buffer() -> None:
    buf = WorkingMemoryBuffer(capacity=3, d_model=16)
    assert len(buf) == 0

    t1 = torch.randn(16)
    t2 = torch.randn(16)
    t3 = torch.randn(16)
    t4 = torch.randn(16)

    buf.push(t1, weight=1.0)
    buf.push(t2, weight=0.8)
    buf.push(t3, weight=0.5)
    assert len(buf) == 3

    # FIFO overflow
    buf.push(t4, weight=0.9)
    assert len(buf) == 3

    mat = buf.get_context_matrix()
    assert mat.shape == (1, 3, 16)

    buf.clear()
    assert len(buf) == 0
    empty_mat = buf.get_context_matrix()
    assert empty_mat.shape == (1, 0, 16)


def test_semantic_graph_core() -> None:
    graph = SemanticGraphCore(embedding_dim=8)
    v1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    node1 = ConceptNode(concept_id="c1", label="Gravity", prototype_vector=v1)
    node2 = ConceptNode(concept_id="c2", label="Falling", prototype_vector=v2)

    graph.add_concept(node1)
    graph.add_concept(node2)
    assert len(graph) == 2

    edge = CausalEdge(source_id="c1", target_id="c2", relation="causes", strength=0.9)
    graph.add_relation(edge)

    related = graph.get_related_concepts("c1")
    assert len(related) == 1
    assert related[0] == ("c2", "causes", 0.9)

    query = torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = graph.query_subgraph(query, top_k=1)
    assert len(results) == 1
    assert results[0].concept_id == "c1"


def test_somatic_valence_engine() -> None:
    engine = SomaticValenceEngine()

    # Successful low loss episode
    marker_success = engine.evaluate_episode(context_loss=0.1, verifier_passed=True)
    assert marker_success.risk_score < 0.2
    assert marker_success.trust_delta > 0.0
    assert marker_success.cognitive_dissonance < 0.1

    # High confidence failure -> high cognitive dissonance
    marker_dissonance = engine.evaluate_episode(context_loss=0.1, verifier_passed=False)
    assert marker_dissonance.risk_score > 0.5
    assert marker_dissonance.cognitive_dissonance > 0.9
    assert marker_dissonance.trust_delta < 0.0


def test_hexa_tier_memory_store() -> None:
    store = HexaTierMemoryStore(embedding_dim=16)
    assert len(store) == 0

    res_ok = VerifierResult(
        passed=True,
        expected="return a + b",
        actual="return a + b",
        error_message=None,
    )
    admitted = store.record_interaction(
        prompt="def add(a, b):",
        solution="return a + b",
        verifier_result=res_ok,
        task_family="arithmetic",
        loss_val=0.05,
    )
    assert admitted is True
    assert len(store) == 1

    records = store.query_top_k("def add(x, y):", top_k=1)
    assert len(records) == 1
    assert records[0].metadata["trust_delta"] > 0
