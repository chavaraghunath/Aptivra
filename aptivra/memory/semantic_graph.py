"""Milestone Genesis: Tier 2 Semantic Hypergraph Core."""

from dataclasses import dataclass, field

import torch


@dataclass
class ConceptNode:
    """Represents a generalized ontological concept abstracted from experience."""

    concept_id: str
    label: str
    prototype_vector: torch.Tensor
    attributes: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class CausalEdge:
    """Represents a directed causal or associative relation between concepts."""

    source_id: str
    target_id: str
    relation: str  # e.g., "causes", "inhibits", "part_of", "prerequisite"
    strength: float = 1.0


class SemanticGraphCore:
    """Neurosymbolic hypergraph storing generalized principles and causal relationships."""

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: list[CausalEdge] = []

    def add_concept(self, node: ConceptNode) -> None:
        """Inserts or updates an ontological concept."""
        self.nodes[node.concept_id] = node

    def add_relation(self, edge: CausalEdge) -> None:
        """Inserts a directed causal or associative relationship."""
        self.edges.append(edge)

    def query_subgraph(self, query_vec: torch.Tensor, top_k: int = 5) -> list[ConceptNode]:
        """Queries concepts by cosine similarity against prototype vectors."""
        if not self.nodes:
            return []
        scores: list[tuple[float, ConceptNode]] = []
        flat_query = query_vec.flatten()
        for node in self.nodes.values():
            flat_proto = node.prototype_vector.flatten()
            sim = torch.cosine_similarity(flat_query, flat_proto, dim=0).item()
            scores.append((float(sim), node))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scores[:top_k]]

    def get_related_concepts(self, concept_id: str) -> list[tuple[str, str, float]]:
        """Returns list of (target_concept_id, relation, strength) for a given concept."""
        results: list[tuple[str, str, float]] = []
        for edge in self.edges:
            if edge.source_id == concept_id:
                results.append((edge.target_id, edge.relation, edge.strength))
        return results

    def __len__(self) -> int:
        return len(self.nodes)
