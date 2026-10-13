"""Milestone Genesis Phase 3: Autonomous Hypothesis Engine."""

from dataclasses import dataclass

import torch

from aptivra.memory.semantic_graph import SemanticGraphCore


@dataclass
class HypothesisCandidate:
    """A scientific or logical hypothesis derived from semantic graph contradiction or vacuum."""

    hypothesis_id: str
    premise: str
    proposed_relation: str
    target_concept: str
    prior_confidence: float
    latent_embedding: torch.Tensor
    hypothesis_type: str = (
        "transitive_causality"  # "contradiction", "transitive_causality", "analogy"
    )


class HypothesisEngine:
    """Discovers knowledge contradictions, causal gaps, and formulates testable hypotheses."""

    def __init__(self, semantic_graph: SemanticGraphCore) -> None:
        self.semantic_graph = semantic_graph

    def discover_hypotheses(self) -> list[HypothesisCandidate]:
        """Scans semantic graph for causal contradictions, transitive gaps, and analogies."""
        hypotheses: list[HypothesisCandidate] = []

        # 1. Contradiction Detection: Scan for opposing relations between same nodes
        edge_map: dict[tuple[str, str], list[str]] = {}
        for edge in self.semantic_graph.edges:
            pair = (edge.source_id, edge.target_id)
            edge_map.setdefault(pair, []).append(edge.relation)

        for (src, tgt), relations in edge_map.items():
            if "causes" in relations and "inhibits" in relations:
                n_src = self.semantic_graph.nodes.get(src)
                n_tgt = self.semantic_graph.nodes.get(tgt)
                emb = (
                    (n_src.prototype_vector + n_tgt.prototype_vector) / 2.0
                    if (n_src and n_tgt)
                    else torch.zeros(self.semantic_graph.embedding_dim)
                )
                hypotheses.append(
                    HypothesisCandidate(
                        hypothesis_id=f"hyp_contradiction_{src}_{tgt}",
                        premise=f"Contradiction detected: '{src}' both causes and inhibits '{tgt}'",
                        proposed_relation="conditionally_modulates",
                        target_concept=tgt,
                        prior_confidence=0.85,
                        latent_embedding=emb,
                        hypothesis_type="contradiction",
                    )
                )

        # 2. Transitive Causal Gap Discovery: If A -> B and B -> C, test A -> C
        nodes = list(self.semantic_graph.nodes.values())
        for e1 in self.semantic_graph.edges:
            for e2 in self.semantic_graph.edges:
                if e1.target_id == e2.source_id and e1.source_id != e2.target_id:
                    a = e1.source_id
                    c = e2.target_id
                    # Check if A -> C already exists
                    has_direct = any(
                        e.source_id == a and e.target_id == c for e in self.semantic_graph.edges
                    )
                    if not has_direct:
                        n_a = self.semantic_graph.nodes.get(a)
                        n_c = self.semantic_graph.nodes.get(c)
                        if n_a and n_c:
                            joint_conf = min(e1.strength, e2.strength) * 0.9
                            premise_str = (
                                f"Transitive deduction: '{a}' influences '{e1.target_id}' "
                                f"which influences '{c}'"
                            )
                            hypotheses.append(
                                HypothesisCandidate(
                                    hypothesis_id=f"hyp_transitive_{a}_{c}",
                                    premise=premise_str,
                                    proposed_relation="transitive_causality",
                                    target_concept=c,
                                    prior_confidence=joint_conf,
                                    latent_embedding=(n_a.prototype_vector + n_c.prototype_vector)
                                    / 2.0,
                                    hypothesis_type="transitive_causality",
                                )
                            )

        # 3. High-Similarity Analogy Discovery: Prototype cosine similarity
        if len(nodes) >= 2:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    n1 = nodes[i]
                    n2 = nodes[j]
                    sim = float(
                        torch.cosine_similarity(
                            n1.prototype_vector.flatten(),
                            n2.prototype_vector.flatten(),
                            dim=0,
                        ).item()
                    )
                    existing = any(
                        (e.source_id == n1.concept_id and e.target_id == n2.concept_id)
                        or (e.source_id == n2.concept_id and e.target_id == n1.concept_id)
                        for e in self.semantic_graph.edges
                    )
                    if sim > 0.4 and not existing:
                        analogy_premise = (
                            f"Analogy: Concept '{n1.label}' correlates with '{n2.label}'"
                        )
                        hypotheses.append(
                            HypothesisCandidate(
                                hypothesis_id=f"hyp_{n1.concept_id}_{n2.concept_id}",
                                premise=analogy_premise,
                                proposed_relation="analogous_property",
                                target_concept=n2.concept_id,
                                prior_confidence=sim,
                                latent_embedding=(n1.prototype_vector + n2.prototype_vector) / 2.0,
                                hypothesis_type="analogy",
                            )
                        )

        return hypotheses
