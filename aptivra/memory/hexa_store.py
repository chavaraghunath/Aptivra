"""Milestone Genesis: Unified Hexa-Tier Memory Store Facade."""

from typing import Any

import torch

from aptivra.memory.semantic_graph import SemanticGraphCore
from aptivra.memory.somatic_valence import SomaticMarker, SomaticValenceEngine
from aptivra.memory.store import EpisodicRecord, EpisodicStore
from aptivra.memory.working_memory import WorkingMemoryBuffer
from aptivra.tasks.base import VerifierResult


class HexaTierMemoryStore:
    """Unified cognitive memory continuum combining all six memory tiers."""

    def __init__(self, embedding_dim: int = 512, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.embedding_dim = embedding_dim
        self.working_memory = WorkingMemoryBuffer(d_model=embedding_dim)
        self.episodic_store = EpisodicStore(embedding_dim=embedding_dim, device=self.device)
        self.semantic_graph = SemanticGraphCore(embedding_dim=embedding_dim)
        self.somatic_engine = SomaticValenceEngine()

    def record_interaction(
        self,
        prompt: str,
        solution: str,
        verifier_result: VerifierResult,
        task_family: str,
        loss_val: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Admits an interaction to episodic memory tagged with somatic valence."""
        marker: SomaticMarker = self.somatic_engine.evaluate_episode(
            context_loss=loss_val, verifier_passed=verifier_result.passed
        )
        meta = metadata or {}
        meta.update(
            {
                "risk_score": marker.risk_score,
                "urgency": marker.urgency,
                "cognitive_dissonance": marker.cognitive_dissonance,
                "trust_delta": marker.trust_delta,
            }
        )

        return self.episodic_store.write_record(
            prompt=prompt,
            target_solution=solution,
            verifier_result=verifier_result,
            task_family=task_family,
            metadata=meta,
        )

    def query_top_k(self, query: str, top_k: int = 5) -> list[EpisodicRecord]:
        """Queries verified episodic records by prompt embedding similarity."""
        return self.episodic_store.query_top_k(query, top_k=top_k)

    def __len__(self) -> int:
        return len(self.episodic_store)
