"""Tier 1 Verifier-Gated Episodic Memory Store."""

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from aptivra.memory.retrieval import VectorEmbeddingIndex
from aptivra.tasks.base import SINGLE_LINE_STOP_CHARACTERS, VerifierResult


@dataclass
class EpisodicRecord:
    """Represents a verified problem-solution execution stored in Tier 1 memory."""

    record_id: str
    task_family: str
    prompt: str
    target_solution: str
    verifier_passed: bool
    retrieval_count: int = 0
    verification_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    answer_stop_characters: tuple[str, ...] = SINGLE_LINE_STOP_CHARACTERS
    training_terminator: str = "\n"


class EpisodicStore:
    """Verifier-Gated Episodic Store holding verified records with vector retrieval."""

    def __init__(self, embedding_dim: int = 128, device: torch.device | None = None) -> None:
        self.records: dict[str, EpisodicRecord] = {}
        self.index = VectorEmbeddingIndex(embedding_dim=embedding_dim, device=device)

    def write_record(
        self,
        prompt: str,
        target_solution: str,
        verifier_result: VerifierResult,
        task_family: str,
        metadata: dict[str, Any] | None = None,
        answer_stop_characters: tuple[str, ...] = SINGLE_LINE_STOP_CHARACTERS,
        training_terminator: str = "\n",
    ) -> bool:
        """Admits record to store IF AND ONLY IF verifier_result.passed is True."""
        if not verifier_result.passed:
            # Strictly reject unverified or failed outputs
            return False

        rec_id = f"t1_{uuid.uuid4().hex[:8]}"
        record = EpisodicRecord(
            record_id=rec_id,
            task_family=task_family,
            prompt=prompt,
            target_solution=target_solution,
            verifier_passed=True,
            metadata=metadata or {},
            answer_stop_characters=answer_stop_characters,
            training_terminator=training_terminator,
        )

        self.records[rec_id] = record

        # Add to vector retrieval index
        emb = self.index.encode(prompt)
        self.index.add_vector(rec_id, emb)

        return True

    def query_top_k(self, query_prompt: str, top_k: int = 5) -> list[EpisodicRecord]:
        """Queries nearest verified records by embedding cosine similarity."""
        if not self.records:
            return []

        query_emb = self.index.encode(query_prompt)
        matches = self.index.search(query_emb, top_k=top_k)

        retrieved: list[EpisodicRecord] = []
        for rec_id, _score in matches:
            if rec_id in self.records:
                record = self.records[rec_id]
                record.retrieval_count += 1
                retrieved.append(record)

        return retrieved

    def get_consolidation_candidates(self, min_retrievals: int = 1) -> list[EpisodicRecord]:
        """Returns frequently retrieved and consistently verified records for consolidation."""
        return [
            rec
            for rec in self.records.values()
            if rec.verifier_passed and rec.retrieval_count >= min_retrievals
        ]

    def __len__(self) -> int:
        return len(self.records)
