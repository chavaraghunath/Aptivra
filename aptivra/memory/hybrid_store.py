"""Milestone 4.2: Hybrid Replay-Consolidated Tier 1 Memory Store ($T_1$-Hybrid)."""

import random
from typing import Any

import torch

from aptivra.memory.store import EpisodicRecord, EpisodicStore
from aptivra.tasks.base import VerifierResult


class HybridMemoryStore:
    """Hybrid store combining T1 consolidation with raw token replay reservoir."""

    def __init__(
        self,
        embedding_dim: int = 128,
        max_raw_replay_capacity: int = 100,
        raw_replay_ratio: float = 0.05,
        device: torch.device | None = None,
    ) -> None:
        self.episodic_store = EpisodicStore(embedding_dim=embedding_dim, device=device)
        self.raw_replay_buffer: list[dict[str, Any]] = []
        self.max_raw_replay_capacity = max_raw_replay_capacity
        self.raw_replay_ratio = raw_replay_ratio

    def write_sample(
        self,
        prompt: str,
        target_solution: str,
        verifier_result: VerifierResult,
        task_family: str,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Admits sample to T1 episodic store if verified, and maintains raw token reservoir."""
        t1_admitted = self.episodic_store.write_record(
            prompt=prompt,
            target_solution=target_solution,
            verifier_result=verifier_result,
            task_family=task_family,
            metadata=metadata,
        )

        if input_ids is not None and labels is not None:
            sample = {
                "input_ids": input_ids.detach().cpu(),
                "labels": labels.detach().cpu(),
                "task_family": task_family,
            }
            if len(self.raw_replay_buffer) < self.max_raw_replay_capacity:
                self.raw_replay_buffer.append(sample)
            else:
                # Reservoir sampling replacement
                replace_idx = random.randint(0, len(self.raw_replay_buffer) - 1)
                self.raw_replay_buffer[replace_idx] = sample

        return t1_admitted

    def query_top_k(self, query_prompt: str, top_k: int = 5) -> list[EpisodicRecord]:
        """Queries nearest verified T1 records by embedding similarity."""
        return self.episodic_store.query_top_k(query_prompt, top_k=top_k)

    def sample_raw_replay(self, num_samples: int) -> list[dict[str, Any]]:
        """Samples random raw token micro-batches from the replay reservoir."""
        if not self.raw_replay_buffer:
            return []
        k = min(num_samples, len(self.raw_replay_buffer))
        return random.sample(self.raw_replay_buffer, k)

    def __len__(self) -> int:
        return len(self.episodic_store)
