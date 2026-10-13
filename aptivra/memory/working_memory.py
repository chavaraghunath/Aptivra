"""Milestone Genesis: Tier 0 Working Memory Attention Cache."""

from dataclasses import dataclass

import torch


@dataclass
class WorkingMemorySlot:
    """Represents a volatile item held in immediate working memory."""

    token_latent: torch.Tensor
    attention_weight: float
    time_created: float
    source_modality: str = "text"


class WorkingMemoryBuffer:
    """High-speed volatile attention cache directly coupled with the Global Workspace."""

    def __init__(self, capacity: int = 256, d_model: int = 512) -> None:
        self.capacity = capacity
        self.d_model = d_model
        self.slots: list[WorkingMemorySlot] = []

    def push(
        self,
        latent: torch.Tensor,
        weight: float = 1.0,
        modality: str = "text",
        timestamp: float = 0.0,
    ) -> None:
        """Pushes a new latent representation into working memory with FIFO decay."""
        if len(self.slots) >= self.capacity:
            self.slots.pop(0)
        self.slots.append(
            WorkingMemorySlot(
                token_latent=latent.detach(),
                attention_weight=weight,
                time_created=timestamp,
                source_modality=modality,
            )
        )

    def get_context_matrix(self) -> torch.Tensor:
        """Returns stacked working memory tensor [1, N, d_model]."""
        if not self.slots:
            return torch.zeros((1, 0, self.d_model))
        latents = [s.token_latent.view(1, -1) for s in self.slots]
        return torch.cat(latents, dim=0).unsqueeze(0)

    def clear(self) -> None:
        """Flushes the working memory buffer."""
        self.slots.clear()

    def __len__(self) -> int:
        return len(self.slots)
