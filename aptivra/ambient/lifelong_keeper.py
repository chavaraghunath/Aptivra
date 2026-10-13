"""Milestone Genesis Phase 4: Lifelong Narrative & Multi-Generational Knowledge Keeper."""

from dataclasses import dataclass

import torch


@dataclass
class EpochSummary:
    """Compressed narrative milestone representing multi-month human-AI interaction epochs."""

    epoch_id: str
    time_window_start: float
    time_window_end: float
    core_values_crystallized: list[str]
    significant_milestones: list[str]
    compressed_latent_gist: torch.Tensor


class LifelongNarrativeKeeper:
    """Maintains autobiographical narrative and generational fidelity across decades."""

    def __init__(self, d_model: int = 512) -> None:
        self.d_model = d_model
        self.epochs: list[EpochSummary] = []
        self.crystallized_values: set[str] = set()

    def compress_epoch(
        self,
        epoch_id: str,
        start_time: float,
        end_time: float,
        milestones: list[str],
        discovered_values: list[str],
        representative_latents: list[torch.Tensor],
    ) -> EpochSummary:
        """Compresses thousands of raw episodic events into a compact autobiographical gist."""
        if representative_latents:
            stacked = torch.stack(representative_latents, dim=0)
            gist = torch.mean(stacked, dim=0)
        else:
            gist = torch.zeros(self.d_model)

        self.crystallized_values.update(discovered_values)

        epoch = EpochSummary(
            epoch_id=epoch_id,
            time_window_start=start_time,
            time_window_end=end_time,
            core_values_crystallized=list(discovered_values),
            significant_milestones=list(milestones),
            compressed_latent_gist=gist,
        )
        self.epochs.append(epoch)
        return epoch

    def generate_symbiotic_reflection(self) -> str:
        """Generates holistic wisdom reflecting co-evolutionary companionship with the user."""
        epoch_count = len(self.epochs)
        val_count = len(self.crystallized_values)
        if self.crystallized_values:
            vals_str = ", ".join(sorted(self.crystallized_values)[:5])
        else:
            vals_str = "integrity"
        return (
            f"Lifelong Keeper: Synchronized over {epoch_count} developmental epochs with "
            f"{val_count} crystallized values [{vals_str}]."
        )
