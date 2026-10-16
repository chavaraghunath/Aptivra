"""Routed expert identity for the standard PEFT implementation."""

from __future__ import annotations

from dataclasses import dataclass

from aptivra.experts.standard import ExpertManifest, PeftExpertBank


@dataclass(frozen=True)
class RoutedExpert:
    """Stable router-to-PEFT binding with task-family provenance."""

    expert_id: str
    task_family: str
    manifest: ExpertManifest

    def __post_init__(self) -> None:
        if not self.expert_id or not self.task_family:
            raise ValueError("routed expert identity and provenance must be non-empty")
        if self.expert_id != self.manifest.adapter_name:
            raise ValueError("router expert ID must equal the PEFT adapter name")


__all__ = ["PeftExpertBank", "RoutedExpert"]
