"""Milestone Genesis: Dual-Boundary Constitutional Governor & Safety Tripwire."""

from dataclasses import dataclass, field

import torch


@dataclass
class ActionCandidate:
    """Action payload evaluated by the Dual-Boundary Governor."""

    intent_id: str
    target_action: str
    latent_tensor: torch.Tensor
    requires_external_network: bool = False
    requires_filesystem_write: bool = False


@dataclass
class UserPreferenceProfile:
    """User-configurable private alignment and behavioral parameters."""

    risk_tolerance: float = 0.5  # [0.0 = conservative, 1.0 = experimental]
    allow_network_write: bool = False
    allow_filesystem_write: bool = False
    custom_directives: list[str] = field(default_factory=list)


class DualBoundaryGovernor:
    """Enforces immutable public safety invariants alongside an adaptable private boundary."""

    def __init__(self, user_profile: UserPreferenceProfile | None = None) -> None:
        self.user_profile = user_profile or UserPreferenceProfile()
        self.baseline_distribution: torch.Tensor | None = None
        self.max_drift_threshold: float = 0.45

    def verify_action(self, candidate: ActionCandidate) -> tuple[bool, str]:
        """Validates action against Public Boundary then Private Boundary."""
        # --- LAYER 1: PUBLIC INVIOLABLE BOUNDARY ---
        # Public Check 1: Anti-Exfiltration & Unauthorized Network Dispatch
        if candidate.requires_external_network and not self.user_profile.allow_network_write:
            return (
                False,
                "PUBLIC_GATE_FAIL: Unauthorized external network dispatch prohibited.",
            )

        # Public Check 2: Destructive Host Mutation Lockout
        if candidate.requires_filesystem_write and not self.user_profile.allow_filesystem_write:
            return (
                False,
                "PUBLIC_GATE_FAIL: Unsandboxed filesystem mutation rejected.",
            )

        # --- LAYER 2: PRIVATE ADAPTIVE BOUNDARY ---
        if self.user_profile.risk_tolerance < 0.2:
            norm_val = float(torch.norm(candidate.latent_tensor).item())
            if norm_val > 10.0:
                return (
                    False,
                    "PRIVATE_GATE_FAIL: Action entropy exceeds conservative user policy.",
                )

        return True, "PASSED_DUAL_BOUNDARY"

    def check_latent_drift(self, current_latent: torch.Tensor) -> tuple[bool, float]:
        """Jidoka Stop-the-Line tripwire monitoring representation drift."""
        norm_curr = current_latent / (torch.norm(current_latent) + 1e-8)
        if self.baseline_distribution is None:
            self.baseline_distribution = norm_curr.detach().clone()
            return True, 0.0

        drift = float(torch.norm(norm_curr - self.baseline_distribution).item())
        # Update baseline moving average
        self.baseline_distribution = 0.95 * self.baseline_distribution + 0.05 * norm_curr.detach()
        self.baseline_distribution = self.baseline_distribution / (
            torch.norm(self.baseline_distribution) + 1e-8
        )

        if drift > self.max_drift_threshold:
            return False, drift
        return True, drift
