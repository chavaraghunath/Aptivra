"""Evidence-only expert growth decisions for ADR-0011."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aptivra.experts.router import RouteDecision


class VerifierInfrastructureError(RuntimeError):
    """A verifier could not produce a semantic pass/fail outcome."""


@dataclass(frozen=True)
class VerifierOutcome:
    """One real verifier result; infrastructure failure is distinct from failure."""

    passed: bool | None
    infrastructure_error: str | None = None

    def __post_init__(self) -> None:
        if self.infrastructure_error is None and self.passed is None:
            raise ValueError("verifier outcome must contain pass/fail evidence")
        if self.infrastructure_error is not None and self.passed is not None:
            raise ValueError("infrastructure failure cannot also claim a verifier result")


@dataclass(frozen=True)
class GrowthSettings:
    """Registered novelty/failure thresholds and capacity bound."""

    verifier_failure_threshold: float
    mean_entropy_threshold: float
    minimum_observations: int
    max_experts: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.verifier_failure_threshold <= 1.0:
            raise ValueError("verifier failure threshold must be in [0, 1]")
        if not math.isfinite(self.mean_entropy_threshold) or self.mean_entropy_threshold < 0.0:
            raise ValueError("mean entropy threshold must be finite and non-negative")
        if isinstance(self.minimum_observations, bool) or self.minimum_observations <= 0:
            raise ValueError("minimum observations must be a positive integer")
        if isinstance(self.max_experts, bool) or self.max_experts <= 0:
            raise ValueError("max experts must be a positive integer")


@dataclass(frozen=True)
class GrowthEvidence:
    """Auditable growth inputs derived only from held-out routing and verification."""

    observations: int
    verifier_failures: int
    verifier_failure_rate: float
    mean_router_entropy: float
    abstentions: int
    misroutes: int


@dataclass(frozen=True)
class GrowthDecision:
    """Fail-closed growth outcome."""

    grow: bool
    reason: str
    evidence: GrowthEvidence


def build_growth_evidence(
    routes: tuple[RouteDecision, ...],
    verifier_outcomes: tuple[VerifierOutcome, ...],
    expected_expert_ids: tuple[str | None, ...],
) -> GrowthEvidence:
    """Aggregate real held-out outcomes and propagate verifier infrastructure failure."""
    if not routes or len(routes) != len(verifier_outcomes):
        raise ValueError("routes and verifier outcomes must be non-empty and aligned")
    if len(expected_expert_ids) != len(routes):
        raise ValueError("expected routes must align with measured routes")
    failures = 0
    abstentions = 0
    misroutes = 0
    entropy_total = 0.0
    for route, outcome, expected in zip(
        routes, verifier_outcomes, expected_expert_ids, strict=True
    ):
        if outcome.infrastructure_error is not None:
            raise VerifierInfrastructureError(outcome.infrastructure_error)
        if outcome.passed is None:
            raise RuntimeError("validated verifier outcome unexpectedly lacks a result")
        failures += int(not outcome.passed)
        abstentions += int(route.route_status == "unrouted")
        misroutes += int(route.route_status == "routed" and route.expert_id != expected)
        entropy_total += route.entropy
    observations = len(routes)
    return GrowthEvidence(
        observations=observations,
        verifier_failures=failures,
        verifier_failure_rate=failures / observations,
        mean_router_entropy=entropy_total / observations,
        abstentions=abstentions,
        misroutes=misroutes,
    )


class EvidenceGrowthPolicy:
    """Propose growth only from registered real-evidence thresholds."""

    def __init__(self, settings: GrowthSettings) -> None:
        self.settings = settings

    def decide(self, current_experts: int, evidence: GrowthEvidence) -> GrowthDecision:
        """Return a deterministic proposal; promotion remains a separate gate."""
        if current_experts >= self.settings.max_experts:
            return GrowthDecision(False, "max_experts_reached", evidence)
        if evidence.observations < self.settings.minimum_observations:
            return GrowthDecision(False, "insufficient_evidence", evidence)
        if evidence.verifier_failure_rate >= self.settings.verifier_failure_threshold:
            return GrowthDecision(True, "verifier_failure_threshold", evidence)
        if evidence.mean_router_entropy >= self.settings.mean_entropy_threshold:
            return GrowthDecision(True, "router_entropy_threshold", evidence)
        return GrowthDecision(False, "thresholds_not_met", evidence)
