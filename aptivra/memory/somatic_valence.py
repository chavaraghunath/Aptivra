"""Milestone Genesis: Tier 4 Somatic Valence & Affective Grounding Core."""

from dataclasses import dataclass


@dataclass
class SomaticMarker:
    """Affective tag encoding risk, urgency, dissonance, and relational trust."""

    risk_score: float  # [0.0 = completely safe, 1.0 = extreme hazard]
    urgency: float  # [0.0 = low priority, 1.0 = critical immediate]
    trust_delta: float  # [-1.0 = trust destroyed, 1.0 = trust reinforced]
    cognitive_dissonance: float  # [0.0 = harmonious, 1.0 = extreme contradiction]


class SomaticValenceEngine:
    """Computes affective markers grounded in Damasio's Somatic Marker hypothesis."""

    def evaluate_episode(self, context_loss: float, verifier_passed: bool) -> SomaticMarker:
        """Evaluates an interaction episode and computes affective valence tags."""
        risk = 0.8 if not verifier_passed else 0.05
        urgency = min(1.0, max(0.0, context_loss * 0.5))

        # Cognitive dissonance spikes when the model is confident (low loss) but fails verification
        if not verifier_passed and context_loss < 0.2:
            dissonance = 0.95
        elif verifier_passed and context_loss > 1.5:
            dissonance = 0.40
        else:
            dissonance = 0.05

        trust = 0.08 if verifier_passed else -0.25

        return SomaticMarker(
            risk_score=risk,
            urgency=urgency,
            trust_delta=trust,
            cognitive_dissonance=dissonance,
        )
