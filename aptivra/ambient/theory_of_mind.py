"""Milestone Genesis: Personal AI Theory-of-Mind Modeling Engine."""

from dataclasses import dataclass


@dataclass
class UserMentalState:
    """Estimated cognitive state of the human user."""

    cognitive_load: float  # [0.0 = relaxed, 1.0 = overloaded]
    current_focus_area: str
    affective_state: str  # "focused", "frustrated", "exploratory"


class TheoryOfMindTracker:
    """Tracks user cognitive load, frustration signals, and conversational intent."""

    def __init__(self) -> None:
        self.state = UserMentalState(
            cognitive_load=0.2,
            current_focus_area="general",
            affective_state="focused",
        )

    def update_from_interaction(
        self, input_length: int, latency_seconds: float, is_correction: bool
    ) -> None:
        """Dynamically adjusts estimated user state based on interaction telemetry."""
        if is_correction:
            self.state.cognitive_load = min(1.0, self.state.cognitive_load + 0.3)
            self.state.affective_state = "frustrated"
        elif latency_seconds < 1.0 and input_length > 100:
            self.state.cognitive_load = min(1.0, self.state.cognitive_load + 0.1)
            self.state.affective_state = "focused"
        else:
            self.state.cognitive_load = max(0.0, self.state.cognitive_load - 0.05)
            self.state.affective_state = "exploratory"
