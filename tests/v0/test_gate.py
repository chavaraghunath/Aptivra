"""Historical v0 assertion for the superseded first-task promotion bypass."""

import numpy as np

from aptivra.v0.gate.promotion_gate import PromotionGate


def test_promotion_measurements_validate_first_task_bypass() -> None:
    gate = PromotionGate(alpha=0.05)
    first_task = gate.evaluate_measurements(
        current_val_accuracy=0.0,
        candidate_val_accuracy=1.0,
        current_losses=np.empty(0),
        candidate_losses=np.empty(0),
    )
    assert first_task.promoted is True
    assert first_task.test_used == "first_task_bypass"
