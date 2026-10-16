"""Evaluation harness and metrics calculation modules."""

from aptivra.eval.harness import ContinualEvalHarness, StandardContinualEvalHarness
from aptivra.eval.metrics import (
    ContinualLearningMetrics,
    compute_backward_transfer,
    compute_confidence_intervals,
    compute_forward_transfer,
    compute_max_forgetting,
)

__all__ = [
    "ContinualLearningMetrics",
    "compute_backward_transfer",
    "compute_forward_transfer",
    "compute_max_forgetting",
    "compute_confidence_intervals",
    "ContinualEvalHarness",
    "StandardContinualEvalHarness",
]
