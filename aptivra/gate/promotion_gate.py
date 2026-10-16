"""Fail-closed paired promotion gate for ADR-0011 consolidation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from aptivra.eval.harness import PredictFunction
from aptivra.tasks.base import TaskInstance


@dataclass(frozen=True)
class PromotionResult:
    """Outcome and complete pre-registered promotion diagnosis."""

    promoted: bool
    p_value: float
    test_used: str
    val_acc_gain: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PromotionGate:
    """Require verifier gain and paired loss non-inferiority with enough evidence."""

    def __init__(
        self,
        alpha: float = 0.05,
        *,
        minimum_regression_samples: int = 5,
        minimum_validation_gain: float = 0.0,
        loss_noninferiority_margin: float = 0.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("promotion alpha must be in (0, 1)")
        if isinstance(minimum_regression_samples, bool) or minimum_regression_samples < 2:
            raise ValueError("promotion requires at least two paired regression samples")
        if not math.isfinite(minimum_validation_gain) or minimum_validation_gain < 0.0:
            raise ValueError("minimum validation gain must be finite and non-negative")
        if not math.isfinite(loss_noninferiority_margin) or loss_noninferiority_margin < 0.0:
            raise ValueError("loss non-inferiority margin must be finite and non-negative")
        self.alpha = alpha
        self.minimum_regression_samples = minimum_regression_samples
        self.minimum_validation_gain = minimum_validation_gain
        self.loss_noninferiority_margin = loss_noninferiority_margin

    def evaluate_candidate(
        self,
        curr_predict_fn: PredictFunction,
        cand_predict_fn: PredictFunction,
        new_task_val_instances: list[TaskInstance],
        held_out_regression_suite: list[TaskInstance],
        verifier_eval_fn: Callable[[PredictFunction, list[TaskInstance]], float],
        loss_eval_fn: Callable[[PredictFunction, list[TaskInstance]], np.ndarray],
    ) -> PromotionResult:
        """Measure current and candidate states through real verifier/loss functions."""
        if not new_task_val_instances:
            raise ValueError("promotion requires non-empty verifier validation evidence")
        current_accuracy = verifier_eval_fn(curr_predict_fn, new_task_val_instances)
        candidate_accuracy = verifier_eval_fn(cand_predict_fn, new_task_val_instances)
        current_losses = loss_eval_fn(curr_predict_fn, held_out_regression_suite)
        candidate_losses = loss_eval_fn(cand_predict_fn, held_out_regression_suite)
        return self.evaluate_measurements(
            current_val_accuracy=current_accuracy,
            candidate_val_accuracy=candidate_accuracy,
            current_losses=current_losses,
            candidate_losses=candidate_losses,
        )

    def evaluate_candidate_with_rollback(
        self,
        current_val_accuracy: float,
        current_losses: np.ndarray,
        cand_predict_fn: PredictFunction,
        new_task_val_instances: list[TaskInstance],
        held_out_regression_suite: list[TaskInstance],
        verifier_eval_fn: Callable[[PredictFunction, list[TaskInstance]], float],
        loss_eval_fn: Callable[[PredictFunction, list[TaskInstance]], np.ndarray],
        rollback_fn: Callable[[], None],
    ) -> PromotionResult:
        """Evaluate a live candidate and atomically invoke caller rollback on rejection."""
        if not new_task_val_instances:
            raise ValueError("promotion requires non-empty verifier validation evidence")
        candidate_accuracy = verifier_eval_fn(cand_predict_fn, new_task_val_instances)
        candidate_losses = loss_eval_fn(cand_predict_fn, held_out_regression_suite)
        result = self.evaluate_measurements(
            current_val_accuracy=current_val_accuracy,
            candidate_val_accuracy=candidate_accuracy,
            current_losses=current_losses,
            candidate_losses=candidate_losses,
        )
        if not result.promoted:
            rollback_fn()
        return result

    def evaluate_measurements(
        self,
        current_val_accuracy: float,
        candidate_val_accuracy: float,
        current_losses: np.ndarray,
        candidate_losses: np.ndarray,
    ) -> PromotionResult:
        """Apply the registered gain and one-sided paired non-inferiority rule."""
        for name, value in (
            ("current validation accuracy", current_val_accuracy),
            ("candidate validation accuracy", candidate_val_accuracy),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        current = np.asarray(current_losses, dtype=np.float64)
        candidate = np.asarray(candidate_losses, dtype=np.float64)
        if current.ndim != 1 or candidate.ndim != 1:
            raise ValueError("Current and candidate loss arrays must be one-dimensional.")
        if len(current) != len(candidate):
            raise ValueError("Current and candidate loss arrays must have equal length.")
        if not np.isfinite(current).all() or not np.isfinite(candidate).all():
            raise ValueError("promotion losses must all be finite")
        gain = candidate_val_accuracy - current_val_accuracy
        if gain <= self.minimum_validation_gain:
            return PromotionResult(
                False,
                1.0,
                "verifier_gain_check",
                gain,
                {
                    "reason": "Rejected: verifier validation gain did not clear the margin",
                    "curr_val_acc": current_val_accuracy,
                    "cand_val_acc": candidate_val_accuracy,
                },
            )
        if len(current) < self.minimum_regression_samples:
            return PromotionResult(
                False,
                0.0,
                "insufficient_paired_evidence",
                gain,
                {
                    "reason": "Rejected: insufficient paired regression evidence",
                    "num_regression_samples": len(current),
                },
            )
        differences = candidate - current
        mean_difference = float(np.mean(differences))
        standard_error = float(stats.sem(differences))
        if standard_error == 0.0 or not math.isfinite(standard_error):
            upper_bound = mean_difference
            p_value = 0.0 if mean_difference <= self.loss_noninferiority_margin else 1.0
        else:
            critical = float(stats.t.ppf(1.0 - self.alpha, df=len(differences) - 1))
            upper_bound = mean_difference + critical * standard_error
            statistic = (mean_difference - self.loss_noninferiority_margin) / standard_error
            p_value = float(stats.t.cdf(statistic, df=len(differences) - 1))
        promoted = upper_bound <= self.loss_noninferiority_margin
        reason = (
            "Promoted: verifier gain and paired non-inferiority passed"
            if promoted
            else "Rejected: paired loss non-inferiority bound exceeded"
        )
        return PromotionResult(
            promoted,
            p_value,
            "paired_noninferiority_t",
            gain,
            {
                "reason": reason,
                "curr_val_acc": current_val_accuracy,
                "cand_val_acc": candidate_val_accuracy,
                "mean_curr_loss": float(np.mean(current)),
                "mean_cand_loss": float(np.mean(candidate)),
                "loss_delta": mean_difference,
                "loss_delta_upper_bound": upper_bound,
                "loss_noninferiority_margin": self.loss_noninferiority_margin,
                "num_regression_samples": len(current),
            },
        )
