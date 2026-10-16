"""Verifiable Promotion Gate with Statistical Regression Guardrails."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aptivra.eval.harness import PredictFunction
from aptivra.gate.stats import StatisticalTester
from aptivra.tasks.base import TaskInstance


@dataclass
class PromotionResult:
    """Outcome and statistical diagnosis of candidate checkpoint promotion evaluation."""

    promoted: bool
    p_value: float
    test_used: str
    val_acc_gain: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PromotionGate:
    """Promotion Gate evaluating validation gain and regression test over held-out suite."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def evaluate_candidate(
        self,
        curr_predict_fn: PredictFunction,
        cand_predict_fn: PredictFunction,
        new_task_val_instances: list[TaskInstance],
        held_out_regression_suite: list[TaskInstance],
        verifier_eval_fn: Callable[[PredictFunction, list[TaskInstance]], float],
        loss_eval_fn: Callable[[PredictFunction, list[TaskInstance]], np.ndarray],
    ) -> PromotionResult:
        """Evaluates candidate checkpoint against current baseline.

        Rule: Promoted iff new-task gain > 0 AND p >= alpha on held-out regression suite.
        """
        curr_val_acc = verifier_eval_fn(curr_predict_fn, new_task_val_instances)
        cand_val_acc = verifier_eval_fn(cand_predict_fn, new_task_val_instances)
        curr_losses = loss_eval_fn(curr_predict_fn, held_out_regression_suite)
        cand_losses = loss_eval_fn(cand_predict_fn, held_out_regression_suite)
        return self.evaluate_measurements(
            current_val_accuracy=curr_val_acc,
            candidate_val_accuracy=cand_val_acc,
            current_losses=curr_losses,
            candidate_losses=cand_losses,
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
        """Evaluate a live candidate against cached promoted-state measurements.

        The caller captures the current measurements and a disk snapshot before
        training mutates the live model.  A rejection invokes ``rollback_fn``
        before this method returns, making rejection an actual state transition.
        """
        candidate_val_accuracy = verifier_eval_fn(
            cand_predict_fn, new_task_val_instances
        )
        candidate_losses = loss_eval_fn(cand_predict_fn, held_out_regression_suite)
        result = self.evaluate_measurements(
            current_val_accuracy=current_val_accuracy,
            candidate_val_accuracy=candidate_val_accuracy,
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
        """Apply the promotion rule to measurements from two distinct states."""
        curr_val_acc = current_val_accuracy
        cand_val_acc = candidate_val_accuracy
        val_acc_gain = cand_val_acc - curr_val_acc

        if val_acc_gain <= 0.0:
            return PromotionResult(
                promoted=False,
                p_value=1.0,
                test_used="val_gain_check",
                val_acc_gain=val_acc_gain,
                diagnostics={
                    "reason": f"No positive val gain ({cand_val_acc:.3f} <= {curr_val_acc:.3f})",
                    "curr_val_acc": curr_val_acc,
                    "cand_val_acc": cand_val_acc,
                },
            )

        # 2. Condition 2: Statistical regression test on held-out suite
        if len(current_losses) == 0:
            # First task in stream: promote automatically if val gain > 0
            return PromotionResult(
                promoted=True,
                p_value=1.0,
                test_used="first_task_bypass",
                val_acc_gain=val_acc_gain,
                diagnostics={"reason": "First task in stream, no regression suite yet."},
            )

        if len(current_losses) != len(candidate_losses):
            raise ValueError("Current and candidate loss arrays must have equal length.")

        curr_losses = current_losses
        cand_losses = candidate_losses

        no_regression, p_val, test_used = StatisticalTester.evaluate_paired_regression(
            current_losses=curr_losses,
            candidate_losses=cand_losses,
            alpha=self.alpha,
        )

        promoted = no_regression and (val_acc_gain > 0.0)

        rej_msg = f"Rejected: Regression (p={p_val:.4f} < {self.alpha})"
        diagnostics: dict[str, Any] = {
            "curr_val_acc": curr_val_acc,
            "cand_val_acc": cand_val_acc,
            "mean_curr_loss": float(np.mean(curr_losses)),
            "mean_cand_loss": float(np.mean(cand_losses)),
            "loss_delta": float(np.mean(cand_losses) - np.mean(curr_losses)),
            "num_regression_samples": len(current_losses),
            "reason": ("Promoted: Positive gain and no regression" if promoted else rej_msg),
        }

        return PromotionResult(
            promoted=promoted,
            p_value=p_val,
            test_used=test_used,
            val_acc_gain=val_acc_gain,
            diagnostics=diagnostics,
        )
