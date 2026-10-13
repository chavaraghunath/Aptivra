"""Unit tests for StatisticalTester and PromotionGate."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aptivra.gate.promotion_gate import PromotionGate
from aptivra.gate.stats import StatisticalTester
from aptivra.tasks.base import TaskInstance


def test_statistical_tester_no_regression() -> None:
    # Candidate losses identical or lower -> no regression (p >= 0.05)
    curr_losses = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4])
    cand_losses = np.array([0.48, 0.58, 0.39, 0.69, 0.49, 0.59, 0.38, 0.49, 0.58, 0.39])

    no_reg, p_val, test_name = StatisticalTester.evaluate_paired_regression(
        curr_losses, cand_losses, alpha=0.05
    )
    assert no_reg is True
    assert p_val >= 0.05
    assert "t_test" in test_name or "wilcoxon" in test_name


def test_statistical_tester_significant_regression() -> None:
    # Candidate losses significantly higher -> regression (p < 0.05)
    curr_losses = np.array([0.2, 0.2, 0.21, 0.19, 0.2, 0.21, 0.2, 0.19, 0.2, 0.21])
    cand_losses = np.array([0.8, 0.85, 0.9, 0.78, 0.82, 0.88, 0.81, 0.79, 0.83, 0.86])

    no_reg, p_val, _test_name = StatisticalTester.evaluate_paired_regression(
        curr_losses, cand_losses, alpha=0.05
    )
    assert no_reg is False
    assert p_val < 0.05


def test_promotion_gate_evaluation() -> None:
    gate = PromotionGate(alpha=0.05)
    val_instances = [
        TaskInstance("1", "T1", "p1", "s1"),
        TaskInstance("2", "T1", "p2", "s2"),
    ]
    reg_instances = [
        TaskInstance("3", "T0", "p3", "s3"),
        TaskInstance("4", "T0", "p4", "s4"),
        TaskInstance("5", "T0", "p5", "s5"),
        TaskInstance("6", "T0", "p6", "s6"),
        TaskInstance("7", "T0", "p7", "s7"),
    ]

    def dummy_verifier_eval(predict_fn: Callable[[str], str], insts: list[TaskInstance]) -> float:
        return float(predict_fn("val_score"))

    def dummy_loss_eval(predict_fn: Callable[[str], str], insts: list[TaskInstance]) -> np.ndarray:
        val = float(predict_fn("loss_val"))
        return np.array([val] * len(insts), dtype=np.float64)

    # Case 1: Candidate has validation gain and no regression -> Promoted
    def curr_pred(prompt: str) -> str:
        return "0.5" if prompt == "val_score" else "0.4"

    def cand_pred(prompt: str) -> str:
        return "0.8" if prompt == "val_score" else "0.35"  # lower loss

    res = gate.evaluate_candidate(
        curr_predict_fn=curr_pred,
        cand_predict_fn=cand_pred,
        new_task_val_instances=val_instances,
        held_out_regression_suite=reg_instances,
        verifier_eval_fn=dummy_verifier_eval,
        loss_eval_fn=dummy_loss_eval,
    )
    assert res.promoted is True

    # Case 2: Candidate has severe regression -> Rejected
    def cand_regressed_pred(prompt: str) -> str:
        return "0.8" if prompt == "val_score" else "0.95"  # much higher loss

    res_reg = gate.evaluate_candidate(
        curr_predict_fn=curr_pred,
        cand_predict_fn=cand_regressed_pred,
        new_task_val_instances=val_instances,
        held_out_regression_suite=reg_instances,
        verifier_eval_fn=dummy_verifier_eval,
        loss_eval_fn=dummy_loss_eval,
    )
    assert res_reg.promoted is False
    assert "Rejected" in res_reg.diagnostics["reason"]


class _BinaryOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_logit = nn.Parameter(torch.tensor(-1.0))

    def predict(self, prompt: str) -> str:
        del prompt
        return "B" if self.output_logit.item() > 0 else "A"


def test_rejected_candidate_rolls_live_model_back(tmp_path: Path) -> None:
    gate = PromotionGate(alpha=0.05)
    model = _BinaryOutputModel()
    snapshot_path = tmp_path / "promoted.pt"
    torch.save(model.state_dict(), snapshot_path)
    pre_candidate_output = model.predict("same prompt")

    new_task = [TaskInstance("new", "new", "same prompt", "B")]
    regression = [
        TaskInstance(str(idx), "old", f"old-{idx}", "A") for idx in range(10)
    ]

    def verifier_eval(predict_fn: Callable[[str], str], insts: list[TaskInstance]) -> float:
        return sum(predict_fn(inst.prompt) == inst.target_solution for inst in insts) / len(insts)

    def loss_eval(predict_fn: Callable[[str], str], insts: list[TaskInstance]) -> np.ndarray:
        return np.asarray(
            [float(predict_fn(inst.prompt) != inst.target_solution) for inst in insts]
        )

    current_val_accuracy = verifier_eval(model.predict, new_task)
    current_losses = loss_eval(model.predict, regression)
    with torch.no_grad():
        model.output_logit.fill_(1.0)
    assert model.predict("same prompt") == "B"

    def rollback() -> None:
        model.load_state_dict(torch.load(snapshot_path, weights_only=True))

    result = gate.evaluate_candidate_with_rollback(
        current_val_accuracy=current_val_accuracy,
        current_losses=current_losses,
        cand_predict_fn=model.predict,
        new_task_val_instances=new_task,
        held_out_regression_suite=regression,
        verifier_eval_fn=verifier_eval,
        loss_eval_fn=loss_eval,
        rollback_fn=rollback,
    )

    assert result.promoted is False
    assert model.predict("same prompt") == pre_candidate_output


def test_promotion_measurements_validate_first_task_and_loss_alignment() -> None:
    gate = PromotionGate(alpha=0.05)
    first_task = gate.evaluate_measurements(
        current_val_accuracy=0.0,
        candidate_val_accuracy=1.0,
        current_losses=np.empty(0),
        candidate_losses=np.empty(0),
    )
    assert first_task.promoted is True
    assert first_task.test_used == "first_task_bypass"

    with np.testing.assert_raises_regex(ValueError, "equal length"):
        gate.evaluate_measurements(
            current_val_accuracy=0.0,
            candidate_val_accuracy=1.0,
            current_losses=np.zeros(5),
            candidate_losses=np.zeros(4),
        )
