"""Fail-closed ADR-0011 promotion-gate boundary tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from aptivra.gate.promotion_gate import PromotionGate
from aptivra.tasks.base import TaskInstance


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"minimum_regression_samples": 1}, "two paired"),
        ({"minimum_regression_samples": True}, "two paired"),
        ({"minimum_validation_gain": -0.1}, "validation gain"),
        ({"minimum_validation_gain": float("inf")}, "validation gain"),
        ({"loss_noninferiority_margin": -0.1}, "margin"),
        ({"loss_noninferiority_margin": float("nan")}, "margin"),
    ],
)
def test_promotion_configuration_rejects_unsafe_boundaries(
    kwargs: dict[str, float | int | bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PromotionGate(**kwargs)


def test_measurement_validation_and_insufficient_evidence_fail_closed() -> None:
    gate = PromotionGate()
    assert PromotionGate(
        minimum_regression_samples=2,
        minimum_validation_gain=0.0,
        loss_noninferiority_margin=0.0,
    ).minimum_regression_samples == 2
    with pytest.raises(ValueError, match="current validation"):
        gate.evaluate_measurements(-0.1, 0.5, np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="candidate validation"):
        gate.evaluate_measurements(0.1, float("nan"), np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="current validation"):
        gate.evaluate_measurements(1.1, 0.5, np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="candidate validation"):
        gate.evaluate_measurements(0.1, -0.1, np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="one-dimensional"):
        gate.evaluate_measurements(0.1, 0.5, np.zeros((1, 5)), np.zeros(5))
    with pytest.raises(ValueError, match="one-dimensional"):
        gate.evaluate_measurements(0.1, 0.5, np.zeros(5), np.zeros((1, 5)))
    with pytest.raises(ValueError, match="equal length"):
        gate.evaluate_measurements(0.1, 0.5, np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError, match="equal length"):
        gate.evaluate_measurements(0.1, 0.5, np.zeros(4), np.zeros(5))
    bad = np.zeros(5)
    bad[2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        gate.evaluate_measurements(0.1, 0.5, bad, np.zeros(5))
    insufficient = gate.evaluate_measurements(0.1, 0.5, np.zeros(4), np.zeros(4))
    assert insufficient.promoted is False
    assert insufficient.p_value == 0.0
    assert insufficient.test_used == "insufficient_paired_evidence"
    assert insufficient.diagnostics["num_regression_samples"] == 4
    with pytest.raises(FrozenInstanceError):
        insufficient.promoted = True  # type: ignore[misc]


def test_paired_noninferiority_uses_upper_bound_not_only_mean() -> None:
    gate = PromotionGate(alpha=0.05, loss_noninferiority_margin=0.0)
    current = np.ones(6)
    clear_gain = gate.evaluate_measurements(
        0.2,
        0.8,
        current,
        np.asarray([0.5, 0.55, 0.6, 0.65, 0.7, 0.75]),
    )
    assert clear_gain.promoted is True
    assert clear_gain.p_value < 0.05
    assert clear_gain.test_used == "paired_noninferiority_t"
    assert clear_gain.diagnostics["loss_delta_upper_bound"] < 0.0
    uncertain = gate.evaluate_measurements(
        0.2,
        0.8,
        current,
        np.asarray([0.1, 0.1, 0.1, 1.8, 1.8, 1.8]),
    )
    assert uncertain.promoted is False
    assert uncertain.diagnostics["loss_delta_upper_bound"] > 0.0
    constant_regression = gate.evaluate_measurements(
        0.2, 0.8, current, np.full(6, 1.1)
    )
    assert constant_regression.promoted is False
    assert constant_regression.p_value == 1.0


def test_paired_noninferiority_reports_exact_registered_statistics() -> None:
    alpha = 0.1
    margin = 0.2
    current = np.asarray([1.0, 1.1, 0.9, 1.2, 0.8])
    candidate = np.asarray([0.9, 1.05, 0.85, 1.0, 0.75])
    result = PromotionGate(
        alpha=alpha,
        minimum_regression_samples=5,
        loss_noninferiority_margin=margin,
    ).evaluate_measurements(0.2, 0.8, current, candidate)
    differences = candidate - current
    mean_difference = float(np.mean(differences))
    standard_error = float(stats.sem(differences))
    critical = float(stats.t.ppf(1.0 - alpha, df=len(differences) - 1))
    expected_upper = mean_difference + critical * standard_error
    expected_statistic = (mean_difference - margin) / standard_error
    expected_p = float(stats.t.cdf(expected_statistic, df=len(differences) - 1))
    assert result.p_value == pytest.approx(expected_p)
    assert result.diagnostics["loss_delta"] == pytest.approx(mean_difference)
    assert result.diagnostics["loss_delta_upper_bound"] == pytest.approx(expected_upper)
    assert result.diagnostics["num_regression_samples"] == 5
    assert result.promoted is (expected_upper <= margin)


def test_candidate_entrypoints_require_verifier_examples_and_skip_rollback_on_pass() -> None:
    gate = PromotionGate()

    def verifier(
        predict: Callable[[str], str], instances: list[TaskInstance]
    ) -> float:
        del predict, instances
        return 1.0

    def losses(
        predict: Callable[[str], str], instances: list[TaskInstance]
    ) -> np.ndarray:
        del predict
        return np.zeros(len(instances))

    def predict(prompt: str) -> str:
        return prompt
    with pytest.raises(ValueError, match="non-empty"):
        gate.evaluate_candidate(predict, predict, [], [], verifier, losses)
    with pytest.raises(ValueError, match="non-empty"):
        gate.evaluate_candidate_with_rollback(
            0.0, np.zeros(5), predict, [], [], verifier, losses, lambda: None
        )

    instances = [TaskInstance(str(index), "family", "prompt", "answer") for index in range(5)]
    rollback_calls = 0

    def rollback() -> None:
        nonlocal rollback_calls
        rollback_calls += 1

    result = gate.evaluate_candidate_with_rollback(
        0.0,
        np.ones(5),
        predict,
        instances,
        instances,
        verifier,
        losses,
        rollback,
    )
    assert result.promoted is True
    assert result.p_value == 0.0
    assert rollback_calls == 0
