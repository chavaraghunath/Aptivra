"""Unit tests for evaluation metrics, harness, task stream isolation, and contamination guards."""

import json
from pathlib import Path

import numpy as np
import pytest

from aptivra.eval.harness import ContinualEvalHarness
from aptivra.eval.metrics import (
    compute_average_accuracy,
    compute_backward_transfer,
    compute_confidence_intervals,
    compute_forward_transfer,
    compute_max_forgetting,
)
from aptivra.tasks.stream import ContinualTaskStream


def test_metrics_calculation() -> None:
    # 3x3 Accuracy Matrix
    # acc_matrix[i, j] = accuracy on task j after training on task i
    acc_matrix = np.array(
        [
            [0.90, 0.00, 0.00],
            [0.80, 0.85, 0.00],
            [0.70, 0.75, 0.80],
        ]
    )
    b = np.array([0.10, 0.15, 0.05])  # Random baseline performance

    # Average Accuracy A_T = mean(acc_matrix[2, :]) = (0.70 + 0.75 + 0.80) / 3 = 0.75
    avg_acc = compute_average_accuracy(acc_matrix)
    assert pytest.approx(avg_acc, abs=1e-4) == 0.75

    # BWT = (1/2) * [(acc_matrix[2,0] - acc_matrix[0,0]) + (acc_matrix[2,1] - acc_matrix[1,1])]
    #     = 0.5 * [(0.70 - 0.90) + (0.75 - 0.85)] = 0.5 * (-0.20 - 0.10) = -0.15
    bwt = compute_backward_transfer(acc_matrix)
    assert pytest.approx(bwt, abs=1e-4) == -0.15

    # FWT = (1/2) * [(acc_matrix[0,1] - b[1]) + (acc_matrix[1,2] - b[2])]
    #     = 0.5 * [(0.00 - 0.15) + (0.00 - 0.05)] = 0.5 * (-0.20) = -0.10
    fwt = compute_forward_transfer(acc_matrix, b)
    assert pytest.approx(fwt, abs=1e-4) == -0.10

    # Max Forgetting per task
    task_names = ["T1", "T2", "T3"]
    forgetting = compute_max_forgetting(acc_matrix, task_names)
    assert pytest.approx(forgetting["T1"], abs=1e-4) == 0.20  # peak 0.90 -> 0.70
    assert pytest.approx(forgetting["T2"], abs=1e-4) == 0.10  # peak 0.85 -> 0.75
    assert pytest.approx(forgetting["T3"], abs=1e-4) == 0.00  # peak 0.80 -> 0.80


def test_confidence_intervals() -> None:
    data = [0.80, 0.82, 0.84, 0.78, 0.81]
    mean, margin, std_err = compute_confidence_intervals(data)
    assert pytest.approx(mean, abs=1e-4) == 0.81
    assert margin > 0.0
    assert std_err > 0.0

    # Single element edge case
    single_mean, single_margin, single_err = compute_confidence_intervals([0.5])
    assert single_mean == 0.5
    assert single_margin == 0.0
    assert single_err == 0.0


def test_task_stream_disjoint_splits_and_contamination_guard() -> None:
    stream = ContinualTaskStream(
        num_train_per_task=20,
        num_val_per_task=10,
        num_test_per_task=10,
        seed=42,
    )
    assert len(stream.families) == 8

    # Test split retrieval and isolation
    split_t1 = stream.get_split("T1_modular_arithmetic")
    assert len(split_t1.train) == 20
    assert len(split_t1.val) == 10
    assert len(split_t1.test) == 10

    # Invalid key check
    with pytest.raises(KeyError, match="not found in stream"):
        stream.get_split("INVALID_TASK_KEY")

    # Test held-out regression suite access up to task index 2
    reg_suite = stream.get_held_out_regression_suite(up_to_task_idx=2)
    assert len(reg_suite) == 30  # Tasks 0, 1, 2 (10 each)

    # Test contamination assertion (passes when candidate split is clean)
    clean_candidate_batch = split_t1.train
    stream.verify_no_contamination(clean_candidate_batch, up_to_task_idx=0)

    # Contamination assertion MUST trigger exception when held-out instance is passed
    contaminated_batch = [reg_suite[0]]
    with pytest.raises(AssertionError, match="VIOLATION DETECTED"):
        stream.verify_no_contamination(contaminated_batch, up_to_task_idx=0)


def test_eval_harness(tmp_path: Path) -> None:
    stream = ContinualTaskStream(
        num_train_per_task=10,
        num_val_per_task=5,
        num_test_per_task=5,
        seed=42,
    )
    harness = ContinualEvalHarness(stream)

    # Dummy predictor function that returns target solution for test instances
    def oracle_predict(prompt: str) -> str:
        # Search all instances for matching prompt
        for fam in stream.families:
            split = stream.get_split(fam.family_id)
            for inst in split.test + split.val:
                if inst.prompt == prompt:
                    return inst.target_solution
        return "WRONG_ANSWER"

    def random_predict(prompt: str) -> str:
        return "RANDOM_GUESS"

    # Test single task evaluation
    acc = harness.evaluate_task(oracle_predict, task_idx=0, use_val_split=False)
    assert acc == 1.0

    acc_val = harness.evaluate_task(oracle_predict, task_idx=0, use_val_split=True)
    assert acc_val == 1.0

    # Test evaluate_matrix for all 8 task cycles
    metrics = harness.evaluate_matrix(
        eval_fn_at_cycle=[oracle_predict] * len(stream.families),
        random_baseline_predict_fn=random_predict,
    )
    assert metrics.average_accuracy == 1.0

    # Test save_metrics_jsonl
    log_file = tmp_path / "metrics.jsonl"
    harness.save_metrics_jsonl(metrics, log_file)
    assert log_file.exists()

    with open(log_file, encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["average_accuracy"] == 1.0
        assert "backward_transfer" in data
