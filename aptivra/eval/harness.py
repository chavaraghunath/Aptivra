"""Continual Learning Evaluation Harness and Logger."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aptivra.eval.metrics import (
    ContinualLearningMetrics,
    compute_average_accuracy,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_max_forgetting,
)
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.stream import ContinualTaskStream

HARNESS_VERSION = "continual-eval-v2"


def _optional_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    """Convert an internal NaN-marked matrix to JSON-safe optional floats."""
    return [
        [None if np.isnan(value) else float(value) for value in row]
        for row in matrix
    ]


def _optional_vector(vector: np.ndarray) -> list[float | None]:
    """Convert an internal NaN-marked vector to JSON-safe optional floats."""
    return [None if np.isnan(value) else float(value) for value in vector]


class ContinualEvalHarness:
    """Evaluation harness running predictions against mechanical verifiers across stream tasks."""

    def __init__(self, task_stream: ContinualTaskStream) -> None:
        self.stream = task_stream
        self.num_tasks = len(self.stream.families)
        self.task_names = [f.family_id for f in self.stream.families]

    def evaluate_task(
        self,
        predict_fn: Callable[[str], str],
        task_idx: int,
        use_val_split: bool = False,
    ) -> float:
        """Evaluates model predict_fn on a specific task family split."""
        family = self.stream.families[task_idx]
        split = self.stream.get_split(family.family_id)
        instances: list[TaskInstance] = split.val if use_val_split else split.test
        verifier = family.verifier

        if not instances:
            raise ValueError(
                f"Cannot measure task '{family.family_id}': evaluation split is empty."
            )

        passed_count = 0
        for inst in instances:
            output = predict_fn(inst.prompt)
            res = verifier.verify(inst, output)
            if res.passed:
                passed_count += 1

        return float(passed_count / len(instances))

    def start_run(self) -> "IncrementalEvaluationRun":
        """Create an incremental run that never retains a live model callback."""
        return IncrementalEvaluationRun(self)

    @staticmethod
    def save_metrics_jsonl(metrics: ContinualLearningMetrics, log_path: Path) -> None:
        """Appends metrics struct as JSON line to log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "average_accuracy": metrics.average_accuracy,
            "backward_transfer": metrics.backward_transfer,
            "forward_transfer": metrics.forward_transfer,
            "max_forgetting": metrics.max_forgetting,
            "total_flops": metrics.total_flops,
            "t1_store_size": metrics.t1_store_size,
            "accuracy_matrix": metrics.accuracy_matrix,
            "random_baseline_accs": metrics.random_baseline_accs,
            "metadata": metrics.metadata,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


class IncrementalEvaluationRun:
    """Records the continual-learning matrix at the time each state exists.

    The untrained baseline is measured before training.  For cycle ``i``, the
    post-training row is measured immediately for tasks ``0..i`` and the
    superdiagonal cell ``R[i, i+1]`` is measured on the next unseen task.  No
    predictor is stored, so later model mutation cannot rewrite earlier rows.
    """

    def __init__(self, harness: ContinualEvalHarness) -> None:
        self.harness = harness
        self.accuracy_matrix = np.full(
            (harness.num_tasks, harness.num_tasks), np.nan, dtype=np.float64
        )
        self.random_baseline_accs = np.full(harness.num_tasks, np.nan, dtype=np.float64)
        self._next_cycle = 0

    def measure_untrained(self, predict_fn: Callable[[str], str]) -> None:
        """Measure the real, initialized-but-untrained baseline ``b_j``."""
        if self._next_cycle != 0 or not np.isnan(self.random_baseline_accs).all():
            raise RuntimeError(
                "The untrained baseline must be measured exactly once before training."
            )
        for task_idx in range(self.harness.num_tasks):
            self.random_baseline_accs[task_idx] = self.harness.evaluate_task(
                predict_fn, task_idx
            )

    def measure_post_cycle(self, cycle_idx: int, predict_fn: Callable[[str], str]) -> None:
        """Measure row ``cycle_idx`` immediately after that task's accepted state."""
        if cycle_idx != self._next_cycle:
            raise ValueError(
                f"Expected cycle {self._next_cycle}, got {cycle_idx}; rows must be chronological."
            )
        for task_idx in range(cycle_idx + 1):
            self.accuracy_matrix[cycle_idx, task_idx] = self.harness.evaluate_task(
                predict_fn, task_idx
            )
        next_task_idx = cycle_idx + 1
        if next_task_idx < self.harness.num_tasks:
            self.accuracy_matrix[cycle_idx, next_task_idx] = self.harness.evaluate_task(
                predict_fn, next_task_idx
            )
        self._next_cycle += 1

    def finalize(self, metadata: dict[str, Any] | None = None) -> ContinualLearningMetrics:
        """Compute metrics, leaving any metric with missing required cells unmeasured."""
        observed_mask = (~np.isnan(self.accuracy_matrix)).tolist()
        run_metadata: dict[str, Any] = dict(metadata or {})
        run_metadata.update(
            {
                "harness_version": HARNESS_VERSION,
                "rows_evaluated_per_cycle": True,
                "evaluation_protocol": "incremental_after_each_task",
                "split_sizes": {
                    "train_per_task": self.harness.stream.num_train_per_task,
                    "val_per_task": self.harness.stream.num_val_per_task,
                    "test_per_task": self.harness.stream.num_test_per_task,
                },
                "task_count": self.harness.num_tasks,
                "random_baseline_measured": bool(
                    not np.isnan(self.random_baseline_accs).any()
                ),
                "forward_transfer_cells_measured": bool(
                    all(
                        not np.isnan(self.accuracy_matrix[idx - 1, idx])
                        for idx in range(1, self.harness.num_tasks)
                    )
                ),
                "observed_accuracy_cells": observed_mask,
            }
        )
        return ContinualLearningMetrics(
            accuracy_matrix=_optional_matrix(self.accuracy_matrix),
            random_baseline_accs=_optional_vector(self.random_baseline_accs),
            average_accuracy=compute_average_accuracy(self.accuracy_matrix),
            backward_transfer=compute_backward_transfer(self.accuracy_matrix),
            forward_transfer=compute_forward_transfer(
                self.accuracy_matrix, self.random_baseline_accs
            ),
            max_forgetting=compute_max_forgetting(
                self.accuracy_matrix, self.harness.task_names
            ),
            metadata=run_metadata,
        )
