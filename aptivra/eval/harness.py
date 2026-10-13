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
            return 0.0

        passed_count = 0
        for inst in instances:
            output = predict_fn(inst.prompt)
            res = verifier.verify(inst, output)
            if res.passed:
                passed_count += 1

        return float(passed_count / len(instances))

    def evaluate_matrix(
        self,
        eval_fn_at_cycle: list[Callable[[str], str]],
        random_baseline_predict_fn: Callable[[str], str] | None = None,
    ) -> ContinualLearningMetrics:
        """Evaluates accuracy matrix R over all cycles T."""
        acc_matrix = np.zeros((self.num_tasks, self.num_tasks), dtype=np.float64)
        b = np.zeros(self.num_tasks, dtype=np.float64)

        # Evaluate random baseline initial performance b_j
        if random_baseline_predict_fn is not None:
            for j in range(self.num_tasks):
                b[j] = self.evaluate_task(random_baseline_predict_fn, j)

        # Evaluate R_{i,j} for cycle i after training on task i
        for i in range(len(eval_fn_at_cycle)):
            predict_fn = eval_fn_at_cycle[i]
            for j in range(i + 1):  # Evaluate on current and all previous tasks
                acc_matrix[i, j] = self.evaluate_task(predict_fn, j)

        avg_acc = compute_average_accuracy(acc_matrix)
        bwt = compute_backward_transfer(acc_matrix)
        fwt = compute_forward_transfer(acc_matrix, b)
        forgetting = compute_max_forgetting(acc_matrix, self.task_names)

        return ContinualLearningMetrics(
            accuracy_matrix=acc_matrix.tolist(),
            random_baseline_accs=b.tolist(),
            average_accuracy=avg_acc,
            backward_transfer=bwt,
            forward_transfer=fwt,
            max_forgetting=forgetting,
        )

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
