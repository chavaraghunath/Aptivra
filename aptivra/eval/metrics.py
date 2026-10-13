"""Continual Learning Metrics Calculation (Lopez-Paz & Ranzato, 2017)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats


@dataclass
class ContinualLearningMetrics:
    """Dataclass holding evaluated metrics for a continual learning run."""

    accuracy_matrix: list[list[float]]
    random_baseline_accs: list[float]
    average_accuracy: float
    backward_transfer: float
    forward_transfer: float
    max_forgetting: dict[str, float]
    total_flops: int = 0
    t1_store_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_average_accuracy(acc_matrix: np.ndarray) -> float:
    """Computes final average accuracy A_T across all tasks after training on last task T."""
    num_tasks = acc_matrix.shape[0]
    return float(np.mean(acc_matrix[num_tasks - 1, :]))


def compute_backward_transfer(acc_matrix: np.ndarray) -> float:
    """Computes Backward Transfer (BWT) = (1 / (T-1)) * sum_{j=1}^{T-1} (R_{T,j} - R_{j,j})."""
    num_tasks = acc_matrix.shape[0]
    if num_tasks <= 1:
        return 0.0

    bwt_sum = 0.0
    for j in range(num_tasks - 1):
        bwt_sum += acc_matrix[num_tasks - 1, j] - acc_matrix[j, j]
    return float(bwt_sum / (num_tasks - 1))


def compute_forward_transfer(acc_matrix: np.ndarray, baseline_accs: np.ndarray) -> float:
    """Computes Forward Transfer (FWT) = (1 / (T-1)) * sum_{j=2}^T (R_{j-1,j} - b_j)."""
    num_tasks = acc_matrix.shape[0]
    if num_tasks <= 1:
        return 0.0

    fwt_sum = 0.0
    for j in range(1, num_tasks):
        fwt_sum += acc_matrix[j - 1, j] - baseline_accs[j]
    return float(fwt_sum / (num_tasks - 1))


def compute_max_forgetting(acc_matrix: np.ndarray, task_names: list[str]) -> dict[str, float]:
    """Computes Maximum Forgetting per task F_j = max_{l in {1..T-1}} (R_{l,j} - R_{T,j})."""
    num_tasks = acc_matrix.shape[0]
    forgetting: dict[str, float] = {}

    for j in range(num_tasks - 1):
        task_name = task_names[j] if j < len(task_names) else f"Task_{j + 1}"
        peak_acc = np.max(acc_matrix[:num_tasks, j])
        final_acc = acc_matrix[num_tasks - 1, j]
        forgetting[task_name] = float(max(0.0, peak_acc - final_acc))

    if num_tasks > 0:
        last_idx = num_tasks - 1
        last_task = task_names[last_idx] if last_idx < len(task_names) else f"Task_{num_tasks}"
        forgetting[last_task] = 0.0

    return forgetting


def compute_confidence_intervals(data: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Computes sample mean and 95% confidence interval (lower, upper, margin)."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, 0.0, 0.0

    std_err = float(scipy.stats.sem(arr))
    margin = float(std_err * scipy.stats.t.ppf((1 + 0.95) / 2.0, df=n - 1))
    return mean, margin, float(std_err)
