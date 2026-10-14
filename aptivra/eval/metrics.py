"""Continual Learning Metrics Calculation (Lopez-Paz & Ranzato, 2017)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats


@dataclass
class ContinualLearningMetrics:
    """Dataclass holding evaluated metrics for a continual learning run."""

    accuracy_matrix: list[list[float | None]]
    random_baseline_accs: list[float | None]
    average_accuracy: float | None
    backward_transfer: float | None
    forward_transfer: float | None
    max_forgetting: dict[str, float | None]
    total_flops: int = 0
    t1_store_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_average_accuracy(
    acc_matrix: np.ndarray, task_indices: list[int] | None = None
) -> float | None:
    """Computes final average accuracy A_T across all tasks after training on last task T."""
    num_tasks = acc_matrix.shape[0]
    indices = task_indices if task_indices is not None else list(range(num_tasks))
    if not indices:
        return None
    final_row = acc_matrix[num_tasks - 1, indices]
    if np.isnan(final_row).any():
        return None
    return float(np.mean(final_row))


def compute_backward_transfer(
    acc_matrix: np.ndarray, task_indices: list[int] | None = None
) -> float | None:
    """Computes Backward Transfer (BWT) = (1 / (T-1)) * sum_{j=1}^{T-1} (R_{T,j} - R_{j,j})."""
    num_tasks = acc_matrix.shape[0]
    indices = task_indices if task_indices is not None else list(range(num_tasks))
    prior_indices = [index for index in indices if index < num_tasks - 1]
    if not prior_indices:
        return None

    required = [acc_matrix[num_tasks - 1, j] for j in prior_indices]
    required.extend(acc_matrix[j, j] for j in prior_indices)
    if np.isnan(np.asarray(required, dtype=np.float64)).any():
        return None

    bwt_sum = 0.0
    for j in prior_indices:
        bwt_sum += acc_matrix[num_tasks - 1, j] - acc_matrix[j, j]
    return float(bwt_sum / len(prior_indices))


def compute_forward_transfer(
    acc_matrix: np.ndarray,
    baseline_accs: np.ndarray,
    task_indices: list[int] | None = None,
) -> float | None:
    """Computes Forward Transfer (FWT) = (1 / (T-1)) * sum_{j=2}^T (R_{j-1,j} - b_j)."""
    num_tasks = acc_matrix.shape[0]
    indices = task_indices if task_indices is not None else list(range(num_tasks))
    future_indices = [index for index in indices if index > 0]
    if not future_indices:
        return None

    required_cells = np.asarray(
        [acc_matrix[j - 1, j] for j in future_indices], dtype=np.float64
    )
    required_baselines = np.asarray(
        [baseline_accs[j] for j in future_indices], dtype=np.float64
    )
    if np.isnan(required_cells).any() or np.isnan(required_baselines).any():
        return None

    fwt_sum = 0.0
    for j in future_indices:
        fwt_sum += acc_matrix[j - 1, j] - baseline_accs[j]
    return float(fwt_sum / len(future_indices))


def compute_max_forgetting(
    acc_matrix: np.ndarray,
    task_names: list[str],
    task_indices: list[int] | None = None,
) -> dict[str, float | None]:
    """Computes Maximum Forgetting per task F_j = max_{l in {1..T-1}} (R_{l,j} - R_{T,j})."""
    num_tasks = acc_matrix.shape[0]
    forgetting: dict[str, float | None] = {}

    indices = task_indices if task_indices is not None else list(range(num_tasks))
    for j in indices:
        task_name = task_names[j] if j < len(task_names) else f"Task_{j + 1}"
        if j == num_tasks - 1:
            forgetting[task_name] = (
                None if np.isnan(acc_matrix[j, j]) else 0.0
            )
            continue
        learned_values = acc_matrix[j:num_tasks, j]
        final_acc = acc_matrix[num_tasks - 1, j]
        if np.isnan(learned_values).any() or np.isnan(final_acc):
            forgetting[task_name] = None
            continue
        peak_acc = np.max(learned_values)
        forgetting[task_name] = float(max(0.0, peak_acc - final_acc))

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
