"""Continual Learning Evaluation Harness and Logger."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from aptivra.eval.metrics import (
    ContinualLearningMetrics,
    compute_average_accuracy,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_max_forgetting,
)
from aptivra.tasks.base import (
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_model_answer,
)
from aptivra.tasks.stream import TaskStreamSplit

HARNESS_VERSION = "continual-eval-v4"
STANDARD_HARNESS_VERSION = "continual-eval-v5"


class PredictFunction(Protocol):
    """Task-aware generation callback used for every harness measurement."""

    def __call__(
        self,
        prompt: str,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> str:
        """Generate at most the runner-configured token bound."""
        ...


class StandardPredictFunction(Protocol):
    """Instance-aware standard-base predictor using decoded task stop rules."""

    def __call__(self, instance: TaskInstance) -> str:
        """Generate text for one task instance without byte-token assumptions."""
        ...


@dataclass(frozen=True)
class BaselineCompute:
    """Finite, non-negative compute and resident-state evidence for one run."""

    target_tokens: int
    optimizer_steps: int
    fisher_steps: int
    generations: int
    estimated_flops: int
    wall_time_seconds: float
    peak_accelerator_memory_bytes: int
    peak_rss_bytes: int
    replay_bytes: int
    ewc_state_bytes: int
    adapter_bytes: int
    total_parameters: int
    peak_trainable_parameters: int

    def __post_init__(self) -> None:
        integer_values = (
            self.target_tokens,
            self.optimizer_steps,
            self.fisher_steps,
            self.generations,
            self.estimated_flops,
            self.peak_accelerator_memory_bytes,
            self.peak_rss_bytes,
            self.replay_bytes,
            self.ewc_state_bytes,
            self.adapter_bytes,
            self.total_parameters,
            self.peak_trainable_parameters,
        )
        if any(isinstance(value, bool) or value < 0 for value in integer_values):
            raise ValueError("baseline compute counters must be non-negative integers")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0.0:
            raise ValueError("baseline wall time must be finite and non-negative")


@dataclass(frozen=True)
class StandardEvaluationCell:
    """One non-pooled continual-evaluation cell and its derived metrics."""

    accuracy_matrix: list[list[float | None]]
    frozen_base_accs: list[float | None]
    average_accuracy: float | None
    backward_transfer: float | None
    forward_transfer: float | None
    max_forgetting: dict[str, float | None]
    aggregate_max_forgetting: float | None


@dataclass(frozen=True)
class StandardBaselineRecord:
    """Versioned lane-3 record binding results to method, seed, and compute."""

    schema_version: str
    method: str
    seed: int
    stream_id: str
    schedule_sha256: str
    in_template: StandardEvaluationCell
    template_ood: StandardEvaluationCell | None
    task_losses: tuple[float, ...]
    compute: BaselineCompute
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != "aptivra.standard_baseline_run.v1":
            raise ValueError("unsupported standard baseline record schema")
        if not self.method:
            raise ValueError("baseline method must be non-empty")
        if len(self.schedule_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.schedule_sha256
        ):
            raise ValueError("schedule_sha256 must be 64 lowercase hexadecimal characters")
        if not self.task_losses or any(not math.isfinite(loss) for loss in self.task_losses):
            raise ValueError("every task must carry a finite loss record")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this immutable evidence record."""
        return asdict(self)


class EvaluationInfrastructureError(RuntimeError):
    """Raised when an evaluation item could not be executed reliably."""


class TaskStream(Protocol):
    """Structural task-stream interface consumed by the evaluator."""

    families: list[TaskFamily]
    num_train_per_task: int
    num_val_per_task: int
    num_test_per_task: int

    def get_split(self, family_id: str) -> TaskStreamSplit:
        """Return the materialized train/validation/test split."""
        ...


def _optional_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    """Convert an internal NaN-marked matrix to JSON-safe optional floats."""
    return [[None if np.isnan(value) else float(value) for value in row] for row in matrix]


def _optional_vector(vector: np.ndarray) -> list[float | None]:
    """Convert an internal NaN-marked vector to JSON-safe optional floats."""
    return [None if np.isnan(value) else float(value) for value in vector]


class ContinualEvalHarness:
    """Evaluation harness running predictions against mechanical verifiers across stream tasks."""

    def __init__(self, task_stream: TaskStream) -> None:
        self.stream = task_stream
        self.num_tasks = len(self.stream.families)
        self.task_names = [f.family_id for f in self.stream.families]
        get_template_ood = getattr(self.stream, "get_template_ood_split", None)
        self.template_ood_task_indices = (
            [
                index
                for index, family in enumerate(self.stream.families)
                if get_template_ood(family.family_id)
            ]
            if callable(get_template_ood)
            else []
        )

    def evaluate_task(
        self,
        predict_fn: PredictFunction,
        task_idx: int,
        use_val_split: bool = False,
    ) -> float:
        """Evaluates model predict_fn on a specific task family split."""
        family = self.stream.families[task_idx]
        split = self.stream.get_split(family.family_id)
        instances: list[TaskInstance] = split.val if use_val_split else split.test
        if not instances:
            raise ValueError(
                f"Cannot measure task '{family.family_id}': evaluation split is empty."
            )

        return self.evaluate_instances(predict_fn, instances)

    def evaluate_template_ood_task(self, predict_fn: PredictFunction, task_idx: int) -> float:
        """Evaluate one code family on its separate held-out-template cell."""
        family = self.stream.families[task_idx]
        get_template_ood = getattr(self.stream, "get_template_ood_split", None)
        if not callable(get_template_ood):
            raise ValueError("The task stream does not provide template-OOD splits.")
        instances: list[TaskInstance] = get_template_ood(family.family_id)
        if not instances:
            raise ValueError(f"Task family '{family.family_id}' has no template-OOD split.")
        return self.evaluate_instances(predict_fn, instances)

    @staticmethod
    def verify_prediction(
        predict_fn: PredictFunction,
        family: TaskFamily,
        instance: TaskInstance,
    ) -> tuple[str, VerifierResult]:
        """Generate, task-truncate, and verify one answer at the harness boundary."""
        raw_output = predict_fn(
            instance.prompt,
            stop_token_ids=family.stop_token_ids,
        )
        result = ContinualEvalHarness.verify_generated_output(family, instance, raw_output)
        return raw_output, result

    @staticmethod
    def verify_generated_output(
        family: TaskFamily, instance: TaskInstance, raw_output: str
    ) -> VerifierResult:
        """Apply canonical answer extraction and verification to generated text."""
        answer = extract_model_answer(raw_output, family.answer_stop_characters)
        result = family.verifier.verify(instance, answer)
        if result.execution_status == "infrastructure_error":
            raise EvaluationInfrastructureError(
                f"Evaluation item '{instance.task_id}' could not run: "
                f"{result.error_message or 'unknown infrastructure failure'}"
            )
        return result

    def predict_answer(
        self,
        predict_fn: PredictFunction,
        instance: TaskInstance,
    ) -> str:
        """Return one extracted answer using the instance family's declared rule."""
        family = self._family(instance.task_family)
        _, answer = self._generate_answer(predict_fn, family, instance)
        return answer

    @staticmethod
    def _generate_answer(
        predict_fn: PredictFunction,
        family: TaskFamily,
        instance: TaskInstance,
    ) -> tuple[str, str]:
        """Apply the family stop rule once to generation and extraction."""
        raw_output = predict_fn(
            instance.prompt,
            stop_token_ids=family.stop_token_ids,
        )
        answer = extract_model_answer(raw_output, family.answer_stop_characters)
        return raw_output, answer

    def evaluate_instances(
        self,
        predict_fn: PredictFunction,
        instances: list[TaskInstance],
    ) -> float:
        """Evaluate generated answers, refusing a score when any item cannot run."""
        if not instances:
            return 1.0
        passed_count = 0
        for instance in instances:
            family = self._family(instance.task_family)
            _, result = self.verify_prediction(predict_fn, family, instance)
            passed_count += int(result.passed)
        return float(passed_count / len(instances))

    def _family(self, family_id: str) -> TaskFamily:
        for family in self.stream.families:
            if family.family_id == family_id:
                return family
        raise KeyError(f"Task family '{family_id}' not found in evaluation stream.")

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
        self.template_ood_accuracy_matrix = np.full(
            (harness.num_tasks, harness.num_tasks), np.nan, dtype=np.float64
        )
        self.template_ood_random_baseline_accs = np.full(
            harness.num_tasks, np.nan, dtype=np.float64
        )
        self._next_cycle = 0

    def measure_untrained(self, predict_fn: PredictFunction) -> None:
        """Measure the real, initialized-but-untrained baseline ``b_j``."""
        if self._next_cycle != 0 or not np.isnan(self.random_baseline_accs).all():
            raise RuntimeError(
                "The untrained baseline must be measured exactly once before training."
            )
        for task_idx in range(self.harness.num_tasks):
            self.random_baseline_accs[task_idx] = self.harness.evaluate_task(predict_fn, task_idx)
        for task_idx in self.harness.template_ood_task_indices:
            self.template_ood_random_baseline_accs[task_idx] = (
                self.harness.evaluate_template_ood_task(predict_fn, task_idx)
            )

    def measure_post_cycle(self, cycle_idx: int, predict_fn: PredictFunction) -> None:
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
        template_ood_indices = set(self.harness.template_ood_task_indices)
        for task_idx in range(cycle_idx + 1):
            if task_idx in template_ood_indices:
                self.template_ood_accuracy_matrix[cycle_idx, task_idx] = (
                    self.harness.evaluate_template_ood_task(predict_fn, task_idx)
                )
        if next_task_idx in template_ood_indices:
            self.template_ood_accuracy_matrix[cycle_idx, next_task_idx] = (
                self.harness.evaluate_template_ood_task(predict_fn, next_task_idx)
            )
        self._next_cycle += 1

    def finalize(self, metadata: dict[str, Any] | None = None) -> ContinualLearningMetrics:
        """Compute metrics, leaving any metric with missing required cells unmeasured."""
        observed_mask = (~np.isnan(self.accuracy_matrix)).tolist()
        headline_indices = getattr(
            self.harness.stream,
            "headline_task_indices",
            list(range(self.harness.num_tasks)),
        )
        family_roles = getattr(
            self.harness.stream,
            "family_roles",
            dict.fromkeys(self.harness.task_names, "skill"),
        )
        split_sizes = {
            "train_per_task": self.harness.stream.num_train_per_task,
            "val_per_task": self.harness.stream.num_val_per_task,
            "test_per_task": self.harness.stream.num_test_per_task,
        }
        template_ood_size = getattr(
            self.harness.stream,
            "num_template_ood_test_per_code_family",
            0,
        )
        if template_ood_size:
            split_sizes["template_ood_test_per_code_family"] = template_ood_size
        run_metadata: dict[str, Any] = dict(metadata or {})
        run_metadata.update(
            {
                "harness_version": HARNESS_VERSION,
                "rows_evaluated_per_cycle": True,
                "evaluation_protocol": "incremental_after_each_task",
                "split_sizes": split_sizes,
                "task_count": self.harness.num_tasks,
                "task_stream_version": getattr(self.harness.stream, "version", "v1"),
                "family_roles": family_roles,
                "headline_family_ids": [
                    self.harness.task_names[index] for index in headline_indices
                ],
                "random_baseline_measured": bool(not np.isnan(self.random_baseline_accs).any()),
                "forward_transfer_cells_measured": bool(
                    all(
                        not np.isnan(self.accuracy_matrix[idx - 1, idx])
                        for idx in range(1, self.harness.num_tasks)
                    )
                ),
                "observed_accuracy_cells": observed_mask,
                "template_ood_evaluation": {
                    "cell_label": "template-OOD",
                    "family_ids": [
                        self.harness.task_names[index]
                        for index in self.harness.template_ood_task_indices
                    ],
                    "pooled_with_in_template_or_headline_metrics": False,
                    "accuracy_matrix": _optional_matrix(self.template_ood_accuracy_matrix),
                    "random_baseline_accs": _optional_vector(
                        self.template_ood_random_baseline_accs
                    ),
                },
            }
        )
        return ContinualLearningMetrics(
            accuracy_matrix=_optional_matrix(self.accuracy_matrix),
            random_baseline_accs=_optional_vector(self.random_baseline_accs),
            average_accuracy=compute_average_accuracy(self.accuracy_matrix, headline_indices),
            backward_transfer=compute_backward_transfer(self.accuracy_matrix, headline_indices),
            forward_transfer=compute_forward_transfer(
                self.accuracy_matrix,
                self.random_baseline_accs,
                headline_indices,
            ),
            max_forgetting=compute_max_forgetting(
                self.accuracy_matrix,
                self.harness.task_names,
                headline_indices,
            ),
            metadata=run_metadata,
        )


def _aggregate_forgetting(values: dict[str, float | None]) -> float | None:
    measured = [value for value in values.values() if value is not None]
    return float(np.mean(np.asarray(measured, dtype=np.float64))) if measured else None


class StandardContinualEvalHarness:
    """ADR-0011 harness boundary with instance-aware decoded stopping."""

    def __init__(self, task_stream: TaskStream) -> None:
        self.stream = task_stream
        self.num_tasks = len(task_stream.families)
        if self.num_tasks == 0:
            raise ValueError("standard continual evaluation requires at least one task")
        self.task_names = [family.family_id for family in task_stream.families]
        get_template_ood = getattr(task_stream, "get_template_ood_split", None)
        self.template_ood_task_indices = (
            [
                index
                for index, family in enumerate(task_stream.families)
                if get_template_ood(family.family_id)
            ]
            if callable(get_template_ood)
            else []
        )

    def _family(self, family_id: str) -> TaskFamily:
        for family in self.stream.families:
            if family.family_id == family_id:
                return family
        raise KeyError(f"Task family '{family_id}' not found in evaluation stream.")

    def evaluate_instances(
        self,
        predict_fn: StandardPredictFunction,
        instances: list[TaskInstance],
    ) -> float:
        """Evaluate instance-aware predictions and fail on verifier infrastructure errors."""
        if not instances:
            raise ValueError("standard evaluation cannot score an empty split")
        passed_count = 0
        for instance in instances:
            family = self._family(instance.task_family)
            raw_output = predict_fn(instance)
            result = ContinualEvalHarness.verify_generated_output(family, instance, raw_output)
            passed_count += int(result.passed)
        return float(passed_count / len(instances))

    def evaluate_task(
        self,
        predict_fn: StandardPredictFunction,
        task_idx: int,
        *,
        use_val_split: bool = False,
    ) -> float:
        """Evaluate a chronological in-template task cell."""
        family = self.stream.families[task_idx]
        split = self.stream.get_split(family.family_id)
        instances = split.val if use_val_split else split.test
        if not instances:
            raise ValueError(
                f"Cannot measure task '{family.family_id}': evaluation split is empty."
            )
        return self.evaluate_instances(predict_fn, instances)

    def evaluate_template_ood_task(
        self,
        predict_fn: StandardPredictFunction,
        task_idx: int,
    ) -> float:
        """Evaluate one separate template-OOD cell without pooling it."""
        family = self.stream.families[task_idx]
        get_template_ood = getattr(self.stream, "get_template_ood_split", None)
        if not callable(get_template_ood):
            raise ValueError("The task stream does not provide template-OOD splits.")
        instances: list[TaskInstance] = get_template_ood(family.family_id)
        if not instances:
            raise ValueError(f"Task family '{family.family_id}' has no template-OOD split.")
        return self.evaluate_instances(predict_fn, instances)

    def start_run(self) -> "StandardIncrementalEvaluationRun":
        """Start one complete standard-base chronological run."""
        return StandardIncrementalEvaluationRun(self)

    @staticmethod
    def save_record_jsonl(record: StandardBaselineRecord, log_path: Path) -> None:
        """Append one finite versioned standard-base result record."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")


class StandardIncrementalEvaluationRun:
    """Fail-closed v5 chronological matrices for an instance-aware predictor."""

    def __init__(self, harness: StandardContinualEvalHarness) -> None:
        self.harness = harness
        shape = (harness.num_tasks, harness.num_tasks)
        self.accuracy_matrix = np.full(shape, np.nan, dtype=np.float64)
        self.frozen_base_accs = np.full(harness.num_tasks, np.nan, dtype=np.float64)
        self.template_ood_accuracy_matrix = np.full(shape, np.nan, dtype=np.float64)
        self.template_ood_frozen_base_accs = np.full(harness.num_tasks, np.nan, dtype=np.float64)
        self._next_cycle = 0

    def measure_frozen(self, predict_fn: StandardPredictFunction) -> None:
        """Measure the frozen pretrained reference vector exactly once."""
        if self._next_cycle != 0 or not np.isnan(self.frozen_base_accs).all():
            raise RuntimeError(
                "The frozen-base vector must be measured exactly once before training."
            )
        for task_idx in range(self.harness.num_tasks):
            self.frozen_base_accs[task_idx] = self.harness.evaluate_task(predict_fn, task_idx)
        for task_idx in self.harness.template_ood_task_indices:
            self.template_ood_frozen_base_accs[task_idx] = self.harness.evaluate_template_ood_task(
                predict_fn, task_idx
            )

    def measure_post_cycle(
        self,
        cycle_idx: int,
        predict_fn: StandardPredictFunction,
    ) -> None:
        """Measure retained, acquired, and next-unseen cells before later mutation."""
        if np.isnan(self.frozen_base_accs).any():
            raise RuntimeError("measure the frozen-base vector before any training cycle")
        if cycle_idx != self._next_cycle:
            raise ValueError(
                f"Expected cycle {self._next_cycle}, got {cycle_idx}; rows must be chronological."
            )
        measured_indices = list(range(cycle_idx + 1))
        if cycle_idx + 1 < self.harness.num_tasks:
            measured_indices.append(cycle_idx + 1)
        template_indices = set(self.harness.template_ood_task_indices)
        for task_idx in measured_indices:
            self.accuracy_matrix[cycle_idx, task_idx] = self.harness.evaluate_task(
                predict_fn, task_idx
            )
            if task_idx in template_indices:
                self.template_ood_accuracy_matrix[cycle_idx, task_idx] = (
                    self.harness.evaluate_template_ood_task(predict_fn, task_idx)
                )
        self._next_cycle += 1

    def _cell(
        self,
        matrix: np.ndarray,
        baseline: np.ndarray,
        indices: list[int],
    ) -> StandardEvaluationCell:
        forgetting = compute_max_forgetting(
            matrix,
            self.harness.task_names,
            indices,
        )
        return StandardEvaluationCell(
            accuracy_matrix=_optional_matrix(matrix),
            frozen_base_accs=_optional_vector(baseline),
            average_accuracy=compute_average_accuracy(matrix, indices),
            backward_transfer=compute_backward_transfer(matrix, indices),
            forward_transfer=compute_forward_transfer(matrix, baseline, indices),
            max_forgetting=forgetting,
            aggregate_max_forgetting=_aggregate_forgetting(forgetting),
        )

    def finalize(
        self,
        *,
        method: str,
        seed: int,
        schedule_sha256: str,
        task_losses: tuple[float, ...],
        compute: BaselineCompute,
        metadata: dict[str, Any] | None = None,
    ) -> StandardBaselineRecord:
        """Require a complete run and derive both non-pooled metric cells."""
        if self._next_cycle != self.harness.num_tasks:
            raise RuntimeError(
                f"Cannot finalize incomplete run: {self._next_cycle}/"
                f"{self.harness.num_tasks} cycles measured."
            )
        if np.isnan(self.frozen_base_accs).any():
            raise RuntimeError("Cannot finalize without a complete frozen-base vector.")
        headline_indices = list(
            getattr(
                self.harness.stream,
                "headline_task_indices",
                list(range(self.harness.num_tasks)),
            )
        )
        family_roles = getattr(
            self.harness.stream,
            "family_roles",
            dict.fromkeys(self.harness.task_names, "skill"),
        )
        split_sizes_by_family = {
            family.family_id: {
                "train": len(self.harness.stream.get_split(family.family_id).train),
                "val": len(self.harness.stream.get_split(family.family_id).val),
                "test": len(self.harness.stream.get_split(family.family_id).test),
            }
            for family in self.harness.stream.families
        }
        stream_id = str(
            getattr(
                self.harness.stream,
                "stream_id",
                getattr(self.harness.stream, "version", "unspecified"),
            )
        )
        run_metadata: dict[str, Any] = dict(metadata or {})
        run_metadata.update(
            {
                "harness_version": STANDARD_HARNESS_VERSION,
                "prediction_boundary": "task_instance_decoded_stops",
                "rows_evaluated_per_cycle": True,
                "evaluation_protocol": "incremental_after_each_task",
                "split_sizes_by_family": split_sizes_by_family,
                "family_roles": family_roles,
                "headline_family_ids": [
                    self.harness.task_names[index] for index in headline_indices
                ],
                "template_ood_pooled_with_in_template": False,
            }
        )
        template_indices = list(self.harness.template_ood_task_indices)
        return StandardBaselineRecord(
            schema_version="aptivra.standard_baseline_run.v1",
            method=method,
            seed=seed,
            stream_id=stream_id,
            schedule_sha256=schedule_sha256,
            in_template=self._cell(
                self.accuracy_matrix,
                self.frozen_base_accs,
                headline_indices,
            ),
            template_ood=(
                self._cell(
                    self.template_ood_accuracy_matrix,
                    self.template_ood_frozen_base_accs,
                    template_indices,
                )
                if template_indices
                else None
            ),
            task_losses=task_losses,
            compute=compute,
            metadata=run_metadata,
        )
