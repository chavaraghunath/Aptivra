"""Unit tests for evaluation metrics, harness, task stream isolation, and contamination guards."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.eval.harness import (
    ContinualEvalHarness,
    EvaluationInfrastructureError,
    IncrementalEvaluationRun,
    PredictFunction,
)
from aptivra.eval.metrics import (
    compute_average_accuracy,
    compute_backward_transfer,
    compute_confidence_intervals,
    compute_forward_transfer,
    compute_max_forgetting,
)
from aptivra.model.config import ModelConfig
from aptivra.tasks import code_sandbox
from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, VerifierResult
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamSplit
from scripts.run_baselines import run_single_baseline


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

    single_task = np.array([[0.5]])
    assert compute_average_accuracy(single_task) == 0.5
    assert compute_backward_transfer(single_task) is None
    assert compute_forward_transfer(single_task, np.array([0.1])) is None
    assert compute_max_forgetting(single_task, ["only"]) == {"only": 0.0}


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
    expected_family_ids = [
        "T1_modular_arithmetic",
        "T2_string_transforms",
        "T3_list_operations",
        "T4_syntax_matching",
        "T5_boolean_logic",
        "T6_sequence_completion",
        "T7_state_machines",
        "T8_python_microsynth",
        "T9_algo_synth",
        "T10_docstring_code",
        "T11_bug_fixer",
        "T12_type_inferrer",
    ]
    assert [family.family_id for family in stream.families] == expected_family_ids

    for family_id in expected_family_ids:
        split = stream.get_split(family_id)
        assert (len(split.train), len(split.val), len(split.test)) == (20, 10, 10)
        train_hashes = {stream._instance_hash(instance) for instance in split.train}
        val_hashes = {stream._instance_hash(instance) for instance in split.val}
        test_hashes = {stream._instance_hash(instance) for instance in split.test}
        assert (len(train_hashes), len(val_hashes), len(test_hashes)) == (20, 10, 10)
        assert train_hashes.isdisjoint(val_hashes)
        assert train_hashes.isdisjoint(test_hashes)
        assert val_hashes.isdisjoint(test_hashes)

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


def test_every_family_unique_generator_fails_clearly_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = ContinualTaskStream(1, 1, 1, seed=42)

    for family in stream.families:
        sole_instance = family.generate_instances(1, seed=42)[0]

        def finite_generator(
            count: int, seed: int, instance: TaskInstance = sole_instance
        ) -> list[TaskInstance]:
            del seed
            return [instance] * count

        with monkeypatch.context() as scoped:
            scoped.setattr(family, "generate_instances", finite_generator)
            with pytest.raises(ValueError) as exc_info:
                family.generate_unique_instances(2, seed=42, max_attempts=2)

        message = str(exc_info.value)
        assert family.family_id in message
        assert "only 1 of 2 requested unique instances after 2 attempts" in message


def test_eval_harness(tmp_path: Path) -> None:
    stream = ContinualTaskStream(
        num_train_per_task=10,
        num_val_per_task=5,
        num_test_per_task=5,
        seed=42,
    )
    harness = ContinualEvalHarness(stream)

    # Dummy predictor function that returns target solution for test instances
    def oracle_predict(prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        del stop_token_ids
        # Search all instances for matching prompt
        for fam in stream.families:
            split = stream.get_split(fam.family_id)
            for inst in split.test + split.val:
                if inst.prompt == prompt:
                    return inst.target_solution
        return "WRONG_ANSWER"

    def random_predict(prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        del prompt, stop_token_ids
        return "RANDOM_GUESS"

    # Test single task evaluation
    acc = harness.evaluate_task(oracle_predict, task_idx=0, use_val_split=False)
    assert acc == 1.0

    acc_val = harness.evaluate_task(oracle_predict, task_idx=0, use_val_split=True)
    assert acc_val == 1.0

    evaluation = harness.start_run()
    evaluation.measure_untrained(random_predict)
    for cycle_idx in range(len(stream.families)):
        evaluation.measure_post_cycle(cycle_idx, oracle_predict)
    metrics = evaluation.finalize({"max_new_tokens": 8})
    assert metrics.average_accuracy == 1.0
    assert metrics.metadata["rows_evaluated_per_cycle"] is True
    assert metrics.metadata["split_sizes"]["test_per_task"] == 5

    # Test save_metrics_jsonl
    log_file = tmp_path / "metrics.jsonl"
    harness.save_metrics_jsonl(metrics, log_file)
    assert log_file.exists()

    with open(log_file, encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["average_accuracy"] == 1.0
        assert "backward_transfer" in data
        assert data["metadata"]["harness_version"] == "continual-eval-v3"


def test_harness_refuses_accuracy_when_verifier_could_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = ContinualTaskStream(1, 1, 1, seed=7)
    harness = ContinualEvalHarness(stream)
    evaluation = harness.start_run()
    family = stream.families[8]
    instance = stream.get_split(family.family_id).test[0]

    def fail_temp_directory(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("no writable temporary directory")

    monkeypatch.setattr(code_sandbox.tempfile, "TemporaryDirectory", fail_temp_directory)

    with pytest.raises(EvaluationInfrastructureError, match="could not run"):
        evaluation.measure_untrained(
            lambda _prompt, stop_token_ids: instance.target_solution
        )

    assert np.isnan(evaluation.random_baseline_accs[8])


class _ExactVerifier(MechanicalVerifier):
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        actual = model_output.strip()
        expected = instance.target_solution
        return VerifierResult(actual == expected, expected, actual)


class _ConflictFamily(TaskFamily):
    def __init__(self, family_id: str, answer: str) -> None:
        super().__init__(family_id, family_id, "conflicting synthetic task")
        self.answer = answer
        self._verifier = _ExactVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        return [
            TaskInstance(f"{self.family_id}-{idx}", self.family_id, "same prompt", self.answer)
            for idx in range(count)
        ]


class _ConflictStream:
    def __init__(self) -> None:
        self.families = [_ConflictFamily("first", "A"), _ConflictFamily("second", "B")]
        self.num_train_per_task = 1
        self.num_val_per_task = 1
        self.num_test_per_task = 4
        self.seed = 7
        self._splits = {
            family.family_id: TaskStreamSplit(
                train=family.generate_instances(1, 7),
                val=family.generate_instances(1, 8),
                test=family.generate_instances(4, 9),
            )
            for family in self.families
        }

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]


class _TinyConflictingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def train_answer(self, answer_idx: int) -> None:
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        target = torch.tensor([answer_idx])
        for _ in range(12):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(self.logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

    def predict(self, prompt: str, *, stop_token_ids: tuple[int, ...]) -> str:
        del prompt, stop_token_ids
        return "A" if int(torch.argmax(self.logits).item()) == 0 else "B"


def test_incremental_matrix_measures_real_forgetting(tmp_path: Path) -> None:
    """Conflicting sequential tasks must produce clearly negative measured BWT."""
    stream = _ConflictStream()
    harness = ContinualEvalHarness(stream)  # type: ignore[arg-type]
    model = _TinyConflictingModel()
    torch.save(model.state_dict(), tmp_path / "tiny-initial.pt")
    evaluation = harness.start_run()
    evaluation.measure_untrained(model.predict)

    model.train_answer(0)
    evaluation.measure_post_cycle(0, model.predict)
    model.train_answer(1)
    evaluation.measure_post_cycle(1, model.predict)

    metrics = evaluation.finalize()
    assert metrics.accuracy_matrix[0][0] == 1.0
    assert metrics.accuracy_matrix[1][0] == 0.0
    assert metrics.backward_transfer is not None
    assert metrics.backward_transfer <= -0.9


def test_baseline_entry_point_records_chronological_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real baseline runner must evaluate each row before the next task trains."""
    events: list[tuple[str, int]] = []
    original_train_task = SequentialBaselineTrainer.train_task
    original_measure_post_cycle = IncrementalEvaluationRun.measure_post_cycle

    def train_task_spy(
        trainer: SequentialBaselineTrainer,
        task_idx: int,
        train_instances: list[TaskInstance],
    ) -> dict[str, object]:
        events.append(("train", task_idx))
        return original_train_task(trainer, task_idx, train_instances)

    def measure_post_cycle_spy(
        evaluation: IncrementalEvaluationRun,
        cycle_idx: int,
        predict_fn: PredictFunction,
    ) -> None:
        events.append(("evaluate", cycle_idx))
        original_measure_post_cycle(evaluation, cycle_idx, predict_fn)

    monkeypatch.setattr(SequentialBaselineTrainer, "train_task", train_task_spy)
    monkeypatch.setattr(IncrementalEvaluationRun, "measure_post_cycle", measure_post_cycle_spy)

    stream = _ConflictStream()
    config = ModelConfig(
        vocab_size=128,
        d_model=8,
        n_layers=1,
        n_heads=1,
        d_ff=16,
        max_seq_len=16,
        dropout=0.0,
        device="cpu",
    )
    output_log = tmp_path / "sequential.jsonl"

    run_single_baseline(
        "sequential",
        config,
        stream,  # type: ignore[arg-type]
        output_log,
        max_new_tokens=1,
    )

    record = json.loads(output_log.read_text(encoding="utf-8").strip())
    matrix = record["accuracy_matrix"]
    observed = record["metadata"]["observed_accuracy_cells"]
    assert events.index(("train", 0)) < events.index(("evaluate", 0))
    assert events.index(("evaluate", 0)) < events.index(("train", 1))
    assert events.index(("train", 1)) < events.index(("evaluate", 1))
    assert record["metadata"]["rows_evaluated_per_cycle"] is True
    assert len(matrix) == len(observed) == 2
    for matrix_row, observed_row in zip(matrix, observed, strict=True):
        for value, was_measured in zip(matrix_row, observed_row, strict=True):
            if not was_measured:
                assert value is None


def test_transfer_metrics_are_unmeasured_when_required_cells_are_missing() -> None:
    stream = _ConflictStream()
    harness = ContinualEvalHarness(stream)  # type: ignore[arg-type]
    evaluation = harness.start_run()
    metrics = evaluation.finalize()

    assert metrics.backward_transfer is None
    assert metrics.forward_transfer is None
    assert metrics.average_accuracy is None
    assert metrics.accuracy_matrix == [[None, None], [None, None]]

    model = _TinyConflictingModel()
    baseline_only = harness.start_run()
    baseline_only.measure_untrained(model.predict)
    assert baseline_only.finalize().forward_transfer is None
    with pytest.raises(RuntimeError, match="exactly once"):
        baseline_only.measure_untrained(model.predict)

    out_of_order = harness.start_run()
    with pytest.raises(ValueError, match="Expected cycle 0"):
        out_of_order.measure_post_cycle(1, model.predict)
