"""Regression tests for task-aware extraction at the harness boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from aptivra.eval.harness import ContinualEvalHarness, PredictFunction
from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.stream import TaskStreamSplit
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


class _SingleFamilyStream:
    """Small stream exercising the real harness entry point without all 12 tasks."""

    def __init__(self, family: TaskFamily, instance: TaskInstance) -> None:
        self.families = [family]
        self.num_train_per_task = 1
        self.num_val_per_task = 1
        self.num_test_per_task = 1
        self.seed = 1
        self._split = TaskStreamSplit([instance], [instance], [instance])

    def get_split(self, family_id: str) -> TaskStreamSplit:
        assert family_id == self.families[0].family_id
        return self._split


CODE_FAMILIES = (
    PythonMicroSynthTaskFamily,
    AlgoSynthTaskFamily,
    DocstringToCodeTaskFamily,
    BugFixerTaskFamily,
    TypeInferrerTaskFamily,
)


@pytest.mark.parametrize("family_type", CODE_FAMILIES)
@pytest.mark.parametrize("trailing_garbage", ["", "ignored generated garbage\nmore"])
def test_code_family_multiline_target_passes_through_real_harness(
    family_type: type[TaskFamily],
    trailing_garbage: str,
) -> None:
    family = family_type()
    instance = family.generate_instances(1, seed=42)[0]
    stream = _SingleFamilyStream(family, instance)
    harness = ContinualEvalHarness(stream)
    received_stop_ids: list[tuple[int, ...]] = []

    def predict(_prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        received_stop_ids.append(stop_token_ids)
        return f"{instance.target_solution}\x00{trailing_garbage}"

    assert "\n" in instance.target_solution
    assert harness.evaluate_task(predict, task_idx=0) == 1.0
    assert received_stop_ids == [(0,)]


@pytest.mark.parametrize("family_type", CODE_FAMILIES)
def test_code_family_near_miss_fails_through_real_harness(
    family_type: type[TaskFamily],
) -> None:
    family = family_type()
    instance = family.generate_instances(1, seed=42)[0]
    harness = ContinualEvalHarness(_SingleFamilyStream(family, instance))

    assert harness.evaluate_task(
        lambda _prompt, stop_token_ids: "not the target\x00ignored",
        task_idx=0,
    ) == 0.0


def test_single_line_family_stops_at_newline_before_garbage() -> None:
    family = ModularArithmeticTaskFamily()
    instance = family.generate_instances(1, seed=42)[0]
    harness = ContinualEvalHarness(_SingleFamilyStream(family, instance))
    received_stop_ids: list[tuple[int, ...]] = []

    def predict(_prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        received_stop_ids.append(stop_token_ids)
        return f"{instance.target_solution}\nignored generated garbage"

    assert harness.evaluate_task(predict, task_idx=0) == 1.0
    assert received_stop_ids == [(0, 10)]


def test_generated_answer_extraction_has_no_non_harness_call_sites() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {
        root / "aptivra" / "eval" / "harness.py",
        root / "aptivra" / "tasks" / "base.py",
    }
    offenders: list[Path] = []
    for directory in (root / "aptivra", root / "scripts"):
        for path in directory.rglob("*.py"):
            if path not in allowed and "extract_model_answer(" in path.read_text(
                encoding="utf-8"
            ):
                offenders.append(path.relative_to(root))

    assert offenders == []


def test_promotion_verifier_routes_generated_text_through_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.run_aptivra import verifier_eval_wrapper

    family = ModularArithmeticTaskFamily()
    instance = family.generate_instances(1, seed=42)[0]
    harness = ContinualEvalHarness(_SingleFamilyStream(family, instance))
    calls: list[tuple[PredictFunction, list[TaskInstance]]] = []
    original = harness.evaluate_instances

    def evaluate_spy(
        predict_fn: PredictFunction,
        instances: list[TaskInstance],
    ) -> float:
        calls.append((predict_fn, instances))
        return original(predict_fn, instances)

    def predict(_prompt: str, stop_token_ids: tuple[int, ...]) -> str:
        del stop_token_ids
        return f"{instance.target_solution}\x00ignored generated garbage"

    monkeypatch.setattr(harness, "evaluate_instances", evaluate_spy)

    assert verifier_eval_wrapper(predict, [instance], harness) == 1.0
    assert calls == [(predict, [instance])]
