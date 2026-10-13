"""Regression tests preventing verifier false positives from near-miss output."""

import pytest

from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.boolean_logic import BooleanLogicTaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.conversational import ConversationalTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.identity import ModelIdentityTaskFamily
from aptivra.tasks.list_operations import ListOperationsTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.tasks.multitasking import MultiTaskingTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.sequence_completion import SequenceCompletionTaskFamily
from aptivra.tasks.skill_dataset import MCPSkillTaskFamily
from aptivra.tasks.state_machines import StateMachineTaskFamily
from aptivra.tasks.string_transforms import StringTransformTaskFamily
from aptivra.tasks.syntax_matching import SyntaxMatchingTaskFamily
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


@pytest.mark.parametrize(
    "family",
    [
        ModularArithmeticTaskFamily(),
        StringTransformTaskFamily(),
        ListOperationsTaskFamily(),
        SyntaxMatchingTaskFamily(),
        BooleanLogicTaskFamily(),
        SequenceCompletionTaskFamily(),
        StateMachineTaskFamily(),
        ModelIdentityTaskFamily(),
        ConversationalTaskFamily(),
        MultiTaskingTaskFamily(),
        MCPSkillTaskFamily("strict", "strict verifier", "# Strict"),
        TypeInferrerTaskFamily(),
    ],
)
def test_exact_verifiers_accept_normalized_target_and_reject_two_near_misses(
    family: object,
) -> None:
    instance = family.generate_instances(1, seed=42)[0]  # type: ignore[attr-defined]
    verifier = family.verifier  # type: ignore[attr-defined]
    normalized_equivalent = f"  {instance.target_solution.replace(chr(10), '   ')}  "
    assert verifier.verify(instance, normalized_equivalent).passed is True
    assert verifier.verify(instance, f"Answer: {instance.target_solution}").passed is False
    contradictory = verifier.verify(
        instance, f"{instance.target_solution} but this may be wrong"
    )
    assert contradictory.passed is False


@pytest.mark.parametrize(
    "family",
    [AlgoSynthTaskFamily(), DocstringToCodeTaskFamily(), BugFixerTaskFamily()],
)
def test_execution_verifiers_accept_correct_code_and_reject_two_invalid_programs(
    family: object,
) -> None:
    instance = family.generate_instances(1, seed=42)[0]  # type: ignore[attr-defined]
    verifier = family.verifier  # type: ignore[attr-defined]
    signature = instance.target_solution.splitlines()[0]
    assert verifier.verify(instance, instance.target_solution).passed is True
    assert verifier.verify(instance, f"{signature}\n    return None").passed is False
    assert verifier.verify(instance, "```python\ndef unfinished():\n    return 1").passed is False

    missing_metadata = TaskInstance(
        task_id=instance.task_id,
        task_family=instance.task_family,
        prompt=instance.prompt,
        target_solution=instance.target_solution,
    )
    assert verifier.verify(missing_metadata, instance.target_solution).passed is False


def test_python_microsynth_requires_exact_value_or_executable_res_assignment() -> None:
    family = PythonMicroSynthTaskFamily()
    instance = family.generate_instances(1, seed=42)[0]
    assert family.verifier.verify(instance, instance.target_solution).passed is True
    assert family.verifier.verify(instance, "x = 0").passed is False
    wrong_value = int(instance.metadata["expected_result"]) + 1
    assert family.verifier.verify(instance, f"res = {wrong_value}").passed is False
