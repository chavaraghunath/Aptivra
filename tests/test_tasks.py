"""Unit tests for all 8 task families and mechanical verifiers."""

from aptivra.tasks.boolean_logic import BooleanLogicTaskFamily
from aptivra.tasks.list_operations import ListOperationsTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.sequence_completion import SequenceCompletionTaskFamily
from aptivra.tasks.state_machines import StateMachineTaskFamily
from aptivra.tasks.string_transforms import StringTransformTaskFamily
from aptivra.tasks.syntax_matching import SyntaxMatchingTaskFamily


def test_modular_arithmetic_verifier() -> None:
    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=10, seed=42)
    assert len(instances) == 10

    for inst in instances:
        # Positive verification
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed, f"Failed positive check: {res.error_message}"

        # Negative verification
        wrong_sol = str(int(inst.target_solution) + 1)
        res_neg = family.verifier.verify(inst, wrong_sol)
        assert not res_neg.passed


def test_string_transforms_verifier() -> None:
    family = StringTransformTaskFamily()
    instances = family.generate_instances(count=10, seed=123)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        res_neg = family.verifier.verify(inst, "INVALID_STRING_OUTPUT")
        assert not res_neg.passed


def test_list_operations_verifier() -> None:
    family = ListOperationsTaskFamily()
    instances = family.generate_instances(count=10, seed=999)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        res_neg = family.verifier.verify(inst, "99999 88888 77777")
        assert not res_neg.passed


def test_syntax_matching_verifier() -> None:
    family = SyntaxMatchingTaskFamily()
    instances = family.generate_instances(count=10, seed=777)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        wrong_sol = "INVALID" if inst.target_solution == "VALID" else "VALID"
        res_neg = family.verifier.verify(inst, wrong_sol)
        assert not res_neg.passed


def test_boolean_logic_verifier() -> None:
    family = BooleanLogicTaskFamily()
    instances = family.generate_instances(count=10, seed=555)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        wrong_sol = "FALSE" if inst.target_solution == "TRUE" else "TRUE"
        res_neg = family.verifier.verify(inst, wrong_sol)
        assert not res_neg.passed


def test_sequence_completion_verifier() -> None:
    family = SequenceCompletionTaskFamily()
    instances = family.generate_instances(count=10, seed=111)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        wrong_sol = str(int(inst.target_solution) + 42)
        res_neg = family.verifier.verify(inst, wrong_sol)
        assert not res_neg.passed


def test_state_machines_verifier() -> None:
    family = StateMachineTaskFamily()
    instances = family.generate_instances(count=10, seed=333)
    assert len(instances) == 10

    for inst in instances:
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        res_neg = family.verifier.verify(inst, "S_NONEXISTENT")
        assert not res_neg.passed


def test_python_microsynth_verifier() -> None:
    family = PythonMicroSynthTaskFamily()
    instances = family.generate_instances(count=5, seed=444)
    assert len(instances) == 5

    for inst in instances:
        # Verify direct integer matching
        res = family.verifier.verify(inst, inst.target_solution)
        assert res.passed

        # Verify executable python block matching
        code_output = f"```python\nres = {inst.target_solution}\n```"
        res_code = family.verifier.verify(inst, code_output)
        assert res_code.passed

        # Security check: verify forbidden imports are caught
        dangerous_code = "import os; res = 1"
        res_danger = family.verifier.verify(inst, dangerous_code)
        assert not res_danger.passed
        assert "Security Error" in (res_danger.error_message or "")


def test_task_instance_determinism() -> None:
    family = ModularArithmeticTaskFamily()
    insts1 = family.generate_instances(count=5, seed=42)
    insts2 = family.generate_instances(count=5, seed=42)
    assert [i.prompt for i in insts1] == [i.prompt for i in insts2]
    assert [i.target_solution for i in insts1] == [i.target_solution for i in insts2]
