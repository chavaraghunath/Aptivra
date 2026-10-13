"""Unit tests for Real-World Code Synthesis Tasks ($T_9$–$T_{12}$)."""

from aptivra.tasks.algo_synth import AlgoSynthTaskFamily, AlgoSynthVerifier
from aptivra.tasks.bug_fixer import BugFixerTaskFamily, BugFixerVerifier
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily, DocstringToCodeVerifier
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily, TypeInferrerVerifier


def test_algo_synth_task_family() -> None:
    fam = AlgoSynthTaskFamily()
    verifier = AlgoSynthVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_docstring_code_task_family() -> None:
    fam = DocstringToCodeTaskFamily()
    verifier = DocstringToCodeVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_bug_fixer_task_family() -> None:
    fam = BugFixerTaskFamily()
    verifier = BugFixerVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True


def test_type_inferrer_task_family() -> None:
    fam = TypeInferrerTaskFamily()
    verifier = TypeInferrerVerifier()
    instances = fam.generate_instances(count=5, seed=42)

    assert len(instances) == 5
    inst = instances[0]
    res = verifier.verify(inst, inst.target_solution)
    assert res.passed is True
