"""Behavioral tests for mutation-only accelerator isolation."""

from types import SimpleNamespace

from conftest import _force_cpu_for_mutation_tests


def _available() -> bool:
    return True


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=_available)),
    )


def test_normal_pytest_keeps_accelerator_discovery_unchanged() -> None:
    torch_module = _fake_torch()

    changed = _force_cpu_for_mutation_tests({}, torch_module)

    assert changed is False
    assert torch_module.cuda.is_available() is True
    assert torch_module.backends.mps.is_available() is True


def test_mutmut_environment_forces_cpu_even_when_marker_is_empty() -> None:
    torch_module = _fake_torch()

    changed = _force_cpu_for_mutation_tests({"MUTANT_UNDER_TEST": ""}, torch_module)

    assert changed is True
    assert torch_module.cuda.is_available() is False
    assert torch_module.backends.mps.is_available() is False
