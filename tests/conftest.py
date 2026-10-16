"""Shared pytest configuration for imports and mutation-safe device discovery."""

import os
import sys
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _force_cpu_for_mutation_tests(
    environment: Mapping[str, str], torch_module: Any
) -> bool:
    """Disable accelerator discovery only inside mutmut-owned pytest processes."""
    if "MUTANT_UNDER_TEST" not in environment:
        return False

    @wraps(torch_module.cuda.is_available)
    def cuda_unavailable() -> bool:
        return False

    @wraps(torch_module.backends.mps.is_available)
    def mps_unavailable() -> bool:
        return False

    torch_module.cuda.is_available = cuda_unavailable
    torch_module.backends.mps.is_available = mps_unavailable
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Prevent forked mutation workers from inheriting initialized GPU state."""
    del config
    _force_cpu_for_mutation_tests(os.environ, torch)
