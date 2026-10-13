"""Task stream, generators, verifiers, and dataset management."""

from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
)
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily

__all__ = [
    "TaskInstance",
    "VerifierResult",
    "MechanicalVerifier",
    "TaskFamily",
    "ContinualTaskStream",
    "AlgoSynthTaskFamily",
    "DocstringToCodeTaskFamily",
    "BugFixerTaskFamily",
    "TypeInferrerTaskFamily",
]
