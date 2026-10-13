"""Continual learning baseline trainers package."""

from aptivra.baselines.base import BaselineTrainer
from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer

__all__ = [
    "BaselineTrainer",
    "SequentialBaselineTrainer",
    "ReplayBaselineTrainer",
    "EWCBaselineTrainer",
    "JointMultitaskBaselineTrainer",
]
