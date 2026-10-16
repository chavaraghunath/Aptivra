"""Continual learning baseline trainers package."""

from aptivra.baselines.base import BaselineTrainer
from aptivra.baselines.ewc import EWCBaselineTrainer
from aptivra.baselines.joint import JointMultitaskBaselineTrainer
from aptivra.baselines.replay import ReplayBaselineTrainer
from aptivra.baselines.sequential import SequentialBaselineTrainer
from aptivra.baselines.standard import (
    BoundedReplayBuffer,
    FrozenBaseTrainer,
    ManualLoRATrainer,
    NaiveSequentialTrainer,
    OnlineEWCTrainer,
    ReplayTrainer,
    StandardBaselineSettings,
    build_five_seed_schedules,
    build_training_schedule,
    load_standard_baseline_settings,
    run_standard_baseline,
    run_standard_baseline_suite,
)

__all__ = [
    "BaselineTrainer",
    "SequentialBaselineTrainer",
    "ReplayBaselineTrainer",
    "EWCBaselineTrainer",
    "JointMultitaskBaselineTrainer",
    "BoundedReplayBuffer",
    "FrozenBaseTrainer",
    "ManualLoRATrainer",
    "NaiveSequentialTrainer",
    "OnlineEWCTrainer",
    "ReplayTrainer",
    "StandardBaselineSettings",
    "build_five_seed_schedules",
    "build_training_schedule",
    "load_standard_baseline_settings",
    "run_standard_baseline",
    "run_standard_baseline_suite",
]
