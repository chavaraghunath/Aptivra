"""Continual Task Stream Manager with Disjoint Splits and Contamination Safeguards."""

import hashlib
from typing import NamedTuple

from aptivra.tasks.algo_synth import AlgoSynthTaskFamily
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.boolean_logic import BooleanLogicTaskFamily
from aptivra.tasks.bug_fixer import BugFixerTaskFamily
from aptivra.tasks.docstring_code import DocstringToCodeTaskFamily
from aptivra.tasks.list_operations import ListOperationsTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.tasks.python_microsynth import PythonMicroSynthTaskFamily
from aptivra.tasks.sequence_completion import SequenceCompletionTaskFamily
from aptivra.tasks.state_machines import StateMachineTaskFamily
from aptivra.tasks.string_transforms import StringTransformTaskFamily
from aptivra.tasks.syntax_matching import SyntaxMatchingTaskFamily
from aptivra.tasks.type_inferrer import TypeInferrerTaskFamily


class TaskStreamSplit(NamedTuple):
    """Container for disjoint splits of a task family."""

    train: list[TaskInstance]
    val: list[TaskInstance]
    test: list[TaskInstance]


class ContinualTaskStream:
    """Manages sequential execution of task families with strict split isolation assertions."""

    def __init__(
        self,
        num_train_per_task: int = 500,
        num_val_per_task: int = 100,
        num_test_per_task: int = 100,
        seed: int = 42,
    ) -> None:
        self.num_train_per_task = num_train_per_task
        self.num_val_per_task = num_val_per_task
        self.num_test_per_task = num_test_per_task
        self.seed = seed

        # Sequential task stream containing 12 task families (T1 - T12)
        self.families: list[TaskFamily] = [
            ModularArithmeticTaskFamily(),
            StringTransformTaskFamily(),
            ListOperationsTaskFamily(),
            SyntaxMatchingTaskFamily(),
            BooleanLogicTaskFamily(),
            SequenceCompletionTaskFamily(),
            StateMachineTaskFamily(),
            PythonMicroSynthTaskFamily(),
            AlgoSynthTaskFamily(),
            DocstringToCodeTaskFamily(),
            BugFixerTaskFamily(),
            TypeInferrerTaskFamily(),
        ]

        self._splits: dict[str, TaskStreamSplit] = {}
        self._held_out_regression_suite: dict[str, list[TaskInstance]] = {}
        self._generate_all_splits()

    @staticmethod
    def _instance_hash(instance: TaskInstance) -> str:
        """Unique content hash of task instance prompt and solution."""
        raw = f"{instance.task_family}:{instance.prompt}:{instance.target_solution}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _generate_all_splits(self) -> None:
        """Generates disjoint train, val, and test splits with explicit isolation assertions."""
        for idx, family in enumerate(self.families):
            train_seed = self.seed * 1000 + idx * 10 + 1
            val_seed = self.seed * 1000 + idx * 10 + 2
            test_seed = self.seed * 1000 + idx * 10 + 3

            # 1. Generate train instances
            train_instances = family.generate_instances(self.num_train_per_task, seed=train_seed)
            train_hashes = {self._instance_hash(inst) for inst in train_instances}

            # 2. Generate val instances, ensuring 0 overlap with train
            val_instances: list[TaskInstance] = []
            val_hashes: set[str] = set()
            val_subseed = val_seed
            while len(val_instances) < self.num_val_per_task:
                cand_instances = family.generate_instances(
                    self.num_val_per_task * 2, seed=val_subseed
                )
                for inst in cand_instances:
                    h = self._instance_hash(inst)
                    if h not in train_hashes and h not in val_hashes:
                        val_hashes.add(h)
                        val_instances.append(inst)
                        if len(val_instances) == self.num_val_per_task:
                            break
                val_subseed += 10000

            # 3. Generate test instances, ensuring 0 overlap with train & val
            test_instances: list[TaskInstance] = []
            test_hashes: set[str] = set()
            test_subseed = test_seed
            while len(test_instances) < self.num_test_per_task:
                cand_instances = family.generate_instances(
                    self.num_test_per_task * 2, seed=test_subseed
                )
                for inst in cand_instances:
                    h = self._instance_hash(inst)
                    if h not in train_hashes and h not in val_hashes and h not in test_hashes:
                        test_hashes.add(h)
                        test_instances.append(inst)
                        if len(test_instances) == self.num_test_per_task:
                            break
                test_subseed += 10000

            # Runtime Assertions: Enforce internal split isolation
            assert train_hashes.isdisjoint(val_hashes), (
                f"Data Contamination: Train and Val overlap in {family.family_id}"
            )
            assert train_hashes.isdisjoint(test_hashes), (
                f"Data Contamination: Train and Test overlap in {family.family_id}"
            )
            assert val_hashes.isdisjoint(test_hashes), (
                f"Data Contamination: Val and Test overlap in {family.family_id}"
            )

            split = TaskStreamSplit(train=train_instances, val=val_instances, test=test_instances)
            self._splits[family.family_id] = split
            # Held-out regression suite uses the test split
            self._held_out_regression_suite[family.family_id] = test_instances

    def get_split(self, family_id: str) -> TaskStreamSplit:
        """Returns the task stream split for a given family ID."""
        if family_id not in self._splits:
            raise KeyError(f"Task family '{family_id}' not found in stream.")
        return self._splits[family_id]

    def get_held_out_regression_suite(self, up_to_task_idx: int) -> list[TaskInstance]:
        """Returns held-out test instances for all tasks up to up_to_task_idx (inclusive)."""
        suite: list[TaskInstance] = []
        for i in range(min(up_to_task_idx + 1, len(self.families))):
            fam_id = self.families[i].family_id
            suite.extend(self._held_out_regression_suite[fam_id])
        return suite

    def verify_no_contamination(
        self, candidate_instances: list[TaskInstance], up_to_task_idx: int
    ) -> None:
        """Strict assertion checking candidate instances against the held-out regression suite."""
        regression_suite = self.get_held_out_regression_suite(up_to_task_idx)
        regression_hashes = {self._instance_hash(inst) for inst in regression_suite}
        candidate_hashes = {self._instance_hash(inst) for inst in candidate_instances}

        intersection = candidate_hashes.intersection(regression_hashes)
        assert len(intersection) == 0, (
            f"VIOLATION DETECTED: {len(intersection)} instances in candidate batch "
            "intersect with the held-out regression suite!"
        )
