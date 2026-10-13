"""State Machine Execution Task Family and Mechanical Verifier."""

import random
from typing import override

from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)


class StateMachineVerifier(MechanicalVerifier):
    """Mechanical verifier for Deterministic Finite Automaton (DFA) state traversal."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class StateMachineTaskFamily(TaskFamily):
    """Generates Deterministic Finite Automaton (DFA) state execution tasks."""

    def __init__(self) -> None:
        super().__init__(
            family_id="T7_state_machines",
            name="State Machines",
            description="Execute state transitions for Deterministic Finite Automata (DFA).",
        )
        self._verifier = StateMachineVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []

        num_states = 4
        states = [f"S{k}" for k in range(num_states)]
        symbols = ["0", "1"]

        for i in range(count):
            # Create random DFA transitions: transition[state][symbol] -> next_state
            transitions: dict[str, dict[str, str]] = {}
            for st in states:
                transitions[st] = {
                    "0": rng.choice(states),
                    "1": rng.choice(states),
                }

            start_state = rng.choice(states)
            input_length = rng.randint(4, 8)
            inputs = [rng.choice(symbols) for _ in range(input_length)]

            # Trace execution
            curr = start_state
            for inp in inputs:
                curr = transitions[curr][inp]

            # Format transitions string for prompt
            trans_str_parts: list[str] = []
            for st in sorted(states):
                t0 = transitions[st]["0"]
                t1 = transitions[st]["1"]
                trans_str_parts.append(f"{st}:(0->{t0},1->{t1})")
            trans_spec = " ".join(trans_str_parts)
            input_seq = "".join(inputs)

            prompt = f"DFA [{trans_spec}] start {start_state} input '{input_seq}' ends at state ="

            instances.append(
                TaskInstance(
                    task_id=f"{self.family_id}_s{seed}_i{i}",
                    task_family=self.family_id,
                    prompt=prompt,
                    target_solution=curr,
                    metadata={"start_state": start_state, "input_len": input_length},
                )
            )

        return instances
