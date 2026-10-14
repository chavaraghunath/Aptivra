"""Version-two task families with broad target and program spaces."""

from __future__ import annotations

import ast
import random
import string
from typing import Any, override

from aptivra.tasks.algo_synth import AlgoSynthVerifier
from aptivra.tasks.base import (
    MULTI_LINE_STOP_CHARACTERS,
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    extract_python_code,
)
from aptivra.tasks.boolean_logic import BooleanLogicVerifier
from aptivra.tasks.bug_fixer import BugFixerVerifier
from aptivra.tasks.docstring_code import DocstringToCodeVerifier
from aptivra.tasks.list_operations import ListOperationsTaskFamily
from aptivra.tasks.modular_arithmetic import ModularArithmeticVerifier
from aptivra.tasks.python_microsynth import PythonMicroSynthVerifier
from aptivra.tasks.sequence_completion import SequenceCompletionVerifier
from aptivra.tasks.state_machines import StateMachineVerifier
from aptivra.tasks.string_transforms import StringTransformTaskFamily
from aptivra.tasks.syntax_matching import SyntaxMatchingVerifier


def _determinacy_metadata(kind: str, **values: object) -> dict[str, Any]:
    """Mark the prompt contract and retain independently checkable source values."""
    return {"determinacy_kind": kind, **values}


def _format_call(function_name: str, value: object) -> str:
    arguments = (
        ", ".join(repr(item) for item in value)
        if isinstance(value, tuple)
        else repr(value)
    )
    return f"{function_name}({arguments})"


def _compact_examples(function_name: str, cases: list[tuple[Any, Any]]) -> str:
    return "; ".join(
        f"{_format_call(function_name, value)}->{expected!r}"
        for value, expected in cases
    )


def _compact_io_examples(cases: list[tuple[Any, Any]]) -> str:
    return ";".join(f"{value!r}=>{expected!r}" for value, expected in cases)


def _instance(
    family: TaskFamily,
    seed: int,
    index: int,
    prompt: str,
    target: str,
    metadata: dict[str, Any] | None = None,
) -> TaskInstance:
    return TaskInstance(
        task_id=f"{family.family_id}_v2_s{seed}_i{index}",
        task_family=family.family_id,
        prompt=prompt,
        target_solution=target,
        metadata=metadata or {},
        answer_stop_characters=family.answer_stop_characters,
        training_terminator=family.training_terminator,
    )


class ModularArithmeticTaskFamilyV2(TaskFamily):
    """Modular arithmetic with a target space large enough for split isolation."""

    def __init__(self) -> None:
        super().__init__(
            "T1_modular_arithmetic",
            "Modular Arithmetic",
            "Evaluate varied modular expressions over large moduli.",
        )
        self._verifier = ModularArithmeticVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            modulus = rng.randint(10_007, 99_991)
            left = rng.randint(1_000, 999_999)
            right = rng.randint(1_000, 999_999)
            mode = rng.randrange(4)
            if mode == 0:
                prompt = f"Evaluate: ({left} + {right}) mod {modulus} ="
                answer = (left + right) % modulus
            elif mode == 1:
                prompt = f"Evaluate: ({left} * {right}) mod {modulus} ="
                answer = (left * right) % modulus
            elif mode == 2:
                offset = rng.randint(2, 97)
                prompt = f"Evaluate: ({left} * {right} + {offset}) mod {modulus} ="
                answer = (left * right + offset) % modulus
            else:
                exponent = rng.randint(2, 31)
                prompt = f"Evaluate the integer power: ({left} ** {exponent}) mod {modulus} ="
                answer = pow(left, exponent, modulus)
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    str(answer),
                    _determinacy_metadata(
                        "modular",
                        mode=mode,
                        modulus=modulus,
                        left=left,
                        right=right,
                        offset=offset if mode == 2 else None,
                        exponent=exponent if mode == 3 else None,
                    ),
                )
            )
        return instances


class StringTransformDiagnosticTaskFamilyV2(StringTransformTaskFamily):
    """Tokenizer-sensitivity diagnostic retained without headline eligibility."""

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            mode = rng.choice(("reverse", "rle", "caesar"))
            length = rng.randint(6, 14)
            shift: int | None = None
            if mode == "reverse":
                source = "".join(rng.choices(string.ascii_lowercase, k=length))
                target = source[::-1]
                prompt = (
                    f"Reverse the characters of '{source}' exactly; output only the reversed "
                    "string ="
                )
            elif mode == "rle":
                characters: list[str] = []
                for _ in range(rng.randint(3, 6)):
                    character = rng.choice(string.ascii_lowercase)
                    characters.extend([character] * rng.randint(1, 4))
                source = "".join(characters)
                target = self._run_length_encode(source)
                prompt = (
                    f"Run-length encode '{source}' as decimal count followed by character "
                    "for each maximal run; concatenate runs with no separator ="
                )
            else:
                shift = rng.randint(1, 10)
                source = "".join(rng.choices(string.ascii_lowercase, k=length))
                target = self._caesar_shift(source, shift)
                prompt = (
                    f"Caesar-shift each lowercase letter in '{source}' forward by {shift}, "
                    "wrapping z to a; output only the shifted string ="
                )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    target,
                    _determinacy_metadata(
                        "string", mode=mode, source=source, shift=shift
                    ),
                )
            )
        return instances


class ListOperationsTaskFamilyV2(ListOperationsTaskFamily):
    """List operations with an explicit, unique output serialization contract."""

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            mode = rng.choice(("sort_asc", "sort_desc", "filter_even", "top_k"))
            size = rng.randint(6, 12)
            numbers = [rng.randint(-50, 100) for _ in range(size)]
            k: int | None = None
            if mode == "sort_asc":
                result = sorted(numbers)
                instruction = "Sort the values in ascending numeric order"
            elif mode == "sort_desc":
                result = sorted(numbers, reverse=True)
                instruction = "Sort the values in descending numeric order"
            elif mode == "filter_even":
                result = [value for value in numbers if value % 2 == 0]
                instruction = "Keep the even values in their original order"
            else:
                k = rng.randint(2, min(4, size))
                result = sorted(numbers, reverse=True)[:k]
                instruction = f"Return the {k} largest values in descending order"
            prompt = (
                f"{instruction}: {numbers}. Output space-separated integers only; empty list "
                "means empty output ="
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    " ".join(map(str, result)),
                    _determinacy_metadata(
                        "list", mode=mode, numbers=numbers, k=k
                    ),
                )
            )
        return instances


class SyntaxMatchingTaskFamilyV2(TaskFamily):
    """Bracket completion with a large, non-binary answer space."""

    def __init__(self) -> None:
        super().__init__(
            "T4_syntax_matching",
            "Syntax Matching",
            "Complete nested mixed-bracket sequences.",
        )
        self._verifier = SyntaxMatchingVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        pairs = {"(": ")", "[": "]", "{": "}"}
        instances: list[TaskInstance] = []
        for index in range(count):
            openings = [rng.choice(tuple(pairs)) for _ in range(rng.randint(8, 16))]
            target = "".join(pairs[item] for item in reversed(openings))
            source = "".join(openings)
            prompt = (
                f"Append the unique closing brackets that balance '{source}' using standard "
                "proper nesting; output only the appended suffix ="
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    target,
                    _determinacy_metadata("brackets", openings=source),
                )
            )
        return instances


def _boolean_expression(rng: random.Random, depth: int) -> str:
    variables = ("A", "B", "C", "D", "E")
    if depth == 0 or rng.random() < 0.18:
        variable = rng.choice(variables)
        return f"(~{variable})" if rng.random() < 0.25 else variable
    left = _boolean_expression(rng, depth - 1)
    right = _boolean_expression(rng, depth - 1)
    return f"({left}{rng.choice(('&', '|', '^'))}{right})"


def _eval_boolean_expression(expression: str, values: dict[str, bool]) -> bool:
    python_expression = expression
    python_expression = (
        python_expression.replace("^", "!=")
        .replace("~", "not ")
        .replace("&", " and ")
        .replace("|", " or ")
    )
    result = eval(  # noqa: S307
        python_expression,
        {"__builtins__": {}},
        values,
    )
    return bool(result)


class BooleanLogicTaskFamilyV2(TaskFamily):
    """Evaluate a Boolean expression as a five-variable truth-table vector."""

    def __init__(self) -> None:
        super().__init__(
            "T5_boolean_logic",
            "Boolean Logic",
            "Return the 32-bit truth table of a five-variable Boolean expression.",
        )
        self._verifier = BooleanLogicVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            expression = _boolean_expression(rng, rng.randint(3, 4))
            bits: list[str] = []
            for assignment in range(32):
                values = {
                    variable: bool(assignment & (1 << (4 - bit)))
                    for bit, variable in enumerate(("A", "B", "C", "D", "E"))
                }
                bits.append("1" if _eval_boolean_expression(expression, values) else "0")
            prompt = (
                f"32-bit truth table {expression}; ops ~ NOT, & AND, | OR, ^ XOR; rows "
                "ABCDE=00000..11111 lexicographic; 1=true,0=false ="
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    "".join(bits),
                    _determinacy_metadata("truth_table", expression=expression),
                )
            )
        return instances


def simple_sequence_continuations(values: list[int]) -> dict[str, tuple[int, int, int]]:
    """Return continuations from a small, named library of simple integer rules."""
    if len(values) < 4:
        return {}
    candidates: dict[str, tuple[int, int, int]] = {}
    differences = [
        right - left for left, right in zip(values, values[1:], strict=False)
    ]
    if len(set(differences)) == 1:
        difference = differences[0]
        current = values[-1]
        continuation: list[int] = []
        for _ in range(3):
            current += difference
            continuation.append(current)
        candidates["arithmetic"] = (
            continuation[0],
            continuation[1],
            continuation[2],
        )

    if values[0] != 0 and all(
        left != 0 and right % left == 0
        for left, right in zip(values, values[1:], strict=False)
    ):
        ratios = [
            right // left for left, right in zip(values, values[1:], strict=False)
        ]
        if len(set(ratios)) == 1:
            ratio = ratios[0]
            current = values[-1]
            geometric: list[int] = []
            for _ in range(3):
                current *= ratio
                geometric.append(current)
            candidates["geometric"] = (
                geometric[0],
                geometric[1],
                geometric[2],
            )

    second_differences = [
        right - left
        for left, right in zip(differences, differences[1:], strict=False)
    ]
    if second_differences and len(set(second_differences)) == 1 and second_differences[0] != 0:
        current = values[-1]
        difference = differences[-1]
        quadratic: list[int] = []
        for _ in range(3):
            difference += second_differences[0]
            current += difference
            quadratic.append(current)
        candidates["quadratic"] = (
            quadratic[0],
            quadratic[1],
            quadratic[2],
        )

    for multiplier in range(1, 5):
        for offset in range(-9, 10):
            if all(
                values[position]
                == values[position - 1] + multiplier * values[position - 2] + offset
                for position in range(2, len(values))
            ):
                extended = list(values)
                for _ in range(3):
                    extended.append(
                        extended[-1] + multiplier * extended[-2] + offset
                    )
                candidates[f"recurrence_m{multiplier}_c{offset}"] = (
                    extended[-3],
                    extended[-2],
                    extended[-1],
                )
    return candidates


class SequenceCompletionTaskFamilyV2(TaskFamily):
    """Sequence inference whose answer is three unseen continuation values."""

    def __init__(self) -> None:
        super().__init__(
            "T6_sequence_completion",
            "Sequence Completion",
            "Infer three continuation values from varied recurrences.",
        )
        self._verifier = SequenceCompletionVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            for _ in range(1_000):
                mode = rng.randrange(4)
                if mode == 0:
                    start = rng.randint(-500, 500)
                    difference = rng.randint(2, 31)
                    sequence = [start + difference * position for position in range(10)]
                elif mode == 1:
                    sequence = [rng.randint(1, 30), rng.randint(1, 30)]
                    multiplier = rng.randint(1, 4)
                    offset = rng.randint(-9, 9)
                    for _ in range(8):
                        sequence.append(
                            sequence[-1] + multiplier * sequence[-2] + offset
                        )
                elif mode == 2:
                    coefficient = rng.randint(2, 19)
                    linear = rng.randint(-20, 20)
                    constant = rng.randint(-100, 100)
                    sequence = [
                        coefficient * n * n + linear * n + constant
                        for n in range(1, 11)
                    ]
                else:
                    start = rng.randint(1, 12)
                    ratio = rng.randint(2, 4)
                    sequence = [start * ratio**position for position in range(10)]
                continuations = simple_sequence_continuations(sequence[:7])
                distinct = set(continuations.values())
                if distinct == {tuple(sequence[7:10])}:
                    break
            else:
                raise RuntimeError("Could not generate an unambiguous simple sequence.")
            prompt_values = ", ".join(str(value) for value in sequence[:7])
            target = ", ".join(str(value) for value in sequence[7:10])
            prompt = (
                "Using the simplest exact rule among arithmetic, integer geometric, "
                "constant-second-difference quadratic, or a[n]=a[n-1]+m*a[n-2]+c with "
                f"m in 1..4 and c in -9..9, predict three values: {prompt_values}, _, _, _ ="
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    target,
                    _determinacy_metadata(
                        "sequence", mode=mode, shown_values=sequence[:7]
                    ),
                )
            )
        return instances


class StateMachineTaskFamilyV2(TaskFamily):
    """DFA execution returning the full state trace instead of one of four labels."""

    def __init__(self) -> None:
        super().__init__(
            "T7_state_machines",
            "State Machines",
            "Execute a DFA and return every visited state.",
        )
        self._verifier = StateMachineVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        states = [f"S{value}" for value in range(8)]
        instances: list[TaskInstance] = []
        for index in range(count):
            transitions = {
                state: (rng.choice(states), rng.choice(states)) for state in states
            }
            start = rng.choice(states)
            symbols = [rng.randrange(2) for _ in range(rng.randint(7, 11))]
            current = start
            trace = [current]
            for symbol in symbols:
                current = transitions[current][symbol]
                trace.append(current)
            spec = " ".join(
                f"{state}:0>{transitions[state][0]},1>{transitions[state][1]}"
                for state in states
            )
            input_symbols = "".join(map(str, symbols))
            prompt = (
                f"DFA [{spec}] start={start} bits={input_symbols}; trace start then state after "
                "each bit, joined by -> ="
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    "->".join(trace),
                    _determinacy_metadata(
                        "dfa",
                        transitions=transitions,
                        start=start,
                        symbols=symbols,
                    ),
                )
            )
        return instances


class PythonMicroSynthTaskFamilyV2(TaskFamily):
    """Synthesize parameterized snippets with unseen executable targets."""

    def __init__(self) -> None:
        super().__init__(
            "T8_python_microsynth",
            "Python Micro-Synthesis",
            "Set res using a parameterized Python computation.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = PythonMicroSynthVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            mode = rng.randrange(5)
            limit = rng.randint(20, 500)
            divisor = rng.randint(2, 17)
            offset = rng.randint(1, 19)
            if mode == 0:
                prompt = (
                    f"Write Python that sets `res` to the integer sum of all multiples of "
                    f"{divisor} in the inclusive range 1..{limit}. Output executable code."
                )
                target = (
                    "res = sum(value for value in range(1, "
                    f"{limit + 1}) if value % {divisor} == 0)"
                )
                expected = sum(value for value in range(1, limit + 1) if value % divisor == 0)
            elif mode == 1:
                prompt = (
                    f"Write Python that sets `res` to the integer count of values in the "
                    f"inclusive range 1..{limit} congruent to {offset % divisor} modulo "
                    f"{divisor}. Output executable code."
                )
                target = (
                    "res = sum(1 for value in range(1, "
                    f"{limit + 1}) if value % {divisor} == {offset % divisor})"
                )
                expected = sum(
                    1
                    for value in range(1, limit + 1)
                    if value % divisor == offset % divisor
                )
            elif mode == 2:
                values = [rng.randint(-30, 50) for _ in range(5)]
                prompt = (
                    f"Write Python that sets `res` to the weighted sum of {values}; the value "
                    f"at zero-based index i has weight i + {offset}. Output executable code."
                )
                target = (
                    f"res = sum((index + {offset}) * value for index, value in "
                    f"enumerate({values}))"
                )
                expected = sum((i + offset) * value for i, value in enumerate(values))
            elif mode == 3:
                values = [rng.randint(-100, 100) for _ in range(7)]
                prompt = (
                    f"Write Python that sets `res` to max(values) - min(values) for {values}. "
                    "Output executable code."
                )
                target = f"res = max({values}) - min({values})"
                expected = max(values) - min(values)
            else:
                values = [rng.randint(-20, 40) for _ in range(7)]
                prompt = (
                    f"Write Python that sets `res` to the sum of x*x for strictly positive "
                    f"values x in {values}. Output executable code."
                )
                target = f"res = sum(value * value for value in {values} if value > 0)"
                expected = sum(value * value for value in values if value > 0)
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    target,
                    _determinacy_metadata(
                        "snippet",
                        mode=mode,
                        expected_result=str(expected),
                        naive_solution="res = None",
                        limit=limit,
                        divisor=divisor,
                        offset=offset,
                        values=values if mode >= 2 else None,
                    ),
                )
            )
        return instances


def _function_name(seed: int, index: int) -> str:
    return f"f_{seed:x}_{index:x}"


def _apply_list_algorithm(
    mode: int,
    values: list[int],
    *,
    divisor: int,
    scale: int,
    threshold: int,
    offset: int,
) -> object:
    if mode == 0:
        return sum(
            value * scale for value in values if value % divisor == offset % divisor
        )
    if mode == 1:
        return sum(1 for value in values if value * scale > threshold + offset)
    if mode == 2:
        return sorted(value * value + offset for value in values if value < threshold)
    if mode == 3:
        return [max(-scale, min(scale, value)) + offset for value in values]
    if mode == 4:
        return [
            (values[index] - values[index - 1]) * scale + threshold + offset
            for index in range(1, len(values))
        ]
    if mode == 5:
        ordered = sorted(set(values), key=lambda value: abs(value - threshold))
        return ordered[: offset % 5]
    if mode == 6:
        return [
            value * scale + offset if index % 2 == 0 else -value + offset
            for index, value in enumerate(values)
        ]
    return sum(abs(value - threshold) * scale for value in values) + offset


def _list_cases(
    mode: int, *, divisor: int, scale: int, threshold: int, offset: int
) -> list[tuple[Any, Any]]:
    values: tuple[list[int], ...] = (
        [],
        [0],
        [1],
        [-1],
        [offset % divisor + divisor],
        [offset % divisor + divisor + 1],
        [(threshold + offset) // scale],
        [(threshold + offset) // scale + 1],
        [0, 1],
        [1, 0],
        [0, 1, 2],
        [3, -1, 4, 1, 5],
        [8, 8, -2, 7],
        [-9, -3, 2, 6],
        [0, 1, -1, 12, -12, 5],
        [threshold - 1, threshold, threshold + 1, offset, -offset],
    )
    return [
        (
            list(case),
            _apply_list_algorithm(
                mode,
                list(case),
                divisor=divisor,
                scale=scale,
                threshold=threshold,
                offset=offset,
            ),
        )
        for case in values
    ]


def _partition_cases(
    cases: list[tuple[Any, Any]],
    naive_outputs: list[object],
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Expose two discriminating cases and retain the rest as held-out evidence."""
    discriminating = [
        case
        for case, naive_output in zip(cases, naive_outputs, strict=True)
        if case[1] != naive_output
    ]
    if len(discriminating) < 2:
        raise RuntimeError("Generated code contract lacks two discriminating examples.")
    discriminating.sort(key=lambda case: len(repr(case[0])) + len(repr(case[1])))
    prompt_cases = discriminating[:2]
    held_out = [case for case in cases if case not in prompt_cases]
    return prompt_cases, held_out


def _apply_broken_list_algorithm(
    mode: int,
    values: list[int],
    *,
    divisor: int,
    scale: int,
    threshold: int,
    offset: int,
) -> object:
    if mode == 0:
        return sum(
            value * scale for value in values if value % divisor != offset % divisor
        )
    if mode == 1:
        return sum(1 for value in values if value * scale <= threshold + offset)
    if mode == 2:
        return sorted(value * value + offset for value in values if value >= threshold)
    if mode == 3:
        return [min(-scale, min(scale, value)) + offset for value in values]
    if mode == 4:
        return [
            (values[index] - values[index - 1]) * scale + threshold + offset
            for index in range(2, len(values))
        ]
    if mode == 5:
        ordered = sorted(set(values), key=lambda value: abs(value - threshold))
        return ordered[(offset % 5) :]
    if mode == 6:
        return [
            value * scale + offset if index % 2 == 1 else -value + offset
            for index, value in enumerate(values)
        ]
    return sum(abs(value - threshold) * scale for value in values) - 1


def _compact_list_contract(metadata: dict[str, Any]) -> str:
    mode = int(metadata["mode"])
    divisor = int(metadata["divisor"])
    scale = int(metadata["scale"])
    threshold = int(metadata["threshold"])
    offset = int(metadata["offset"])
    if mode == 0:
        return f"sum {scale}*x where x%{divisor}={offset % divisor}"
    if mode == 1:
        return f"count x where {scale}*x>{threshold + offset}"
    if mode == 2:
        return f"sorted x*x+{offset} where x<{threshold}"
    if mode == 3:
        return f"each clamp(x,-{scale},{scale})+{offset}"
    if mode == 4:
        return f"adjacent differences *{scale} +{threshold + offset}"
    if mode == 5:
        return f"first {offset % 5} unique x ordered by abs(x-{threshold})"
    if mode == 6:
        return f"even i: {scale}*x+{offset}; odd i: {offset}-x"
    return f"sum abs(x-{threshold})*{scale} +{offset}"


class AlgoSynthTaskFamilyV2(TaskFamily):
    """Algorithm synthesis from a compositional list-program grammar."""

    def __init__(self) -> None:
        super().__init__(
            "T9_algo_synth",
            "Algorithmic Synthesis",
            "Synthesize parameterized list algorithms across varied control flow.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier: MechanicalVerifier = AlgoSynthVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            fn = _function_name(seed, index)
            divisor = rng.randint(2, 19)
            scale = rng.randint(2, 13)
            threshold = rng.randint(-15, 25)
            offset = rng.randint(1, 997)
            mode = rng.randrange(8)
            if mode == 0:
                remainder = offset % divisor
                description = (
                    f"sum values congruent to {remainder} mod {divisor}, scaled by {scale}"
                )
                solution = (
                    f"def {fn}(xs):\n    return sum(x * {scale} for x in xs if "
                    f"x % {divisor} == {remainder})"
                )
            elif mode == 1:
                boundary = threshold + offset
                description = f"count values whose {scale}x product exceeds {boundary}"
                solution = (
                    f"def {fn}(xs):\n    return sum(1 for x in xs if "
                    f"x * {scale} > {boundary})"
                )
            elif mode == 2:
                description = (
                    f"return sorted squares plus {offset} for values below {threshold}"
                )
                solution = (
                    f"def {fn}(xs):\n    return sorted(x * x + {offset} for x in xs "
                    f"if x < {threshold})"
                )
            elif mode == 3:
                description = (
                    f"return each value clamped to [-{scale}, {scale}], then add {offset}"
                )
                solution = (
                    f"def {fn}(xs):\n    return [max(-{scale}, min({scale}, x)) + "
                    f"{offset} for x in xs]"
                )
            elif mode == 4:
                description = (
                    f"scale adjacent differences by {scale}, then add {threshold + offset}"
                )
                solution = (
                    f"def {fn}(xs):\n    return [(xs[i]-xs[i-1])*{scale}+"
                    f"{threshold + offset} for i in range(1,len(xs))]"
                )
            elif mode == 5:
                take = offset % 5
                description = (
                    f"return the first {take} unique values ordered by distance from {threshold}"
                )
                solution = (
                    f"def {fn}(xs):\n    return sorted(set(xs), key=lambda x: "
                    f"abs(x - {threshold}))[:{take}]"
                )
            elif mode == 6:
                description = (
                    f"return alternating signed values scaled by {scale}, offset by {offset}"
                )
                solution = (
                    f"def {fn}(xs):\n    return [x*{scale}+{offset} if i%2==0 "
                    f"else {offset}-x for i,x in enumerate(xs)]"
                )
            else:
                description = (
                    f"sum absolute deviations from {threshold}, scaled by {scale}, plus {offset}"
                )
                solution = (
                    f"def {fn}(xs):\n    return sum(abs(x - {threshold}) * {scale} "
                    f"for x in xs) + {offset}"
                )
            cases = _list_cases(
                mode,
                divisor=divisor,
                scale=scale,
                threshold=threshold,
                offset=offset,
            )
            prompt_cases, held_out_cases = _partition_cases(
                cases, [case[0] for case in cases]
            )
            naive_solution = f"def {fn}(xs):\n    return xs"
            prompt = (
                f"Write {fn}(xs): {description}. Examples: "
                f"{_compact_examples(fn, prompt_cases)}\n"
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    solution,
                    _determinacy_metadata(
                        "function",
                        fn_name=fn,
                        test_cases=prompt_cases + held_out_cases,
                        prompt_cases=prompt_cases,
                        held_out_cases=held_out_cases,
                        naive_solution=naive_solution,
                        mode=mode,
                        divisor=divisor,
                        scale=scale,
                        threshold=threshold,
                        offset=offset,
                        description=description,
                    ),
                )
            )
        return instances


def _apply_string_algorithm(
    mode: int,
    text: str,
    *,
    step: int,
    limit: int,
    character: str,
) -> object:
    if mode == 0:
        return text[limit % step :: step]
    if mode == 1:
        rotation = step % max(1, len(text))
        return text[rotation:] + text[:rotation]
    if mode == 2:
        return "".join(value for value in text if value.lower() != character)[:limit]
    if mode == 3:
        return text[:limit].lower().count(character)
    if mode == 4:
        return " ".join(reversed(text.split()[:limit]))
    if mode == 5:
        return text[:limit][1::2][::-1]
    if mode == 6:
        return "".join(
            character if value.lower() in "aeiou" else value for value in text[:limit]
        )
    return text + str(len(text) * step)


def _string_cases(
    mode: int, *, step: int, limit: int, character: str
) -> list[tuple[Any, Any]]:
    values = [
        "a b",
        "a b c",
        "Aptivra",
        "banana split",
        "one two three four five six",
        f"{character}A{character}eiou-{character.upper()}",
        f"{character * 3}xyz-{character.upper()}",
        "xyZxy0123456789",
        "",
    ]
    if mode == 0:
        values.append((string.ascii_letters * 4)[: limit % step + step * 2 + 3])
    elif mode in {2, 3, 5, 6}:
        values.append(("b" * limit) + character + "aeiouXYZ")
    elif mode == 4:
        values.append(" ".join(f"w{index}" for index in range(limit + 2)))
    return [
        (
            value,
            _apply_string_algorithm(
                mode,
                value,
                step=step,
                limit=limit,
                character=character,
            ),
        )
        for value in values
    ]


class DocstringToCodeTaskFamilyV2(TaskFamily):
    """Docstring synthesis from a compositional string-program grammar."""

    def __init__(self) -> None:
        super().__init__(
            "T10_docstring_code",
            "Docstring-to-Code Synthesis",
            "Synthesize varied string algorithms from precise docstrings.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = DocstringToCodeVerifier()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            fn = _function_name(seed, index)
            step = rng.randint(2, 97)
            limit = rng.randint(3, 997)
            character = rng.choice(string.ascii_lowercase)
            mode = rng.randrange(8)
            if mode == 0:
                start = limit % step
                doc = f"Return every {step}th character starting at index {start}."
                solution = f"def {fn}(text):\n    return text[{start}::{step}]"
            elif mode == 1:
                doc = f"Rotate left by {step} characters."
                solution = (
                    f"def {fn}(text):\n    k = {step} % max(1, len(text))\n"
                    "    return text[k:] + text[:k]"
                )
            elif mode == 2:
                doc = f"Remove '{character}' ignoring case, then keep {limit} characters."
                solution = (
                    f"def {fn}(text):\n    return ''.join(c for c in text if "
                    f"c.lower() != '{character}')[:{limit}]"
                )
            elif mode == 3:
                doc = f"Count '{character}' ignoring case in the first {limit} characters."
                solution = (
                    f"def {fn}(text):\n    return text[:{limit}].lower().count('{character}')"
                )
            elif mode == 4:
                doc = f"Reverse at most the first {limit} whitespace-separated words."
                solution = (
                    f"def {fn}(text):\n    return ' '.join(reversed(text.split()[:{limit}]))"
                )
            elif mode == 5:
                doc = f"Return odd-index characters from the first {limit}, reversed."
                solution = f"def {fn}(text):\n    return text[:{limit}][1::2][::-1]"
            elif mode == 6:
                doc = f"Replace vowels with '{character}' in the first {limit} characters."
                solution = (
                    f"def {fn}(text):\n    return ''.join('{character}' if c.lower() in "
                    f"'aeiou' else c for c in text[:{limit}])"
                )
            else:
                doc = f"Append the decimal length multiplied by {step}."
                solution = f"def {fn}(text):\n    return text + str(len(text) * {step})"
            cases = _string_cases(
                mode,
                step=step,
                limit=limit,
                character=character,
            )
            prompt_cases, held_out_cases = _partition_cases(
                cases, [case[0] for case in cases]
            )
            naive_solution = f"def {fn}(text):\n    return text"
            prompt = (
                f'def {fn}(text):\n    """{doc} Examples: '
                f'{_compact_io_examples(prompt_cases)}"""\n'
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    solution,
                    _determinacy_metadata(
                        "function",
                        fn_name=fn,
                        test_cases=prompt_cases + held_out_cases,
                        prompt_cases=prompt_cases,
                        held_out_cases=held_out_cases,
                        naive_solution=naive_solution,
                        mode=mode,
                        step=step,
                        limit=limit,
                        character=character,
                    ),
                )
            )
        return instances


class BugFixerTaskFamilyV2(AlgoSynthTaskFamilyV2):
    """Bug repair over the same broad list-algorithm grammar with real faults."""

    def __init__(self) -> None:
        TaskFamily.__init__(
            self,
            "T11_bug_fixer",
            "Python Bug Fixer",
            "Repair parameterized boundary, predicate, and operator faults.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = BugFixerVerifier()

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        correct = super().generate_instances(count, seed)
        instances: list[TaskInstance] = []
        for index, item in enumerate(correct):
            mode = int(item.metadata["mode"])
            divisor = int(item.metadata["divisor"])
            scale = int(item.metadata["scale"])
            threshold = int(item.metadata["threshold"])
            offset = int(item.metadata["offset"])
            broken = item.target_solution
            if mode == 0:
                broken = broken.replace(" == ", " != ", 1)
            elif mode == 1:
                broken = broken.replace(" > ", " <= ", 1)
            elif mode == 2:
                broken = broken.replace(" < ", " >= ", 1)
            elif mode == 3:
                broken = broken.replace("max(-", "min(-", 1)
            elif mode == 4:
                broken = broken.replace("range(1,", "range(2,", 1)
            elif mode == 5:
                take = offset % 5
                broken = broken.replace(f"[:{take}]", f"[{take}:]", 1)
            elif mode == 6:
                broken = broken.replace("==0", "==1", 1)
            else:
                broken = broken.rsplit(" + ", maxsplit=1)[0] + " - 1"
            cases = list(item.metadata["test_cases"])
            broken_outputs = [
                _apply_broken_list_algorithm(
                    mode,
                    list(case[0]),
                    divisor=divisor,
                    scale=scale,
                    threshold=threshold,
                    offset=offset,
                )
                for case in cases
            ]
            prompt_cases, held_out_cases = _partition_cases(cases, broken_outputs)
            prompt = (
                f"Fix {item.metadata['fn_name']}(xs): {_compact_list_contract(item.metadata)}. "
                f"Ex: {_compact_io_examples(prompt_cases)}\n"
                f"{broken}\n"
            )
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    item.target_solution,
                    _determinacy_metadata(
                        "function",
                        fn_name=item.metadata["fn_name"],
                        test_cases=prompt_cases + held_out_cases,
                        prompt_cases=prompt_cases,
                        held_out_cases=held_out_cases,
                        naive_solution=broken,
                        mode=mode,
                        divisor=divisor,
                        scale=scale,
                        threshold=threshold,
                        offset=offset,
                    ),
                )
            )
        return instances


class TypeInferrerVerifierV2(MechanicalVerifier):
    """Accept formatting variants of the uniquely specified annotated program."""

    @override
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        extracted = extract_python_code(model_output)
        if extracted is None:
            return VerifierResult(
                False,
                instance.target_solution,
                model_output.strip(),
                "Unterminated code fence.",
            )
        try:
            expected_tree = ast.parse(instance.target_solution)
            actual_tree = ast.parse(extracted)
        except SyntaxError as error:
            return VerifierResult(
                False,
                instance.target_solution,
                extracted,
                f"Invalid Python: {error.msg}.",
            )
        passed = ast.dump(actual_tree, include_attributes=False) == ast.dump(
            expected_tree, include_attributes=False
        )
        return VerifierResult(
            passed,
            instance.target_solution,
            extracted,
            None if passed else "Annotations or preserved function body do not match the contract.",
        )


class TypeInferrerTaskFamilyV2(TaskFamily):
    """Infer annotations over varied typed scalar, collection, and text programs."""

    def __init__(self) -> None:
        super().__init__(
            "T12_type_inferrer",
            "Python Type Annotations Inferrer",
            "Infer annotations for varied Python function structures.",
            answer_stop_characters=MULTI_LINE_STOP_CHARACTERS,
            training_terminator="\x00",
        )
        self._verifier = TypeInferrerVerifierV2()

    @property
    @override
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    @override
    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for index in range(count):
            fn = _function_name(seed, index)
            constant = rng.randint(2, 99_991)
            mode = rng.randrange(8)
            templates: tuple[
                tuple[str, str, str, tuple[str, ...], str, list[tuple[Any, Any]]],
                ...,
            ] = (
                (
                    "(value)",
                    f"return value + {constant}",
                    "(value: int) -> int",
                    ("int",),
                    "int",
                    [(3, 3 + constant), (-2, -2 + constant)],
                ),
                (
                    "(value)",
                    f"return value / {constant}",
                    "(value: float) -> float",
                    ("float",),
                    "float",
                    [(3.5, 3.5 / constant), (-2.0, -2.0 / constant)],
                ),
                (
                    "(text)",
                    f"return text + '-{constant}'",
                    "(text: str) -> str",
                    ("str",),
                    "str",
                    [("ab", f"ab-{constant}"), ("", f"-{constant}")],
                ),
                (
                    "(items)",
                    f"return len(items) + {constant}",
                    "(items: list[int]) -> int",
                    ("list[int]",),
                    "int",
                    [([1, 2], 2 + constant), ([], constant)],
                ),
                (
                    "(items)",
                    f"return [x + {constant} for x in items]",
                    "(items: list[int]) -> list[int]",
                    ("list[int]",),
                    "list[int]",
                    [([1, -2], [1 + constant, -2 + constant]), ([], [])],
                ),
                (
                    "(text)",
                    f"return text[:{constant}]",
                    "(text: str) -> str",
                    ("str",),
                    "str",
                    [("Aptivra", "Aptivra"[:constant]), ("", "")],
                ),
                (
                    "(left, right)",
                    f"return left[:{constant}] == right[:{constant}]",
                    "(left: str, right: str) -> bool",
                    ("str", "str"),
                    "bool",
                    [(('abc', 'abc'), True), (('abc', 'abd'), constant < 3)],
                ),
                (
                    "(items)",
                    f"return tuple(items[:{constant}])",
                    "(items: list[str]) -> tuple[str, ...]",
                    ("list[str]",),
                    "tuple[str, ...]",
                    [(["a", "b"], ("a", "b")), ([], ())],
                ),
            )
            signature, body, typed_signature, argument_types, return_type, cases = templates[
                mode
            ]
            prompt = (
                "Infer most-specific PEP585 types; keep body. "
                f"Ex: {_compact_io_examples(cases)}\n"
                f"def {fn}{signature}:\n    {body}\n"
            )
            untyped_source = f"def {fn}{signature}:\n    {body}"
            target = f"def {fn}{typed_signature}:\n    {body}"
            naive_signature = signature.replace(")", ": object)", 1)
            if "," in signature:
                naive_signature = "(left: object, right: object)"
            naive = f"def {fn}{naive_signature} -> object:\n    {body}"
            instances.append(
                _instance(
                    self,
                    seed,
                    index,
                    prompt,
                    target,
                    _determinacy_metadata(
                        "typed_function",
                        fn_name=fn,
                        prompt_cases=cases,
                        held_out_cases=cases,
                        naive_solution=naive,
                        argument_types=list(argument_types),
                        return_type=return_type,
                        untyped_source=untyped_source,
                        mode=mode,
                    ),
                )
            )
        return instances


def v2_families() -> list[TaskFamily]:
    """Return the ordered v2 stream family registry."""
    return [
        ModularArithmeticTaskFamilyV2(),
        StringTransformDiagnosticTaskFamilyV2(),
        ListOperationsTaskFamilyV2(),
        SyntaxMatchingTaskFamilyV2(),
        BooleanLogicTaskFamilyV2(),
        SequenceCompletionTaskFamilyV2(),
        StateMachineTaskFamilyV2(),
        PythonMicroSynthTaskFamilyV2(),
        AlgoSynthTaskFamilyV2(),
        DocstringToCodeTaskFamilyV2(),
        BugFixerTaskFamilyV2(),
        TypeInferrerTaskFamilyV2(),
    ]
