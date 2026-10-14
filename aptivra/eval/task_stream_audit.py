"""Deterministic audit and policy gate for versioned continual task streams."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from aptivra.eval.config import load_experiment_settings
from aptivra.inference.chat_canonical import IDENTITY_CHAT_TEMPLATES
from aptivra.tasks.base import TaskFamily, TaskInstance
from aptivra.tasks.code_sandbox import (
    SandboxBatchRequest,
    SandboxResult,
    verify_source_batch,
)
from aptivra.tasks.conversational import ConversationalTaskFamily
from aptivra.tasks.identity import ModelIdentityTaskFamily
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamSplit
from aptivra.tasks.structure import structural_hash
from aptivra.tasks.v2 import _eval_boolean_expression, simple_sequence_continuations

AUDIT_SCHEMA = "aptivra-task-stream-audit-v2"
CODE_FAMILY_IDS = {
    "T8_python_microsynth",
    "T9_algo_synth",
    "T10_docstring_code",
    "T11_bug_fixer",
    "T12_type_inferrer",
}


@dataclass(frozen=True)
class AuditPolicy:
    """Configured upper bounds enforced for one stream version."""

    max_target_overlap_count: int
    max_structural_overlap_count: int
    max_identity_chat_contamination_count: int
    max_uniform_chance: float
    max_trivial_baseline_accuracy: float
    max_ambiguous_item_count: int


@dataclass(frozen=True)
class AuditPlanEntry:
    """One stream config, recorded output, and its policy."""

    stream_config: Path
    output: Path
    policy: AuditPolicy


@dataclass(frozen=True)
class AuditPlan:
    """Validated multi-version audit plan."""

    baseline_sample_size: int
    streams: tuple[AuditPlanEntry, ...]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping.")
    return value


def _nonnegative_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{key}' must be a non-negative integer.")
    return value


def _unit_float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"'{key}' must be numeric.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"'{key}' must be between 0 and 1.")
    return result


def load_audit_plan(path: Path | str) -> AuditPlan:
    """Load a validated audit plan without resolving paths outside the repository."""
    plan_path = Path(path)
    root = _mapping(yaml.safe_load(plan_path.read_text(encoding="utf-8")), "root")
    audit = _mapping(root.get("task_stream_audit"), "task_stream_audit")
    baseline_sample_size = _nonnegative_int(audit, "baseline_sample_size")
    if baseline_sample_size < 1:
        raise ValueError("'baseline_sample_size' must be at least 1.")
    raw_streams = audit.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("'streams' must be a non-empty list.")
    entries: list[AuditPlanEntry] = []
    for index, raw_entry in enumerate(raw_streams):
        entry = _mapping(raw_entry, f"streams[{index}]")
        config_value = entry.get("stream_config")
        output_value = entry.get("output")
        if not isinstance(config_value, str) or not config_value:
            raise ValueError("'stream_config' must be a non-empty path string.")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("'output' must be a non-empty path string.")
        raw_policy = _mapping(entry.get("policy"), "policy")
        entries.append(
            AuditPlanEntry(
                stream_config=Path(config_value),
                output=Path(output_value),
                policy=AuditPolicy(
                    max_target_overlap_count=_nonnegative_int(
                        raw_policy, "max_target_overlap_count"
                    ),
                    max_structural_overlap_count=_nonnegative_int(
                        raw_policy, "max_structural_overlap_count"
                    ),
                    max_identity_chat_contamination_count=_nonnegative_int(
                        raw_policy, "max_identity_chat_contamination_count"
                    ),
                    max_uniform_chance=_unit_float(
                        raw_policy, "max_uniform_chance"
                    ),
                    max_trivial_baseline_accuracy=_unit_float(
                        raw_policy, "max_trivial_baseline_accuracy"
                    ),
                    max_ambiguous_item_count=_nonnegative_int(
                        raw_policy, "max_ambiguous_item_count"
                    ),
                ),
            )
        )
    return AuditPlan(baseline_sample_size, tuple(entries))


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _split_metrics(instances: list[TaskInstance]) -> dict[str, object]:
    prompt_count = len({item.prompt for item in instances})
    targets = [item.target_solution for item in instances]
    target_counts = Counter(targets)
    lengths = sorted(len(target.encode("utf-8")) for target in targets)
    distinct_targets = len(target_counts)
    return {
        "instances": len(instances),
        "distinct_prompts": prompt_count,
        "distinct_targets": distinct_targets,
        "uniform_target_chance": (
            1.0 / distinct_targets if distinct_targets else None
        ),
        "empirical_most_frequent_rate": (
            max(target_counts.values()) / len(targets) if targets else None
        ),
        "target_length_bytes": {
            "min": min(lengths, default=0),
            "median": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
    }


def _training_contamination_strings() -> set[str]:
    strings: set[str] = set()
    for family in (ModelIdentityTaskFamily(), ConversationalTaskFamily()):
        for instance in family.generate_instances(32, seed=42):
            strings.update((instance.prompt, instance.target_solution))
    for question, prompt, answer in IDENTITY_CHAT_TEMPLATES:
        strings.update((question, prompt, answer))
    return strings


def _trigrams(value: str) -> set[str]:
    normalized = " ".join(value.lower().split())
    if len(normalized) < 3:
        return {normalized}
    return {normalized[index : index + 3] for index in range(len(normalized) - 2)}


def _similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _score_predictions(
    family: TaskFamily,
    instances: list[TaskInstance],
    predictions: list[str],
) -> float:
    passed = 0
    for instance, prediction in zip(instances, predictions, strict=True):
        result = family.verifier.verify(instance, prediction)
        if result.execution_status == "infrastructure_error":
            raise RuntimeError(
                f"Baseline verification infrastructure failed for '{instance.task_id}': "
                f"{result.error_message or 'unknown failure'}"
            )
        passed += int(result.passed)
    return passed / len(instances) if instances else 0.0


def trivial_baseline_scores(
    family: TaskFamily,
    split: TaskStreamSplit,
    sample_size: int,
) -> dict[str, object]:
    """Score train-only trivial predictors on a deterministic test prefix."""
    if not split.train:
        raise ValueError(f"Family '{family.family_id}' has no training instances.")
    sample = split.test[:sample_size]
    target_counts = Counter(item.target_solution for item in split.train)
    most_frequent = min(
        target_counts,
        key=lambda target: (-target_counts[target], target),
    )
    train_grams = [_trigrams(item.prompt) for item in split.train]
    nearest_predictions: list[str] = []
    for item in sample:
        grams = _trigrams(item.prompt)
        nearest_index = max(
            range(len(split.train)),
            key=lambda index: (_similarity(grams, train_grams[index]), -index),
        )
        nearest_predictions.append(split.train[nearest_index].target_solution)
    return {
        "sample_size": len(sample),
        "most_frequent_answer": _score_predictions(
            family, sample, [most_frequent] * len(sample)
        ),
        "copy_prompt": _score_predictions(
            family, sample, [item.prompt for item in sample]
        ),
        "nearest_training_prompt": _score_predictions(
            family, sample, nearest_predictions
        ),
        "nearest_similarity": "character_trigram_jaccard",
    }


def _function_call(function_name: str, value: object) -> str:
    arguments = (
        ", ".join(repr(item) for item in value)
        if isinstance(value, tuple)
        else repr(value)
    )
    return f"{function_name}({arguments})"


def _cases_are_visible(
    family_id: str,
    prompt: str,
    function_name: str,
    cases: list[tuple[Any, Any]],
) -> bool:
    if family_id == "T9_algo_synth":
        rendered = (
            f"{_function_call(function_name, value)}->{expected!r}"
            for value, expected in cases
        )
    else:
        rendered = (f"{value!r}=>{expected!r}" for value, expected in cases)
    return all(
        example in prompt for example in rendered
    )


def _run_length_encode(value: str) -> str:
    if not value:
        return ""
    encoded: list[str] = []
    run_length = 1
    for index in range(1, len(value)):
        if value[index] == value[index - 1]:
            run_length += 1
        else:
            encoded.append(f"{run_length}{value[index - 1]}")
            run_length = 1
    encoded.append(f"{run_length}{value[-1]}")
    return "".join(encoded)


def _direct_expected(instance: TaskInstance, kind: str) -> str | None:
    metadata = instance.metadata
    if kind == "modular":
        modular_mode = int(metadata["mode"])
        left = int(metadata["left"])
        right = int(metadata["right"])
        modulus = int(metadata["modulus"])
        if modular_mode == 0:
            modular_result = left + right
        elif modular_mode == 1:
            modular_result = left * right
        elif modular_mode == 2:
            modular_result = left * right + int(metadata["offset"])
        else:
            return str(pow(left, int(metadata["exponent"]), modulus))
        return str(modular_result % modulus)
    if kind == "string":
        string_mode = str(metadata["mode"])
        source = str(metadata["source"])
        if string_mode == "reverse":
            return source[::-1]
        if string_mode == "rle":
            return _run_length_encode(source)
        shift = int(metadata["shift"])
        return "".join(
            chr(ord("a") + (ord(character) - ord("a") + shift) % 26)
            for character in source
        )
    if kind == "list":
        list_mode = str(metadata["mode"])
        numbers = [int(value) for value in metadata["numbers"]]
        if list_mode == "sort_asc":
            list_result = sorted(numbers)
        elif list_mode == "sort_desc":
            list_result = sorted(numbers, reverse=True)
        elif list_mode == "filter_even":
            list_result = [value for value in numbers if value % 2 == 0]
        else:
            list_result = sorted(numbers, reverse=True)[: int(metadata["k"])]
        return " ".join(map(str, list_result))
    if kind == "brackets":
        pairs = {"(": ")", "[": "]", "{": "}"}
        return "".join(pairs[value] for value in reversed(str(metadata["openings"])))
    if kind == "truth_table":
        expression = str(metadata["expression"])
        bits: list[str] = []
        for assignment in range(32):
            values = {
                variable: bool(assignment & (1 << (4 - bit)))
                for bit, variable in enumerate(("A", "B", "C", "D", "E"))
            }
            bits.append("1" if _eval_boolean_expression(expression, values) else "0")
        return "".join(bits)
    if kind == "dfa":
        transitions = metadata["transitions"]
        if not isinstance(transitions, dict):
            return None
        current = str(metadata["start"])
        trace = [current]
        for raw_symbol in metadata["symbols"]:
            choices = transitions.get(current)
            if not isinstance(choices, tuple | list) or len(choices) != 2:
                return None
            current = str(choices[int(raw_symbol)])
            trace.append(current)
        return "->".join(trace)
    return None


def _direct_prompt_exposes_contract(instance: TaskInstance, kind: str) -> bool:
    metadata = instance.metadata
    if kind == "modular":
        modular_mode = int(metadata["mode"])
        left = int(metadata["left"])
        right = int(metadata["right"])
        modulus = int(metadata["modulus"])
        if modular_mode == 0:
            expected_prompt = f"Evaluate: ({left} + {right}) mod {modulus} ="
        elif modular_mode == 1:
            expected_prompt = f"Evaluate: ({left} * {right}) mod {modulus} ="
        elif modular_mode == 2:
            expected_prompt = (
                f"Evaluate: ({left} * {right} + {int(metadata['offset'])}) mod "
                f"{modulus} ="
            )
        else:
            expected_prompt = (
                f"Evaluate the integer power: ({left} ** {int(metadata['exponent'])}) "
                f"mod {modulus} ="
            )
        return instance.prompt == expected_prompt
    elif kind == "string":
        source = str(metadata["source"])
        string_mode = str(metadata["mode"])
        if string_mode == "reverse":
            expected_prompt = (
                f"Reverse the characters of '{source}' exactly; output only the reversed "
                "string ="
            )
        elif string_mode == "rle":
            expected_prompt = (
                f"Run-length encode '{source}' as decimal count followed by character "
                "for each maximal run; concatenate runs with no separator ="
            )
        else:
            expected_prompt = (
                f"Caesar-shift each lowercase letter in '{source}' forward by "
                f"{int(metadata['shift'])}, wrapping z to a; output only the shifted string ="
            )
        return instance.prompt == expected_prompt
    elif kind == "list":
        numbers = metadata["numbers"]
        list_mode = str(metadata["mode"])
        if list_mode == "sort_asc":
            instruction = "Sort the values in ascending numeric order"
        elif list_mode == "sort_desc":
            instruction = "Sort the values in descending numeric order"
        elif list_mode == "filter_even":
            instruction = "Keep the even values in their original order"
        else:
            instruction = f"Return the {int(metadata['k'])} largest values in descending order"
        expected_prompt = (
            f"{instruction}: {numbers}. Output space-separated integers only; empty list "
            "means empty output ="
        )
        return instance.prompt == expected_prompt
    elif kind == "brackets":
        expected_prompt = (
            f"Append the unique closing brackets that balance '{metadata['openings']}' using "
            "standard proper nesting; output only the appended suffix ="
        )
        return instance.prompt == expected_prompt
    elif kind == "truth_table":
        expected_prompt = (
            f"32-bit truth table {metadata['expression']}; ops ~ NOT, & AND, | OR, ^ XOR; rows "
            "ABCDE=00000..11111 lexicographic; 1=true,0=false ="
        )
        return instance.prompt == expected_prompt
    elif kind == "dfa":
        transitions = metadata["transitions"]
        if not isinstance(transitions, dict):
            return False
        spec = " ".join(
            f"{state}:0>{transitions[state][0]},1>{transitions[state][1]}"
            for state in (f"S{value}" for value in range(8))
        )
        symbols = "".join(map(str, metadata["symbols"]))
        expected_prompt = (
            f"DFA [{spec}] start={metadata['start']} bits={symbols}; trace start then state after "
            "each bit, joined by -> ="
        )
        return instance.prompt == expected_prompt
    else:
        return False


def _annotation_contract(source: str) -> tuple[list[str], str] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    function = tree.body[0]
    if function.returns is None or any(arg.annotation is None for arg in function.args.args):
        return None
    annotations: list[str] = []
    for argument in function.args.args:
        assert argument.annotation is not None
        annotations.append(ast.unparse(argument.annotation))
    return (
        annotations,
        ast.unparse(function.returns),
    )


def _runtime_type_name(value: object) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list) and value:
        item_types = {_runtime_type_name(item) for item in value}
        if len(item_types) == 1 and None not in item_types:
            return f"list[{next(iter(item_types))}]"
    if isinstance(value, tuple):
        item_types = {_runtime_type_name(item) for item in value}
        if len(item_types) == 1 and None not in item_types:
            return f"tuple[{next(iter(item_types))}, ...]"
    return None


def _contract_from_examples(
    cases: list[tuple[Any, Any]],
) -> tuple[list[str], str] | None:
    if not cases:
        return None
    raw_arguments, raw_result = cases[0]
    arguments = raw_arguments if isinstance(raw_arguments, tuple) else (raw_arguments,)
    argument_types = [_runtime_type_name(argument) for argument in arguments]
    result_type = _runtime_type_name(raw_result)
    if result_type is None or any(value is None for value in argument_types):
        return None
    return [str(value) for value in argument_types], result_type


def _execution_failed(result: object) -> bool:
    execution_status = getattr(result, "execution_status", "completed")
    if execution_status == "infrastructure_error":
        raise RuntimeError("Determinacy verification infrastructure failed.")
    return not bool(getattr(result, "passed", False))


def _snippet_expected(instance: TaskInstance) -> str | None:
    metadata = instance.metadata
    mode = metadata.get("mode")
    if not isinstance(mode, int) or isinstance(mode, bool):
        return None
    limit = int(metadata["limit"])
    divisor = int(metadata["divisor"])
    offset = int(metadata["offset"])
    if mode == 0:
        if str(limit) not in instance.prompt or str(divisor) not in instance.prompt:
            return None
        value = sum(number for number in range(1, limit + 1) if number % divisor == 0)
    elif mode == 1:
        if not all(
            str(number) in instance.prompt
            for number in (limit, offset % divisor, divisor)
        ):
            return None
        value = sum(
            1
            for number in range(1, limit + 1)
            if number % divisor == offset % divisor
        )
    else:
        raw_values = metadata.get("values")
        if not isinstance(raw_values, list) or not all(
            isinstance(number, int) and not isinstance(number, bool)
            for number in raw_values
        ):
            return None
        values = [int(number) for number in raw_values]
        if repr(values) not in instance.prompt:
            return None
        if mode == 2:
            value = sum((index + offset) * number for index, number in enumerate(values))
        elif mode == 3:
            value = max(values) - min(values)
        elif mode == 4:
            value = sum(number * number for number in values if number > 0)
        else:
            return None
    return str(value)


def _compact_examples(function_name: str, cases: list[tuple[Any, Any]]) -> str:
    return "; ".join(
        f"{_function_call(function_name, value)}->{expected!r}"
        for value, expected in cases
    )


def _compact_io_examples(cases: list[tuple[Any, Any]]) -> str:
    return ";".join(f"{value!r}=>{expected!r}" for value, expected in cases)


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


def _list_contract_description(metadata: dict[str, Any]) -> str:
    mode = int(metadata["mode"])
    divisor = int(metadata["divisor"])
    scale = int(metadata["scale"])
    threshold = int(metadata["threshold"])
    offset = int(metadata["offset"])
    if mode == 0:
        return f"sum values congruent to {offset % divisor} mod {divisor}, scaled by {scale}"
    if mode == 1:
        return f"count values whose {scale}x product exceeds {threshold + offset}"
    if mode == 2:
        return f"return sorted squares plus {offset} for values below {threshold}"
    if mode == 3:
        return f"return each value clamped to [-{scale}, {scale}], then add {offset}"
    if mode == 4:
        return f"scale adjacent differences by {scale}, then add {threshold + offset}"
    if mode == 5:
        return (
            f"return the first {offset % 5} unique values ordered by distance from "
            f"{threshold}"
        )
    if mode == 6:
        return f"return alternating signed values scaled by {scale}, offset by {offset}"
    return f"sum absolute deviations from {threshold}, scaled by {scale}, plus {offset}"


def _string_contract_doc(metadata: dict[str, Any]) -> str:
    mode = int(metadata["mode"])
    step = int(metadata["step"])
    limit = int(metadata["limit"])
    character = str(metadata["character"])
    if mode == 0:
        return f"Return every {step}th character starting at index {limit % step}."
    if mode == 1:
        return f"Rotate left by {step} characters."
    if mode == 2:
        return f"Remove '{character}' ignoring case, then keep {limit} characters."
    if mode == 3:
        return f"Count '{character}' ignoring case in the first {limit} characters."
    if mode == 4:
        return f"Reverse at most the first {limit} whitespace-separated words."
    if mode == 5:
        return f"Return odd-index characters from the first {limit}, reversed."
    if mode == 6:
        return f"Replace vowels with '{character}' in the first {limit} characters."
    return f"Append the decimal length multiplied by {step}."


def _code_prompt_exposes_contract(
    instance: TaskInstance,
    kind: str,
    function_name: str | None,
    prompt_cases: list[tuple[Any, Any]],
    naive: str,
) -> bool:
    metadata = instance.metadata
    if kind == "snippet":
        mode = int(metadata["mode"])
        limit = int(metadata["limit"])
        divisor = int(metadata["divisor"])
        offset = int(metadata["offset"])
        values = metadata.get("values")
        if mode == 0:
            expected_prompt = (
                f"Write Python that sets `res` to the integer sum of all multiples of "
                f"{divisor} in the inclusive range 1..{limit}. Output executable code."
            )
        elif mode == 1:
            expected_prompt = (
                f"Write Python that sets `res` to the integer count of values in the "
                f"inclusive range 1..{limit} congruent to {offset % divisor} modulo "
                f"{divisor}. Output executable code."
            )
        elif mode == 2:
            expected_prompt = (
                f"Write Python that sets `res` to the weighted sum of {values}; the value "
                f"at zero-based index i has weight i + {offset}. Output executable code."
            )
        elif mode == 3:
            expected_prompt = (
                f"Write Python that sets `res` to max(values) - min(values) for {values}. "
                "Output executable code."
            )
        else:
            expected_prompt = (
                f"Write Python that sets `res` to the sum of x*x for strictly positive "
                f"values x in {values}. Output executable code."
            )
        return instance.prompt == expected_prompt
    if function_name is None:
        return False
    if instance.task_family == "T9_algo_synth":
        expected_prompt = (
            f"Write {function_name}(xs): {_list_contract_description(metadata)}. Examples: "
            f"{_compact_examples(function_name, prompt_cases)}\n"
        )
    elif instance.task_family == "T10_docstring_code":
        expected_prompt = (
            f'def {function_name}(text):\n    """{_string_contract_doc(metadata)} Examples: '
            f'{_compact_io_examples(prompt_cases)}"""\n'
        )
    elif instance.task_family == "T11_bug_fixer":
        expected_prompt = (
            f"Fix {function_name}(xs): {_compact_list_contract(metadata)}. "
            f"Ex: {_compact_io_examples(prompt_cases)}\n{naive}\n"
        )
    elif instance.task_family == "T12_type_inferrer":
        untyped_source = metadata.get("untyped_source")
        if not isinstance(untyped_source, str):
            return False
        expected_prompt = (
            "Infer most-specific PEP585 types; keep body. "
            f"Ex: {_compact_io_examples(prompt_cases)}\n{untyped_source}\n"
        )
    else:
        return False
    return instance.prompt == expected_prompt


def _code_determinacy_plan(
    instance: TaskInstance, kind: str
) -> tuple[tuple[str, str] | None, list[SandboxBatchRequest]]:
    metadata = instance.metadata
    if kind == "snippet":
        expected = _snippet_expected(instance)
        naive = metadata.get("naive_solution")
        if (
            not isinstance(expected, str)
            or expected != metadata.get("expected_result")
            or not isinstance(naive, str)
        ):
            return (
                "ambiguous",
                "snippet prompt lacks executable determinacy evidence",
            ), []
        if not _code_prompt_exposes_contract(instance, kind, None, [], naive):
            return ("ambiguous", "snippet prompt does not expose its full contract"), []
        return None, [
            SandboxBatchRequest("snippet", instance.target_solution, expected=expected),
            SandboxBatchRequest("snippet", naive, expected=expected),
        ]

    function_name = metadata.get("fn_name")
    prompt_cases = metadata.get("prompt_cases")
    held_out_cases = metadata.get("held_out_cases")
    naive = metadata.get("naive_solution")
    if (
        not isinstance(function_name, str)
        or not isinstance(prompt_cases, list)
        or len(prompt_cases) < 2
        or not isinstance(held_out_cases, list)
        or not held_out_cases
        or not isinstance(naive, str)
    ):
        return (
            "ambiguous",
            "code prompt lacks two examples, held-out cases, or a naive variant",
        ), []
    if not _cases_are_visible(
        instance.task_family, instance.prompt, function_name, prompt_cases
    ):
        return ("ambiguous", "declared examples are not visible in the prompt"), []
    if not _code_prompt_exposes_contract(
        instance, kind, function_name, prompt_cases, naive
    ):
        return ("ambiguous", "natural-language code contract is incomplete"), []

    all_cases = prompt_cases + held_out_cases
    requests = [
        SandboxBatchRequest(
            "function",
            instance.target_solution,
            function_name=function_name,
            test_cases=all_cases,
        )
    ]
    if kind == "typed_function":
        expected_contract = (
            [str(value) for value in metadata.get("argument_types", [])],
            str(metadata.get("return_type", "")),
        )
        if _contract_from_examples(prompt_cases) != expected_contract:
            return (
                "ambiguous",
                "typed examples do not uniquely expose the declared annotation contract",
            ), []
        if _annotation_contract(instance.target_solution) != expected_contract:
            return (
                "invalid",
                "target annotations do not match the stated type contract",
            ), []
        if _annotation_contract(naive) == expected_contract:
            return (
                "ambiguous",
                "naive annotations also match the stated type contract",
            ), []
        return None, requests

    for case in prompt_cases:
        requests.append(
            SandboxBatchRequest(
                "function",
                naive,
                function_name=function_name,
                test_cases=[case],
            )
        )
    return None, requests


def _interpret_code_results(
    kind: str, results: list[SandboxResult]
) -> tuple[str, str]:
    if not results or _execution_failed(results[0]):
        return "invalid", "target fails prompt-visible or held-out cases"
    if kind == "snippet":
        if len(results) != 2:
            return "invalid", "snippet determinacy execution result is incomplete"
        if not _execution_failed(results[1]):
            return "ambiguous", "naive snippet also satisfies the stated computation"
        return "valid", "target passes and naive snippet fails the prompt computation"
    if kind == "typed_function":
        return "valid", "target executes examples and uniquely matches the stated annotations"
    if any(not _execution_failed(result) for result in results[1:]):
        return "ambiguous", "naive or buggy program passes a stated example"
    return "valid", "target passes stated/held-out cases and naive program fails each example"


def _determinacy_outcome(instance: TaskInstance) -> tuple[str, str]:
    kind = instance.metadata.get("determinacy_kind")
    if not isinstance(kind, str):
        return "ambiguous", "no machine-checkable prompt determinacy contract"
    if kind == "sequence":
        raw_values = instance.metadata.get("shown_values")
        if not isinstance(raw_values, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_values
        ):
            return "invalid", "sequence metadata is malformed"
        candidates = simple_sequence_continuations(raw_values)
        continuations = set(candidates.values())
        target = ", ".join(map(str, next(iter(continuations)))) if len(continuations) == 1 else None
        if len(continuations) != 1:
            return "ambiguous", f"simple rule library yields {len(continuations)} continuations"
        if target != instance.target_solution:
            return "invalid", "unique simple-rule continuation does not match target"
        shown = ", ".join(map(str, raw_values))
        expected_prompt = (
            "Using the simplest exact rule among arithmetic, integer geometric, "
            "constant-second-difference quadratic, or a[n]=a[n-1]+m*a[n-2]+c with "
            f"m in 1..4 and c in -9..9, predict three values: {shown}, _, _, _ ="
        )
        if instance.prompt != expected_prompt:
            return "ambiguous", "sequence rule library or shown terms are absent from prompt"
        return "valid", "simple rule library yields one continuation"
    if kind in {"snippet", "function", "typed_function"}:
        raise RuntimeError("Code determinacy must be evaluated through the batch path.")
    expected = _direct_expected(instance, kind)
    if expected is None:
        return "invalid", f"unknown or malformed determinacy kind '{kind}'"
    if not _direct_prompt_exposes_contract(instance, kind):
        return "ambiguous", "prompt omits data or serialization needed to determine the answer"
    if expected != instance.target_solution:
        return "invalid", "independently recomputed answer does not match target"
    return "valid", "prompt contract independently recomputes to the target"


def _determinacy_metrics(instances: list[TaskInstance]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    outcomes: list[tuple[str, str] | None] = [None] * len(instances)
    plans: list[tuple[int, str, list[SandboxBatchRequest]]] = []
    for index, instance in enumerate(instances):
        kind = instance.metadata.get("determinacy_kind")
        if isinstance(kind, str) and kind in {"snippet", "function", "typed_function"}:
            precomputed, requests = _code_determinacy_plan(instance, kind)
            if precomputed is not None:
                outcomes[index] = precomputed
            else:
                plans.append((index, kind, requests))
        else:
            outcomes[index] = _determinacy_outcome(instance)

    for chunk_start in range(0, len(plans), 32):
        chunk = plans[chunk_start : chunk_start + 32]
        requests = [request for _, _, plan in chunk for request in plan]
        results = verify_source_batch(requests)
        result_offset = 0
        for index, kind, plan in chunk:
            next_offset = result_offset + len(plan)
            outcomes[index] = _interpret_code_results(
                kind, results[result_offset:next_offset]
            )
            result_offset = next_offset

    for instance, result in zip(instances, outcomes, strict=True):
        assert result is not None
        outcome, reason = result
        counts[outcome] += 1
        if outcome != "valid" and len(failures) < 5:
            failures.append({"task_id": instance.task_id, "outcome": outcome, "reason": reason})
    return {
        "checked_items": len(instances),
        "valid_items": counts["valid"],
        "ambiguous_items": counts["ambiguous"],
        "invalid_items": counts["invalid"],
        "failure_examples": failures,
    }


def _family_audit(
    stream: ContinualTaskStream,
    family: TaskFamily,
    sample_size: int,
    contamination_strings: set[str],
) -> dict[str, object]:
    split = stream.get_split(family.family_id)
    train_prompts = {item.prompt for item in split.train}
    test_prompts = {item.prompt for item in split.test}
    train_targets = {item.target_solution for item in split.train}
    test_targets = {item.target_solution for item in split.test}
    structural_applicable = family.family_id in CODE_FAMILY_IDS
    train_structures = {
        fingerprint
        for item in split.train
        if (fingerprint := structural_hash(item.target_solution)) is not None
    }
    test_structures = {
        fingerprint
        for item in split.test
        if (fingerprint := structural_hash(item.target_solution)) is not None
    }
    contamination: dict[str, dict[str, int]] = {}
    for split_name, instances in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        contamination[split_name] = {
            "prompt_matches": sum(
                item.prompt in contamination_strings for item in instances
            ),
            "target_matches": sum(
                item.target_solution in contamination_strings for item in instances
            ),
        }
    return {
        "family_id": family.family_id,
        "role": stream.family_roles[family.family_id],
        "headline_eligible": family.family_id
        in {
            stream.families[index].family_id for index in stream.headline_task_indices
        },
        "splits": {
            "train": _split_metrics(split.train),
            "val": _split_metrics(split.val),
            "test": _split_metrics(split.test),
        },
        "train_test_overlap": {
            "exact_prompts": len(train_prompts & test_prompts),
            "exact_targets": len(train_targets & test_targets),
            "structural_targets": (
                len(train_structures & test_structures)
                if structural_applicable
                else None
            ),
            "distinct_train_structures": (
                len(train_structures) if structural_applicable else None
            ),
            "distinct_test_structures": (
                len(test_structures) if structural_applicable else None
            ),
        },
        "identity_chat_contamination": contamination,
        "trivial_baselines": trivial_baseline_scores(family, split, sample_size),
        "determinacy": _determinacy_metrics(split.train + split.val + split.test),
    }


def audit_stream(
    stream: ContinualTaskStream,
    policy: AuditPolicy,
    *,
    baseline_sample_size: int,
    stream_config: str,
) -> dict[str, object]:
    """Compute a complete deterministic audit record and policy verdict."""
    contamination_strings = _training_contamination_strings()
    families = [
        _family_audit(
            stream,
            family,
            baseline_sample_size,
            contamination_strings,
        )
        for family in stream.families
    ]
    violations: list[str] = []
    for family in families:
        family_id = str(family["family_id"])
        overlap = family["train_test_overlap"]
        assert isinstance(overlap, dict)
        target_overlap = overlap["exact_targets"]
        structural_overlap = overlap["structural_targets"]
        if isinstance(target_overlap, int) and target_overlap > policy.max_target_overlap_count:
            violations.append(
                f"{family_id}: target overlap {target_overlap} exceeds "
                f"{policy.max_target_overlap_count}"
            )
        if (
            isinstance(structural_overlap, int)
            and structural_overlap > policy.max_structural_overlap_count
        ):
            violations.append(
                f"{family_id}: structural overlap {structural_overlap} exceeds "
                f"{policy.max_structural_overlap_count}"
            )
        splits = family["splits"]
        assert isinstance(splits, dict)
        test_metrics = splits["test"]
        assert isinstance(test_metrics, dict)
        chance = test_metrics["uniform_target_chance"]
        if isinstance(chance, float) and chance > policy.max_uniform_chance:
            violations.append(
                f"{family_id}: uniform chance {chance:.6f} exceeds "
                f"{policy.max_uniform_chance:.6f}"
            )
        contamination = family["identity_chat_contamination"]
        assert isinstance(contamination, dict)
        contamination_count = 0
        for counts in contamination.values():
            assert isinstance(counts, dict)
            contamination_count += int(counts["prompt_matches"])
            contamination_count += int(counts["target_matches"])
        if contamination_count > policy.max_identity_chat_contamination_count:
            violations.append(
                f"{family_id}: identity/chat contamination {contamination_count} exceeds "
                f"{policy.max_identity_chat_contamination_count}"
            )
        baselines = family["trivial_baselines"]
        assert isinstance(baselines, dict)
        for baseline in (
            "most_frequent_answer",
            "copy_prompt",
            "nearest_training_prompt",
        ):
            score = baselines[baseline]
            if (
                isinstance(score, float)
                and score > policy.max_trivial_baseline_accuracy
            ):
                violations.append(
                    f"{family_id}: {baseline} accuracy {score:.6f} exceeds "
                    f"{policy.max_trivial_baseline_accuracy:.6f}"
                )
        determinacy = family["determinacy"]
        assert isinstance(determinacy, dict)
        ambiguous_count = int(determinacy["ambiguous_items"])
        invalid_count = int(determinacy["invalid_items"])
        failure_count = ambiguous_count + invalid_count
        if failure_count > policy.max_ambiguous_item_count:
            violations.append(
                f"{family_id}: ambiguous/invalid item count {failure_count} exceeds "
                f"{policy.max_ambiguous_item_count}"
            )
    return {
        "schema": AUDIT_SCHEMA,
        "stream_version": stream.version,
        "stream_config": stream_config,
        "seed": stream.seed,
        "split_sizes": {
            "train": stream.num_train_per_task,
            "val": stream.num_val_per_task,
            "test": stream.num_test_per_task,
        },
        "policy": asdict(policy),
        "passed": not violations,
        "violations": violations,
        "families": families,
    }


def build_plan_records(plan: AuditPlan) -> list[tuple[AuditPlanEntry, dict[str, object]]]:
    """Materialize and audit every stream named by a plan."""
    records: list[tuple[AuditPlanEntry, dict[str, object]]] = []
    for entry in plan.streams:
        settings = load_experiment_settings(entry.stream_config)
        stream = settings.make_stream()
        records.append(
            (
                entry,
                audit_stream(
                    stream,
                    entry.policy,
                    baseline_sample_size=plan.baseline_sample_size,
                    stream_config=entry.stream_config.as_posix(),
                ),
            )
        )
    return records


def render_record(record: dict[str, object]) -> str:
    """Render stable JSON suitable for checked-in evidence."""
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def write_or_check_plan(plan: AuditPlan, *, check: bool) -> None:
    """Write records or verify checked-in records and enforce every policy."""
    failures: list[str] = []
    for entry, record in build_plan_records(plan):
        rendered = render_record(record)
        if check:
            if not entry.output.exists():
                failures.append(f"missing recorded audit: {entry.output}")
            elif entry.output.read_text(encoding="utf-8") != rendered:
                failures.append(f"recorded audit is stale: {entry.output}")
        else:
            entry.output.parent.mkdir(parents=True, exist_ok=True)
            entry.output.write_text(rendered, encoding="utf-8")
        violations = record["violations"]
        assert isinstance(violations, list)
        failures.extend(str(violation) for violation in violations)
    if failures:
        raise RuntimeError("Task stream audit failed:\n" + "\n".join(failures))
