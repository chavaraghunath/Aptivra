"""Regression tests for the versioned v2 continual task stream."""

import hashlib
import json
from pathlib import Path

import pytest

from aptivra.eval.config import load_experiment_settings
from aptivra.eval.harness import ContinualEvalHarness
from aptivra.tasks.code_sandbox import verify_function_source
from aptivra.tasks.stream import ContinualTaskStream
from aptivra.tasks.structure import structural_hash
from aptivra.tasks.v2 import (
    BugFixerTaskFamilyV2,
    TypeInferrerTaskFamilyV2,
    simple_sequence_continuations,
)


def _v1_snapshot() -> tuple[int, int, str]:
    stream = ContinualTaskStream()
    records: list[dict[str, object]] = []
    for family in stream.families:
        split = stream.get_split(family.family_id)
        for split_name, instances in (
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ):
            for instance in instances:
                records.append(
                    {
                        "family_id": family.family_id,
                        "split": split_name,
                        "task_id": instance.task_id,
                        "task_family": instance.task_family,
                        "prompt": instance.prompt,
                        "target_solution": instance.target_solution,
                        "metadata": instance.metadata,
                        "answer_stop_characters": instance.answer_stop_characters,
                        "training_terminator": instance.training_terminator,
                    }
                )
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return len(records), len(payload), hashlib.sha256(payload).hexdigest()


def test_v1_default_stream_is_byte_stable() -> None:
    assert _v1_snapshot() == (
        10_800,
        3_894_503,
        "81c88937debbc55c6e34e9ce42d89749301d0d2ea37829e514cbabcc740b4a19",
    )


def test_config_selects_v2_and_rejects_unknown_version(tmp_path: Path) -> None:
    settings = load_experiment_settings("configs/task_stream_v2.yaml")
    assert settings.task_stream.version == "v2"
    assert settings.make_stream().version == "v2"

    invalid = tmp_path / "invalid-version.yaml"
    invalid.write_text(
        """task_stream:
  version: future
  num_train_per_task: 10
  num_val_per_task: 5
  num_test_per_task: 5
  seed: 42
  max_generation_attempts: 10
evaluation:
  max_new_tokens: 128
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        load_experiment_settings(invalid)


def test_v2_is_deterministic_and_seed_sensitive() -> None:
    first = ContinualTaskStream(10, 5, 5, seed=42, version="v2")
    repeated = ContinualTaskStream(10, 5, 5, seed=42, version="v2")
    alternate = ContinualTaskStream(10, 5, 5, seed=43, version="v2")

    first_targets = [
        item.target_solution
        for family in first.families
        for item in first.get_split(family.family_id).train
    ]
    repeated_targets = [
        item.target_solution
        for family in repeated.families
        for item in repeated.get_split(family.family_id).train
    ]
    alternate_targets = [
        item.target_solution
        for family in alternate.families
        for item in alternate.get_split(family.family_id).train
    ]
    assert first_targets == repeated_targets
    assert first_targets != alternate_targets


def test_structural_hash_ignores_names_but_not_algorithms() -> None:
    first = "def alpha(values):\n    return sum(item for item in values)"
    renamed = "def beta(numbers):\n    return sum(number for number in numbers)"
    changed = "def beta(numbers):\n    return max(number for number in numbers)"

    assert structural_hash(first) == structural_hash(renamed)
    assert structural_hash(first) != structural_hash(changed)
    assert structural_hash("def broken(") is None


def test_v2_targets_are_unseen_and_code_targets_verify_in_child_runner() -> None:
    stream = ContinualTaskStream(20, 10, 10, seed=42, version="v2")
    for family in stream.families:
        split = stream.get_split(family.family_id)
        assert {item.target_solution for item in split.train}.isdisjoint(
            item.target_solution for item in split.test
        )
    for family in stream.families[7:11]:
        instance = stream.get_split(family.family_id).test[0]
        assert family.verifier.verify(instance, instance.target_solution).passed is True


def test_bug_fixer_prompts_state_intent_and_two_discriminating_examples() -> None:
    family = BugFixerTaskFamilyV2()
    by_mode = {
        int(item.metadata["mode"]): item
        for item in family.generate_instances(128, seed=42101)
    }
    assert set(by_mode) == set(range(8))

    for instance in by_mode.values():
        prompt_cases = instance.metadata["prompt_cases"]
        assert len(prompt_cases) == 2
        assert instance.prompt.startswith("Fix ")
        assert instance.prompt.splitlines()[0].count("=>") >= 2
        function_name = instance.metadata["fn_name"]
        target_result = verify_function_source(
            instance.target_solution,
            function_name,
            prompt_cases + instance.metadata["held_out_cases"],
        )
        assert target_result.passed is True
        for case in prompt_cases:
            broken_result = verify_function_source(
                instance.metadata["naive_solution"], function_name, [case]
            )
            assert broken_result.passed is False


def test_sequence_rule_library_exposes_alternative_simple_continuations() -> None:
    candidates = simple_sequence_continuations([-5, -4, -2, 1])
    assert candidates["quadratic"] == (5, 10, 16)
    assert candidates["recurrence_m1_c7"] == (6, 14, 27)


def test_type_inference_prompt_is_explicit_and_verifier_accepts_formatting() -> None:
    family = TypeInferrerTaskFamilyV2()
    instance = family.generate_instances(1, seed=42111)[0]
    compact = instance.target_solution.replace(":\n    ", ": ")

    assert "Infer most-specific PEP585 types" in instance.prompt
    assert instance.prompt.splitlines()[0].count("=>") >= 2
    assert family.verifier.verify(instance, compact).passed is True


def test_bug_fixer_verifier_accepts_distinct_correct_program_on_held_out_cases() -> None:
    family = BugFixerTaskFamilyV2()
    instance = next(
        item
        for item in family.generate_instances(128, seed=42101)
        if item.metadata["mode"] == 7
    )
    function_name = instance.metadata["fn_name"]
    alternative = (
        f"def {function_name}(xs):\n"
        f"    total = {instance.metadata['offset']}\n"
        "    for value in xs:\n"
        f"        total += abs(value - {instance.metadata['threshold']}) * "
        f"{instance.metadata['scale']}\n"
        "    return total"
    )

    assert alternative != instance.target_solution
    assert family.verifier.verify(instance, alternative).passed is True


def test_v2_diagnostic_is_measured_but_excluded_from_headline_aggregates() -> None:
    stream = ContinualTaskStream(1, 1, 1, seed=42, version="v2")
    assert stream.family_roles["T2_string_transforms"] == (
        "tokenizer_sensitivity_diagnostic"
    )
    assert 1 not in stream.headline_task_indices

    run = ContinualEvalHarness(stream).start_run()
    run.accuracy_matrix[:] = 0.5
    run.random_baseline_accs[:] = 0.0
    run.accuracy_matrix[-1, 1] = 1.0
    with_high_diagnostic = run.finalize()
    run.accuracy_matrix[-1, 1] = 0.0
    with_low_diagnostic = run.finalize()

    assert with_high_diagnostic.average_accuracy == with_low_diagnostic.average_accuracy
    assert "T2_string_transforms" not in with_high_diagnostic.max_forgetting
    assert with_high_diagnostic.metadata["task_count"] == 12
    assert "T2_string_transforms" not in with_high_diagnostic.metadata[
        "headline_family_ids"
    ]
