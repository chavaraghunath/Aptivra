"""Tests for deterministic task-stream audit evidence and policy gates."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aptivra.eval.task_stream_audit import (
    AuditPolicy,
    _determinacy_metrics,
    audit_stream,
    load_audit_plan,
    write_or_check_plan,
)
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.identity import ModelIdentityTaskFamily
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamSplit


def test_recorded_v1_and_v2_audits_are_current_and_pass_policy() -> None:
    plan = load_audit_plan("configs/task_stream_audit.yaml")
    write_or_check_plan(plan, check=True)

    for entry in plan.streams:
        record = json.loads(entry.output.read_text(encoding="utf-8"))
        assert record["passed"] is True
        assert len(record["families"]) == 12


def test_strict_v2_policy_rejects_known_v1_defects() -> None:
    stream = ContinualTaskStream(20, 10, 10, seed=42, version="v1")
    policy = AuditPolicy(
        max_target_overlap_count=0,
        max_structural_overlap_count=0,
        max_identity_chat_contamination_count=0,
        max_uniform_chance=0.05,
        max_trivial_baseline_accuracy=0.1,
        max_ambiguous_item_count=0,
    )
    record = audit_stream(
        stream,
        policy,
        baseline_sample_size=5,
        stream_config="test-v1",
    )
    assert record["passed"] is False
    assert record["violations"]


def test_audit_detects_identity_training_string_contamination() -> None:
    stream = ContinualTaskStream(2, 1, 1, seed=42, version="v2")
    family = stream.families[0]
    split = stream.get_split(family.family_id)
    identity = ModelIdentityTaskFamily().generate_instances(1, seed=42)[0]
    contaminated = replace(identity, task_family=family.family_id)
    stream._splits[family.family_id] = TaskStreamSplit(
        train=split.train,
        val=split.val,
        test=[contaminated],
    )
    policy = AuditPolicy(0, 0, 0, 0.05, 0.1, 0)

    record = audit_stream(
        stream,
        policy,
        baseline_sample_size=1,
        stream_config="contaminated-test",
    )

    assert record["passed"] is False
    assert any("identity/chat contamination" in item for item in record["violations"])


def test_audit_plan_rejects_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "bad-audit.yaml"
    path.write_text(
        """task_stream_audit:
  baseline_sample_size: 0
  streams: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline_sample_size"):
        load_audit_plan(path)


def test_determinacy_audit_rejects_prompt_hidden_examples() -> None:
    stream = ContinualTaskStream(1, 1, 1, seed=42, version="v2")
    instance = stream.get_split("T11_bug_fixer").test[0]
    obscured = replace(instance, prompt="# Repair this function without a specification.")

    metrics = _determinacy_metrics([obscured])

    assert metrics["valid_items"] == 0
    assert metrics["ambiguous_items"] == 1
    assert metrics["invalid_items"] == 0


def test_determinacy_audit_counts_multiple_simple_sequence_continuations() -> None:
    instance = TaskInstance(
        task_id="ambiguous-sequence",
        task_family="T6_sequence_completion",
        prompt=(
            "Using the simplest exact rule among arithmetic, integer geometric, "
            "constant-second-difference quadratic, or a[n]=a[n-1]+m*a[n-2]+c with "
            "m in 1..4 and c in -9..9, predict three values: -5, -4, -2, 1, _, _, _ ="
        ),
        target_solution="5, 10, 16",
        metadata={"determinacy_kind": "sequence", "shown_values": [-5, -4, -2, 1]},
    )

    metrics = _determinacy_metrics([instance])

    assert metrics["ambiguous_items"] == 1
    assert metrics["invalid_items"] == 0
