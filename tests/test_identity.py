"""Acceptance tests for manifest-generated identity tasks."""

import pytest

from aptivra.inference.chat_canonical import (
    ReleaseMetric,
    assert_permitted_identity_target,
    canonical_identity_records,
)
from aptivra.tasks.identity import ModelIdentityTaskFamily


def test_model_identity_task_family() -> None:
    family = ModelIdentityTaskFamily()
    assert family.family_id == "model_identity_capabilities"

    instances = family.generate_instances(count=6, seed=42)
    assert len(instances) == 6
    assert all("<|im_start|>assistant\n" in item.prompt for item in instances)
    assert all("github" not in item.target_solution.lower() for item in instances)

    result = family.verifier.verify(instances[1], instances[1].target_solution)
    assert result.passed is True
    assert not family.verifier.verify(instances[1], "A sanitized near miss").passed

    capabilities = next(
        record for record in canonical_identity_records() if record.prompt_intent == "capabilities"
    )
    assert capabilities.answer.startswith("No release evidence is available")


@pytest.mark.parametrize(
    ("forbidden", "reason"),
    [
        ("I am autonomous.", "autonomous"),
        ("I use a Top-2 MoE.", "Top-2 MoE"),
        ("I am a 200M model.", "200M"),
        ("I am a 200 million parameter model.", "200M"),
        ("I guarantee zero forgetting.", "zero forgetting"),
        ("I guarantee zero catastrophic forgetting.", "zero forgetting"),
        ("I have 5,716 skills.", "skill count"),
        ("I have 8+ MCP skills.", "skill count"),
        ("I support native execution.", "native execution"),
        ("My C-engine is ready.", "native execution"),
        ("My throughput is 10 tokens per second.", "throughput"),
        ("The runtime transfers 10 MB/s.", "throughput"),
        ("My unbound accuracy is 99%.", "unbound metric"),
        ("See GitHub for details.", "source-host reference"),
    ],
)
def test_forbidden_identity_targets_fail_generation(
    forbidden: str,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=f"forbidden claim: {reason}"):
        assert_permitted_identity_target(forbidden)


@pytest.mark.parametrize("term", ["accuracy", "bpb", "bwt", "fwt", "metric", "score"])
def test_each_unbound_metric_term_fails_generation(term: str) -> None:
    with pytest.raises(ValueError, match="forbidden claim: unbound metric"):
        assert_permitted_identity_target(f"The reported {term} is 0.5.")


def test_forbidden_checker_accepts_only_precise_near_misses() -> None:
    for permitted in (
        "It is not a Top-2 mixture-of-experts model.",
        "It has 199 million parameters.",
        "It has 201M parameters.",
        "It is skilled at concise answers.",
        "It runs in a C implementation.",
    ):
        assert_permitted_identity_target(permitted)


def test_bound_metric_must_name_the_exact_release_metric() -> None:
    metric = ReleaseMetric(
        name="exact_match_accuracy",
        value=0.75,
        unit="ratio",
    )
    assert_permitted_identity_target(
        "Bound release evidence reports exact_match_accuracy=0.75 ratio.",
        allowed_metric=metric,
    )
    with pytest.raises(ValueError, match="forbidden claim: release metric binding"):
        assert_permitted_identity_target(
            "Bound release evidence reports another score.",
            allowed_metric=metric,
        )
