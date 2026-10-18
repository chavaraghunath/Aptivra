"""ADR-0011 lane-5 router, evidence, rollback, and restart acceptance tests."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from aptivra.consolidation.cycle import ConsolidationCycle, load_accepted_cycle
from aptivra.experts.expert import RoutedExpert
from aptivra.experts.router import (
    RouteDecision,
    SequenceTop1Router,
    masked_mean_frozen_embeddings,
    route_and_activate_fallback,
)
from aptivra.experts.spawner import (
    EvidenceGrowthPolicy,
    GrowthSettings,
    VerifierInfrastructureError,
    VerifierOutcome,
    build_growth_evidence,
)
from aptivra.experts.standard import (
    ExpertManifest,
    PeftExpertBank,
    StandardExpertSettings,
    load_standard_expert_settings,
)
from aptivra.gate.promotion_gate import PromotionResult
from aptivra.inference.standard_base import (
    StandardBase,
    load_standard_base,
    load_standard_base_settings,
)
from aptivra.tasks.base import TaskInstance
from aptivra.v0.experts.expert import ExpertModule
from tests.test_standard_base import _tiny_snapshot


def _settings() -> StandardExpertSettings:
    settings = load_standard_expert_settings()
    return settings.model_copy(
        update={
            "feature": settings.feature.model_copy(update={"enabled": True}),
            "router": settings.router.model_copy(
                update={
                    "confidence_threshold": 0.8,
                    "learning_rate": 0.1,
                    "max_optimizer_steps": 20,
                    "replay_examples_per_expert": 2,
                }
            ),
            "growth": settings.growth.model_copy(
                update={"minimum_observations": 2}
            ),
            "consolidation": settings.consolidation.model_copy(
                update={
                    "adapter_learning_rate": 0.05,
                    "max_adapter_optimizer_steps": 2,
                    "max_target_tokens": 32,
                    "minimum_regression_samples": 5,
                }
            ),
        }
    )


def _base(tmp_path: Path) -> StandardBase:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    return load_standard_base(
        load_standard_base_settings(config_path), snapshot_path=snapshot, device="cpu"
    )


def _instance(task_id: str, family: str = "new") -> TaskInstance:
    return TaskInstance(task_id, family, "prompt", "answer")


def _real_features(bank: PeftExpertBank, texts: list[str]) -> torch.Tensor:
    encoded = bank.base.tokenizer(
        texts,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return masked_mean_frozen_embeddings(
        bank.model,
        cast(torch.Tensor, encoded["input_ids"]),
        cast(torch.Tensor, encoded["attention_mask"]),
    )


def test_router_uses_masked_real_frozen_embeddings_and_abstains() -> None:
    embedding = torch.nn.Embedding(8, 2)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 2.0],
                    [9.0, 9.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )
    embedding.weight.requires_grad_(False)

    class FrozenEmbeddingModel:
        def get_input_embeddings(self) -> torch.nn.Module:
            return embedding

    inputs = torch.tensor([[1, 2, 3], [1, 3, 3]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    features = masked_mean_frozen_embeddings(FrozenEmbeddingModel(), inputs, mask)
    assert torch.equal(features, torch.tensor([[1.5, 1.0], [2.0, 0.0]]))

    router = SequenceTop1Router(2, confidence_threshold=0.8)
    router.add_candidate("old", "old-family")
    router.promote_candidate()
    router.add_candidate("new", "new-family")
    with torch.no_grad():
        router.weights[0].copy_(torch.tensor([1.0, 0.0]))
        router.weights[1].copy_(torch.tensor([-1.0, 0.0]))
    abstained, routed = router.route(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
    assert abstained.route_status == "unrouted"
    assert abstained.expert_id is None
    assert routed.route_status == "routed"
    assert routed.expert_id == "old"
    assert abstained.entropy > 0.0
    assert routed.entropy > 0.0
    with pytest.raises(FrozenInstanceError):
        routed.confidence = 0.0  # type: ignore[misc]

    exact = SequenceTop1Router(1, 0.5)
    exact.add_candidate("first", "first")
    exact.promote_candidate()
    exact.add_candidate("second", "second")
    with torch.no_grad():
        exact.weights[0].zero_()
        exact.weights[1].zero_()
        exact.biases[0].fill_(math.log(4.0))
        exact.biases[1].zero_()
    exact_confidence = float(exact(torch.ones(1, 1))[0, 0].item())
    exact.confidence_threshold = exact_confidence
    exact_decision = exact.route(torch.ones(1, 1))[0]
    assert exact_decision.route_status == "routed"
    assert torch.allclose(
        exact.logits(torch.ones(1, 1)), torch.tensor([[math.log(4.0), 0.0]])
    )

    extreme = SequenceTop1Router(1, 0.5)
    extreme.add_candidate("positive", "positive")
    extreme.promote_candidate()
    extreme.add_candidate("negative", "negative")
    with torch.no_grad():
        extreme.weights[0].fill_(1_000.0)
        extreme.weights[1].fill_(-1_000.0)
    assert math.isfinite(extreme.route(torch.ones(1, 1))[0].entropy)

    class RecordingBank:
        def __init__(self) -> None:
            self.active: str | None = "unexpected"
            self.action = "none"

        def activate(self, adapter_name: str) -> None:
            self.active = adapter_name
            self.action = "activate"

        def deactivate(self) -> None:
            self.active = None
            self.action = "deactivate"

    bank = RecordingBank()
    fallback = route_and_activate_fallback(bank, router, torch.tensor([[0.0, 1.0]]))
    assert fallback.route_status == "unrouted"
    assert bank.active is None
    assert bank.action == "deactivate"
    selected = route_and_activate_fallback(bank, router, torch.tensor([[2.0, 0.0]]))
    assert selected.expert_id == "old"
    assert bank.active == "old"
    assert bank.action == "activate"


def test_router_replay_calibrates_only_new_row_and_exposes_misroute() -> None:
    router = SequenceTop1Router(2, confidence_threshold=0.7)
    router.add_candidate("old", "old-family")
    with torch.no_grad():
        router.weights[0].copy_(torch.tensor([-1.0, 0.0]))
    router.promote_candidate()
    old_row = router.weights[0].detach().clone()
    router.add_candidate("new", "new-family")
    result, _optimizer = router.fit_candidate(
        torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        torch.tensor([[-2.0, 0.0], [-3.0, 0.0]]),
        torch.tensor([0, 0]),
        learning_rate=0.2,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_steps=30,
    )
    assert result.current_examples == 2
    assert result.replay_examples == 2
    assert np.isfinite(result.loss)
    assert _optimizer.defaults["lr"] == 0.2
    assert _optimizer.defaults["weight_decay"] == 0.01
    assert _optimizer.defaults["betas"] == (0.9, 0.999)
    assert _optimizer.defaults["eps"] == 1e-8
    with pytest.raises(FrozenInstanceError):
        result.loss = 0.0  # type: ignore[misc]
    assert torch.equal(router.weights[0], old_row)
    assert router.weights[0].requires_grad is False
    assert router.route(torch.tensor([[2.0, 0.0]]))[0].expert_id == "new"
    assert router.route(torch.tensor([[-2.0, 0.0]]))[0].expert_id == "old"
    misroute = router.route(torch.tensor([[2.0, 0.0]]))[0]
    evidence = build_growth_evidence(
        (misroute, misroute),
        (VerifierOutcome(False), VerifierOutcome(True)),
        ("old", "old"),
    )
    assert evidence.misroutes == 2
    assert evidence.abstentions == 0
    assert evidence.verifier_failures == 1
    assert evidence.verifier_failure_rate == 0.5
    assert evidence.mean_router_entropy == pytest.approx(misroute.entropy)
    decision = EvidenceGrowthPolicy(
        GrowthSettings(0.5, 10.0, 2, 4)
    ).decide(2, evidence)
    assert decision.grow is True
    assert decision.reason == "verifier_failure_threshold"


def test_growth_propagates_real_verifier_infrastructure_failure() -> None:
    route = SequenceTop1Router(1, 0.5)
    route.add_candidate("only", "family")
    decision = route.route(torch.ones(1, 1))[0]
    with pytest.raises(VerifierInfrastructureError, match="sandbox timed out"):
        build_growth_evidence(
            (decision,),
            (VerifierOutcome(None, "sandbox timed out"),),
            ("only",),
        )


def test_router_input_and_configuration_guardrails() -> None:
    embedding = torch.nn.Embedding(4, 2)

    class EmbeddingModel:
        def get_input_embeddings(self) -> torch.nn.Module:
            return embedding

    model = EmbeddingModel()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="rank-2"):
        masked_mean_frozen_embeddings(model, ids[0], mask[0])
    with pytest.raises(ValueError, match="rank-2"):
        masked_mean_frozen_embeddings(model, ids.unsqueeze(0), mask.unsqueeze(0))
    with pytest.raises(ValueError, match="equal rank-2"):
        masked_mean_frozen_embeddings(model, ids, torch.ones(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="equal rank-2"):
        masked_mean_frozen_embeddings(model, ids, torch.ones(1, 1, dtype=torch.long))
    with pytest.raises(TypeError, match="torch.long"):
        masked_mean_frozen_embeddings(model, ids.float(), mask)
    with pytest.raises(ValueError, match="zero or one"):
        masked_mean_frozen_embeddings(model, ids, torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="zero or one"):
        masked_mean_frozen_embeddings(model, ids, torch.tensor([[0, -1]]))
    with pytest.raises(TypeError, match="boolean or integer"):
        masked_mean_frozen_embeddings(model, ids, mask.float())
    with pytest.raises(ValueError, match="all-padding"):
        masked_mean_frozen_embeddings(model, ids, torch.zeros_like(mask))
    with pytest.raises(RuntimeError, match="frozen"):
        masked_mean_frozen_embeddings(model, ids, mask)
    embedding.weight.requires_grad_(False)
    assert masked_mean_frozen_embeddings(model, ids, mask).shape == (1, 2)

    with pytest.raises(ValueError, match="hidden size"):
        SequenceTop1Router(0, 0.5)
    with pytest.raises(ValueError, match="hidden size"):
        SequenceTop1Router(-1, 0.5)
    with pytest.raises(ValueError, match="threshold"):
        SequenceTop1Router(2, 0.0)
    with pytest.raises(ValueError, match="threshold"):
        SequenceTop1Router(2, -0.1)
    with pytest.raises(ValueError, match="threshold"):
        SequenceTop1Router(2, 1.1)
    assert SequenceTop1Router(2, 0.5).confidence_threshold == 0.5
    assert SequenceTop1Router(2, 1.0).confidence_threshold == 1.0
    wide = SequenceTop1Router(1_000, 0.5)
    assert wide.add_candidate("wide", "wide") == 0
    assert wide.logits(torch.ones(1, 1_000)).shape == (1, 1)


def test_router_topology_and_fit_guardrails() -> None:
    router = SequenceTop1Router(2, 0.5)
    with pytest.raises(RuntimeError, match="no expert rows"):
        router.logits(torch.ones(1, 2))
    with pytest.raises(ValueError, match="non-empty"):
        router.add_candidate("", "family")
    first_index = router.add_candidate("old", "old-family")
    assert isinstance(first_index, int)
    assert first_index == 0
    with pytest.raises(ValueError, match="already exists"):
        router.add_candidate("old", "other")
    with pytest.raises(ValueError, match="wrong shape"):
        router.logits(torch.ones(1, 3))
    with pytest.raises(ValueError, match="wrong shape"):
        router.logits(torch.ones(1, 1))
    with pytest.raises(ValueError, match="wrong shape"):
        router.logits(torch.ones(1, 1, 2))
    router.promote_candidate()
    assert router.weights[0].requires_grad is False
    assert router.biases[0].requires_grad is False
    with pytest.raises(RuntimeError, match="no trainable"):
        router.promote_candidate()
    with pytest.raises(RuntimeError, match="no trainable"):
        router.candidate_parameters()
    second_index = router.add_candidate("new", "new-family")
    assert second_index == 1
    assert router.weights[0].requires_grad is False
    assert router.biases[0].requires_grad is False
    valid_current = torch.ones(3, 2)
    valid_replay = -torch.ones(1, 2)
    valid_labels = torch.tensor([0])
    cases = [
        (torch.ones(1, 3), valid_replay, valid_labels, 0.1, 1, "current"),
        (torch.ones(1, 1), valid_replay, valid_labels, 0.1, 1, "current"),
        (torch.ones(1, 1, 2), valid_replay, valid_labels, 0.1, 1, "current"),
        (torch.ones(2), valid_replay, valid_labels, 0.1, 1, "current"),
        (torch.empty(0, 2), valid_replay, valid_labels, 0.1, 1, "current-task"),
        (valid_current, torch.ones(1, 3), valid_labels, 0.1, 1, "replay router"),
        (valid_current, torch.ones(1, 1), valid_labels, 0.1, 1, "replay router"),
        (valid_current, torch.ones(1, 1, 2), valid_labels, 0.1, 1, "replay router"),
        (valid_current, torch.ones(2), valid_labels, 0.1, 1, "replay router"),
        (valid_current, valid_replay, torch.tensor([], dtype=torch.long), 0.1, 1, "align"),
        (valid_current, valid_replay, torch.tensor([0, 0]), 0.1, 1, "align"),
        (
            valid_current,
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.long),
            0.1,
            1,
            "requires replay",
        ),
        (valid_current, valid_replay, torch.tensor([1]), 0.1, 1, "prior frozen"),
        (valid_current, valid_replay, torch.tensor([-1]), 0.1, 1, "prior frozen"),
        (valid_current, valid_replay, valid_labels, 0.0, 1, "learning rate"),
        (valid_current, valid_replay, valid_labels, -0.1, 1, "learning rate"),
        (valid_current, valid_replay, valid_labels, 0.1, 0, "max steps"),
        (valid_current, valid_replay, valid_labels, 0.1, -1, "max steps"),
    ]
    for current, replay, labels, rate, steps, message in cases:
        with pytest.raises((ValueError, RuntimeError), match=message):
            router.fit_candidate(
                current,
                replay,
                labels,
                learning_rate=rate,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                max_steps=steps,
            )
    optimizer_values = {
        "learning_rate": 0.1,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_steps": 1,
    }
    for field, replacement, message in [
        ("weight_decay", -0.1, "weight decay"),
        ("weight_decay", 1.1, "weight decay"),
        ("adam_beta1", 0.0, "beta1"),
        ("adam_beta1", -0.1, "beta1"),
        ("adam_beta1", 1.0, "beta1"),
        ("adam_beta1", 1.1, "beta1"),
        ("adam_beta2", 0.0, "beta2"),
        ("adam_beta2", -0.1, "beta2"),
        ("adam_beta2", 1.0, "beta2"),
        ("adam_beta2", 1.1, "beta2"),
        ("adam_epsilon", 0.0, "epsilon"),
        ("adam_epsilon", -0.1, "epsilon"),
        ("adam_epsilon", 1.1, "epsilon"),
    ]:
        with pytest.raises(ValueError, match=message):
            router.fit_candidate(
                valid_current,
                valid_replay,
                valid_labels,
                **(optimizer_values | {field: replacement}),
            )
    accepted_boundaries = optimizer_values | {
        "weight_decay": 1.0,
        "adam_epsilon": 1.0,
    }
    boundary_result, _ = router.fit_candidate(
        valid_current,
        valid_replay,
        valid_labels,
        **accepted_boundaries,
    )
    assert boundary_result.optimizer_steps == 1
    class NoopBank:
        def activate(self, adapter_name: str) -> None:
            del adapter_name

        def deactivate(self) -> None:
            return None

    with pytest.raises(ValueError, match="exactly one"):
        route_and_activate_fallback(NoopBank(), router, torch.ones(2, 2))
    with pytest.raises(ValueError, match="exactly one"):
        route_and_activate_fallback(NoopBank(), router, torch.ones(1))
    with pytest.raises(ValueError, match="exactly one"):
        route_and_activate_fallback(NoopBank(), router, torch.ones(0, 2))
    with pytest.raises(ValueError, match="exactly one"):
        route_and_activate_fallback(NoopBank(), router, torch.ones(1, 1, 2))

    three_rows = SequenceTop1Router(2, 0.5)
    for index in range(3):
        three_rows.add_candidate(str(index), str(index))
        if index < 2:
            three_rows.promote_candidate()
    candidate_weight, candidate_bias = three_rows.candidate_parameters()
    assert candidate_weight is three_rows.weights[2]
    assert candidate_bias is three_rows.biases[2]
    fitted, _optimizer = three_rows.fit_candidate(
        torch.ones(3, 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1]),
        learning_rate=0.1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0,
        max_steps=1,
    )
    assert fitted.current_examples == 3
    assert _optimizer.defaults["weight_decay"] == 0.0
    assert _optimizer.defaults["eps"] == 1.0
    three_rows.promote_candidate()
    assert three_rows.frozen_rows == [True, True, True]

    first_only = SequenceTop1Router(2, 0.5)
    first_only.add_candidate("first", "first")
    unseen = first_only.route(-torch.ones(1, 2))[0]
    assert unseen.route_status == "unrouted"
    assert unseen.fallback_probability == pytest.approx(0.5)
    initial_row = first_only.weights[0].detach().clone()
    first_result, _first_optimizer = first_only.fit_candidate(
        torch.ones(1, 2),
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        learning_rate=0.1,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_steps=1,
    )
    assert first_result.replay_examples == 0
    assert first_result.loss > 0.0
    assert not torch.equal(first_only.weights[0], initial_row)
    with pytest.raises(ValueError, match="align"):
        first_only.fit_candidate(
            torch.ones(1, 2),
            torch.empty(0, 2),
            torch.tensor(0),
            learning_rate=0.1,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_steps=1,
        )


def test_growth_evidence_and_policy_guardrails() -> None:
    frozen_outcome = VerifierOutcome(True)
    with pytest.raises(FrozenInstanceError):
        frozen_outcome.passed = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="pass/fail"):
        VerifierOutcome(None)
    with pytest.raises(ValueError, match="cannot also"):
        VerifierOutcome(True, "error")
    for args, message in [
        ((-0.1, 1.0, 1, 1), "failure threshold"),
        ((1.1, 1.0, 1, 1), "failure threshold"),
        ((0.5, -0.1, 1, 1), "entropy threshold"),
        ((0.5, float("inf"), 1, 1), "entropy threshold"),
        ((0.5, 1.0, -1, 1), "minimum observations"),
        ((0.5, 1.0, 0, 1), "minimum observations"),
        ((0.5, 1.0, 1, -1), "max experts"),
        ((0.5, 1.0, 1, 0), "max experts"),
    ]:
        with pytest.raises(ValueError, match=message):
            GrowthSettings(*args)
    assert GrowthSettings(0.0, 0.0, 1, 1).max_experts == 1
    router = SequenceTop1Router(1, 0.5)
    router.add_candidate("one", "family")
    route = router.route(torch.ones(1, 1))[0]
    with pytest.raises(ValueError, match="non-empty and aligned"):
        build_growth_evidence((), (), ())
    with pytest.raises(ValueError, match="non-empty and aligned"):
        build_growth_evidence((route,), (), ("one",))
    with pytest.raises(ValueError, match="expected routes"):
        build_growth_evidence((route,), (VerifierOutcome(True),), ())
    with pytest.raises(ValueError, match="expected routes"):
        build_growth_evidence((route,), (VerifierOutcome(True),), ("one", "two"))
    passed = build_growth_evidence(
        (route, route),
        (VerifierOutcome(True), VerifierOutcome(True)),
        ("one", "one"),
    )
    with pytest.raises(FrozenInstanceError):
        passed.observations = 0  # type: ignore[misc]
    policy = EvidenceGrowthPolicy(GrowthSettings(0.5, 1.0, 2, 2))
    with pytest.raises(FrozenInstanceError):
        policy.settings.max_experts = 3  # type: ignore[misc]
    maxed = policy.decide(2, passed)
    assert maxed.reason == "max_experts_reached"
    assert maxed.grow is False
    with pytest.raises(FrozenInstanceError):
        maxed.grow = True  # type: ignore[misc]
    assert policy.decide(3, passed).reason == "max_experts_reached"
    insufficient = EvidenceGrowthPolicy(GrowthSettings(0.5, 1.0, 3, 4)).decide(
        1, passed
    )
    assert insufficient.reason == "insufficient_evidence"
    assert insufficient.grow is False
    more_than_minimum = EvidenceGrowthPolicy(GrowthSettings(1.0, 10.0, 2, 4)).decide(
        1,
        passed.__class__(3, 0, 0.0, 0.0, 0, 0),
    )
    assert more_than_minimum.reason == "thresholds_not_met"
    entropy_policy = EvidenceGrowthPolicy(GrowthSettings(1.0, 0.0, 2, 4))
    entropy_decision = entropy_policy.decide(1, passed)
    assert entropy_decision.reason == "router_entropy_threshold"
    assert entropy_decision.grow is True
    above_entropy = EvidenceGrowthPolicy(GrowthSettings(1.0, 0.1, 2, 4)).decide(
        1,
        passed.__class__(2, 0, 0.0, 0.2, 0, 0),
    )
    assert above_entropy.reason == "router_entropy_threshold"
    quiet_policy = EvidenceGrowthPolicy(GrowthSettings(1.0, 10.0, 2, 4))
    quiet_decision = quiet_policy.decide(1, passed)
    assert quiet_decision.reason == "thresholds_not_met"
    assert quiet_decision.grow is False
    above_failure = EvidenceGrowthPolicy(GrowthSettings(0.25, 10.0, 2, 4)).decide(
        1,
        passed.__class__(2, 1, 0.5, 0.0, 0, 0),
    )
    assert above_failure.reason == "verifier_failure_threshold"
    abstained = RouteDecision("unrouted", None, 0.4, 0.2, (0.5, 0.5))
    abstained_evidence = build_growth_evidence(
        (abstained,), (VerifierOutcome(True),), ("one",)
    )
    assert abstained_evidence.abstentions == 1
    assert abstained_evidence.misroutes == 0
    lexical_route = RouteDecision("routed", "z", 0.9, 0.1, (0.1, 0.9))
    equal_distinct_expected = "".join(("z", ""))
    assert equal_distinct_expected == lexical_route.expert_id
    assert equal_distinct_expected is not lexical_route.expert_id
    lexical_evidence = build_growth_evidence(
        (lexical_route,), (VerifierOutcome(True),), (equal_distinct_expected,)
    )
    assert lexical_evidence.misroutes == 0
    actually_misrouted = build_growth_evidence(
        (lexical_route,), (VerifierOutcome(True),), ("a",)
    )
    assert actually_misrouted.misroutes == 1


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("router", "confidence_threshold", 0.0),
        ("router", "confidence_threshold", 1.1),
        ("router", "learning_rate", 0.0),
        ("router", "learning_rate", 1.1),
        ("router", "weight_decay", -0.1),
        ("router", "weight_decay", 1.1),
        ("router", "adam_beta1", 0.0),
        ("router", "adam_beta1", 1.0),
        ("router", "adam_beta2", 0.0),
        ("router", "adam_beta2", 1.0),
        ("router", "adam_epsilon", 0.0),
        ("router", "adam_epsilon", 1.1),
        ("router", "max_optimizer_steps", 10_001),
        ("router", "replay_examples_per_expert", 10_001),
        ("growth", "verifier_failure_threshold", -0.1),
        ("growth", "verifier_failure_threshold", 1.1),
        ("growth", "mean_entropy_threshold", -0.1),
        ("growth", "minimum_observations", 100_001),
        ("consolidation", "adapter_learning_rate", 0.0),
        ("consolidation", "adapter_learning_rate", 1.1),
        ("consolidation", "adapter_weight_decay", -0.1),
        ("consolidation", "adapter_weight_decay", 1.1),
        ("consolidation", "adapter_adam_beta1", 0.0),
        ("consolidation", "adapter_adam_beta1", 1.0),
        ("consolidation", "adapter_adam_beta2", 0.0),
        ("consolidation", "adapter_adam_beta2", 1.0),
        ("consolidation", "adapter_adam_epsilon", 0.0),
        ("consolidation", "adapter_adam_epsilon", 1.1),
        ("consolidation", "max_adapter_optimizer_steps", 100_001),
        ("consolidation", "replay_capacity", 50_001),
        ("consolidation", "replay_ratio", 0.0),
        ("consolidation", "replay_ratio", 1.1),
        ("consolidation", "replay_sampling_seed", -1),
        ("consolidation", "replay_sampling_seed", True),
        ("consolidation", "replay_sampling_seed", 2**31),
        ("consolidation", "replay_eviction_policy", "fifo"),
        ("consolidation", "promotion_alpha", 0.0),
        ("consolidation", "promotion_alpha", 1.0),
        ("consolidation", "minimum_regression_samples", 1),
        ("consolidation", "minimum_validation_gain", -0.1),
        ("consolidation", "minimum_validation_gain", 1.1),
        ("consolidation", "loss_noninferiority_margin", -0.1),
    ],
)
def test_lane5_expert_configuration_boundaries(
    section: str, field: str, replacement: object
) -> None:
    settings = load_standard_expert_settings()
    nested = getattr(settings, section)
    payload = settings.model_dump()
    payload[section] = nested.model_dump() | {field: replacement}
    with pytest.raises(ValueError):
        StandardExpertSettings.model_validate(payload)


def test_lane5_expert_configuration_accepts_registered_inclusive_bounds() -> None:
    settings = load_standard_expert_settings()
    payload = settings.model_dump()
    payload["consolidation"] |= {
        "replay_capacity": 1,
        "replay_per_family_minimum": 1,
        "replay_sampling_seed": 2**31 - 1,
    }
    validated = StandardExpertSettings.model_validate(payload)
    assert validated.consolidation.replay_capacity == 1
    assert validated.consolidation.replay_per_family_minimum == 1
    assert validated.consolidation.replay_sampling_seed == 2**31 - 1

    payload["consolidation"]["replay_per_family_minimum"] = 2
    with pytest.raises(ValueError, match="exceeds capacity"):
        StandardExpertSettings.model_validate(payload)


def test_routed_expert_requires_manifest_identity_and_provenance() -> None:
    manifest = ExpertManifest(
        schema_version="aptivra.peft_expert_manifest.v1",
        adapter_name="expert",
        adapter_subdir="expert",
        base_repository="local/base",
        base_revision="a" * 40,
        base_config_sha256="b" * 64,
        base_weights_sha256="c" * 64,
        base_fingerprint="d" * 64,
        rank=16,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        parameter_count=1,
        peft_version="0.21.2",
        peft_license="Apache-2.0",
        content_sha256="e" * 64,
    )
    routed = RoutedExpert("expert", "family", manifest)
    assert routed.task_family == "family"
    with pytest.raises(FrozenInstanceError):
        routed.task_family = "changed"  # type: ignore[misc]
    equal_but_distinct = "".join(("ex", "pert"))
    assert equal_but_distinct == manifest.adapter_name
    assert equal_but_distinct is not manifest.adapter_name
    assert RoutedExpert(equal_but_distinct, "family", manifest).expert_id == "expert"
    with pytest.raises(ValueError, match="non-empty"):
        RoutedExpert("expert", "", manifest)
    with pytest.raises(ValueError, match="equal"):
        RoutedExpert("other", "family", manifest)
    with pytest.raises(ValueError, match="equal"):
        RoutedExpert("a", "family", manifest)


def test_standard_expert_path_rejects_historical_v0_model() -> None:
    legacy = ExpertModule(expert_id=0, d_model=4, d_expert=2)
    with pytest.raises(TypeError, match="StandardBase"):
        PeftExpertBank.start_candidate(  # type: ignore[arg-type]
            legacy, _settings(), "candidate"
        )


def test_rejected_cycle_restores_every_state_rng_and_hash(tmp_path: Path) -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path / "base"), settings, "new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("new", "new")
    cycle = ConsolidationCycle(bank, router, settings)
    snapshot = cycle.snapshot()
    current = [_instance("current-1"), _instance("current-2")]
    current_features = _real_features(bank, [item.prompt for item in current])
    cycle.train_candidate(
        current,
        current_features,
        torch.empty((0, 16)),
        torch.empty((0,), dtype=torch.long),
    )
    cycle.replay.add_instances(current)
    cycle.admit_verified_memory(tuple(current), (VerifierOutcome(True), VerifierOutcome(False)))
    random.random()
    np.random.random()
    torch.rand(2)
    cycle.bank.deactivate()
    cycle.bank.model.train(not cycle.bank.model.training)
    cycle.router.train(not cycle.router.training)
    changed_settings = settings.model_copy(
        update={
            "router": settings.router.model_copy(
                update={"confidence_threshold": 0.9}
            )
        }
    )
    cycle.settings = changed_settings
    cycle.bank.settings = changed_settings
    rejected = PromotionResult(False, 1.0, "verifier_gain_check", 0.0)
    restored_hashes = cycle.finalize(
        snapshot, rejected, current, tmp_path / "must-not-exist"
    )
    assert restored_hashes == snapshot.hashes
    assert cycle.state_hashes() == snapshot.hashes
    assert all(
        parameter.requires_grad
        for parameter in cycle.bank.adapter_parameters("new")
    )
    assert cycle.router.weights[-1].requires_grad is True
    assert cycle.settings == settings
    assert cycle.bank.settings == settings
    adapter_layer_states = {
        module.disable_adapters
        for module in cycle.bank.model.modules()
        if isinstance(getattr(module, "disable_adapters", None), bool)
    }
    assert adapter_layer_states == {False}
    assert not (tmp_path / "must-not-exist").exists()


def test_row_and_mature_expert_freeze_during_real_candidate_training(tmp_path: Path) -> None:
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path / "base"), settings, "old")
    old_artifact = tmp_path / "old"
    bank.promote_candidate(old_artifact)
    bank.add_candidate("new")
    old_adapter_hash = bank.adapter_parameter_sha256("old")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("old", "old")
    router.promote_candidate()
    old_row = router.weights[0].detach().clone()
    router.add_candidate("new", "new")
    cycle = ConsolidationCycle(bank, router, settings)
    current = [_instance("new-1"), _instance("new-2")]
    replay = [_instance("old-1", "old"), _instance("old-2", "old")]
    cycle.train_candidate(
        current,
        _real_features(bank, [item.prompt for item in current]),
        _real_features(bank, [item.prompt for item in replay]),
        torch.tensor([0, 0]),
    )
    assert bank.adapter_parameter_sha256("old") == old_adapter_hash
    assert all(not parameter.requires_grad for parameter in bank.adapter_parameters("old"))
    assert torch.equal(router.weights[0], old_row)
    assert router.weights[0].requires_grad is False


def _persist_accepted_cycle(
    tmp_path: Path,
) -> tuple[ConsolidationCycle, Path, dict[str, str]]:
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path / "live"), settings, "old")
    bank.promote_candidate(tmp_path / "seed-old")
    bank.add_candidate("new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("old", "old")
    router.promote_candidate()
    router.add_candidate("new", "new")
    cycle = ConsolidationCycle(bank, router, settings)
    snapshot = cycle.snapshot()
    current = [_instance("accepted-1"), _instance("accepted-2")]
    features = _real_features(bank, [item.prompt for item in current])
    cycle.train_candidate(
        current,
        features,
        _real_features(bank, ["historical prompt", "historical replay"]),
        torch.tensor([0, 0]),
    )
    cycle.admit_verified_memory(tuple(current), (VerifierOutcome(True), VerifierOutcome(True)))
    promotion = cycle.evaluate_promotion(
        tuple(VerifierOutcome(False) for _ in range(5)),
        tuple(VerifierOutcome(True) for _ in range(5)),
        np.full(5, 0.8),
        np.full(5, 0.4),
    )
    assert promotion.promoted is True
    artifact = tmp_path / "accepted"
    hashes = cycle.finalize(snapshot, promotion, current, artifact)
    return cycle, artifact, hashes


def test_accepted_cycle_reloads_with_matching_hashes(tmp_path: Path) -> None:
    cycle, artifact, hashes = _persist_accepted_cycle(tmp_path)
    settings = _settings()

    restarted = load_accepted_cycle(_base(tmp_path / "restart"), settings, artifact)
    assert restarted.state_hashes == hashes
    assert restarted.bank.adapter_parameter_sha256("old") == cycle.bank.adapter_parameter_sha256(
        "old"
    )
    assert restarted.bank.adapter_parameter_sha256("new") == cycle.bank.adapter_parameter_sha256(
        "new"
    )
    assert restarted.bank.adapter_names == ("new", "old")
    assert restarted.router.expert_ids == ["old", "new"]
    assert restarted.router.frozen_rows == [True, True]
    assert restarted.replay.records == cycle.replay.records
    assert restarted.memory == cycle.memory


def test_accepted_cycle_survives_process_restart(tmp_path: Path) -> None:
    _cycle, artifact, hashes = _persist_accepted_cycle(tmp_path)
    restart_code = """
import json
import sys
from pathlib import Path
from aptivra.consolidation.cycle import load_accepted_cycle
from tests.test_router_consolidation import _base, _settings
reloaded = load_accepted_cycle(_base(Path(sys.argv[1])), _settings(), Path(sys.argv[2]))
print(json.dumps(reloaded.state_hashes, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", restart_code, str(tmp_path / "process"), str(artifact)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    assert json.loads(completed.stdout) == hashes


def test_promotion_rejects_insufficient_evidence_and_propagates_failure(
    tmp_path: Path,
) -> None:
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path), settings, "new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("new", "new")
    cycle = ConsolidationCycle(bank, router, settings)
    insufficient = cycle.evaluate_promotion(
        (VerifierOutcome(False),),
        (VerifierOutcome(True),),
        np.asarray([0.8]),
        np.asarray([0.4]),
    )
    assert insufficient.promoted is False
    assert insufficient.test_used == "insufficient_paired_evidence"
    with pytest.raises(VerifierInfrastructureError, match="verifier crashed"):
        cycle.evaluate_promotion(
            (VerifierOutcome(False),),
            (VerifierOutcome(None, "verifier crashed"),),
            np.asarray([0.8]),
            np.asarray([0.4]),
        )
