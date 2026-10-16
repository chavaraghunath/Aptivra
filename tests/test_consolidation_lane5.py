"""Additional mutation-strength tests for ADR-0011 consolidation transactions."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

import aptivra.consolidation.cycle as cycle_module
from aptivra.consolidation.cycle import (
    AcceptedCycle,
    ConsolidationCycle,
    ConsolidationSnapshot,
    ConsolidationTrainingResult,
    VerifiedMemoryRecord,
    load_accepted_cycle,
    state_sha256,
)
from aptivra.experts.router import SequenceTop1Router
from aptivra.experts.spawner import VerifierInfrastructureError, VerifierOutcome
from aptivra.experts.standard import PeftExpertBank
from aptivra.gate.promotion_gate import PromotionResult
from aptivra.inference.standard_base import collate_task_instances
from tests.test_router_consolidation import _base, _instance, _real_features, _settings


def _cycle(tmp_path: Path) -> ConsolidationCycle:
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path), settings, "new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("new", "new")
    return ConsolidationCycle(bank, router, settings)


def _three_expert_cycle(tmp_path: Path) -> ConsolidationCycle:
    settings = _settings()
    bank = PeftExpertBank.start_candidate(_base(tmp_path / "base"), settings, "old-a")
    bank.promote_candidate(tmp_path / "old-a")
    bank.add_candidate("old-b")
    bank.promote_candidate(tmp_path / "old-b")
    bank.add_candidate("new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("old-a", "old-a")
    router.promote_candidate()
    router.add_candidate("old-b", "old-b")
    router.promote_candidate()
    router.add_candidate("new", "new")
    return ConsolidationCycle(bank, router, settings)


def test_state_hash_is_type_shape_and_value_sensitive() -> None:
    first = state_sha256(
        {
            "tensor": torch.tensor(1.0),
            "array": np.asarray([1], dtype=np.int64),
            "list": [1],
            "tuple": (1,),
            "set": {1},
            "none": None,
            "bytes": b"one",
        }
    )
    assert first == state_sha256(
        {
            "tensor": torch.tensor(1.0),
            "array": np.asarray([1], dtype=np.int64),
            "list": [1],
            "tuple": (1,),
            "set": {1},
            "none": None,
            "bytes": b"one",
        }
    )
    assert first != state_sha256({"tensor": torch.tensor([1.0])})
    assert first != state_sha256({"tensor": torch.tensor(2.0)})
    assert state_sha256([1]) != state_sha256([])
    assert state_sha256([1]) != state_sha256([2])
    assert state_sha256((1,)) != state_sha256(())
    assert state_sha256((1,)) != state_sha256((2,))
    assert state_sha256({1}) != state_sha256(set())
    assert state_sha256({1}) != state_sha256({2})
    with pytest.raises(TypeError, match="cannot hash"):
        state_sha256(object())


def test_rng_capture_and_restore_cover_mps_and_cuda_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mps_rng = torch.tensor([7], dtype=torch.uint8)
    cuda_rng = torch.tensor([9], dtype=torch.uint8)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mps_rng)
    cpu_state = cycle_module._capture_rng_state(torch.device("cpu"))
    assert cpu_state["torch_mps"] is None
    assert cpu_state["torch_cuda"] is None
    xpu_state = cycle_module._capture_rng_state(torch.device("xpu"))
    assert xpu_state["torch_mps"] is None
    mps_state = cycle_module._capture_rng_state(torch.device("mps"))
    assert torch.equal(mps_state["torch_mps"], mps_rng)
    assert mps_state["torch_cuda"] is None

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_rng)
    cpu_state = cycle_module._capture_rng_state(torch.device("cpu"))
    assert cpu_state["torch_mps"] is None
    assert cpu_state["torch_cuda"] is None
    unavailable_mps_state = cycle_module._capture_rng_state(torch.device("mps"))
    assert unavailable_mps_state["torch_cuda"] is None
    cuda_state = cycle_module._capture_rng_state(torch.device("cuda"))
    assert cuda_state["torch_mps"] is None
    assert torch.equal(cuda_state["torch_cuda"], cuda_rng)

    restored: list[tuple[str, torch.Tensor]] = []
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda value: restored.append(("mps", value))
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((device.type, value)),
    )
    cycle_module._restore_rng_state(torch.device("mps"), mps_state)
    cycle_module._restore_rng_state(torch.device("cuda"), cuda_state)
    assert restored == [("mps", mps_rng), ("cuda", cuda_rng)]
    restored.clear()
    cpu_state["torch_mps"] = mps_rng
    cpu_state["torch_cuda"] = cuda_rng
    cycle_module._restore_rng_state(torch.device("cpu"), cpu_state)
    assert restored == []
    cycle_module._restore_rng_state(torch.device("xpu"), mps_state)
    cuda_only_state = dict(cpu_state)
    cuda_only_state["torch_mps"] = None
    cuda_only_state["torch_cuda"] = cuda_rng
    cycle_module._restore_rng_state(torch.device("mps"), cuda_only_state)
    assert restored == []


def test_adapter_enablement_requires_one_consistent_state() -> None:
    empty = cast(
        PeftExpertBank,
        SimpleNamespace(model=SimpleNamespace(modules=lambda: [])),
    )
    inconsistent = cast(
        PeftExpertBank,
        SimpleNamespace(
            model=SimpleNamespace(
                modules=lambda: [
                    SimpleNamespace(disable_adapters=False),
                    SimpleNamespace(disable_adapters=True),
                ]
            )
        ),
    )
    for bank in (empty, inconsistent):
        with pytest.raises(RuntimeError, match="inconsistent"):
            cycle_module._adapters_enabled(bank)


def test_cycle_constructor_and_memory_fail_closed(tmp_path: Path) -> None:
    settings = _settings()
    disabled = settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": False})}
    )
    bank = PeftExpertBank.start_candidate(_base(tmp_path / "base"), settings, "new")
    router = SequenceTop1Router(16, settings.router.confidence_threshold)
    router.add_candidate("new", "new")
    with pytest.raises(RuntimeError, match="disabled"):
        ConsolidationCycle(bank, router, disabled)

    wrong = SequenceTop1Router(16, settings.router.confidence_threshold)
    wrong.add_candidate("other", "new")
    with pytest.raises(ValueError, match="must match"):
        ConsolidationCycle(bank, wrong, settings)
    lexical_low = SequenceTop1Router(16, settings.router.confidence_threshold)
    lexical_low.add_candidate("a", "new")
    with pytest.raises(ValueError, match="must match"):
        ConsolidationCycle(bank, lexical_low, settings)
    equal_distinct = SequenceTop1Router(16, settings.router.confidence_threshold)
    distinct_name = bytearray(b"new").decode()
    assert distinct_name == bank.trainable_adapter
    assert distinct_name is not bank.trainable_adapter
    equal_distinct.add_candidate(distinct_name, "new")
    assert ConsolidationCycle(bank, equal_distinct, settings).bank is bank
    router.promote_candidate()
    with pytest.raises(ValueError, match="trainable"):
        ConsolidationCycle(bank, router, settings)

    multi = PeftExpertBank.start_candidate(
        _base(tmp_path / "multi"), settings, "old"
    )
    multi.promote_candidate(tmp_path / "old-artifact")
    multi.add_candidate("new")
    incomplete = SequenceTop1Router(16, settings.router.confidence_threshold)
    incomplete.add_candidate("new", "new")
    with pytest.raises(ValueError, match="map exactly"):
        ConsolidationCycle(multi, incomplete, settings)
    superset = SequenceTop1Router(16, settings.router.confidence_threshold)
    superset.add_candidate("extra", "extra")
    superset.promote_candidate()
    superset.add_candidate("new", "new")
    with pytest.raises(ValueError, match="map exactly"):
        ConsolidationCycle(bank, superset, settings)

    cycle = _cycle(tmp_path / "memory")
    assert cycle.adapter_optimizer.defaults["lr"] == settings.consolidation.adapter_learning_rate
    assert cycle.adapter_optimizer.defaults["weight_decay"] == (
        settings.consolidation.adapter_weight_decay
    )
    assert cycle.adapter_optimizer.defaults["betas"] == (
        settings.consolidation.adapter_adam_beta1,
        settings.consolidation.adapter_adam_beta2,
    )
    assert cycle.adapter_optimizer.defaults["eps"] == (
        settings.consolidation.adapter_adam_epsilon
    )
    instance = _instance("one")
    with pytest.raises(ValueError, match="align"):
        cycle.admit_verified_memory((instance,), ())
    with pytest.raises(ValueError, match="align"):
        cycle.admit_verified_memory((), (VerifierOutcome(True),))
    cycle.admit_verified_memory((instance,), (VerifierOutcome(False),))
    assert cycle.memory == ()
    cycle.admit_verified_memory((instance,), (VerifierOutcome(True),))
    cycle.admit_verified_memory((instance,), (VerifierOutcome(True),))
    assert len(cycle.memory) == 1
    assert cycle.memory[0] == VerifiedMemoryRecord("one", "new", "prompt", "answer")
    with pytest.raises(FrozenInstanceError):
        cycle.memory[0].prompt = "changed"
    with pytest.raises(VerifierInfrastructureError, match="failed"):
        cycle.admit_verified_memory(
            (instance,), (VerifierOutcome(None, "verifier failed"),)
        )


def test_training_and_promotion_input_bounds(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path)
    with pytest.raises(ValueError, match="current-task"):
        cycle.train_candidate(
            [], torch.empty(0, 16), torch.empty(0, 16), torch.empty(0, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="align"):
        cycle.evaluate_promotion(
            (VerifierOutcome(False),),
            (),
            np.ones(5),
            np.zeros(5),
        )
    with pytest.raises(ValueError, match="align"):
        cycle.evaluate_promotion(
            (),
            (VerifierOutcome(True),),
            np.ones(5),
            np.zeros(5),
        )
    with pytest.raises(ValueError, match="non-empty"):
        cycle.evaluate_promotion((), (), np.ones(5), np.zeros(5))

    mixed_accuracy = cycle_module._verifier_accuracy(
        (VerifierOutcome(True), VerifierOutcome(False))
    )
    assert mixed_accuracy == 0.5


def test_single_step_training_result_is_exact_and_frozen(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path)
    instance = _instance("single")
    batch = collate_task_instances(cycle.bank.base, [instance])
    with torch.no_grad():
        expected_loss = float(
            cycle.bank.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
                return_dict=True,
            ).loss.item()
        )
    expected_tokens = int((batch.labels[:, 1:] != -100).sum().item())
    result = cycle.train_candidate(
        [instance],
        _real_features(cycle.bank, [instance.prompt]),
        torch.empty(0, 16),
        torch.empty(0, dtype=torch.long),
    )
    assert isinstance(result, ConsolidationTrainingResult)
    assert result.adapter_loss == pytest.approx(expected_loss)
    assert math.isfinite(result.adapter_loss)
    assert result.adapter_optimizer_steps == 1
    assert result.target_tokens == expected_tokens
    with pytest.raises(FrozenInstanceError):
        result.target_tokens = 0  # type: ignore[misc]


def test_snapshot_and_accepted_cycle_records_are_frozen(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path / "snapshot")
    snapshot = cycle.snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.hashes = {}  # type: ignore[misc]
    artifact, _accepted = _accepted_artifact(tmp_path / "accepted")
    loaded = load_accepted_cycle(_base(tmp_path / "restart"), _settings(), artifact)
    assert isinstance(loaded, AcceptedCycle)
    with pytest.raises(FrozenInstanceError):
        loaded.memory = ()  # type: ignore[misc]


def test_restore_rejects_invalid_active_adapter_count_and_hash(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path)
    snapshot = cycle.snapshot()
    for active in ((), ("new", "new")):
        payload = dict(snapshot.payload)
        payload["active_adapters"] = active
        invalid = ConsolidationSnapshot(payload, snapshot.hashes)
        with pytest.raises(RuntimeError, match="exactly one"):
            cycle.restore(invalid)
    wrong_hashes = dict(snapshot.hashes)
    wrong_hashes["memory"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash verification"):
        cycle.restore(ConsolidationSnapshot(snapshot.payload, wrong_hashes))


def test_target_token_cap_aborts_before_optimizer_step(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path)
    limited = cycle.settings.model_copy(
        update={
            "consolidation": cycle.settings.consolidation.model_copy(
                update={"max_target_tokens": 1}
            )
        }
    )
    limited_cycle = ConsolidationCycle(cycle.bank, cycle.router, limited)
    current = [_instance("too-many")]
    with pytest.raises(RuntimeError, match="target-token cap"):
        limited_cycle.train_candidate(
            current,
            _real_features(cycle.bank, ["prompt"]),
            torch.empty(0, 16),
            torch.empty(0, dtype=torch.long),
        )


def test_router_replay_requires_every_prior_row_and_exact_counts(tmp_path: Path) -> None:
    current = [_instance("current")]
    valid_labels = torch.tensor([0, 0, 1, 1])
    valid_cycle = _three_expert_cycle(tmp_path / "valid")
    valid_result = valid_cycle.train_candidate(
        current,
        _real_features(valid_cycle.bank, ["current"]),
        _real_features(valid_cycle.bank, ["a", "b", "c", "d"]),
        valid_labels,
    )
    assert valid_result.router.replay_examples == 4

    for name, labels, message in [
        ("wrong-set", torch.tensor([1, 1]), "every prior"),
        ("too-few", torch.tensor([0, 1]), "count"),
        ("too-many", torch.tensor([0, 0, 0, 1, 1]), "count"),
    ]:
        cycle = _three_expert_cycle(tmp_path / name)
        replay_text = [str(index) for index in range(labels.shape[0])]
        with pytest.raises(ValueError, match=message):
            cycle.train_candidate(
                current,
                _real_features(cycle.bank, ["current"]),
                _real_features(cycle.bank, replay_text),
                labels,
            )


def test_finalize_rejects_existing_acceptance_path(tmp_path: Path) -> None:
    cycle = _cycle(tmp_path / "cycle")
    snapshot = cycle.snapshot()
    output = tmp_path / "existing"
    output.mkdir()
    promoted = PromotionResult(True, 1.0, "test", 1.0)
    with pytest.raises(FileExistsError, match="immutable"):
        cycle.finalize(snapshot, promoted, [_instance("one")], output)


def _accepted_artifact(tmp_path: Path) -> tuple[Path, ConsolidationCycle]:
    cycle = _cycle(tmp_path / "live")
    snapshot = cycle.snapshot()
    current = [_instance("one"), _instance("two")]
    cycle.train_candidate(
        current,
        _real_features(cycle.bank, ["prompt", "prompt"]),
        torch.empty(0, 16),
        torch.empty(0, dtype=torch.long),
    )
    cycle.admit_verified_memory(tuple(current), (VerifierOutcome(True), VerifierOutcome(True)))
    result = cycle.evaluate_promotion(
        tuple(VerifierOutcome(False) for _ in range(5)),
        tuple(VerifierOutcome(True) for _ in range(5)),
        np.ones(5),
        np.zeros(5),
    )
    artifact = tmp_path / "accepted"
    cycle.finalize(snapshot, result, current, artifact)
    return artifact, cycle


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("state.pt", "state-file"),
        ("replay.json", "replay"),
        ("memory.json", "memory"),
        ("config.json", "config"),
        ("promotion.json", "promotion-evidence"),
    ],
)
def test_restart_rejects_tampered_cycle_files(
    tmp_path: Path, filename: str, message: str
) -> None:
    artifact, _cycle_value = _accepted_artifact(tmp_path / filename.replace(".", "-"))
    path = artifact / filename
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=message):
        load_accepted_cycle(_base(tmp_path / f"restart-{filename}"), _settings(), artifact)


def test_restart_rejects_manifest_schema_base_and_settings_mismatch(tmp_path: Path) -> None:
    artifact, _cycle_value = _accepted_artifact(tmp_path / "manifest")
    manifest_path = artifact / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, value, message in [
        ("schema_version", "wrong", "schema"),
        ("base_fingerprint", "0" * 64, "base fingerprint"),
        ("expert_manifests", [], "no promoted experts"),
    ]:
        copy = tmp_path / f"copy-{field}"
        shutil.copytree(artifact, copy)
        changed = dict(original)
        changed[field] = value
        (copy / "manifest.json").write_text(
            json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=message):
            load_accepted_cycle(_base(tmp_path / f"base-{field}"), _settings(), copy)

    wrong_hash = tmp_path / "copy-expert-hash"
    shutil.copytree(artifact, wrong_hash)
    changed = dict(original)
    changed["hashes"] = {**original["hashes"], "experts": "0" * 64}
    (wrong_hash / "manifest.json").write_text(
        json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expert hash"):
        load_accepted_cycle(_base(tmp_path / "base-expert-hash"), _settings(), wrong_hash)

    changed_settings = _settings().model_copy(
        update={
            "router": _settings().router.model_copy(update={"confidence_threshold": 0.9})
        }
    )
    with pytest.raises(ValueError, match="settings"):
        load_accepted_cycle(_base(tmp_path / "base-settings"), changed_settings, artifact)


@pytest.mark.parametrize(
    ("hash_name", "message"),
    [
        ("state_file", "state-file"),
        ("replay", "replay"),
        ("memory", "memory"),
        ("config", "config"),
        ("promotion_evidence", "promotion-evidence"),
        ("experts", "expert hash"),
        ("router", "router hash"),
        ("optimizer", "optimizer hash"),
        ("rng", "RNG hash"),
    ],
)
def test_restart_hash_checks_reject_both_lexical_directions(
    tmp_path: Path, hash_name: str, message: str
) -> None:
    artifact, _cycle_value = _accepted_artifact(tmp_path / "source")
    original = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for label, invalid_hash in (("low", "0" * 64), ("high", "f" * 64)):
        copy = tmp_path / f"{hash_name}-{label}"
        shutil.copytree(artifact, copy)
        changed = dict(original)
        changed["hashes"] = {**original["hashes"], hash_name: invalid_hash}
        (copy / "manifest.json").write_text(
            json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=message):
            load_accepted_cycle(
                _base(tmp_path / f"base-{hash_name}-{label}"), _settings(), copy
            )


def test_accepted_json_is_canonical_and_expert_order_is_enforced(tmp_path: Path) -> None:
    artifact, _cycle_value = _accepted_artifact(tmp_path / "source")
    for filename in (
        "replay.json",
        "memory.json",
        "config.json",
        "promotion.json",
        "manifest.json",
    ):
        path = artifact / filename
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == json.dumps(parsed, sort_keys=True) + "\n"

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["expert_manifests"]) == 1
    manifest["expert_manifests"][0]["adapter_name"] = "different"
    manifest["expert_manifests"][0]["adapter_subdir"] = "different"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expert/router order"):
        load_accepted_cycle(_base(tmp_path / "restart"), _settings(), artifact)
