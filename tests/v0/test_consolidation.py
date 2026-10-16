"""Unit tests for Consolidation Cycle."""

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from aptivra.eval.config import seed_experiment
from aptivra.model.config import ModelConfig
from aptivra.tasks.base import VerifierResult
from aptivra.tasks.modular_arithmetic import ModularArithmeticTaskFamily
from aptivra.v0.consolidation.cycle import ConsolidationConfig, ConsolidationCycle


def test_consolidation_cycle_execution() -> None:
    model_cfg = ModelConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64, device="cpu"
    )
    cons_cfg = ConsolidationConfig(
        use_t1_memory=True,
        use_expert_growth=True,
        use_replay_buffer=True,
        use_ewc_penalty=True,
        epochs=1,
    )
    trainer = ConsolidationCycle(model_cfg, cons_cfg)

    family = ModularArithmeticTaskFamily()
    instances = family.generate_instances(count=8, seed=42)

    res = trainer.run_cycle(task_idx=0, task_family=family.family_id, train_instances=instances)
    assert "loss" in res
    assert res["flops"] > 0


def test_weight_snapshot_restores_backbone_and_expert_topology(tmp_path: Path) -> None:
    model_cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=32,
        device="cpu",
    )
    cycle = ConsolidationCycle(model_cfg, ConsolidationConfig(epochs=1))
    cycle.spawner.config.entropy_threshold = -1.0
    family = ModularArithmeticTaskFamily()
    instance = family.generate_instances(1, seed=42)[0]
    cycle.t1_store.write_record(
        instance.prompt,
        instance.target_solution,
        VerifierResult(True, instance.target_solution, instance.target_solution),
        instance.task_family,
    )
    cycle.t1_store.query_top_k(instance.prompt, top_k=1)
    cycle.run_cycle(0, instance.task_family, [instance])
    assert cycle.router.num_experts == 2
    assert cycle.experts[0].is_mature is True
    expected_state = {
        name: value.detach().clone() for name, value in cycle.model.state_dict().items()
    }
    snapshot = tmp_path / "weights.pt"
    cycle.save_weight_snapshot(snapshot)

    with torch.no_grad():
        for parameter in cycle.model.parameters():
            parameter.add_(10.0)
    cycle.router.spawn_expert_row()

    cycle.restore_weight_snapshot(snapshot)

    assert cycle.router.num_experts == 2
    assert cycle.router.frozen_mask == [True, False]
    assert len(cycle.experts) == 2
    assert cycle.experts[0].is_mature is True
    for name, value in cycle.model.state_dict().items():
        assert torch.equal(value, expected_state[name])


def test_promotion_snapshot_restores_all_learned_state_but_retains_spent_flops(
    tmp_path: Path,
) -> None:
    seed_experiment(17)
    cycle = _tiny_cycle()
    family = ModularArithmeticTaskFamily()
    first = family.generate_instances(2, seed=1)
    cycle.run_cycle(0, family.family_id, first)
    cycle.fisher_matrices.append(
        {name: torch.ones_like(value) for name, value in cycle.model.named_parameters()}
    )
    cycle.optpar_matrices.append(
        {name: value.detach().clone() for name, value in cycle.model.named_parameters()}
    )
    verified = VerifierResult(True, first[0].target_solution, first[0].target_solution)
    cycle.t1_store.write_record(
        first[0].prompt,
        first[0].target_solution,
        verified,
        first[0].task_family,
    )
    cycle.t1_store.query_top_k(first[0].prompt, top_k=1)

    expected = {
        "model": copy.deepcopy(cycle.model.state_dict()),
        "router": copy.deepcopy(cycle.router.state_dict()),
        "experts": copy.deepcopy(cycle.experts.state_dict()),
        "optimizer": copy.deepcopy(cycle.optimizer.state_dict()),
        "replay": copy.deepcopy(cycle.replay_buffer),
        "fisher": copy.deepcopy(cycle.fisher_matrices),
        "optpar": copy.deepcopy(cycle.optpar_matrices),
        "records": copy.deepcopy(cycle.t1_store.records),
        "vectors": cycle.t1_store.index.vectors.detach().clone(),
        "record_ids": list(cycle.t1_store.index.record_ids),
        "encoder": copy.deepcopy(cycle.t1_store.index.encoder.state_dict()),
        "index_optimizer": copy.deepcopy(cycle.t1_store.index.optimizer.state_dict()),
    }
    snapshot = tmp_path / "promotion.pt"
    cycle.save_promotion_snapshot(snapshot)
    expected_torch_rng = torch.get_rng_state().clone()
    expected_python_rng = random.getstate()
    expected_numpy_rng = np.random.get_state()
    pre_candidate_flops = cycle.total_flops

    rejected = family.generate_instances(2, seed=2)
    cycle.t1_store.write_record(
        rejected[0].prompt,
        rejected[0].target_solution,
        verified,
        rejected[0].task_family,
    )
    cycle.fisher_matrices.append({"candidate": torch.ones(1)})
    cycle.optpar_matrices.append({"candidate": torch.ones(1)})
    cycle.spawner.config.entropy_threshold = -1.0
    cycle.run_cycle(1, family.family_id, rejected)
    spent_flops = cycle.total_flops
    random.random()
    np.random.random()

    cycle.restore_promotion_snapshot(snapshot)

    _assert_nested_equal(cycle.model.state_dict(), expected["model"])
    _assert_nested_equal(cycle.router.state_dict(), expected["router"])
    _assert_nested_equal(cycle.experts.state_dict(), expected["experts"])
    _assert_nested_equal(cycle.optimizer.state_dict(), expected["optimizer"])
    assert cycle.replay_buffer == expected["replay"]
    _assert_nested_equal(cycle.fisher_matrices, expected["fisher"])
    _assert_nested_equal(cycle.optpar_matrices, expected["optpar"])
    assert cycle.t1_store.records == expected["records"]
    assert torch.equal(cycle.t1_store.index.vectors, expected["vectors"])
    assert cycle.t1_store.index.record_ids == expected["record_ids"]
    _assert_nested_equal(cycle.t1_store.index.encoder.state_dict(), expected["encoder"])
    _assert_nested_equal(
        cycle.t1_store.index.optimizer.state_dict(), expected["index_optimizer"]
    )
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)
    assert random.getstate() == expected_python_rng
    restored_numpy_rng = np.random.get_state()
    assert restored_numpy_rng[0] == expected_numpy_rng[0]
    assert np.array_equal(restored_numpy_rng[1], expected_numpy_rng[1])
    assert restored_numpy_rng[2:] == expected_numpy_rng[2:]
    assert cycle.total_flops == spent_flops > pre_candidate_flops


def test_promotion_snapshot_rng_is_device_independent_and_replays_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_experiment(991)
    cycle = _tiny_cycle()
    snapshot = tmp_path / "rng-device-independent.pt"
    cycle.save_promotion_snapshot(snapshot)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)
    random.random()
    np.random.random()
    torch.rand(4)

    real_torch_load = torch.load
    raw_snapshot = real_torch_load(snapshot, map_location="cpu", weights_only=True)
    assert isinstance(raw_snapshot, dict)
    assert not _contains_tensor(raw_snapshot["rng_state"])

    requested_map_locations: list[object] = []

    def simulated_accelerator_load(*args: Any, **kwargs: Any) -> Any:
        requested_map_location = kwargs.get("map_location")
        requested_map_locations.append(requested_map_location)
        kwargs["map_location"] = "cpu"
        loaded = real_torch_load(*args, **kwargs)
        assert isinstance(loaded, dict)

        # Emulate the old load path's indiscriminate accelerator relocation while
        # remaining runnable on a CPU-only host. The fixed RNG payload has no tensors
        # to relocate, and the fixed restore path never requests the model device.
        if requested_map_location is cycle.device:
            if "rng_state" in loaded:
                loaded["rng_state"] = _move_tensors_to_meta(loaded["rng_state"])
            for legacy_key in (
                "torch_rng_state",
                "cuda_rng_states",
                "mps_rng_state",
                "numpy_rng_state",
            ):
                if legacy_key in loaded:
                    loaded[legacy_key] = _move_tensors_to_meta(loaded[legacy_key])
        return loaded

    monkeypatch.setattr(torch, "load", simulated_accelerator_load)
    cycle.restore_promotion_snapshot(snapshot)

    assert requested_map_locations == ["cpu"]
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_rejected_cycle_has_same_next_cycle_as_never_seen_candidate(tmp_path: Path) -> None:
    family = ModularArithmeticTaskFamily()
    rejected_task = family.generate_instances(2, seed=20)
    next_task = family.generate_instances(2, seed=30)

    seed_experiment(123)
    rejected_cycle = _tiny_cycle()
    rejected_cycle.spawner.config.entropy_threshold = -1.0
    snapshot = tmp_path / "before-rejected.pt"
    rejected_cycle.save_promotion_snapshot(snapshot)
    rejected_cycle.run_cycle(0, family.family_id, rejected_task)
    rejected_flops = rejected_cycle.total_flops
    rejected_cycle.restore_promotion_snapshot(snapshot)
    next_result = rejected_cycle.run_cycle(1, family.family_id, next_task)
    rejected_state = copy.deepcopy(rejected_cycle.model.state_dict())
    rejected_router = copy.deepcopy(rejected_cycle.router.state_dict())
    rejected_experts = copy.deepcopy(rejected_cycle.experts.state_dict())
    rejected_optimizer = copy.deepcopy(rejected_cycle.optimizer.state_dict())

    seed_experiment(123)
    control_cycle = _tiny_cycle()
    control_cycle.spawner.config.entropy_threshold = -1.0
    control_result = control_cycle.run_cycle(1, family.family_id, next_task)

    assert next_result["loss"] == control_result["loss"]
    _assert_nested_equal(rejected_state, control_cycle.model.state_dict())
    _assert_nested_equal(rejected_router, control_cycle.router.state_dict())
    _assert_nested_equal(rejected_experts, control_cycle.experts.state_dict())
    _assert_nested_equal(rejected_optimizer, control_cycle.optimizer.state_dict())
    assert rejected_cycle.replay_buffer == control_cycle.replay_buffer
    assert rejected_cycle.router.frozen_mask == control_cycle.router.frozen_mask
    assert len(rejected_cycle.experts) == len(control_cycle.experts)
    assert rejected_cycle.total_flops == control_cycle.total_flops + rejected_flops


def _tiny_cycle() -> ConsolidationCycle:
    model_cfg = ModelConfig(
        vocab_size=128,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=32,
        dropout=0.1,
        device="cpu",
    )
    return ConsolidationCycle(
        model_cfg,
        ConsolidationConfig(
            use_t1_memory=True,
            use_expert_growth=True,
            use_replay_buffer=True,
            use_ewc_penalty=True,
            epochs=1,
        ),
    )


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value)
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_tensor(item) for item in value)
    return False


def _move_tensors_to_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to("meta")
    if isinstance(value, dict):
        return {key: _move_tensors_to_meta(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_meta(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_meta(item) for item in value)
    return value
