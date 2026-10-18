"""Focused mutation proofs for the bounded MPS pilot."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import aptivra.eval.mps_pilot as pilot
from aptivra.inference.standard_base import CausalBatch
from aptivra.tasks.base import MechanicalVerifier, TaskFamily, TaskInstance, exact_match_result
from aptivra.tasks.stream import TaskStreamSplit


class _Verifier(MechanicalVerifier):
    def verify(self, instance: TaskInstance, model_output: str) -> Any:
        return exact_match_result(instance, model_output)


class _Family(TaskFamily):
    def __init__(self, family_id: str) -> None:
        super().__init__(family_id, family_id, "mutation fixture")
        self._verifier = _Verifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        return [
            TaskInstance(
                f"{self.family_id}-{seed}-{index}",
                self.family_id,
                f"two words {index}",
                "ok",
            )
            for index in range(count)
        ]


class _Stream:
    def __init__(self, count: int = 8) -> None:
        self.families = [_Family("a"), _Family("b")]
        self._splits = {
            family.family_id: TaskStreamSplit(
                train=family.generate_instances(count, 1),
                val=family.generate_instances(count, 2),
                test=family.generate_instances(count, 3),
            )
            for family in self.families
        }

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1] * len(text.split())


def _settings() -> pilot.MpsPilotSettings:
    return pilot.load_mps_pilot_settings("configs/h3_mps_pilot.yaml")


def test_preflight_accepts_matching_issue_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    issues = settings.streams.chronological_issues
    stream = _Stream(count=43)
    audit = {"passed": True}
    manifest = {
        "schema_version": 3,
        "audit": dict(audit),
        "record_count": int("258"),
        "record_shards": ["one"],
    }

    audit_checks: list[bool] = []
    monkeypatch.setattr(
        pilot,
        "write_or_check_plan",
        lambda _plan, *, check: audit_checks.append(check),
    )
    monkeypatch.setattr(
        pilot,
        "load_audit_plan",
        lambda _: SimpleNamespace(
            streams=(
                SimpleNamespace(
                    stream_config=settings.streams.synthetic_v2.config,
                    output=settings.streams.synthetic_v2.audit_record,
                ),
            )
        ),
    )
    monkeypatch.setattr(pilot, "_require_hash", lambda _p, expected, _d: expected)

    def json_mapping(path: Path) -> dict[str, Any]:
        if path == settings.streams.synthetic_v2.audit_record:
            return {"passed": True, "violations": [], "families": [], "stream_version": "v2"}
        if path == issues.manifest:
            return manifest
        return audit

    monkeypatch.setattr(pilot, "_json_mapping", json_mapping)
    monkeypatch.setattr(
        pilot,
        "load_github_issue_builder_config",
        lambda _: SimpleNamespace(
            raw_dataset_path=Path(issues.raw_data.as_posix()),
            manifest_path=Path(issues.manifest.as_posix()),
            audit_path=Path(issues.audit_record.as_posix()),
        ),
    )
    monkeypatch.setattr(
        pilot,
        "GitHubIssuesContinualStream",
        lambda *_: SimpleNamespace(
            families=stream.families,
            audit_summary=lambda: {
                family.family_id: {
                    "coverage": {"meets_threshold": True},
                    "overlap": {"train_test": 0, "train_val": 0, "val_test": 0},
                }
                for family in stream.families
            },
            get_split=stream.get_split,
        ),
    )

    result = pilot.validate_claim_streams(settings)

    assert result["chronological_issues"]["record_count"] == 258
    assert audit_checks == [True]


def test_process_sampling_enforces_ps_success_and_parses_cpu_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def run(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(stdout="7 1 12.5 4096 process name\n")

    monkeypatch.setattr(pilot.subprocess, "run", run)

    assert pilot._process_samples() == (
        pilot.ProcessSample(7, 1, 12.5, 4096 * 1024, "process name"),
    )
    assert calls == [{"check": True, "capture_output": True, "text": True, "timeout": 10.0}]


def test_quiet_wait_doubles_backoff_until_the_configured_ceiling() -> None:
    settings = _settings().quiet_machine.model_copy(
        update={
            "timeout_seconds": 30.0,
            "initial_backoff_seconds": 1.0,
            "maximum_backoff_seconds": 8.0,
        }
    )
    clock = [0.0]
    sleeps: list[float] = []
    samples = iter(({"quiet": False}, {"quiet": False}, {"quiet": False}, {"quiet": True}))

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    result = pilot.wait_for_quiet_machine(
        settings,
        sleep=sleep,
        monotonic=lambda: clock[0],
        sample=lambda _: next(samples),
    )

    assert sleeps == [1.0, 2.0, 4.0]
    assert result["wait_seconds"] == 7.0


def test_quiet_retry_uses_elapsed_duration_for_global_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "execution": settings.execution.model_copy(update={"wall_time_limit_seconds": 10.0})
        }
    )
    monkeypatch.setattr(
        pilot,
        "wait_for_quiet_machine",
        lambda _: (_ for _ in ()).throw(pilot.QuietMachineTimeoutError(2.0, [])),
    )
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: 20.0)

    progress_path = tmp_path / "progress.json"
    with pytest.raises(TimeoutError, match="wall-time cap"):
        pilot._quiet_before_measurement(
            settings,
            {"quiet_gate_events": []},
            progress_path,
            5.0,
        )
    assert pilot._json_mapping(progress_path)["elapsed_wall_time_seconds"] == 15.0


def test_calibration_full_ft_method_state_is_exact() -> None:
    settings = _settings()
    model = torch.nn.Linear(3, 2)
    base = cast(Any, SimpleNamespace(model=model))

    frozen_optimizer, frozen_router, frozen = pilot._configure_calibration_method(
        "frozen_base", base, settings
    )
    assert frozen_optimizer is frozen_router is None
    assert frozen == {}
    assert all(not parameter.requires_grad for parameter in model.parameters())

    naive_optimizer, naive_router, naive = pilot._configure_calibration_method(
        "naive_sequential", base, settings
    )
    assert isinstance(naive_optimizer, torch.optim.AdamW)
    assert naive_router is None
    assert naive == {}
    assert all(parameter.requires_grad for parameter in model.parameters())

    replay_optimizer, replay_router, replay = pilot._configure_calibration_method(
        "replay", base, settings
    )
    assert isinstance(replay_optimizer, torch.optim.AdamW)
    assert replay_router is None
    assert replay == {"replay_capacity_records": 500}

    dynamic_method = "".join(("online", "_ewc"))
    assert dynamic_method == "online_ewc"
    optimizer, router, metadata = pilot._configure_calibration_method(
        cast(Any, dynamic_method), base, settings
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert router is None
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    assert metadata["ewc_state_bytes"] == 2 * parameter_bytes
    anchor, fisher = metadata["ewc_state"]
    assert set(anchor) == set(fisher) == {"weight", "bias"}
    assert all(torch.count_nonzero(value) == 0 for value in fisher.values())


class _TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.adapters = torch.nn.ModuleDict()
        self.active = ""

    def add_adapter(self, name: str) -> None:
        self.adapters[name] = torch.nn.Linear(2, 1, bias=False)

    def set_active(self, name: str) -> None:
        self.active = name


def test_manual_lora_trains_only_the_last_resident_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "calibration": settings.calibration.model_copy(update={"resident_adapter_count": 3})
        }
    )
    model = _TinyLoRAModel()
    monkeypatch.setattr(
        pilot.ManualLoRATrainer,
        "_inject",
        lambda _model, _config: [model],
    )

    optimizer, router, metadata = pilot._configure_calibration_method(
        "manual_lora", cast(Any, SimpleNamespace(model=model)), settings
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert router is None
    assert metadata == {"manual_lora_layer_count": 1, "resident_adapters": 3}
    assert model.active == "resident_0002"
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert {
        name: parameter.requires_grad
        for name, adapter in model.adapters.items()
        for parameter in adapter.parameters()
    } == {
        "resident_0000": False,
        "resident_0001": False,
        "resident_0002": True,
    }


def test_aptivra_calibration_builds_and_freezes_the_exact_resident_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "calibration": settings.calibration.model_copy(
                update={"resident_adapter_count": 3}
            )
        }
    )
    adapter_parameters = {
        f"resident_{index:04d}": torch.nn.Parameter(torch.tensor(float(index + 1)))
        for index in range(3)
    }
    calls: list[tuple[str, str]] = []

    class Bank:
        def __init__(self) -> None:
            self.trainable_adapter: str | None = "resident_0000"
            self.mature_adapters: set[str] = set()
            self.adapter_names = ["resident_0000"]

        def activate(self, name: str) -> None:
            calls.append(("activate", name))

        def add_candidate(self, name: str) -> None:
            calls.append(("bank_add", name))
            self.adapter_names.append(name)
            self.trainable_adapter = name

        def adapter_parameters(self, name: str) -> list[torch.nn.Parameter]:
            return [adapter_parameters[name]]

        def actual_parameter_count(self, name: str) -> int:
            assert name == "resident_0002"
            return 1

    bank = Bank()
    router_parameter = torch.nn.Parameter(torch.tensor(4.0))

    class Router:
        def __init__(self, hidden_size: int, threshold: float) -> None:
            calls.append(("router_init", f"{hidden_size}:{threshold}"))

        def to(self, device: torch.device) -> Router:
            calls.append(("router_to", device.type))
            return self

        def add_candidate(self, name: str, family: str) -> None:
            calls.append(("router_add", f"{name}:{family}"))

        def promote_candidate(self) -> None:
            calls.append(("router_promote", ""))

        def candidate_parameters(self) -> list[torch.nn.Parameter]:
            return [router_parameter]

    base = cast(
        Any,
        SimpleNamespace(
            model=torch.nn.Linear(2, 2),
            device=torch.device("cpu"),
            settings=SimpleNamespace(architecture=SimpleNamespace(hidden_size=2)),
        ),
    )
    monkeypatch.setattr(pilot.PeftExpertBank, "start_candidate", lambda *_: bank)
    monkeypatch.setattr(pilot, "SequenceTop1Router", Router)

    optimizer, router, metadata = pilot._configure_calibration_method(
        "aptivra_full", base, settings
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(router, Router)
    assert bank.trainable_adapter == "resident_0002"
    assert bank.mature_adapters == {"resident_0000", "resident_0001"}
    assert bank.adapter_names == ["resident_0000", "resident_0001", "resident_0002"]
    assert metadata["trainable_adapter_parameters"] == 1
    assert metadata["resident_adapters"] == 3
    assert calls == [
        ("router_init", "2:0.65"),
        ("router_to", "cpu"),
        ("router_add", "resident_0000:historical_0000"),
        ("activate", "resident_0000"),
        ("router_promote", ""),
        ("bank_add", "resident_0001"),
        ("router_add", "resident_0001:historical_0001"),
        ("router_to", "cpu"),
        ("activate", "resident_0001"),
        ("router_promote", ""),
        ("bank_add", "resident_0002"),
        ("router_add", "resident_0002:historical_0002"),
        ("router_to", "cpu"),
    ]


def test_prompt_features_builds_binary_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def masked(_model: Any, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        captured["ids"] = ids
        captured["mask"] = mask
        return mask.to(torch.float32)

    monkeypatch.setattr(pilot, "masked_mean_frozen_embeddings", masked)
    base = cast(
        Any,
        SimpleNamespace(
            tokenizer=_Tokenizer(),
            model=object(),
            device=torch.device("cpu"),
        ),
    )
    instances = [
        TaskInstance("short", "a", "one", "ok"),
        TaskInstance("long", "a", "one two three", "ok"),
    ]

    result = pilot._prompt_features(base, instances)

    assert captured["ids"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert captured["mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert torch.equal(result, captured["mask"].to(torch.float32))


def test_adapter_training_rejects_nonfinite_loss_before_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    bank = SimpleNamespace(
        trainable_adapter="candidate",
        activate=lambda _: None,
        adapter_parameters=lambda _: [parameter],
    )
    batch = CausalBatch(
        input_ids=torch.ones((1, 2), dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        labels=torch.ones((1, 2), dtype=torch.long),
        task_ids=("one",),
    )
    monkeypatch.setattr(pilot, "collate_task_instances", lambda *_: batch)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda *_: (parameter * torch.tensor(float("nan")), torch.empty(0)),
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        pilot._train_adapter_steps(
            cast(Any, SimpleNamespace()),
            cast(Any, bank),
            [TaskInstance("one", "a", "one", "ok")],
            1,
            1,
            1,
            1,
            0.01,
        )


def test_diagonal_fisher_uses_mean_of_per_example_squared_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    base = cast(Any, SimpleNamespace(model=model, device=torch.device("cpu")))
    instances = [
        TaskInstance("positive", "tiny", "a", "one"),
        TaskInstance("negative", "tiny", "b", "two"),
    ]
    monkeypatch.setattr(
        pilot,
        "collate_task_instances",
        lambda _base, rows: SimpleNamespace(task_ids=(rows[0].task_id,)),
    )
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda _base, batch: (
            parameter.sum() if batch.task_ids == ("positive",) else -parameter.sum(),
            torch.empty(0),
        ),
    )

    fisher, _ = pilot._diagonal_fisher(base, instances)

    assert fisher["weight"].item() == 1.0


def test_diagonal_fisher_rejects_squared_gradient_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    parameter = model.weight
    base = cast(Any, SimpleNamespace(model=model, device=torch.device("cpu")))
    instance = TaskInstance("overflow", "tiny", "a", "one")
    monkeypatch.setattr(
        pilot,
        "collate_task_instances",
        lambda *_: SimpleNamespace(task_ids=("overflow",)),
    )
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda *_: (parameter.sum() * 1.0e20, torch.empty(0)),
    )

    with pytest.raises(FloatingPointError, match="Fisher contains"):
        pilot._diagonal_fisher(base, [instance])


def test_ewc_penalty_uses_conventional_half_scaling() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    base = cast(Any, SimpleNamespace(model=model, device=torch.device("cpu")))

    penalty = pilot._ewc_penalty(
        base,
        {"weight": torch.tensor([[1.0]])},
        {"weight": torch.tensor([[3.0]])},
        4.0,
    )

    assert penalty.item() == 24.0


def test_bounded_pair_rejects_index_equal_to_family_count() -> None:
    with pytest.raises(ValueError, match="distinct real tasks"):
        pilot._BoundedPairStream(cast(Any, _Stream()), (0, 2), 1, 1)

    reversed_pair = pilot._BoundedPairStream(cast(Any, _Stream()), (1, 0), 1, 1)
    assert [family.family_id for family in reversed_pair.families] == ["b", "a"]


class _FakeMpsParameter:
    requires_grad = True
    device = torch.device("mps")

    def requires_grad_(self, enabled: bool) -> _FakeMpsParameter:
        self.requires_grad = enabled
        return self

    def numel(self) -> int:
        return 3


class _FakeFrozenModel:
    def __init__(self) -> None:
        self.parameter = _FakeMpsParameter()

    def parameters(self) -> list[_FakeMpsParameter]:
        return [self.parameter]

    def eval(self) -> _FakeFrozenModel:
        return self


class _Sampler:
    def __enter__(self) -> _Sampler:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def record(self) -> dict[str, Any]:
        return {"process_rss_peak_bytes": 100, "mps_driver_peak_bytes": 90}


@pytest.mark.parametrize(
    ("ending_swap", "expected_growth", "expected_compromised", "expected_throughput"),
    ((125, 25, True, None), (105, 5, False, 24.0), (90, 0, False, 24.0)),
)
def test_frozen_calibration_pads_and_accounts_for_swap_and_throughput(
    ending_swap: int,
    expected_growth: int,
    expected_compromised: bool,
    expected_throughput: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "calibration": settings.calibration.model_copy(
                update={
                    "memory_cap_bytes": 80,
                    "swap_throughput_reporting_threshold_bytes": 20,
                }
            )
        }
    )
    base = SimpleNamespace(
        device=torch.device("mps"),
        model=_FakeFrozenModel(),
        tokenizer=_Tokenizer(),
    )
    selected = _Stream().get_split("a").train[:2]
    initial_batch = CausalBatch(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        labels=torch.ones((2, 4), dtype=torch.long),
        task_ids=tuple(item.task_id for item in selected),
    )
    seen_lengths: list[int] = []
    seen_padding: list[list[int]] = []
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(pilot, "longest_distinct_examples", lambda *_: (selected, [12, 4]))
    monkeypatch.setattr(pilot, "collate_task_instances", lambda *_: initial_batch)

    def loss(_base: Any, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch.input_ids.shape == batch.attention_mask.shape == batch.labels.shape == (2, 12)
        assert torch.all(batch.attention_mask[:, :4] == 1)
        assert torch.all(batch.attention_mask[:, 4:] == 0)
        assert torch.all(batch.labels[:, :4] == 1)
        assert torch.all(batch.labels[:, 4:] == -100)
        seen_lengths.append(batch.input_ids.shape[1])
        seen_padding.append(batch.attention_mask[:, -1].tolist())
        return torch.tensor(2.0), torch.empty(0)

    monkeypatch.setattr(pilot, "_decoder_loss", loss)
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    monkeypatch.setattr(pilot, "_mps_values", lambda: (7, 13))
    class LowRssSampler(_Sampler):
        def record(self) -> dict[str, Any]:
            return {"process_rss_peak_bytes": 50, "mps_driver_peak_bytes": 90}

    monkeypatch.setattr(
        pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: LowRssSampler()
    )
    swaps = iter((100, ending_swap))
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: next(swaps))
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 75.0)
    counter = iter(float(value) for value in range(20))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(counter))
    generations: list[tuple[str, int]] = []

    def generate(_base: Any, instance: TaskInstance, *, max_new_tokens: int) -> Any:
        generations.append((instance.task_id, max_new_tokens))
        return SimpleNamespace(generated_token_ids=(1,), stop_reason="eos")

    monkeypatch.setattr(pilot, "generate_task_instance_answer", generate)

    frozen_method = "_".join(("frozen", "base"))
    canonical_frozen = "frozen_base"
    assert frozen_method == canonical_frozen and frozen_method is not canonical_frozen
    record = pilot.run_calibration_attempt(
        cast(Any, frozen_method),
        "synthetic_v2",
        cast(Any, _Stream()),
        2,
        settings,
        {"quiet": True},
    )

    assert record["status"] == "completed"
    assert record["accumulation"] == "not_applicable"
    assert record["gradient_checkpointing"] == "not_applicable"
    assert record["backward"] == "not_applicable"
    assert record["optimizer"] == "not_applicable"
    assert record["accepted"] is False
    assert record["memory"]["mps_driver_step_peak_bytes"] == 13
    assert record["system_swap_growth_bytes"] == expected_growth
    assert record["throughput_compromised"] is expected_compromised
    assert record["tokens_per_second"] == expected_throughput
    assert record["losses"] == [2.0, 2.0, 2.0]
    assert record["step_times_seconds"] == [1.0, 1.0, 1.0]
    assert record["generation"]["wall_time_seconds"] == 1.0
    assert record["wall_time_seconds"] == 10.0
    assert generations == [(selected[0].task_id, record["generation"]["maximum_answer_tokens"])]
    assert seen_lengths == [12, 12, 12, 12]
    assert seen_padding == [[0, 0]] * 4


def test_training_calibration_accounts_for_accumulation_warmup_and_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "calibration": settings.calibration.model_copy(
                update={
                    "memory_cap_bytes": 100,
                    "swap_throughput_reporting_threshold_bytes": 20,
                }
            )
        }
    )

    class TrainingModel(_FakeFrozenModel):
        def train(self) -> TrainingModel:
            return self

    base = SimpleNamespace(
        device=torch.device("mps"),
        model=TrainingModel(),
        tokenizer=_Tokenizer(),
    )
    selected = _Stream().get_split("a").train[:8]
    collated_ids: list[tuple[str, ...]] = []

    def collate(_base: Any, batch: list[TaskInstance]) -> CausalBatch:
        collated_ids.append(tuple(item.task_id for item in batch))
        return CausalBatch(
            input_ids=torch.ones((len(batch), 5), dtype=torch.long),
            attention_mask=torch.ones((len(batch), 5), dtype=torch.long),
            labels=torch.ones((len(batch), 5), dtype=torch.long),
            task_ids=tuple(item.task_id for item in batch),
        )

    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Optimizer:
        def __init__(self) -> None:
            self.zero_calls = 0
            self.step_calls = 0

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            parameter.grad = None
            self.zero_calls += 1

        def step(self) -> None:
            self.step_calls += 1

    optimizer = Optimizer()
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(pilot, "longest_distinct_examples", lambda *_: (selected, [5] * 8))
    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    pad_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_pad = torch.nn.functional.pad

    def pad(tensor: torch.Tensor, padding: tuple[int, ...], **kwargs: Any) -> torch.Tensor:
        pad_calls.append((tuple(tensor.shape), padding))
        return real_pad(tensor, padding, **kwargs)

    monkeypatch.setattr(pilot.torch.nn.functional, "pad", pad)
    monkeypatch.setattr(
        pilot,
        "_configure_calibration_method",
        lambda *_: (optimizer, None, {}),
    )
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda *_: (parameter * 2.0, torch.empty((4, 5, 1))),
    )
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    monkeypatch.setattr(pilot, "_mps_values", lambda: (7, 13))
    monkeypatch.setattr(pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: _Sampler())
    swaps = iter((100, 120))
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: next(swaps))
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 75.0)
    counter = iter(float(value) for value in range(30))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(counter))

    record = pilot.run_calibration_attempt(
        "replay",
        "synthetic_v2",
        cast(Any, _Stream()),
        4,
        settings,
        _quiet_record := {"quiet": True},
    )

    assert record["status"] == "completed"
    assert record["quiet_machine"] is _quiet_record
    assert record["accumulation"] == 2
    assert record["backward"] is True
    assert record["optimizer"] == "AdamW"
    assert record["trainable_parameters"] == 3
    assert record["losses"] == [2.0, 2.0, 2.0]
    assert record["finite_losses"] is True
    assert record["step_times_seconds"] == [1.0, 1.0, 1.0]
    assert record["median_step_time_seconds"] == 1.0
    assert record["tokens_per_second"] == 40.0
    assert record["throughput_compromised"] is False
    assert record["accepted"] is True
    assert record["system_swap_growth_bytes"] == 20
    assert record["memory"]["mps_current_peak_bytes"] == 7
    assert record["memory"]["mps_driver_step_peak_bytes"] == 13
    assert record["no_cpu_fallback"] is True
    assert optimizer.zero_calls == optimizer.step_calls == 4
    assert pad_calls == []
    assert collated_ids == [
        tuple(item.task_id for item in selected[:4]),
        tuple(item.task_id for item in selected[4:]),
    ]


def test_aptivra_calibration_records_exact_router_loss_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    class Model(_FakeFrozenModel):
        def train(self) -> Model:
            return self

    base = SimpleNamespace(device=torch.device("mps"), model=Model(), tokenizer=_Tokenizer())
    selected = _Stream().get_split("a").train[:8]
    loss_parameter = torch.nn.Parameter(torch.tensor(2.0))
    router_parameter = torch.nn.Parameter(torch.tensor(0.5))
    optimizer_calls: list[tuple[str, bool | None]] = []

    class Optimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            optimizer_calls.append(("zero", set_to_none))
            loss_parameter.grad = None
            router_parameter.grad = None

        def step(self) -> None:
            optimizer_calls.append(("step", None))

    pooled_rows: list[torch.Tensor] = []

    class Router:
        num_experts = 4

        def parameters(self) -> list[torch.nn.Parameter]:
            return [router_parameter]

        def classification_logits(self, pooled: torch.Tensor) -> torch.Tensor:
            pooled_rows.append(pooled.clone())
            columns = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            return router_parameter * columns.expand(pooled.shape[0], -1)

    router = Router()

    def collate(_base: Any, rows: list[TaskInstance]) -> CausalBatch:
        return CausalBatch(
            input_ids=torch.ones((len(rows), 5), dtype=torch.long),
            attention_mask=torch.ones((len(rows), 5), dtype=torch.long),
            labels=torch.ones((len(rows), 5), dtype=torch.long),
            task_ids=tuple(row.task_id for row in rows),
        )

    decoder_batches: list[CausalBatch] = []

    def decoder(_base: Any, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_batches.append(batch)
        hidden = torch.arange(len(batch.task_ids) * 6, dtype=torch.float32).reshape(
            len(batch.task_ids), 3, 2
        )
        return loss_parameter.square(), hidden

    sampler_arguments: list[tuple[torch.device, float, float]] = []

    def sampler(
        device: torch.device,
        interval: float,
        *,
        swap_sample_interval_seconds: float,
    ) -> _Sampler:
        sampler_arguments.append((device, interval, swap_sample_interval_seconds))
        return _Sampler()

    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(
        pilot, "longest_distinct_examples", lambda *_: (selected, [12] * len(selected))
    )
    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_configure_calibration_method",
        lambda *_: (Optimizer(), router, {"bank": object(), "resident_adapters": 3}),
    )
    monkeypatch.setattr(pilot, "_decoder_loss", decoder)
    real_full = torch.full

    def cpu_full(size: Any, fill_value: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("device", None)
        return real_full(size, fill_value, **kwargs)

    monkeypatch.setattr(pilot.torch, "full", cpu_full)
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    monkeypatch.setattr(pilot, "_mps_values", lambda: (7, 13))
    monkeypatch.setattr(pilot, "MemoryPeakSampler", sampler)
    swaps = iter((100, 100))
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: next(swaps))
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 75.0)
    clock = iter((10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 16.0))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))

    record = pilot.run_calibration_attempt(
        "aptivra_full",
        "synthetic_v2",
        cast(Any, _Stream()),
        4,
        settings,
        {"quiet": True},
    )

    columns = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_router_loss = torch.nn.functional.cross_entropy(
        0.5 * columns.expand(4, -1), torch.full((4,), 4, dtype=torch.long)
    ).item()
    assert record["status"] == "completed"
    assert record["trainable_parameters"] == 4
    assert record["accumulation"] == 2
    assert isinstance(record["accumulation"], int)
    assert record["gradient_checkpointing"] is False
    assert record["backward"] is True
    assert record["optimizer"] == "AdamW"
    assert record["method_state"] == {"resident_adapters": 3}
    assert record["losses"] == pytest.approx([4.0 + expected_router_loss] * 3)
    assert record["step_times_seconds"] == [0.5, 0.5, 0.5]
    assert record["median_step_time_seconds"] == 0.5
    assert record["tokens_per_second"] == 192.0
    assert record["wall_time_seconds"] == 6.0
    assert record["no_cpu_fallback"] is True
    assert sampler_arguments == [
        (base.device, settings.calibration.memory_sample_interval_seconds, 0.5)
    ]
    assert len(decoder_batches) == 8
    assert all(batch.input_ids.shape == (4, 12) for batch in decoder_batches)
    assert all(torch.all(batch.attention_mask[:, -7:] == 0) for batch in decoder_batches)
    assert all(torch.all(batch.labels[:, -7:] == -100) for batch in decoder_batches)
    expected_pooled = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2).mean(dim=1)
    assert len(pooled_rows) == 8
    assert all(torch.equal(row, expected_pooled) for row in pooled_rows)
    assert optimizer_calls == [("zero", True), ("step", None)] * 4
    assert loss_parameter.grad is not None
    assert loss_parameter.grad.item() == pytest.approx(4.0)


def test_online_ewc_calibration_initializes_fisher_and_adds_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    class Model(_FakeFrozenModel):
        def train(self) -> Model:
            return self

    base = SimpleNamespace(device=torch.device("mps"), model=Model(), tokenizer=_Tokenizer())
    selected = _Stream().get_split("a").train[:8]
    loss_parameter = torch.nn.Parameter(torch.tensor(2.0))
    fisher = {"weight": torch.tensor([0.0])}
    anchor = {"weight": torch.tensor([1.0])}
    zero_grad_flags: list[bool] = []

    class Optimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            zero_grad_flags.append(set_to_none)
            loss_parameter.grad = None

        def step(self) -> None:
            return None

    def collate(_base: Any, rows: list[TaskInstance]) -> CausalBatch:
        return CausalBatch(
            input_ids=torch.ones((len(rows), 4), dtype=torch.long),
            attention_mask=torch.ones((len(rows), 4), dtype=torch.long),
            labels=torch.ones((len(rows), 4), dtype=torch.long),
            task_ids=tuple(row.task_id for row in rows),
        )

    penalties: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]] = []

    def penalty(
        _base: Any,
        seen_anchor: dict[str, torch.Tensor],
        seen_fisher: dict[str, torch.Tensor],
        weight: float,
    ) -> torch.Tensor:
        penalties.append((seen_anchor, seen_fisher, weight))
        return loss_parameter * 3.0

    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(
        pilot, "longest_distinct_examples", lambda *_: (selected, [4] * len(selected))
    )
    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_configure_calibration_method",
        lambda *_: (
            Optimizer(),
            None,
            {
                "ewc_state": (anchor, fisher),
                "ewc_state_bytes": 16,
                "ewc_penalty_lambda": 400.0,
            },
        ),
    )
    monkeypatch.setattr(
        pilot,
        "_diagonal_fisher",
        lambda *_: ({"weight": torch.tensor([2.5])}, 5.296557),
    )
    monkeypatch.setattr(pilot, "_ewc_penalty", penalty)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda *_: (loss_parameter.square(), torch.empty((4, 4, 1))),
    )
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    monkeypatch.setattr(pilot, "_mps_values", lambda: (7, 13))
    monkeypatch.setattr(pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: _Sampler())
    swaps = iter((100, 100))
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: next(swaps))
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 75.0)
    clock = iter(float(value) for value in range(10))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))

    online_method = "_".join(("online", "ewc"))
    canonical_online = "online_ewc"
    assert online_method == canonical_online and online_method is not canonical_online
    record = pilot.run_calibration_attempt(
        cast(Any, online_method),
        "chronological_issues",
        cast(Any, _Stream()),
        4,
        settings,
        {"quiet": True},
    )

    assert record["status"] == "completed"
    assert record["method_state"] == {
        "ewc_state_bytes": 16,
        "ewc_penalty_lambda": 400.0,
        "fisher_initialization_loss": 5.296557,
        "fisher_examples": 8,
        "fisher_estimator": "mean_per_example_squared_gradient",
    }
    assert torch.equal(fisher["weight"], torch.tensor([2.5]))
    assert record["losses"] == [10.0, 10.0, 10.0]
    assert len(penalties) == 8
    assert all(item == (anchor, fisher, 400.0) for item in penalties)
    assert zero_grad_flags == [True] * 5
    assert loss_parameter.grad is not None
    assert loss_parameter.grad.item() == pytest.approx(7.0)


def test_training_calibration_enables_checkpointing_only_at_micro_batch_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    class Model:
        def __init__(self) -> None:
            self.enabled = False
            self.config = SimpleNamespace(use_cache=True)

        def gradient_checkpointing_enable(self) -> None:
            self.enabled = True

        def parameters(self) -> list[Any]:
            return []

    model = Model()
    base = SimpleNamespace(device=torch.device("mps"), model=model)
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(
        pilot,
        "longest_distinct_examples",
        lambda *_: (_ for _ in ()).throw(RuntimeError("stop after checkpoint setup")),
    )
    base.device = torch.device("cpu")
    cpu_record = pilot.run_calibration_attempt(
        "naive_sequential",
        "synthetic_v2",
        cast(Any, _Stream()),
        1,
        settings,
        {"quiet": True},
    )
    assert cpu_record["status"] == "failed"
    assert cpu_record["failure"]["type"] == "RuntimeError"
    base.device = torch.device("mps")
    record = pilot.run_calibration_attempt(
        "naive_sequential",
        "synthetic_v2",
        cast(Any, _Stream()),
        1,
        settings,
        {"quiet": True},
        checkpointing=True,
    )
    assert record["status"] == "failed"
    assert record["gradient_checkpointing"] is True
    assert model.enabled is True
    assert model.config.use_cache is False

    for method, micro_batch in (("frozen_base", 1), ("naive_sequential", 2)):
        model.enabled = False
        model.config.use_cache = True
        record = pilot.run_calibration_attempt(
            cast(Any, method),
            "synthetic_v2",
            cast(Any, _Stream()),
            micro_batch,
            settings,
            {"quiet": True},
            checkpointing=True,
        )
        assert record["status"] == "failed"
        assert model.enabled is False
        assert model.config.use_cache is True


class _RepeatSampler(_Sampler):
    def record(self) -> dict[str, Any]:
        return {"sampled": True}


def test_naive_repeatability_visits_wrapped_batches_and_records_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    synthetic_batches = settings.repeatability.micro_batches.synthetic_v2.model_copy(
        update={"naive_sequential": 2}
    )
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={
                    "evaluation_examples_per_task": 1,
                    "effective_batch_size": 2,
                    "micro_batches": settings.repeatability.micro_batches.model_copy(
                        update={"synthetic_v2": synthetic_batches}
                    ),
                }
            ),
            "cycle_caps": settings.cycle_caps.model_copy(
                update={"adapter_optimizer_steps": 3, "adapter_target_tokens": 100}
            ),
        }
    )
    model = torch.nn.Linear(1, 1, bias=False)
    base = SimpleNamespace(model=model, device=torch.device("cpu"))
    selected_ids: list[tuple[str, ...]] = []

    def collate(_base: Any, selected: list[TaskInstance]) -> CausalBatch:
        selected_ids.append(tuple(item.task_id for item in selected))
        return CausalBatch(
            input_ids=torch.ones((2, 2), dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            labels=torch.ones((2, 2), dtype=torch.long),
            task_ids=tuple(item.task_id for item in selected),
        )

    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    sampler_arguments: list[tuple[torch.device, float, float]] = []

    def sampler(
        device: torch.device,
        interval: float,
        *,
        swap_sample_interval_seconds: float,
    ) -> _RepeatSampler:
        sampler_arguments.append((device, interval, swap_sample_interval_seconds))
        return _RepeatSampler()

    monkeypatch.setattr(pilot, "MemoryPeakSampler", sampler)
    monkeypatch.setattr(pilot, "_evaluate_pair", lambda *_: [1.0, 1.0])
    monkeypatch.setattr(pilot, "_evaluate_template_ood_pair", lambda *_: None)
    monkeypatch.setattr(pilot, "collate_task_instances", collate)
    monkeypatch.setattr(
        pilot,
        "_decoder_loss",
        lambda *_: (model.weight.sum(), torch.empty(0)),
    )
    times = iter((10.0, 16.0))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(times))

    record = pilot.run_repeatability_cell(
        "naive_sequential",
        "synthetic_v2",
        cast(Any, _Stream()),
        (0, 1),
        17,
        0,
        settings,
        {"quiet": True},
    )

    assert record["status"] == "completed"
    assert record["metrics"]["optimizer_steps"] == 6
    assert record["wall_time_seconds"] == 6.0
    assert len(selected_ids) == 6
    assert len({task_id for batch in selected_ids[:3] for task_id in batch}) == 6
    assert sampler_arguments == [
        (base.device, settings.calibration.memory_sample_interval_seconds, 0.5)
    ]


def test_aptivra_repeatability_forwards_identity_and_marks_each_learning_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={"evaluation_examples_per_task": 1}
            )
        }
    )
    base = SimpleNamespace(model=torch.nn.Linear(1, 1), device=torch.device("cpu"))
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    cleaned: list[Any] = []
    monkeypatch.setattr(pilot, "_cleanup_base", cleaned.append)
    monkeypatch.setattr(pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: _RepeatSampler())
    captured: dict[str, Any] = {}

    def run_pair(
        _base: Any,
        _pair: Any,
        _settings: Any,
        *,
        stream_name: str,
        seed: int,
    ) -> dict[str, Any]:
        captured.update(stream_name=stream_name, seed=seed)
        return {
            "accuracy_matrix": [[0.4, 0.1], [0.3, 0.2]],
            "frozen_base_accuracy": [0.1, 0.2],
        }

    monkeypatch.setattr(pilot, "_run_aptivra_pair", run_pair)
    times = iter((2.0, 5.5))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(times))
    record = pilot.run_repeatability_cell(
        "aptivra_full",
        "synthetic_v2",
        cast(Any, _Stream()),
        (0, 1),
        23,
        4,
        settings,
        {"quiet": True},
    )
    assert captured == {"stream_name": "synthetic_v2", "seed": 23}
    assert record["status"] == "completed"
    assert record["seed"] == 23
    assert record["repeat_index"] == 4
    assert record["metrics"]["trained_task_above_frozen_base"] == [True, False]
    assert record["metrics"]["demonstrably_learned"] is False
    assert record["memory"] == {"sampled": True}
    assert record["wall_time_seconds"] == 3.5
    assert cleaned == [base]


def test_repeatability_redacts_failure_and_always_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={"evaluation_examples_per_task": 1}
            )
        }
    )
    monkeypatch.setattr(pilot, "seed_experiment", lambda _: None)
    monkeypatch.setattr(
        pilot,
        "_base_for_pilot",
        lambda _: (_ for _ in ()).throw(RuntimeError("private model detail")),
    )
    cleaned: list[Any] = []
    monkeypatch.setattr(pilot, "_cleanup_base", cleaned.append)
    times = iter((2.0, 3.0))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(times))
    record = pilot.run_repeatability_cell(
        "aptivra_full",
        "synthetic_v2",
        cast(Any, _Stream()),
        (0, 1),
        23,
        4,
        settings,
        {"quiet": True},
    )
    assert record["status"] == "failed"
    assert record["failure"]["type"] == "RuntimeError"
    assert "private model detail" not in str(record)
    assert record["wall_time_seconds"] == 1.0
    assert cleaned == [None]


@dataclass
class _Promotion:
    promoted: bool = False


@dataclass
class _RouterResult:
    loss: float = 0.0


def test_aptivra_pair_processes_both_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={"evaluation_examples_per_task": 1}
            )
        }
    )
    stream = pilot._BoundedPairStream(cast(Any, _Stream(count=10)), (0, 1), 2, 1)
    bank = SimpleNamespace(
        trainable_adapter="pilot_0000",
        mature_adapters=set(),
        add_candidate=lambda name: setattr(bank, "trainable_adapter", name),
        deactivate=lambda: None,
        activate=lambda _name: None,
    )

    class Router:
        num_experts = 1

        def fit_candidate(self, *_args: Any, **_kwargs: Any) -> tuple[_RouterResult, None]:
            return _RouterResult(), None

    expert = SimpleNamespace(
        feature=SimpleNamespace(
            enabled=False,
            model_copy=lambda **_: SimpleNamespace(enabled=True),
        ),
        model_copy=lambda **_: expert,
        router=SimpleNamespace(
            confidence_threshold=0.5,
            replay_examples_per_expert=1,
            learning_rate=0.1,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_optimizer_steps=1,
        ),
        consolidation=SimpleNamespace(
            adapter_learning_rate=0.1,
            promotion_alpha=0.05,
            minimum_regression_samples=1,
            minimum_validation_gain=0.0,
            loss_noninferiority_margin=0.0,
        ),
    )
    monkeypatch.setattr(pilot, "load_standard_expert_settings", lambda _: expert)
    monkeypatch.setattr(pilot.PeftExpertBank, "start_candidate", lambda *_: bank)
    monkeypatch.setattr(pilot, "SequenceTop1Router", lambda *_: Router())
    monkeypatch.setattr(pilot, "_add_router_candidate_on_device", lambda *_: None)
    monkeypatch.setattr(pilot, "_evaluate_pair", lambda *_: [0.5, 0.5])
    monkeypatch.setattr(pilot, "_evaluate_one_task", lambda *_: 0.5)
    monkeypatch.setattr(pilot, "_evaluate_template_ood_pair", lambda *_: None)
    monkeypatch.setattr(pilot, "_losses", lambda *_: np.asarray([1.0]))
    monkeypatch.setattr(
        pilot,
        "_train_adapter_steps",
        lambda *_: {
            "losses": [0.5],
            "target_tokens": 7,
            "optimizer_steps": 1,
        },
    )
    monkeypatch.setattr(pilot, "_prompt_features", lambda *_: torch.ones((1, 1)))
    monkeypatch.setattr(
        pilot,
        "_router_replay_batch",
        lambda *_: (torch.empty((0, 1)), torch.empty(0, dtype=torch.long)),
    )
    monkeypatch.setattr(pilot, "_routing_probe", lambda *_: {"coverage": 1.0})
    monkeypatch.setattr(
        pilot,
        "PromotionGate",
        lambda *_args, **_kwargs: SimpleNamespace(evaluate_measurements=lambda *_: _Promotion()),
    )
    monkeypatch.setattr(pilot, "_remove_candidate", lambda *_: None)
    monkeypatch.setattr(
        pilot,
        "generate_task_instance_answer",
        lambda *_a, **_k: SimpleNamespace(text="ok"),
    )

    result = pilot._run_aptivra_pair(
        cast(
            Any,
            SimpleNamespace(
                device=torch.device("cpu"),
                settings=SimpleNamespace(architecture=SimpleNamespace(hidden_size=1)),
            ),
        ),
        stream,
        settings,
        stream_name="synthetic_v2",
        seed=17,
    )

    assert len(result["promotions"]) == 2
    assert result["target_tokens"] == 14
    assert result["optimizer_steps"] == 2


def test_aptivra_pair_promotes_replays_and_verifies_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={"evaluation_examples_per_task": 2}
            )
        }
    )
    stream = pilot._BoundedPairStream(cast(Any, _Stream(count=10)), (0, 1), 2, 1)

    class Bank:
        def __init__(self) -> None:
            self.trainable_adapter: str | None = "pilot_0000"
            self.mature_adapters: set[str] = set()
            self.activations: list[str | None] = []

        def add_candidate(self, name: str) -> None:
            self.trainable_adapter = name

        def deactivate(self) -> None:
            self.activations.append(None)

        def activate(self, name: str) -> None:
            self.activations.append(name)

    bank = Bank()

    class Router:
        def __init__(self) -> None:
            self.num_experts = 1
            self.promotions = 0

        def fit_candidate(self, *_args: Any, **_kwargs: Any) -> tuple[_RouterResult, None]:
            return _RouterResult(), None

        def promote_candidate(self) -> None:
            self.promotions += 1

        def route(self, features: torch.Tensor) -> list[Any]:
            return [
                SimpleNamespace(route_status="routed", expert_id="pilot_0000")
                for _ in range(features.shape[0])
            ]

    router = Router()
    expert = SimpleNamespace(
        feature=SimpleNamespace(
            enabled=False,
            model_copy=lambda **_: SimpleNamespace(enabled=True),
        ),
        model_copy=lambda **_: expert,
        router=SimpleNamespace(
            confidence_threshold=0.5,
            replay_examples_per_expert=1,
            learning_rate=0.1,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.99,
            adam_epsilon=1e-8,
            max_optimizer_steps=3,
        ),
        consolidation=SimpleNamespace(
            adapter_learning_rate=0.1,
            promotion_alpha=0.05,
            minimum_regression_samples=1,
            minimum_validation_gain=0.0,
            loss_noninferiority_margin=0.0,
        ),
    )
    added: list[tuple[str, str]] = []
    combined_sizes: list[int] = []
    replay_sizes: list[int] = []
    loss_sizes: list[int] = []
    feature_sizes: list[int] = []
    monkeypatch.setattr(pilot, "load_standard_expert_settings", lambda _: expert)
    monkeypatch.setattr(pilot.PeftExpertBank, "start_candidate", lambda *_: bank)
    monkeypatch.setattr(pilot, "SequenceTop1Router", lambda *_: router)

    def add_router(_router: Any, name: str, family: str, _device: Any) -> None:
        added.append((name, family))
        if len(added) > 1:
            router.num_experts += 1

    monkeypatch.setattr(pilot, "_add_router_candidate_on_device", add_router)
    monkeypatch.setattr(pilot, "_evaluate_pair", lambda *_: [0.75, 0.5])
    monkeypatch.setattr(pilot, "_evaluate_one_task", lambda *_: 0.5)
    monkeypatch.setattr(pilot, "_evaluate_template_ood_pair", lambda *_: None)
    def losses(_base: Any, rows: list[TaskInstance]) -> np.ndarray:
        loss_sizes.append(len(rows))
        return np.asarray([1.0, 2.0])

    monkeypatch.setattr(pilot, "_losses", losses)

    def train(_base: Any, _bank: Any, instances: list[TaskInstance], *_args: Any) -> Any:
        combined_sizes.append(len(instances))
        return {"losses": [0.5], "target_tokens": len(instances), "optimizer_steps": 1}

    monkeypatch.setattr(pilot, "_train_adapter_steps", train)
    def prompt_features(_base: Any, rows: list[TaskInstance]) -> torch.Tensor:
        feature_sizes.append(len(rows))
        return torch.ones((len(rows), 1))

    monkeypatch.setattr(pilot, "_prompt_features", prompt_features)

    def replay_batch(_base: Any, rows: list[TaskInstance], _limit: int) -> Any:
        replay_sizes.append(len(rows))
        return torch.empty((0, 1)), torch.empty(0, dtype=torch.long)

    monkeypatch.setattr(pilot, "_router_replay_batch", replay_batch)
    monkeypatch.setattr(pilot, "_routing_probe", lambda *_: {"coverage": 1.0})
    monkeypatch.setattr(
        pilot,
        "PromotionGate",
        lambda *_args, **_kwargs: SimpleNamespace(
            evaluate_measurements=lambda *_: _Promotion(promoted=True)
        ),
    )
    monkeypatch.setattr(
        pilot,
        "_predict_with_adapter_assignments",
        lambda _base, _bank, rows, _ids, **_kwargs: ["ok"] * len(rows),
    )

    result = pilot._run_aptivra_pair(
        cast(
            Any,
            SimpleNamespace(
                device=torch.device("cpu"),
                settings=SimpleNamespace(architecture=SimpleNamespace(hidden_size=1)),
            ),
        ),
        stream,
        settings,
        stream_name="synthetic_v2",
        seed=17,
    )

    assert added == [("pilot_0000", "a"), ("pilot_0001", "b")]
    assert combined_sizes == [10, 15]
    assert replay_sizes == [0, 10]
    assert loss_sizes == [2, 2, 4, 4]
    assert feature_sizes == [8, 2, 8, 2]
    assert bank.mature_adapters == {"pilot_0000", "pilot_0001"}
    assert bank.trainable_adapter is None
    assert router.promotions == 2
    assert result["optimizer_steps"] == 2
    assert result["target_tokens"] == 25
    assert result["verified_memory_count"] == 4
    assert result["replay_records"] == 20
    assert result["router_experts"] == 2
    assert result["template_ood_examples_per_task"] == 0


def test_routed_evaluation_activates_adapter_and_abstention_uses_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active: list[str | None] = [None]
    bank = SimpleNamespace(
        activate=lambda adapter: active.__setitem__(0, adapter),
        deactivate=lambda: active.__setitem__(0, None),
    )
    monkeypatch.setattr(
        pilot,
        "_base_predict_many",
        lambda _base, instances, _tokens: [
            f"{active[0] or 'base'}:{instance.task_id}" for instance in instances
        ],
    )
    instances = [
        TaskInstance("fallback", "family", "p", "a"),
        TaskInstance("routed", "family", "p", "a"),
    ]

    outputs = pilot._predict_with_adapter_assignments(
        cast(Any, object()),
        cast(Any, bank),
        instances,
        [None, "expert-1"],
        max_new_tokens=4,
        batch_size=1,
    )

    assert outputs == ["base:fallback", "expert-1:routed"]


def test_compute_estimate_matches_independent_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "h3_estimate": settings.h3_estimate.model_copy(
                update={"v1_regression_seed_count": 2}
            )
        }
    )
    streams = {
        "synthetic_v2": cast(Any, _Stream(count=1)),
        "chronological_issues": cast(Any, _Stream(count=1)),
    }
    v1 = _Stream(count=1)
    third = _Family("c")
    v1.families.append(third)
    v1._splits["c"] = TaskStreamSplit(
        train=third.generate_instances(1, 1),
        val=third.generate_instances(1, 2),
        test=third.generate_instances(1, 3),
    )
    monkeypatch.setattr(
        pilot,
        "load_experiment_settings",
        lambda _: SimpleNamespace(make_stream=lambda: v1),
    )
    completed_status = "".join(("com", "pleted"))
    canonical_completed = "completed"
    assert completed_status == canonical_completed and completed_status is not canonical_completed
    calibration = [
        {
            "status": completed_status,
            "method": method,
            "stream": stream.encode().decode(),
            "tokens_per_second": 4.0,
            **(
                {"generation": {"wall_time_seconds": 2.0 if stream == "synthetic_v2" else 3.0}}
                if method == "frozen_base"
                else {}
            ),
        }
        for method in pilot._METHODS
        for stream in streams
    ]
    calibration.append(
        {
            "status": "abandoned",
            "method": "frozen_base",
            "stream": "synthetic_v2",
            "tokens_per_second": 1e-9,
        }
    )

    profiles = [
        {
            "status": completed_status,
            "stream": b"synthetic_v2".decode(),
            "batched_seconds_per_example": 2.0,
        },
        {
            "status": completed_status,
            "stream": b"chronological_issues".decode(),
            "batched_seconds_per_example": 3.0,
        },
    ]
    profiles.append(
        {
            "status": "abandoned",
            "stream": "synthetic_v2",
            "batched_seconds_per_example": 999.0,
        }
    )
    result = pilot.compute_estimate(
        calibration,
        profiles,
        streams,
        cast(Any, SimpleNamespace(tokenizer=_Tokenizer())),
        settings,
    )

    estimate = settings.h3_estimate
    evaluation_cap = settings.cycle_caps.evaluation_examples_per_seen_task * 3
    v1_evaluation_cap = settings.cycle_caps.evaluation_examples_per_seen_task * 6
    per_method_stream_training = 10 / 4
    main_training = 12 * per_method_stream_training * estimate.main_seed_count
    main_evaluation = estimate.main_method_count * estimate.main_seed_count * evaluation_cap * 5.0
    v1_training = 6 * (15 / 4) * estimate.v1_regression_seed_count
    v1_evaluation = (
        estimate.v1_regression_method_count
        * estimate.v1_regression_seed_count
        * v1_evaluation_cap
        * 2.5
    )
    ablation_cells = sum(item.seeds for item in estimate.ablations) * 2
    ablation_training = ablation_cells * per_method_stream_training
    ablation_evaluation = ablation_cells * evaluation_cap * 2.5
    arithmetic = result["arithmetic"]

    assert result["exact_training_sequence_tokens"] == {
        "synthetic_v2": 10,
        "chronological_issues": 10,
        "v1_regression": 15,
    }
    assert result["evaluation_generation_caps_by_stream"] == {
        "synthetic_v2": evaluation_cap,
        "chronological_issues": evaluation_cap,
    }
    assert arithmetic["main_training_seconds"] == main_training
    assert arithmetic["main_evaluation_seconds_at_cap"] == main_evaluation
    assert arithmetic["v1_regression_training_seconds"] == v1_training
    assert arithmetic["v1_regression_evaluation_seconds_at_cap"] == v1_evaluation
    assert arithmetic["ablation_training_seconds"] == ablation_training
    assert arithmetic["ablation_evaluation_seconds_at_cap"] == ablation_evaluation
    assert arithmetic["total_seconds"] == sum(
        (
            main_training,
            main_evaluation,
            v1_training,
            v1_evaluation,
            ablation_training,
            ablation_evaluation,
        )
    )


def test_compute_estimate_fails_closed_without_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    streams = {
        "synthetic_v2": cast(Any, _Stream(count=1)),
        "chronological_issues": cast(Any, _Stream(count=1)),
    }
    monkeypatch.setattr(
        pilot,
        "load_experiment_settings",
        lambda _: SimpleNamespace(make_stream=lambda: _Stream(count=1)),
    )
    result = pilot.compute_estimate(
        [], [], streams, cast(Any, SimpleNamespace(tokenizer=_Tokenizer())), settings
    )

    assert result["status"] == "incomplete"
    assert "frozen_base:synthetic_v2" in result["missing_measurements"]
    assert "frozen_base:chronological_issues" in result["missing_measurements"]
    assert "batched_generation:chronological_issues" in result["missing_measurements"]
    assert "Infinity" not in json.dumps(result, allow_nan=False)


def test_compute_estimate_uses_per_cell_medians_and_unequal_stream_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    synthetic = _Stream(count=1)
    issues = _Stream(count=2)
    third = _Family("c")
    issues.families.append(third)
    issues._splits["c"] = TaskStreamSplit(
        train=third.generate_instances(2, 1),
        val=third.generate_instances(2, 2),
        test=third.generate_instances(2, 3),
    )
    v1 = _Stream(count=1)
    v1.families = v1.families[:1]
    monkeypatch.setattr(
        pilot,
        "load_experiment_settings",
        lambda _: SimpleNamespace(make_stream=lambda: v1),
    )
    streams = {
        "synthetic_v2": cast(Any, synthetic),
        "chronological_issues": cast(Any, issues),
    }
    calibration: list[dict[str, Any]] = []
    expected_seconds_per_token: dict[str, float] = {}
    for method_index, method in enumerate(pilot._METHODS, start=1):
        for stream_index, stream in enumerate(streams, start=1):
            median_rate = float(2 * method_index * stream_index)
            expected_seconds_per_token[f"{method}:{stream}"] = 1.0 / median_rate
            for rate in (median_rate / 2.0, median_rate, median_rate * 10.0):
                recorded_method = method.encode().decode()
                assert recorded_method == method and recorded_method is not method
                calibration.append(
                    {
                        "status": "completed",
                        "method": recorded_method,
                        "stream": stream,
                        "tokens_per_second": rate,
                    }
                )
    calibration.append(
        {
            "status": "failed",
            "method": "frozen_base",
            "stream": "synthetic_v2",
            "tokens_per_second": 1e-9,
        }
    )
    profiles = [
        *(
            {
                "status": "completed",
                "stream": "synthetic_v2",
                "batched_seconds_per_example": value,
            }
            for value in (0.1, 0.3, 10.0)
        ),
        *(
            {
                "status": "completed",
                "stream": "chronological_issues",
                "batched_seconds_per_example": value,
            }
            for value in (0.2, 0.5, 20.0)
        ),
        {
            "status": "failed",
            "stream": "synthetic_v2",
            "batched_seconds_per_example": 999.0,
        },
    ]
    result = pilot.compute_estimate(
        calibration,
        profiles,
        streams,
        cast(Any, SimpleNamespace(tokenizer=_Tokenizer())),
        settings,
    )
    assert result["status"] == "completed"
    assert result["exact_training_sequence_tokens"] == {
        "synthetic_v2": 10,
        "chronological_issues": 30,
        "v1_regression": 5,
    }
    assert result["exact_training_target_tokens"] == {
        "synthetic_v2": 4,
        "chronological_issues": 12,
        "v1_regression": 2,
    }
    assert result["measured_seconds_per_sequence_token_by_method_stream"] == (
        expected_seconds_per_token
    )
    assert result["measured_batched_generation_seconds_per_example_by_stream"] == {
        "synthetic_v2": 0.3,
        "chronological_issues": 0.5,
    }
    cap = settings.cycle_caps.evaluation_examples_per_seen_task
    assert result["evaluation_generation_caps_by_stream"] == {
        "synthetic_v2": cap * 3,
        "chronological_issues": cap * 6,
    }
    assert all(
        isinstance(value, int)
        for value in result["evaluation_generation_caps_by_stream"].values()
    )
    per_cell = result["per_method_stream_cell_seconds"]
    assert per_cell == {
        key: (10 if key.endswith(":synthetic_v2") else 30) * seconds
        for key, seconds in expected_seconds_per_token.items()
    }
    arithmetic = result["arithmetic"]
    assert arithmetic["total_hours"] == pytest.approx(arithmetic["total_seconds"] / 3600.0)
    assert arithmetic["total_seconds"] == pytest.approx(
        arithmetic["main_training_seconds"]
        + arithmetic["main_evaluation_seconds_at_cap"]
        + arithmetic["v1_regression_training_seconds"]
        + arithmetic["v1_regression_evaluation_seconds_at_cap"]
        + arithmetic["ablation_training_seconds"]
        + arithmetic["ablation_evaluation_seconds_at_cap"]
    )


@pytest.mark.parametrize("bad", (0.0, -1.0, math.nan, math.inf))
def test_compute_estimate_rejects_each_nonpositive_or_nonfinite_rate(
    monkeypatch: pytest.MonkeyPatch,
    bad: float,
) -> None:
    settings = _settings()
    streams = {
        "synthetic_v2": cast(Any, _Stream(count=1)),
        "chronological_issues": cast(Any, _Stream(count=1)),
    }
    monkeypatch.setattr(
        pilot,
        "load_experiment_settings",
        lambda _: SimpleNamespace(make_stream=lambda: _Stream(count=1)),
    )
    calibration = [
        {
            "status": "completed",
            "method": method,
            "stream": stream,
            "tokens_per_second": (
                bad if (method, stream) == ("frozen_base", "synthetic_v2") else 4.0
            ),
        }
        for method in pilot._METHODS
        for stream in streams
    ]
    profiles = [
        {
            "status": "completed",
            "stream": stream,
            "batched_seconds_per_example": 1.0,
        }
        for stream in streams
    ]
    result = pilot.compute_estimate(
        calibration,
        profiles,
        streams,
        cast(Any, SimpleNamespace(tokenizer=_Tokenizer())),
        settings,
    )
    assert result["status"] == "incomplete"
    assert result["missing_measurements"] == ["frozen_base:synthetic_v2"]
    assert "arithmetic" not in result
