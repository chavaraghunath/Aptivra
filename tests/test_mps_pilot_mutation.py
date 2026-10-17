"""Focused mutation proofs for the bounded MPS pilot."""

from __future__ import annotations

import json
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
    stream = _Stream(count=1)
    audit = {"passed": True}
    manifest = {
        "schema_version": 3,
        "audit": audit,
        "record_count": 6,
        "record_shards": ["one"],
    }

    monkeypatch.setattr(pilot, "write_or_check_plan", lambda *args, **kwargs: None)
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
            raw_dataset_path=issues.raw_data,
            manifest_path=issues.manifest,
            audit_path=issues.audit_record,
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

    assert result["chronological_issues"]["record_count"] == 6


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

    with pytest.raises(TimeoutError, match="wall-time cap"):
        pilot._quiet_before_measurement(
            settings,
            {"quiet_gate_events": []},
            tmp_path / "progress.json",
            0.0,
        )


def test_calibration_full_ft_method_state_is_exact() -> None:
    settings = _settings()
    model = torch.nn.Linear(3, 2)
    base = cast(Any, SimpleNamespace(model=model))

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
    assert {
        name: parameter.requires_grad
        for name, adapter in model.adapters.items()
        for parameter in adapter.parameters()
    } == {
        "resident_0000": False,
        "resident_0001": False,
        "resident_0002": True,
    }


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
            0.01,
        )


def test_bounded_pair_rejects_index_equal_to_family_count() -> None:
    with pytest.raises(ValueError, match="distinct real tasks"):
        pilot._BoundedPairStream(cast(Any, _Stream()), (0, 2), 1, 1)


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
    ("ending_swap", "expected_compromised", "expected_throughput"),
    ((125, True, None), (105, False, 10.0)),
)
def test_frozen_calibration_pads_and_accounts_for_swap_and_throughput(
    ending_swap: int,
    expected_compromised: bool,
    expected_throughput: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "calibration": settings.calibration.model_copy(
                update={"swap_throughput_reporting_threshold_bytes": 20}
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
    monkeypatch.setattr(pilot, "_base_for_pilot", lambda _: cast(Any, base))
    monkeypatch.setattr(pilot, "_cleanup_base", lambda _: None)
    monkeypatch.setattr(pilot, "longest_distinct_examples", lambda *_: (selected, [5, 4]))
    monkeypatch.setattr(pilot, "collate_task_instances", lambda *_: initial_batch)

    def loss(_base: Any, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        seen_lengths.append(batch.input_ids.shape[1])
        return torch.tensor(2.0), torch.empty(0)

    monkeypatch.setattr(pilot, "_decoder_loss", loss)
    monkeypatch.setattr(pilot, "_synchronize", lambda _: None)
    monkeypatch.setattr(pilot, "_mps_values", lambda: (7, 13))
    monkeypatch.setattr(pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: _Sampler())
    swaps = iter((100, ending_swap))
    monkeypatch.setattr(pilot, "system_swap_used_bytes", lambda: next(swaps))
    monkeypatch.setattr(pilot, "system_free_memory_percentage", lambda: 75.0)
    counter = iter(float(value) for value in range(20))
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(
        pilot,
        "generate_task_instance_answer",
        lambda *_args, **_kwargs: SimpleNamespace(generated_token_ids=(1,), stop_reason="eos"),
    )

    record = pilot.run_calibration_attempt(
        "frozen_base",
        "synthetic_v2",
        cast(Any, _Stream()),
        2,
        settings,
        {"quiet": True},
    )

    assert record["status"] == "completed"
    assert record["memory"]["mps_driver_step_peak_bytes"] == 13
    assert record["system_swap_growth_bytes"] == ending_swap - 100
    assert record["throughput_compromised"] is expected_compromised
    assert record["tokens_per_second"] == expected_throughput
    assert seen_lengths == [5, 5, 5, 5]


class _RepeatSampler(_Sampler):
    def record(self) -> dict[str, Any]:
        return {"sampled": True}


def test_naive_repeatability_visits_wrapped_batches_and_records_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    settings = settings.model_copy(
        update={
            "repeatability": settings.repeatability.model_copy(
                update={
                    "training_examples_per_task": 4,
                    "evaluation_examples_per_task": 1,
                    "optimizer_steps_per_task": 3,
                    "micro_batch_size": 2,
                }
            )
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
    monkeypatch.setattr(pilot, "MemoryPeakSampler", lambda *_args, **_kwargs: _RepeatSampler())
    monkeypatch.setattr(
        pilot,
        "generate_task_instance_answer",
        lambda *_a, **_k: SimpleNamespace(text="ok"),
    )
    monkeypatch.setattr(pilot, "_evaluate_pair", lambda *_: [1.0, 1.0])
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
    assert selected_ids[:3] == [
        ("a-1-0", "a-1-1"),
        ("a-1-2", "a-1-3"),
        ("a-1-0", "a-1-1"),
    ]
    assert len(selected_ids) == 6


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
                update={
                    "training_examples_per_task": 2,
                    "evaluation_examples_per_task": 1,
                    "optimizer_steps_per_task": 1,
                }
            )
        }
    )
    stream = pilot._BoundedPairStream(cast(Any, _Stream()), (0, 1), 2, 1)
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
    monkeypatch.setattr(pilot, "_losses", lambda *_: np.asarray([1.0]))
    monkeypatch.setattr(pilot, "_train_adapter_steps", lambda *_: ([0.5], 7))
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
    )

    assert len(result["promotions"]) == 2
    assert result["target_tokens"] == 14
    assert result["optimizer_steps"] == 2


def test_compute_estimate_matches_independent_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    streams = {
        "synthetic_v2": cast(Any, _Stream(count=1)),
        "chronological_issues": cast(Any, _Stream(count=1)),
    }
    v1 = _Stream(count=1)
    monkeypatch.setattr(
        pilot,
        "load_experiment_settings",
        lambda _: SimpleNamespace(make_stream=lambda: v1),
    )
    calibration = [
        {
            "status": "completed",
            "method": method,
            "stream": stream,
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

    result = pilot.compute_estimate(
        calibration,
        streams,
        cast(Any, SimpleNamespace(tokenizer=_Tokenizer())),
        settings,
    )

    estimate = settings.h3_estimate
    evaluation_cap = settings.cycle_caps.evaluation_examples_per_seen_task * 3
    per_method_stream_training = 10 / 4
    main_training = 12 * per_method_stream_training * estimate.main_seed_count
    main_evaluation = estimate.main_method_count * estimate.main_seed_count * evaluation_cap * 5.0
    v1_training = 6 * per_method_stream_training * estimate.v1_regression_seed_count
    v1_evaluation = (
        estimate.v1_regression_method_count
        * estimate.v1_regression_seed_count
        * evaluation_cap
        * 2.5
    )
    ablation_cells = sum(item.seeds for item in estimate.ablations) * 2
    ablation_training = ablation_cells * per_method_stream_training
    ablation_evaluation = ablation_cells * evaluation_cap * 2.5
    arithmetic = result["arithmetic"]

    assert result["exact_training_sequence_tokens"] == {
        "synthetic_v2": 10,
        "chronological_issues": 10,
        "v1_regression": 10,
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
        [], streams, cast(Any, SimpleNamespace(tokenizer=_Tokenizer())), settings
    )

    assert result["status"] == "incomplete"
    assert "frozen_base:synthetic_v2" in result["missing_measurements"]
    assert "frozen_generation:chronological_issues" in result["missing_measurements"]
    assert "Infinity" not in json.dumps(result, allow_nan=False)
