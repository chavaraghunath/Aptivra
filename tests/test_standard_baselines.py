"""Tiny-fixture acceptance tests for ADR-0011 lane 3."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml

import aptivra.baselines.standard as standard_baselines
from aptivra.baselines.standard import (
    BaselineMethod,
    BoundedReplayBuffer,
    ManualLoRALinear,
    ManualLoRATrainer,
    OnlineEWCTrainer,
    ReplayRecord,
    StandardBaselineSettings,
    build_five_seed_schedules,
    build_training_schedule,
    create_standard_baseline_trainer,
    full_logit_loss_from_hidden,
    load_standard_baseline_settings,
    run_standard_baseline,
    run_standard_baseline_suite,
    target_only_loss_from_hidden,
)
from aptivra.eval.harness import (
    BaselineCompute,
    ContinualEvalHarness,
    StandardBaselineRecord,
    StandardContinualEvalHarness,
)
from aptivra.inference.standard_base import (
    StandardBase,
    collate_task_instances,
    load_standard_base,
    load_standard_base_settings,
)
from aptivra.tasks.base import (
    MechanicalVerifier,
    TaskFamily,
    TaskInstance,
    VerifierResult,
    exact_match_result,
)
from aptivra.tasks.github_issues import GitHubIssuesContinualStream, content_hash
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamSplit
from tests.test_standard_base import _tiny_snapshot


class _ExactVerifier(MechanicalVerifier):
    def verify(self, instance: TaskInstance, model_output: str) -> VerifierResult:
        return exact_match_result(instance, model_output)


class _TinyFamily(TaskFamily):
    def __init__(self, family_id: str, prompt: str, answer: str) -> None:
        super().__init__(family_id, family_id, "tiny lane-3 fixture")
        self.prompt = prompt
        self.answer = answer
        self._verifier = _ExactVerifier()

    @property
    def verifier(self) -> MechanicalVerifier:
        return self._verifier

    def generate_instances(self, count: int, seed: int) -> list[TaskInstance]:
        return [
            TaskInstance(
                task_id=f"{self.family_id}-{seed}-{index}",
                task_family=self.family_id,
                prompt=self.prompt,
                target_solution=self.answer,
            )
            for index in range(count)
        ]


class _TinyV2Stream:
    version = "v2"
    stream_id = "task-stream-v2-tiny-fixture"

    def __init__(self, *, train_count: int = 4) -> None:
        self.families = [
            _TinyFamily("tiny_skill", "short prompt", "answer"),
            _TinyFamily("tiny_code", "longer prompt", "target"),
        ]
        self.num_train_per_task = train_count
        self.num_val_per_task = 1
        self.num_test_per_task = 1
        self.num_template_ood_test_per_code_family = 1
        self.family_roles = {family.family_id: "skill" for family in self.families}
        self.headline_task_indices = [0, 1]
        self._splits = {
            family.family_id: TaskStreamSplit(
                train=family.generate_instances(train_count, 10 + index),
                val=family.generate_instances(1, 20 + index),
                test=family.generate_instances(1, 30 + index),
            )
            for index, family in enumerate(self.families)
        }
        self._template_ood = {
            "tiny_code": [
                TaskInstance(
                    "tiny-code-ood",
                    "tiny_code",
                    "short longer prompt",
                    "target",
                )
            ]
        }

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._splits[family_id]

    def get_template_ood_split(self, family_id: str) -> list[TaskInstance]:
        return self._template_ood.get(family_id, [])


class _TinyStreamWithoutTemplateOOD:
    def __init__(self, stream: _TinyV2Stream) -> None:
        self.families = stream.families
        self.num_train_per_task = stream.num_train_per_task
        self.num_val_per_task = stream.num_val_per_task
        self.num_test_per_task = stream.num_test_per_task
        self._stream = stream

    def get_split(self, family_id: str) -> TaskStreamSplit:
        return self._stream.get_split(family_id)


def _enabled_settings(**updates: object) -> StandardBaselineSettings:
    settings = load_standard_baseline_settings()
    training = settings.training.model_copy(update={"max_new_tokens": 1})
    values: dict[str, object] = {
        "feature": settings.feature.model_copy(update={"enabled": True}),
        "training": training,
    }
    values.update(updates)
    return settings.model_copy(update=values)


def _tiny_base(tmp_path: Path) -> StandardBase:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    return load_standard_base(
        load_standard_base_settings(config_path),
        snapshot_path=snapshot,
        device="cpu",
    )


def _long_context_tiny_base(tmp_path: Path) -> StandardBase:
    """Reuse the tiny weights with enough RoPE context for every real v2 prompt."""
    config_path, snapshot = _tiny_snapshot(tmp_path)
    model_config_path = snapshot / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["max_position_embeddings"] = 128
    model_config_path.write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    standard_base = config["standard_base"]
    config_bytes = model_config_path.read_bytes()
    standard_base["source"]["config"] = {
        "path": "config.json",
        "bytes": len(config_bytes),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    standard_base["architecture"]["max_position_embeddings"] = 128
    standard_base["runtime"].update(
        {"max_prompt_tokens": 126, "max_sequence_tokens": 128, "max_new_tokens": 1}
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_standard_base(
        load_standard_base_settings(config_path),
        snapshot_path=snapshot,
        device="cpu",
    )


def _instance(task_id: str, family: str, prompt: str = "short") -> TaskInstance:
    return TaskInstance(task_id, family, prompt, "answer")


def test_packaged_config_is_strict_disabled_and_complete(tmp_path: Path) -> None:
    settings = load_standard_baseline_settings()
    assert load_standard_baseline_settings(
        "configs/smollm2_baselines.yaml"
    ) == settings
    assert settings.feature.enabled is False
    assert settings.methods == [
        "frozen_base",
        "naive_sequential",
        "replay",
        "online_ewc",
        "manual_lora",
    ]
    assert len(settings.seeds) == len(set(settings.seeds)) == 5
    assert settings.training.effective_batch_size == 8
    assert settings.replay.capacity == 500
    assert settings.online_ewc.state_byte_limit == 3_000_000_000
    assert settings.manual_lora.rank == 16
    assert set(settings.manual_lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    raw = yaml.safe_load(Path("configs/smollm2_baselines.yaml").read_text(encoding="utf-8"))
    invalid = copy.deepcopy(raw)
    invalid["smollm2_baselines"]["seeds"][-1] = invalid["smollm2_baselines"]["seeds"][0]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="five unique"):
        load_standard_baseline_settings(path)

    wrong_root = tmp_path / "wrong-root.yaml"
    wrong_root.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contain only"):
        load_standard_baseline_settings(wrong_root)

    mismatched_batch = settings.training.model_copy(update={"micro_batch_size": 4})
    with pytest.raises(ValueError, match=r"micro_batch_size \*"):
        standard_baselines.BaselineTrainingSettings.model_validate(mismatched_batch.model_dump())
    nonstandard_effective = settings.training.model_copy(
        update={"micro_batch_size": 4, "gradient_accumulation_steps": 1, "effective_batch_size": 4}
    )
    with pytest.raises(ValueError, match="effective_batch_size=8"):
        standard_baselines.BaselineTrainingSettings.model_validate(
            nonstandard_effective.model_dump()
        )
    with pytest.raises(ValueError, match="cannot exceed capacity"):
        standard_baselines.ReplaySettings.model_validate(
            settings.replay.model_dump()
            | {"capacity": 1, "per_family_minimum": 2}
        )


def test_five_seed_schedule_is_reproducible_shared_and_seed_sensitive() -> None:
    settings = _enabled_settings()
    stream = _TinyV2Stream(train_count=8)
    first = build_five_seed_schedules(settings, stream)
    repeated = build_five_seed_schedules(settings, stream)

    assert first == repeated
    assert len({schedule.sha256 for schedule in first}) == 5
    assert all(schedule.order(0, 0) != first[0].order(0, 0) for schedule in first[1:])
    assert build_training_schedule(
        stream,
        seed=settings.seeds[0],
        epochs_per_task=1,
    ).order(0, 0) == first[0].order(0, 0)
    with pytest.raises(IndexError, match="no task"):
        first[0].order(99, 0)
    with pytest.raises(ValueError, match="non-negative"):
        build_training_schedule(stream, seed=-1, epochs_per_task=1)


def test_target_only_loss_is_exactly_full_logit_loss_in_float64_cpu(
    tmp_path: Path,
) -> None:
    base = _tiny_base(tmp_path)
    base.model.double()
    batch = collate_task_instances(
        base,
        [
            TaskInstance("one", "tiny", "short prompt", "answer"),
            TaskInstance("two", "tiny", "short", "target"),
        ],
    )
    outputs = base.model.model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        return_dict=True,
    )

    target_only = target_only_loss_from_hidden(base.model, outputs.last_hidden_state, batch.labels)
    full = full_logit_loss_from_hidden(base.model, outputs.last_hidden_state, batch.labels)

    assert target_only.dtype == torch.float64
    assert torch.equal(target_only, full)

    no_targets = torch.full_like(batch.labels, -100)
    with pytest.raises(ValueError, match="supervised target"):
        target_only_loss_from_hidden(base.model, outputs.last_hidden_state, no_targets)
    with pytest.raises(
        ValueError,
        match="^full-logit reference requires at least one target token$",
    ):
        full_logit_loss_from_hidden(base.model, outputs.last_hidden_state, no_targets)


def test_replay_reservoir_is_deterministic_stratified_and_fails_closed() -> None:
    defaults = load_standard_baseline_settings().replay
    settings = defaults.model_copy(
        update={"capacity": 2, "per_family_minimum": 1, "byte_limit": 10_000}
    )
    first = BoundedReplayBuffer(settings)
    second = BoundedReplayBuffer(settings)
    records = [
        _instance("a1", "a"),
        _instance("a2", "a"),
        _instance("b1", "b"),
    ]
    first.add_instances(records[:2])
    first.add_instances(records[2:])
    second.add_instances(records)
    assert first.records == second.records
    assert {record.task_family for record in first.records} == {"a", "b"}
    assert len(first.records) == 2
    assert 0 < first.size_bytes <= settings.byte_limit
    assert first.sample(1, context="same") == first.sample(1, context="same")

    before = first.records
    with pytest.raises(MemoryError, match="per-family minimum"):
        first.add_instances([_instance("c1", "c")])
    assert first.records == before

    tiny_limit = defaults.model_copy(
        update={"capacity": 2, "per_family_minimum": 1, "byte_limit": 1}
    )
    with pytest.raises(MemoryError, match="exceeds"):
        BoundedReplayBuffer(tiny_limit).add_instances([_instance("large", "a")])
    with pytest.raises(ValueError, match="non-negative"):
        first.sample(-1, context="bad")

    first._records = cast(tuple[ReplayRecord, ...], (torch.tensor([1]),))
    with pytest.raises(RuntimeError, match="accelerator tensors"):
        first.assert_within_bounds()


def test_online_ewc_retains_one_consolidated_state_and_enforces_bound(
    tmp_path: Path,
) -> None:
    stream = _TinyV2Stream(train_count=2)
    settings = _enabled_settings()
    schedule = build_training_schedule(stream, seed=settings.seeds[0], epochs_per_task=1)
    base = _tiny_base(tmp_path / "bounded")
    required_bytes = 2 * sum(
        parameter.numel() * parameter.element_size() for parameter in base.model.parameters()
    )
    too_small = settings.model_copy(
        update={
            "online_ewc": settings.online_ewc.model_copy(
                update={"state_byte_limit": required_bytes - 1}
            )
        }
    )
    with pytest.raises(MemoryError, match=r"Fisher\+anchor"):
        OnlineEWCTrainer(base, too_small, schedule)

    trainer = OnlineEWCTrainer(_tiny_base(tmp_path / "online"), settings, schedule)
    first = stream.get_split("tiny_skill").train
    second = stream.get_split("tiny_code").train
    assert math.isfinite(trainer.train_task(0, first))
    first_count = trainer.state_tensor_count
    first_bytes = trainer.ewc_state_bytes
    assert first_count > 0
    assert math.isfinite(trainer.train_task(1, second))
    assert trainer.state_tensor_count == first_count
    assert trainer.ewc_state_bytes == first_bytes
    assert isinstance(trainer.fisher, dict)
    assert isinstance(trainer.anchor, dict)

    trainer.settings = settings.model_copy(
        update={
            "online_ewc": settings.online_ewc.model_copy(
                update={"state_byte_limit": first_bytes - 1}
            )
        }
    )
    with pytest.raises(RuntimeError, match="live state"):
        _ = trainer.ewc_state_bytes


def test_manual_lora_wraps_all_targets_freezes_base_and_uses_no_peft(
    tmp_path: Path,
) -> None:
    settings = _enabled_settings()
    stream = _TinyV2Stream(train_count=2)
    schedule = build_training_schedule(stream, seed=settings.seeds[0], epochs_per_task=1)
    base = _tiny_base(tmp_path)
    original = {
        name: parameter.detach().clone() for name, parameter in base.model.named_parameters()
    }
    trainer = ManualLoRATrainer(base, settings, schedule)

    assert len(trainer.wrapped_layers) == 7
    assert all(isinstance(layer, ManualLoRALinear) for layer in trainer.wrapped_layers)
    assert math.isfinite(trainer.train_task(0, stream.get_split("tiny_skill").train))
    assert trainer.adapter_bytes > 0
    assert set(trainer.family_adapters) == {"tiny_skill"}
    for name, parameter in base.model.named_parameters():
        if ".adapters." not in name:
            original_name = name.replace(".base_layer", "")
            assert torch.equal(parameter, original[original_name])
            assert parameter.requires_grad is False
    assert "peft" not in Path("pyproject.toml").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    "method",
    ["frozen_base", "naive_sequential", "replay", "online_ewc", "manual_lora"],
)
def test_every_baseline_runs_end_to_end_through_v5_harness(
    tmp_path: Path,
    method: BaselineMethod,
) -> None:
    settings = _enabled_settings()
    stream = _TinyV2Stream(train_count=2)
    record = run_standard_baseline(
        method,
        _tiny_base(tmp_path / method),
        stream,
        settings,
        seed=settings.seeds[0],
    )

    assert record.schema_version == "aptivra.standard_baseline_run.v1"
    assert record.method == method
    assert record.seed == settings.seeds[0]
    assert record.stream_id == stream.stream_id
    assert len(record.in_template.accuracy_matrix) == 2
    assert record.in_template.aggregate_max_forgetting is not None
    assert record.template_ood is not None
    assert record.template_ood.accuracy_matrix[1][1] is not None
    assert record.metadata["harness_version"] == "continual-eval-v5"
    assert record.metadata["prediction_boundary"] == "task_instance_decoded_stops"
    assert record.metadata["manual_lora_uses_peft"] is False
    assert all(math.isfinite(loss) for loss in record.task_losses)
    assert record.compute.generations > 0
    assert record.compute.peak_rss_bytes > 0
    assert math.isfinite(record.compute.wall_time_seconds)
    if method == "frozen_base":
        assert record.compute.optimizer_steps == 0
        assert record.compute.target_tokens == 0
        assert record.task_losses == (0.0, 0.0)
    else:
        assert record.compute.optimizer_steps == 2
        assert record.compute.target_tokens > 0
    if method == "replay":
        assert record.compute.replay_bytes > 0
    if method == "online_ewc":
        assert record.compute.ewc_state_bytes > 0
        assert record.compute.fisher_steps == 4
    if method == "manual_lora":
        assert record.compute.adapter_bytes > 0
    if method == "frozen_base":
        log_path = tmp_path / "nested" / "evidence" / "baseline.jsonl"
        StandardContinualEvalHarness.save_record_jsonl(record, log_path)
        StandardContinualEvalHarness.save_record_jsonl(record, log_path)
        expected_line = json.dumps(
            record.to_dict(),
            sort_keys=True,
            allow_nan=False,
        )
        assert log_path.read_text(encoding="utf-8") == f"{expected_line}\n{expected_line}\n"
        payload = json.loads(expected_line)
        assert payload["schedule_sha256"] == record.schedule_sha256
        with pytest.raises(ValueError, match="Out of range float values"):
            StandardContinualEvalHarness.save_record_jsonl(
                replace(record, metadata={"not_finite": float("nan")}),
                tmp_path / "nonfinite.jsonl",
            )

        with pytest.raises(ValueError, match="schema"):
            replace(record, schema_version="unknown")
        with pytest.raises(ValueError, match="non-empty"):
            replace(record, method="")
        with pytest.raises(ValueError, match="lowercase hexadecimal"):
            replace(record, schedule_sha256="G" * 64)
        with pytest.raises(ValueError, match="finite loss"):
            replace(record, task_losses=(float("nan"),))


def test_nonfinite_loss_and_incomplete_harness_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _enabled_settings()
    stream = _TinyV2Stream(train_count=2)
    schedule = build_training_schedule(stream, seed=settings.seeds[0], epochs_per_task=1)
    trainer = create_standard_baseline_trainer(
        "naive_sequential", _tiny_base(tmp_path), settings, schedule
    )
    monkeypatch.setattr(
        standard_baselines,
        "target_only_causal_loss",
        lambda _base, _batch: torch.tensor(float("nan"), requires_grad=True),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_task(0, stream.get_split("tiny_skill").train)

    run = StandardContinualEvalHarness(stream).start_run()
    compute = BaselineCompute(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 1, 0)
    with pytest.raises(RuntimeError, match="incomplete"):
        run.finalize(
            method="frozen_base",
            seed=settings.seeds[0],
            schedule_sha256="a" * 64,
            task_losses=(0.0, 0.0),
            compute=compute,
        )


def test_legacy_harness_template_ood_dispatch_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _TinyV2Stream()
    harness = ContinualEvalHarness(stream)

    def oracle(prompt: str, *, stop_token_ids: tuple[int, ...]) -> str:
        del prompt, stop_token_ids
        return "deliberately wrong"

    assert harness.evaluate_template_ood_task(oracle, 1) == 0.0

    unavailable = ContinualEvalHarness(_TinyStreamWithoutTemplateOOD(stream))
    with pytest.raises(
        ValueError,
        match="^The task stream does not provide template-OOD splits\\.$",
    ):
        unavailable.evaluate_template_ood_task(oracle, 1)

    monkeypatch.setattr(stream, "get_template_ood_split", lambda _family_id: [])
    empty = ContinualEvalHarness(stream)
    with pytest.raises(
        ValueError,
        match="^Task family 'tiny_code' has no template-OOD split\\.$",
    ):
        empty.evaluate_template_ood_task(oracle, 1)


def _write_tiny_github_stream(tmp_path: Path) -> GitHubIssuesContinualStream:
    tmp_path.mkdir(parents=True)
    raw_path = tmp_path / "issues.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    number = 0
    for window_id, prompt, target in (
        ("2019", "short prompt", "answer"),
        ("2020", "longer prompt", "target"),
    ):
        for split in ("train", "test"):
            number += 1
            record_id = f"record-{number}"
            digest = content_hash(prompt, target)
            metadata = {
                "record_id": record_id,
                "repo": "local/tiny",
                "number": number,
                "html_url": f"https://example.invalid/{number}",
                "event_created_at": f"{window_id}-01-01T00:00:00Z",
                "issue_updated_at": f"{window_id}-01-01T00:00:00Z",
                "issue_closed_at": f"{window_id}-01-01T00:00:00Z",
                "window_id": window_id,
                "split": split,
                "source_labels": [target],
                "raw_target_labels": [target],
                "canonical_labels": [target],
                "target": target,
                "content_hash": digest,
                "label_edit_event_count": 0,
            }
            records.append(metadata)
            raw_rows.append(
                {
                    "record_id": record_id,
                    "prompt": prompt,
                    "target_solution": target,
                    "metadata": metadata,
                }
            )
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "seed": 17,
                "raw_dataset_path": str(raw_path),
                "records": records,
                "windows": [
                    {"id": "2019", "start": "2019-01-01", "end": "2020-01-01"},
                    {"id": "2020", "start": "2020-01-01", "end": "2021-01-01"},
                ],
                "audit": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return GitHubIssuesContinualStream(manifest_path, raw_path)


def test_frozen_baseline_runs_on_real_github_stream_loader(tmp_path: Path) -> None:
    settings = _enabled_settings()
    stream = _write_tiny_github_stream(tmp_path / "stream")
    record = run_standard_baseline(
        "frozen_base",
        _tiny_base(tmp_path / "base"),
        stream,
        settings,
        seed=settings.seeds[0],
    )

    assert record.stream_id == "github-issues-chronological-v3"
    assert record.template_ood is None
    assert len(record.in_template.accuracy_matrix) == 2
    assert record.metadata["split_sizes_by_family"] == {
        "github_issues_2019": {"train": 1, "val": 0, "test": 1},
        "github_issues_2020": {"train": 1, "val": 0, "test": 1},
    }


def test_frozen_baseline_runs_on_complete_real_v2_stream(tmp_path: Path) -> None:
    """Prove the production v2 registry, role split, and OOD cells use the v5 harness."""
    stream_config = yaml.safe_load(
        Path("configs/task_stream_v2.yaml").read_text(encoding="utf-8")
    )["task_stream"]
    stream = ContinualTaskStream(
        num_train_per_task=1,
        num_val_per_task=1,
        num_test_per_task=1,
        num_template_ood_test_per_code_family=1,
        seed=stream_config["seed"],
        max_generation_attempts=stream_config["max_generation_attempts"],
        version="v2",
        family_roles=stream_config["family_roles"],
        headline_roles=stream_config["headline_roles"],
    )
    settings = _enabled_settings()
    record = run_standard_baseline(
        "frozen_base",
        _long_context_tiny_base(tmp_path / "base"),
        stream,
        settings,
        seed=settings.seeds[0],
    )

    assert record.stream_id == "v2"
    assert len(record.in_template.accuracy_matrix) == 12
    assert record.template_ood is not None
    assert len(record.metadata["headline_family_ids"]) == 11
    assert "T2_string_transforms" not in record.metadata["headline_family_ids"]
    assert set(record.template_ood.max_forgetting) == {
        "T8_python_microsynth",
        "T9_algo_synth",
        "T10_docstring_code",
        "T11_bug_fixer",
        "T12_type_inferrer",
    }


def test_disabled_suite_rejects_before_factories_are_called() -> None:
    settings = load_standard_baseline_settings()

    def forbidden_base(_method: BaselineMethod, _seed: int) -> StandardBase:
        raise AssertionError("base allocation occurred while feature was disabled")

    def forbidden_stream(_seed: int) -> _TinyV2Stream:
        raise AssertionError("stream allocation occurred while feature was disabled")

    with pytest.raises(
        RuntimeError,
        match="^ADR-0011 standard baselines are disabled by configuration$",
    ):
        run_standard_baseline_suite(
            settings,
            base_factory=forbidden_base,
            stream_factory=forbidden_stream,
        )


def test_enabled_suite_runs_exact_shared_five_by_five_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _enabled_settings()
    streams: dict[int, _TinyV2Stream] = {}
    base_calls: list[tuple[BaselineMethod, int]] = []
    run_calls: list[tuple[BaselineMethod, int, str, object, object]] = []

    def stream_factory(seed: int) -> _TinyV2Stream:
        stream = _TinyV2Stream(train_count=8)
        streams[seed] = stream
        return stream

    def base_factory(method: BaselineMethod, seed: int) -> StandardBase:
        base_calls.append((method, seed))
        return cast(StandardBase, object())

    def fake_run(
        method: BaselineMethod,
        base: StandardBase,
        stream: object,
        passed_settings: StandardBaselineSettings,
        *,
        seed: int,
        schedule: standard_baselines.TrainingSchedule | None = None,
    ) -> StandardBaselineRecord:
        assert schedule is not None
        assert passed_settings is settings
        run_calls.append((method, seed, schedule.sha256, base, stream))
        return cast(StandardBaselineRecord, (method, seed))

    monkeypatch.setattr(standard_baselines, "run_standard_baseline", fake_run)
    records = run_standard_baseline_suite(
        settings,
        base_factory=base_factory,
        stream_factory=stream_factory,
    )

    expected_calls = [
        (method, seed) for seed in settings.seeds for method in settings.methods
    ]
    assert base_calls == expected_calls
    assert list(records) == expected_calls
    assert [(method, seed) for method, seed, *_ in run_calls] == expected_calls
    for seed in settings.seeds:
        calls_for_seed = [call for call in run_calls if call[1] == seed]
        assert len({call[2] for call in calls_for_seed}) == 1
        assert all(call[4] is streams[seed] for call in calls_for_seed)
        assert len({id(call[3]) for call in calls_for_seed}) == len(settings.methods)

    fixed_schedule = build_training_schedule(
        _TinyV2Stream(train_count=8),
        seed=settings.seeds[0],
        epochs_per_task=settings.training.epochs_per_task,
    )

    def colliding_schedule(
        _stream: object,
        *,
        seed: int,
        epochs_per_task: int,
    ) -> standard_baselines.TrainingSchedule:
        assert epochs_per_task == settings.training.epochs_per_task
        return replace(fixed_schedule, seed=seed)

    monkeypatch.setattr(standard_baselines, "build_training_schedule", colliding_schedule)
    with pytest.raises(
        ValueError,
        match="^different seeds produced the same training data order$",
    ):
        run_standard_baseline_suite(
            settings,
            base_factory=base_factory,
            stream_factory=stream_factory,
        )


def test_five_seed_builder_and_frozen_trainer_fail_closed(tmp_path: Path) -> None:
    settings = _enabled_settings()
    colliding_stream = _TinyV2Stream(train_count=1)
    with pytest.raises(
        ValueError,
        match=(
            "^configured seeds produced colliding data orders; "
            "use seeds with distinct schedules$"
        ),
    ):
        build_five_seed_schedules(settings, colliding_stream)

    stream = _TinyV2Stream(train_count=2)
    schedule = build_training_schedule(
        stream,
        seed=settings.seeds[0],
        epochs_per_task=settings.training.epochs_per_task,
    )
    base = _tiny_base(tmp_path)
    trainer = create_standard_baseline_trainer("frozen_base", base, settings, schedule)
    assert all(not parameter.requires_grad for parameter in base.model.parameters())
    assert trainer.train_task(0, stream.get_split("tiny_skill").train) == 0.0
    with pytest.raises(
        ValueError,
        match="^cannot advance frozen baseline over an empty task split$",
    ):
        trainer.train_task(0, [])


def test_record_validation_rejects_nonfinite_and_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="wall time"):
        BaselineCompute(0, 0, 0, 0, 0, float("nan"), 0, 0, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        BaselineCompute(-1, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 1, 0)
