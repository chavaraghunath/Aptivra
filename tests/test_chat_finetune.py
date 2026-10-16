"""Focused mutation proofs for lane-6 target-only identity fine-tuning."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import aptivra.inference.chat_finetune as chat_finetune
from aptivra.experts.standard import (
    ExpertManifest,
    StandardExpertSettings,
    load_standard_expert_settings,
)
from aptivra.inference.chat_canonical import (
    IdentityManifest,
    load_identity_manifest,
)
from aptivra.inference.chat_finetune import (
    IdentityTrainingBatch,
    identity_training_instances,
    materialize_identity_batch,
    train_identity_adapter,
)
from aptivra.inference.chat_prompt import (
    END_OF_TEXT,
    IM_END,
    IM_START,
    IdentitySettings,
    load_identity_settings,
)
from aptivra.inference.standard_base import StandardBase
from aptivra.tasks.base import TaskInstance


def _enabled_identity_settings(*, max_steps: int, epochs: int) -> IdentitySettings:
    settings = load_identity_settings()
    training = settings.training.model_copy(
        update={"max_optimizer_steps": max_steps, "epochs": epochs}
    )
    return settings.model_copy(
        update={
            "feature": settings.feature.model_copy(update={"enabled": True}),
            "training": training,
        }
    )


def _enabled_expert_settings() -> StandardExpertSettings:
    settings = load_standard_expert_settings()
    return settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": True})}
    )


class _ScriptedTokenizer:
    eos_token_id = 0

    def __init__(self, *, prompt_ids: list[int], target_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.target_ids = target_ids

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        special = {IM_START: [1], IM_END: [2], END_OF_TEXT: [0]}
        if text in special:
            return special[text]
        if text == "prompt":
            return list(self.prompt_ids)
        if text == f"answer{IM_END}{END_OF_TEXT}":
            return list(self.target_ids)
        raise AssertionError(f"unexpected tokenization input: {text!r}")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {IM_START: 1, IM_END: 2, END_OF_TEXT: 0}[token]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return {1: IM_START, 2: IM_END, 0: END_OF_TEXT}[token_ids[0]]


def _batch_base(
    tokenizer: _ScriptedTokenizer,
    *,
    sequence_limit: int,
) -> StandardBase:
    return cast(
        StandardBase,
        SimpleNamespace(
            tokenizer=tokenizer,
            settings=SimpleNamespace(
                runtime=SimpleNamespace(max_sequence_tokens=sequence_limit)
            ),
            device=torch.device("cpu"),
        ),
    )


def _batch_settings(*, sequence_limit: int) -> IdentitySettings:
    settings = _enabled_identity_settings(max_steps=1, epochs=1)
    return settings.model_copy(
        update={
            "training": settings.training.model_copy(
                update={"max_sequence_tokens": sequence_limit}
            )
        }
    )


def _batch_instance() -> TaskInstance:
    return TaskInstance("identity-test", "identity", "prompt", "answer")


def test_training_instances_pass_exact_count_and_zero_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_identity_manifest()
    expert_settings = _enabled_expert_settings()
    calls: list[tuple[str, int, int]] = []

    class _IdentityFamily:
        def __init__(
            self,
            received_manifest: IdentityManifest,
            *,
            expert_settings: StandardExpertSettings,
        ) -> None:
            assert received_manifest is manifest
            assert expert_settings is not None

        def generate_instances(self, count: int, *, seed: int) -> list[TaskInstance]:
            calls.append(("identity", count, seed))
            return [
                TaskInstance(f"identity-{index}", "identity", "prompt", "answer")
                for index in range(count)
            ]

    class _ConversationFamily:
        def __init__(
            self,
            received_manifest: IdentityManifest,
            *,
            expert_settings: StandardExpertSettings,
        ) -> None:
            assert received_manifest is manifest
            assert expert_settings is not None

        def generate_instances(self, count: int, *, seed: int) -> list[TaskInstance]:
            calls.append(("conversation", count, seed))
            return [
                TaskInstance(f"conversation-{index}", "identity", "prompt", "answer")
                for index in range(count)
            ]

    monkeypatch.setattr(chat_finetune, "ModelIdentityTaskFamily", _IdentityFamily)
    monkeypatch.setattr(chat_finetune, "ConversationalTaskFamily", _ConversationFamily)

    instances = identity_training_instances(
        manifest,
        expert_settings=expert_settings,
    )

    assert calls == [("identity", 5, 0), ("conversation", 5, 0)]
    assert [item.task_id for item in instances] == [
        *(f"identity-{index}" for index in range(5)),
        *(f"conversation-{index}" for index in range(5)),
    ]


def test_batch_rejects_lexically_greater_incorrect_suffix() -> None:
    tokenizer = _ScriptedTokenizer(prompt_ids=[10], target_ids=[9, 0])

    with pytest.raises(ValueError, match="message delimiter followed by EOS"):
        materialize_identity_batch(
            _batch_base(tokenizer, sequence_limit=10),
            _batch_instance(),
            _batch_settings(sequence_limit=10),
        )


def test_batch_token_limit_uses_sum_for_overlapping_bit_lengths() -> None:
    tokenizer = _ScriptedTokenizer(prompt_ids=[10, 11, 12], target_ids=[7, 2, 0])

    with pytest.raises(ValueError, match="prompt=3, target=3, total=6, limit=5"):
        materialize_identity_batch(
            _batch_base(tokenizer, sequence_limit=5),
            _batch_instance(),
            _batch_settings(sequence_limit=5),
        )


class _CountingInstances:
    def __init__(self, count: int) -> None:
        self.items = tuple(
            TaskInstance(f"task-{index}", "identity", "prompt", "answer")
            for index in range(count)
        )
        self.iterations = 0
        self.yields = 0

    def __iter__(self) -> Iterator[TaskInstance]:
        self.iterations += 1
        for item in self.items:
            self.yields += 1
            yield item


class _TrainingModel:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.calls = 0
        self.train_calls = 0

    def train(self) -> None:
        self.train_calls += 1

    def __call__(self, **kwargs: object) -> object:
        assert kwargs["use_cache"] is False
        assert kwargs["return_dict"] is True
        self.calls += 1
        return SimpleNamespace(loss=(self.parameter - 1.0).square())


class _TrainingBank:
    def __init__(self, *, hashes: tuple[str, str]) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.5))
        self.model = _TrainingModel(self.parameter)
        self.hashes = hashes
        self.hash_calls = 0
        self.promoted_to: Path | str | None = None
        self.manifest = cast(
            ExpertManifest,
            SimpleNamespace(adapter_name="identity", content_sha256="a" * 64),
        )

    def adapter_parameters(self, adapter_name: str) -> list[torch.nn.Parameter]:
        assert adapter_name == "identity"
        return [self.parameter]

    def base_parameter_sha256(self) -> str:
        value = self.hashes[min(self.hash_calls, len(self.hashes) - 1)]
        self.hash_calls += 1
        return value

    def promote_candidate(self, output_dir: Path | str) -> ExpertManifest:
        self.promoted_to = output_dir
        return self.manifest


class _RecordingOptimizer:
    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
    ) -> None:
        self.parameters = tuple(parameters)
        assert lr > 0.0
        assert weight_decay >= 0.0
        self.zero_grad_calls: list[bool] = []
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_grad_calls.append(set_to_none)
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def step(self) -> None:
        self.step_calls += 1


def _install_training_fakes(
    monkeypatch: pytest.MonkeyPatch,
    instances: _CountingInstances,
    *,
    hashes: tuple[str, str] = ("a" * 64, "a" * 64),
) -> tuple[_TrainingBank, list[_RecordingOptimizer]]:
    bank = _TrainingBank(hashes=hashes)
    optimizers: list[_RecordingOptimizer] = []

    def start_candidate(
        cls: type[object],
        base: StandardBase,
        settings: StandardExpertSettings,
        adapter_name: str,
    ) -> _TrainingBank:
        del cls, base, settings
        assert adapter_name == "identity"
        return bank

    def make_optimizer(
        parameters: Sequence[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
    ) -> _RecordingOptimizer:
        optimizer = _RecordingOptimizer(parameters, lr=lr, weight_decay=weight_decay)
        optimizers.append(optimizer)
        return optimizer

    def validate_tokenizer(tokenizer: object, template: object) -> dict[str, int]:
        del tokenizer, template
        return {IM_START: 1, IM_END: 2, END_OF_TEXT: 0}

    def generated_instances(
        manifest: IdentityManifest,
        *,
        expert_settings: StandardExpertSettings,
    ) -> _CountingInstances:
        del manifest, expert_settings
        return instances

    monkeypatch.setattr(
        chat_finetune.PeftExpertBank,
        "start_candidate",
        classmethod(start_candidate),
    )
    monkeypatch.setattr(chat_finetune.torch.optim, "AdamW", make_optimizer)
    monkeypatch.setattr(
        chat_finetune,
        "validate_chat_tokenizer",
        validate_tokenizer,
    )
    monkeypatch.setattr(
        chat_finetune,
        "identity_training_instances",
        generated_instances,
    )
    batch = IdentityTrainingBatch(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        labels=torch.tensor([[-100, 2]], dtype=torch.long),
        task_id="task",
        prompt_tokens=1,
        target_tokens=7,
    )
    def materialize_batch(
        base: StandardBase,
        instance: TaskInstance,
        settings: IdentitySettings,
    ) -> IdentityTrainingBatch:
        del base, instance, settings
        return batch

    monkeypatch.setattr(
        chat_finetune,
        "materialize_identity_batch",
        materialize_batch,
    )
    return bank, optimizers


def test_training_zeroes_gradients_with_exact_set_to_none_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _CountingInstances(1)
    bank, optimizers = _install_training_fakes(monkeypatch, instances)
    result = train_identity_adapter(
        cast(StandardBase, SimpleNamespace(tokenizer=object())),
        _enabled_expert_settings(),
        _enabled_identity_settings(max_steps=1, epochs=1),
        load_identity_manifest(),
        tmp_path / "zero-grad",
    )

    assert result.optimizer_steps == 1
    assert bank.model.calls == 1
    assert len(optimizers) == 1
    assert optimizers[0].zero_grad_calls == [True]


@pytest.mark.parametrize(
    (
        "instance_count",
        "max_steps",
        "epochs",
        "expected_iterations",
        "expected_yields",
    ),
    [(2, 3, 4, 2, 4), (2, 2, 4, 1, 2), (3, 1, 4, 1, 2)],
)
def test_training_budget_breaks_at_exact_inner_and_outer_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instance_count: int,
    max_steps: int,
    epochs: int,
    expected_iterations: int,
    expected_yields: int,
) -> None:
    instances = _CountingInstances(instance_count)
    bank, _ = _install_training_fakes(monkeypatch, instances)

    result = train_identity_adapter(
        cast(StandardBase, SimpleNamespace(tokenizer=object())),
        _enabled_expert_settings(),
        _enabled_identity_settings(max_steps=max_steps, epochs=epochs),
        load_identity_manifest(),
        tmp_path / "bounded",
    )

    assert result.optimizer_steps == max_steps
    assert result.target_tokens == max_steps * 7
    assert bank.model.calls == max_steps
    assert instances.iterations == expected_iterations
    assert instances.yields == expected_yields


def test_training_rejects_zero_steps_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _CountingInstances(0)
    _install_training_fakes(monkeypatch, instances)

    with pytest.raises(RuntimeError, match="completed no optimizer steps"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1, epochs=2),
            load_identity_manifest(),
            tmp_path / "empty",
        )


def test_training_rejects_lexically_smaller_changed_base_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _CountingInstances(1)
    _install_training_fakes(
        monkeypatch,
        instances,
        hashes=("f" * 64, "0" * 64),
    )

    with pytest.raises(RuntimeError, match="modified frozen base bytes"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1, epochs=1),
            load_identity_manifest(),
            tmp_path / "changed-base",
        )
