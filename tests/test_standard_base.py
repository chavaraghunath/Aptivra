"""Offline acceptance tests for ADR-0011 lanes 0 through 2."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest
import torch
import yaml
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import aptivra.inference.standard_base as standard_base
from aptivra.inference.batched_generation import generate_task_instance_answers
from aptivra.inference.standard_base import (
    RuntimeSettings,
    StandardBase,
    collate_task_instances,
    forward_task_batch,
    generate_task_answer,
    generate_task_instance_answer,
    load_standard_base,
    load_standard_base_settings,
    resolve_standard_base_device,
    verify_snapshot,
)
from aptivra.tasks.base import TaskInstance

_ROOT = Path(__file__).resolve().parents[1]
_REVISION = "a" * 40
_TINY_MODEL_SEED = 20_261_016
_MPS_TOLERANCE_PATH = Path(__file__).with_name("standard_base_mps_tolerances.json")


@pytest.fixture(autouse=True)
def _offline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access is forbidden in standard-base tests")

    monkeypatch.setattr(socket, "create_connection", fail_network)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _tensor_inventory(path: Path) -> tuple[int, int, str]:
    parameter_count = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        names = list(tensors.keys())
        dtypes = {str(tensors.get_slice(name).get_dtype()) for name in names}
        for name in names:
            parameter_count += math.prod(tensors.get_slice(name).get_shape())
    assert len(dtypes) == 1
    return parameter_count, len(names), dtypes.pop()


def _tiny_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    snapshot = (
        tmp_path
        / "models--local--tiny-llama"
        / "snapshots"
        / _REVISION
    )
    snapshot.mkdir(parents=True)
    vocab = {
        "<eos>": 0,
        "<unk>": 1,
        "short": 2,
        "longer": 3,
        "prompt": 4,
        "answer": 5,
        "target": 6,
        "x": 7,
        "y": 8,
        "z": 9,
        "one": 10,
        "two": 11,
        "three": 12,
        "four": 13,
        "five": 14,
        "six": 15,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<eos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(snapshot)
    tokenizer_config_path = snapshot / "tokenizer_config.json"
    saved_tokenizer_config = json.loads(
        tokenizer_config_path.read_text(encoding="utf-8")
    )
    saved_tokenizer_config["added_tokens_decoder"] = {
        "0": {"content": "<eos>"},
        "1": {"content": "<unk>"},
    }
    tokenizer_config_path.write_text(
        json.dumps(saved_tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(model_config)
    generator = torch.Generator(device="cpu").manual_seed(_TINY_MODEL_SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1.0)
            else:
                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=model_config.initializer_range,
                    generator=generator,
                )
    model.tie_weights()
    model.save_pretrained(snapshot, safe_serialization=True)
    del model

    parameter_count, tensor_count, tensor_dtype = _tensor_inventory(
        snapshot / "model.safetensors"
    )
    tokenizer_config = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    raw_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer_assets = [
        _file_record(path)
        for path in sorted(snapshot.glob("*.json"))
        if path.name not in {"config.json", "generation_config.json"}
    ]
    config = {
        "standard_base": {
            "schema_version": "aptivra.standard_base_config.v1",
            "feature": {"enabled": True},
            "schema_boundary": {
                "legacy_identity": "v0-byte-level",
                "legacy_evaluation_schema": "aptivra.model_card_eval.v4",
                "legacy_provenance_schema": "aptivra.conversion_provenance.v1",
                "new_identity": "smollm2-standard-base-v1",
                "new_fingerprint_schema": "aptivra.standard_base_fingerprint.v1",
            },
            "source": {
                "repository": "local/tiny-llama",
                "revision": _REVISION,
                "config": _file_record(snapshot / "config.json"),
                "weights": _file_record(snapshot / "model.safetensors"),
                "tokenizer_assets": tokenizer_assets,
            },
            "architecture": {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": raw_config["hidden_size"],
                "intermediate_size": raw_config["intermediate_size"],
                "num_hidden_layers": raw_config["num_hidden_layers"],
                "num_attention_heads": raw_config["num_attention_heads"],
                "num_key_value_heads": raw_config["num_key_value_heads"],
                "vocab_size": raw_config["vocab_size"],
                "max_position_embeddings": raw_config["max_position_embeddings"],
                "bos_token_id": raw_config["bos_token_id"],
                "eos_token_id": raw_config["eos_token_id"],
                "tie_word_embeddings": True,
                "unique_parameter_count": parameter_count,
                "tensor_count": tensor_count,
                "tensor_dtype": tensor_dtype,
                "separate_lm_head_weight": False,
            },
            "tokenizer": {
                "tokenizer_class": tokenizer_config["tokenizer_class"],
                "bos_token": "<eos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": None,
                "chat_template": None,
                "standalone_chat_template": False,
                "required_added_tokens": {0: "<eos>", 1: "<unk>"},
            },
            "runtime": {
                "dtype": "float32",
                "max_prompt_tokens": 24,
                "max_sequence_tokens": 32,
                "max_new_tokens": 8,
                "stop_policy": "eos_only",
            },
        }
    }
    config_path = tmp_path / "tiny-standard-base.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, snapshot


@pytest.fixture
def tiny_base(tmp_path: Path) -> StandardBase:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    return load_standard_base(
        load_standard_base_settings(config_path),
        snapshot_path=snapshot,
        device="cpu",
    )


def _task(task_id: str, prompt: str, target: str) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        task_family="test",
        prompt=prompt,
        target_solution=target,
    )


def test_default_contract_records_pinned_facts_and_is_disabled() -> None:
    settings = load_standard_base_settings()

    assert settings.feature.enabled is False
    assert settings.schema_boundary.legacy_identity == "v0-byte-level"
    assert settings.schema_boundary.new_identity == "smollm2-standard-base-v1"
    assert settings.source.repository == "HuggingFaceTB/SmolLM2-360M"
    assert settings.source.revision == "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    assert settings.source.config.sha256 == (
        "34f7801487078de7e434e19162c497e5cc6ff397080e40e8586627cb68a5168a"
    )
    assert settings.source.weights.sha256 == (
        "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f"
    )
    assert settings.architecture.architectures == ["LlamaForCausalLM"]
    assert settings.architecture.unique_parameter_count == 361_821_120
    assert settings.runtime.dtype == "float32"
    assert settings.runtime.stop_policy == "eos_only"


def test_frozen_v0_artifact_hashes_are_literal_regressions() -> None:
    assert _sha256(_ROOT / "aptivra/eval/model_card_heldout_v1.json") == (
        "0bf112e9be7a5d3ccb0f6f6b3d82a833634a735cda7bbf06cf1d3dec0c6a27a3"
    )
    assert _sha256(_ROOT / "aptivra/eval/model_card_heldout_text_v1.txt") == (
        "bb3329d0ecc4b20d65fefb87f0fd6a4b380da0d6f562805d13a9b8c265921f18"
    )


def test_feature_flag_fails_before_snapshot_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("disabled feature reached snapshot verification")

    monkeypatch.setattr(standard_base, "verify_snapshot", forbidden)
    with pytest.raises(RuntimeError) as error:
        load_standard_base(load_standard_base_settings(), device="cpu")
    assert str(error.value) == (
        "ADR-0011 standard-base path is disabled until final local acceptance"
    )


@pytest.mark.parametrize("mutation", ["revision", "hash", "architecture"])
def test_preflight_rejects_wrong_identity_before_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    if mutation == "revision":
        settings = settings.model_copy(
            update={
                "source": settings.source.model_copy(update={"revision": "b" * 40})
            }
        )
    elif mutation == "hash":
        settings = settings.model_copy(
            update={
                "source": settings.source.model_copy(
                    update={
                        "weights": settings.source.weights.model_copy(
                            update={"sha256": "0" * 64}
                        )
                    }
                )
            }
        )
    else:
        settings = settings.model_copy(
            update={
                "architecture": settings.architecture.model_copy(
                    update={"hidden_size": settings.architecture.hidden_size + 8}
                )
            }
        )

    def allocation_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("model allocation occurred before preflight rejected input")

    monkeypatch.setattr(
        standard_base.AutoModelForCausalLM,
        "from_pretrained",
        allocation_forbidden,
    )
    with pytest.raises(ValueError):
        load_standard_base(settings, snapshot_path=snapshot, device="cpu")


def test_verified_tiny_stock_load_is_offline_fp32_and_tied(tmp_path: Path) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)

    verified = verify_snapshot(settings, snapshot)
    base = load_standard_base(settings, snapshot_path=snapshot, device="cpu")

    assert verified.repository == "local/tiny-llama"
    assert verified.revision == _REVISION
    assert verified.unique_parameter_count == settings.architecture.unique_parameter_count
    assert verified.tensor_count == settings.architecture.tensor_count
    assert isinstance(base.model, LlamaForCausalLM)
    assert {parameter.dtype for parameter in base.model.parameters()} == {torch.float32}
    assert (
        base.model.get_input_embeddings().weight
        is base.model.get_output_embeddings().weight
    )
    assert base.tokenizer.pad_token_id is None


def test_stock_loader_forwards_every_security_and_placement_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    reference = load_standard_base(settings, snapshot_path=snapshot, device="cpu")
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    def config_spy(*args: object, **kwargs: object) -> object:
        calls["config"] = (args, kwargs)
        return reference.model.config

    def tokenizer_spy(*args: object, **kwargs: object) -> object:
        calls["tokenizer"] = (args, kwargs)
        return reference.tokenizer

    def model_spy(*args: object, **kwargs: object) -> object:
        calls["model"] = (args, kwargs)
        return reference.model

    placement_calls: list[dict[str, object]] = []
    real_to = reference.model.to

    def placement_spy(*args: object, **kwargs: object) -> torch.nn.Module:
        assert not args
        placement_calls.append(kwargs)
        return real_to(**kwargs)

    monkeypatch.setattr(standard_base.AutoConfig, "from_pretrained", config_spy)
    monkeypatch.setattr(standard_base.AutoTokenizer, "from_pretrained", tokenizer_spy)
    monkeypatch.setattr(
        standard_base.AutoModelForCausalLM,
        "from_pretrained",
        model_spy,
    )
    monkeypatch.setattr(reference.model, "to", placement_spy)

    loaded = load_standard_base(settings, snapshot_path=snapshot)

    assert calls["config"] == (
        (snapshot.absolute(),),
        {"local_files_only": True, "trust_remote_code": False},
    )
    assert calls["tokenizer"] == (
        (snapshot.absolute(),),
        {
            "use_fast": False,
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )
    model_args, model_kwargs = calls["model"]
    assert model_args == (snapshot.absolute(),)
    assert model_kwargs == {
        "config": loaded.model.config,
        "dtype": torch.float32,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert loaded.snapshot.path == snapshot.absolute()
    assert loaded.device == resolve_standard_base_device("auto")
    assert loaded.model.training is False
    assert placement_calls == [
        {"device": resolve_standard_base_device("auto"), "dtype": torch.float32}
    ]


def test_stock_loader_rejects_every_post_preflight_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, snapshot_path = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    reference = load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")

    def arrange(
        patcher: pytest.MonkeyPatch,
        *,
        config: object = None,
        tokenizer: object = None,
        model: object = None,
        snapshot: object = None,
    ) -> None:
        resolved_config = reference.model.config if config is None else config
        resolved_tokenizer = reference.tokenizer if tokenizer is None else tokenizer
        resolved_model = reference.model if model is None else model
        resolved_snapshot = reference.snapshot if snapshot is None else snapshot
        patcher.setattr(
            standard_base,
            "verify_snapshot",
            lambda *args, **kwargs: resolved_snapshot,
        )
        patcher.setattr(
            standard_base.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: resolved_config,
        )
        patcher.setattr(
            standard_base.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: resolved_tokenizer,
        )
        patcher.setattr(
            standard_base.AutoModelForCausalLM,
            "from_pretrained",
            lambda *args, **kwargs: resolved_model,
        )

    with monkeypatch.context() as patcher:
        arrange(patcher, config=object())
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == "verified snapshot did not load as stock LlamaConfig"

    wrong_class = settings.model_copy(
        update={
            "tokenizer": settings.tokenizer.model_copy(
                update={"tokenizer_class": "WrongTokenizer"}
            )
        }
    )
    with monkeypatch.context() as patcher:
        arrange(patcher)
        with pytest.raises(ValueError) as error:
            load_standard_base(wrong_class, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "loaded tokenizer class mismatch: expected WrongTokenizer, "
            f"got {type(reference.tokenizer).__name__}"
        )

    wrong_vocab = settings.model_copy(
        update={
            "architecture": settings.architecture.model_copy(
                update={"vocab_size": settings.architecture.vocab_size + 1}
            )
        }
    )
    with monkeypatch.context() as patcher:
        arrange(patcher)
        with pytest.raises(ValueError) as error:
            load_standard_base(wrong_vocab, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "loaded tokenizer vocabulary does not match the fingerprint"
        )

    wrong_eos = settings.model_copy(
        update={
            "architecture": settings.architecture.model_copy(update={"eos_token_id": 1})
        }
    )
    with monkeypatch.context() as patcher:
        arrange(patcher)
        with pytest.raises(ValueError) as error:
            load_standard_base(wrong_eos, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "loaded tokenizer EOS ID does not match the architecture"
        )

    with monkeypatch.context() as patcher:
        arrange(patcher)
        patcher.setattr(reference.tokenizer, "pad_token", "<eos>")
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == "standard-base tokenizer must not define a pad token"

    with monkeypatch.context() as patcher:
        arrange(patcher, model=object())
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "verified snapshot did not load as stock LlamaForCausalLM"
        )

    with monkeypatch.context() as patcher:
        arrange(patcher)
        reference.model.half()
        patcher.setattr(reference.model, "to", lambda *args, **kwargs: reference.model)
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "standard-base loader failed to cast every floating parameter to fp32"
        )
    reference.model.float()

    original_output_weight = reference.model.lm_head.weight
    reference.model.lm_head.weight = torch.nn.Parameter(original_output_weight.detach().clone())
    with monkeypatch.context() as patcher:
        arrange(patcher)
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "standard-base loader did not preserve tied input/output weights"
        )
    reference.model.lm_head.weight = reference.model.get_input_embeddings().weight

    wrong_count_snapshot = replace(
        reference.snapshot,
        unique_parameter_count=reference.snapshot.unique_parameter_count + 1,
    )
    with monkeypatch.context() as patcher:
        arrange(patcher, snapshot=wrong_count_snapshot)
        with pytest.raises(ValueError) as error:
            load_standard_base(settings, snapshot_path=snapshot_path, device="cpu")
        assert str(error.value) == (
            "loaded unique parameter count does not match the verified snapshot"
        )


def test_masked_collation_preserves_target_eos_and_masks_padding(
    tiny_base: StandardBase,
) -> None:
    batch = collate_task_instances(
        tiny_base,
        [
            _task("short", "short prompt", "answer"),
            _task("long", "longer prompt one two", "target three"),
        ],
    )
    eos_id = tiny_base.tokenizer.eos_token_id
    assert batch.task_ids == ("short", "long")
    assert torch.all(batch.labels[batch.attention_mask == 0] == -100)
    assert torch.all(batch.input_ids[batch.attention_mask == 0] == eos_id)
    for row in range(batch.input_ids.shape[0]):
        final_real_position = int(batch.attention_mask[row].sum().item()) - 1
        assert batch.input_ids[row, final_real_position].item() == eos_id
        assert batch.labels[row, final_real_position].item() == eos_id
        assert batch.attention_mask[row, final_real_position].item() == 1
    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.long
    assert batch.labels.dtype == torch.long
    assert batch.input_ids.device == tiny_base.device
    assert batch.attention_mask.device == tiny_base.device
    assert batch.labels.device == tiny_base.device


def test_collation_accepts_exact_limits_and_rejects_each_limit_independently(
    tiny_base: StandardBase,
) -> None:
    empty_target = replace(
        _task("empty-target", "short", ""),
        training_terminator="",
    )
    empty_target_batch = collate_task_instances(tiny_base, [empty_target])
    assert empty_target_batch.input_ids.tolist() == [[2, 0]]
    assert empty_target_batch.labels.tolist() == [[-100, 0]]

    exact = replace(
        tiny_base,
        settings=tiny_base.settings.model_copy(
            update={
                "runtime": RuntimeSettings(
                    dtype="float32",
                    max_prompt_tokens=3,
                    max_sequence_tokens=5,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    accepted = collate_task_instances(
        exact,
        [_task("boundary", "one two three", "answer")],
    )
    assert accepted.attention_mask.sum().item() == 5

    prompt_only = replace(
        exact,
        settings=exact.settings.model_copy(
            update={
                "runtime": RuntimeSettings(
                    dtype="float32",
                    max_prompt_tokens=2,
                    max_sequence_tokens=8,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="prompt=3"):
        collate_task_instances(
            prompt_only,
            [_task("prompt-only", "one two three", "answer")],
        )

    sequence_only = replace(
        exact,
        settings=exact.settings.model_copy(
            update={
                "runtime": RuntimeSettings.model_construct(
                    dtype="float32",
                    max_prompt_tokens=4,
                    max_sequence_tokens=5,
                    max_new_tokens=1,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="total=6"):
        collate_task_instances(
            sequence_only,
            [_task("sequence-only", "one two three", "answer target")],
        )


def test_raw_task_forward_is_finite(tiny_base: StandardBase) -> None:
    batch = collate_task_instances(
        tiny_base,
        [_task("forward", "short prompt", "answer")],
    )

    output = forward_task_batch(tiny_base, batch)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (*batch.input_ids.shape, len(tiny_base.tokenizer))


def test_forward_passes_the_complete_raw_batch_contract(tiny_base: StandardBase) -> None:
    batch = collate_task_instances(
        tiny_base,
        [_task("forward-contract", "short prompt", "answer")],
    )
    calls: list[dict[str, object]] = []
    sentinel = object()

    class RecordingModel:
        def __call__(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return sentinel

    recording_base = replace(tiny_base, model=cast(Any, RecordingModel()))

    assert forward_task_batch(recording_base, batch) is sentinel
    assert calls == [
        {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
            "return_dict": True,
        }
    ]


def test_oversize_task_fails_with_identity_and_measured_lengths(
    tiny_base: StandardBase,
) -> None:
    constrained = StandardBase(
        model=tiny_base.model,
        tokenizer=tiny_base.tokenizer,
        settings=tiny_base.settings.model_copy(
            update={
                "runtime": RuntimeSettings(
                    dtype="float32",
                    max_prompt_tokens=2,
                    max_sequence_tokens=4,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
        snapshot=tiny_base.snapshot,
        device=tiny_base.device,
    )

    with pytest.raises(
        ValueError,
        match=r"task 'oversize'.*prompt=3, target=2, total=5, max_prompt=2, max_sequence=4",
    ):
        collate_task_instances(
            constrained,
            [_task("oversize", "one two three", "answer")],
        )


def test_cpu_generation_is_non_degenerate_and_repeatable(tiny_base: StandardBase) -> None:
    outcomes = [generate_task_answer(tiny_base, "short prompt") for _ in range(3)]

    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0].generated_token_ids == (4, 4, 4, 4, 4, 4, 4, 4)
    assert outcomes[0].stop_reason == "length"
    assert outcomes[0].text == "prompt prompt prompt prompt prompt prompt prompt prompt"


def test_batched_generation_is_output_identical_to_scalar(tiny_base: StandardBase) -> None:
    instances = (
        TaskInstance("one", "tiny", "short", "answer"),
        TaskInstance("two", "tiny", "longer prompt", "target"),
        TaskInstance("three", "tiny", "one two three", "answer"),
    )

    scalar = tuple(generate_task_instance_answer(tiny_base, item) for item in instances)
    batched = generate_task_instance_answers(tiny_base, instances)

    assert batched == scalar


def test_batched_generation_removes_padding_after_per_row_terminator(
    tiny_base: StandardBase,
) -> None:
    class TerminatorTokenizer:
        eos_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [2] if text == "short" else [3, 2]

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            pieces = {0: "<eos>", 5: "answer", 9: "\n"}
            return "".join(pieces[token] for token in token_ids)

    class TerminatorGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = [5, 9] if input_ids.shape[0] == 1 else [5, 9, 0]
            appended = torch.tensor(
                [suffix] * input_ids.shape[0],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, appended), dim=1)

    base = replace(
        tiny_base,
        model=cast(Any, TerminatorGenerator()),
        tokenizer=cast(Any, TerminatorTokenizer()),
    )
    instances = (
        TaskInstance("one", "tiny", "short", "answer"),
        TaskInstance("two", "tiny", "longer", "answer"),
    )

    scalar = tuple(generate_task_instance_answer(base, item) for item in instances)
    batched = generate_task_instance_answers(base, instances)

    assert batched == scalar
    assert batched[0].generated_token_ids == (5, 9)


def test_generation_removes_only_the_terminal_eos(tiny_base: StandardBase) -> None:
    class AnswerThenEosGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = torch.tensor([[5, 0]], dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    result = generate_task_answer(
        replace(tiny_base, model=cast(Any, AnswerThenEosGenerator())),
        "short prompt",
        max_new_tokens=2,
    )

    assert result.generated_token_ids == (5, 0)
    assert result.stop_reason == "eos"
    assert result.text == "answer"


def test_generation_recognizes_large_eos_and_keeps_all_preceding_tokens(
    tiny_base: StandardBase,
) -> None:
    class LargeEosTokenizer:
        eos_token_id = 1000

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del text
            assert add_special_tokens is False
            return [1]

        def decode(
            self, token_ids: list[int], *, skip_special_tokens: bool
        ) -> str:
            assert skip_special_tokens is False
            return { (5,): "partial", (5, 6): "complete"}[tuple(token_ids)]

    class AnswerThenEosGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = torch.tensor(
                [[5, 6, 1000]], dtype=torch.long, device=input_ids.device
            )
            return torch.cat((input_ids, suffix), dim=1)

    result = generate_task_answer(
        replace(
            tiny_base,
            model=cast(Any, AnswerThenEosGenerator()),
            tokenizer=cast(Any, LargeEosTokenizer()),
        ),
        "prompt",
        max_new_tokens=3,
    )

    assert result.stop_reason == "eos"
    assert result.text == "complete"

    class NoEosGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = torch.tensor([[5, 6]], dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    no_eos = generate_task_answer(
        replace(
            tiny_base,
            model=cast(Any, NoEosGenerator()),
            tokenizer=cast(Any, LargeEosTokenizer()),
        ),
        "prompt",
        max_new_tokens=2,
    )
    assert no_eos.stop_reason == "length"
    assert no_eos.text == "complete"


def test_generation_passes_strict_greedy_contract_and_reports_length(
    tiny_base: StandardBase,
) -> None:
    calls: list[dict[str, object]] = []

    class RecordingGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            calls.append(kwargs)
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = torch.tensor([[5]], dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    recording_base = replace(tiny_base, model=cast(Any, RecordingGenerator()))
    result = generate_task_answer(recording_base, "short prompt", max_new_tokens=1)

    assert result.generated_token_ids == (5,)
    assert result.stop_reason == "length"
    assert result.text == "answer"
    assert len(calls) == 1
    assert calls[0]["do_sample"] is False
    assert calls[0]["max_new_tokens"] == 1
    assert calls[0]["eos_token_id"] == 0
    assert calls[0]["pad_token_id"] == 0
    assert calls[0]["use_cache"] is True
    assert torch.equal(
        cast(torch.Tensor, calls[0]["attention_mask"]),
        torch.ones((1, 2), dtype=torch.long),
    )
    assert cast(torch.Tensor, calls[0]["input_ids"]).dtype == torch.long
    assert cast(torch.Tensor, calls[0]["input_ids"]).device == tiny_base.device


def test_generation_accepts_exact_length_boundaries_and_rejects_each_overrun(
    tiny_base: StandardBase,
) -> None:
    exact = replace(
        tiny_base,
        settings=tiny_base.settings.model_copy(
            update={
                "runtime": RuntimeSettings(
                    dtype="float32",
                    max_prompt_tokens=3,
                    max_sequence_tokens=5,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    assert generate_task_answer(exact, "one two three", max_new_tokens=2).stop_reason == (
        "length"
    )
    assert generate_task_answer(exact, "one", max_new_tokens=1).stop_reason == "length"

    prompt_only = replace(
        exact,
        settings=exact.settings.model_copy(
            update={
                "runtime": RuntimeSettings(
                    dtype="float32",
                    max_prompt_tokens=2,
                    max_sequence_tokens=8,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="prompt=3"):
        generate_task_answer(prompt_only, "one two three", max_new_tokens=1)

    sequence_only = replace(
        exact,
        settings=exact.settings.model_copy(
            update={
                "runtime": RuntimeSettings.model_construct(
                    dtype="float32",
                    max_prompt_tokens=4,
                    max_sequence_tokens=5,
                    max_new_tokens=2,
                    stop_policy="eos_only",
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="requested_new=2"):
        generate_task_answer(sequence_only, "one two three four", max_new_tokens=2)

    bitwise_blind_spot = replace(
        exact,
        settings=exact.settings.model_copy(
            update={
                "runtime": RuntimeSettings.model_construct(
                    dtype="float32",
                    max_prompt_tokens=3,
                    max_sequence_tokens=3,
                    max_new_tokens=1,
                    stop_policy="eos_only",
                    task_stop_policy="decoded_text",
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="requested_new=1"):
        generate_task_answer(bitwise_blind_spot, "one two three", max_new_tokens=1)


def test_decoded_stopping_boundaries_and_runtime_contracts(
    tiny_base: StandardBase,
) -> None:
    calls: list[tuple[list[int], bool]] = []

    class RecordingTokenizer:
        def decode(
            self, token_ids: list[int], *, skip_special_tokens: bool
        ) -> str:
            calls.append((token_ids, skip_special_tokens))
            return "\nanswer"

    with pytest.raises(ValueError, match="non-negative"):
        standard_base.DecodedTextStoppingCriteria(
            cast(Any, RecordingTokenizer()),
            prompt_length=-1,
            stop_sequences=("\n",),
        )
    criterion = standard_base.DecodedTextStoppingCriteria(
        cast(Any, RecordingTokenizer()),
        prompt_length=0,
        stop_sequences=("\n",),
    )
    decision = criterion(torch.tensor([[5, 6]]), None)
    assert decision.tolist() == [True]
    assert calls == [([5, 6], False)]
    assert standard_base._first_decoded_terminator("\nanswer", ("\n",)) == (
        0,
        "\n",
    )

    many_unique = tuple(f"stop-{index}" for index in range(300))
    assert standard_base._validated_stop_sequences(many_unique) == many_unique

    result = standard_base.GenerationResult("answer", (5,), "length")
    with pytest.raises(AttributeError):
        result.text = "changed"  # type: ignore[misc]

    batch = collate_task_instances(tiny_base, [_task("frozen", "prompt", "answer")])
    with pytest.raises(AttributeError):
        batch.task_ids = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        tiny_base.device = torch.device("cpu")  # type: ignore[misc]
    with pytest.raises(AttributeError):
        tiny_base.snapshot.revision = "b" * 40  # type: ignore[misc]


def test_file_fingerprint_accepts_one_character_and_rejects_empty_path() -> None:
    fingerprint = standard_base.FileFingerprint(
        path="x", bytes=1, sha256="a" * 64
    )
    assert fingerprint.path == "x"
    with pytest.raises(ValueError):
        standard_base.FileFingerprint(path="", bytes=1, sha256="a" * 64)


def test_standard_base_runtime_annotations_and_keyword_only_contracts() -> None:
    for value in vars(standard_base).values():
        if getattr(value, "__module__", None) != standard_base.__name__:
            continue
        if inspect.isfunction(value):
            get_type_hints(value)
        elif inspect.isclass(value):
            get_type_hints(value)
            for member in vars(value).values():
                candidate = (
                    member.__func__
                    if isinstance(member, classmethod | staticmethod)
                    else member
                )
                if inspect.isfunction(candidate):
                    get_type_hints(candidate)

    keyword_only = {
        standard_base.DecodedTextStoppingCriteria.__init__: {
            "prompt_length",
            "stop_sequences",
        },
        standard_base.generate_task_answer: {"max_new_tokens", "stop_sequences"},
        standard_base.generate_task_instance_answer: {"max_new_tokens"},
    }
    for function, expected_names in keyword_only.items():
        assert {
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        } == expected_names


@pytest.mark.parametrize("value", [True, 0, 9, 1.5, "2"])
def test_generation_rejects_invalid_length(
    tiny_base: StandardBase,
    value: object,
) -> None:
    with pytest.raises(ValueError) as error:
        generate_task_answer(
            tiny_base,
            "short prompt",
            max_new_tokens=cast(Any, value),
        )
    expected = (
        "max_new_tokens must be an integer"
        if isinstance(value, bool) or not isinstance(value, int)
        else "max_new_tokens must be between 1 and 8"
    )
    assert str(error.value) == expected


def test_empty_batch_and_unknown_prompt_fail_clearly(tiny_base: StandardBase) -> None:
    with pytest.raises(ValueError) as batch_error:
        collate_task_instances(tiny_base, [])
    assert str(batch_error.value) == "cannot collate an empty task batch"
    with pytest.raises(ValueError) as generation_error:
        generate_task_answer(tiny_base, "")
    assert str(generation_error.value) == "generation prompt tokenized to zero tokens"
    with pytest.raises(ValueError) as prompt_error:
        collate_task_instances(tiny_base, [_task("empty-prompt", "", "answer")])
    assert str(prompt_error.value) == (
        "task 'empty-prompt' prompt tokenized to zero tokens"
    )


def test_missing_integer_eos_fails_before_materialization(tiny_base: StandardBase) -> None:
    class MissingEosTokenizer:
        eos_token_id = None

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del text, add_special_tokens
            return [1]

    invalid = replace(
        tiny_base,
        tokenizer=cast(Any, MissingEosTokenizer()),
    )
    with pytest.raises(ValueError) as collation_error:
        collate_task_instances(invalid, [_task("missing-eos", "prompt", "answer")])
    assert str(collation_error.value) == (
        "standard-base tokenizer must define an integer EOS ID"
    )
    with pytest.raises(ValueError) as generation_error:
        generate_task_answer(invalid, "prompt", max_new_tokens=1)
    assert str(generation_error.value) == (
        "standard-base tokenizer must define an integer EOS ID"
    )


def test_tokenization_and_tensor_construction_arguments_are_explicit(
    tiny_base: StandardBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encode_calls: list[tuple[str, bool]] = []
    tensor_calls: list[dict[str, object]] = []
    real_encode = tiny_base.tokenizer.encode
    real_tensor = torch.tensor

    def encode_spy(text: str, *, add_special_tokens: bool) -> list[int]:
        encode_calls.append((text, add_special_tokens))
        return cast(list[int], real_encode(text, add_special_tokens=add_special_tokens))

    def tensor_spy(*args: object, **kwargs: object) -> torch.Tensor:
        tensor_calls.append(kwargs)
        return real_tensor(*args, **kwargs)

    class EosGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = real_tensor([[0]], dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, suffix), dim=1)

    monkeypatch.setattr(tiny_base.tokenizer, "encode", encode_spy)
    monkeypatch.setattr(standard_base.torch, "tensor", tensor_spy)

    collate_task_instances(
        tiny_base,
        [_task("arguments", "short prompt", "answer")],
    )
    generate_task_answer(
        replace(tiny_base, model=cast(Any, EosGenerator())),
        "short prompt",
        max_new_tokens=1,
    )

    assert encode_calls == [
        ("short prompt", False),
        ("answer\n", False),
        ("short prompt", False),
    ]
    assert tensor_calls == [
        {"dtype": torch.long, "device": tiny_base.device},
        {"dtype": torch.long, "device": tiny_base.device},
        {"dtype": torch.long, "device": tiny_base.device},
        {"dtype": torch.long, "device": tiny_base.device},
    ]


def test_strict_config_validation_rejects_unknown_and_unsafe_values(
    tmp_path: Path,
) -> None:
    config_path, _ = _tiny_snapshot(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases: list[tuple[Callable[[dict[str, object]], None], str]] = [
        (
            lambda document: cast(dict[str, object], document["standard_base"]).__setitem__(
                "unknown", True
            ),
            "Extra inputs are not permitted",
        ),
        (
            lambda document: cast(
                dict[str, object],
                cast(dict[str, object], document["standard_base"])["source"],
            ).__setitem__("revision", "main"),
            "40-character",
        ),
        (
            lambda document: cast(
                dict[str, object],
                cast(dict[str, object], document["standard_base"])["runtime"],
            ).__setitem__("dtype", "float16"),
            "float32",
        ),
    ]
    for index, (mutate, message) in enumerate(cases):
        candidate = copy.deepcopy(raw)
        mutate(candidate)
        path = tmp_path / f"invalid-{index}.yaml"
        path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_standard_base_settings(path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source", "repository"), "invalid", "owner/name"),
        (("source", "config", "path"), "../config.json", "safe filenames"),
        (("source", "config", "sha256"), "A" * 64, "64 lowercase"),
        (("source", "config", "path"), "other.json", "must be config.json"),
        (("source", "weights", "path"), "other.bin", "must be model.safetensors"),
        (
            ("architecture", "hidden_size"),
            15,
            "divisible by num_attention_heads",
        ),
        (
            ("architecture", "num_key_value_heads"),
            3,
            "divisible by num_key_value_heads",
        ),
        (("tokenizer", "required_added_tokens", 0), "", "must be non-empty"),
        (("runtime", "max_prompt_tokens"), 32, "less than max_sequence_tokens"),
        (("runtime", "max_new_tokens"), 9, "plus max_new_tokens"),
        (
            ("architecture", "max_position_embeddings"),
            31,
            "exceeds the architecture context length",
        ),
    ],
)
def test_nested_contract_validation_rejects_invalid_values(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    config_path, _ = _tiny_snapshot(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target: Any = raw["standard_base"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    candidate = tmp_path / "nested-invalid.yaml"
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_standard_base_settings(candidate)


def test_config_root_and_manifest_validation_are_fail_closed(tmp_path: Path) -> None:
    invalid_root = tmp_path / "invalid-root.yaml"
    invalid_root.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError) as root_error:
        load_standard_base_settings(str(invalid_root))
    assert str(root_error.value) == "standard-base config root must be a YAML mapping"

    wrong_key = tmp_path / "wrong-key.yaml"
    wrong_key.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError) as key_error:
        load_standard_base_settings(wrong_key)
    assert str(key_error.value) == (
        "standard-base config root must contain only 'standard_base'"
    )

    config_path, _ = _tiny_snapshot(tmp_path / "fixture")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = raw["standard_base"]["source"]
    source["tokenizer_assets"] = [
        source["tokenizer_assets"][0],
        source["tokenizer_assets"][0],
    ]
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        load_standard_base_settings(duplicate)


def test_preflight_helpers_fail_with_exact_actionable_diagnostics(tmp_path: Path) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path / "fixture")
    settings = load_standard_base_settings(config_path)

    wrong_revision = settings.model_copy(
        update={"source": settings.source.model_copy(update={"revision": "b" * 40})}
    )
    with pytest.raises(ValueError) as revision_error:
        verify_snapshot(wrong_revision, snapshot)
    assert str(revision_error.value) == (
        f"snapshot revision mismatch: expected {'b' * 40}, got {_REVISION}"
    )

    with pytest.raises(
        ValueError,
        match="^snapshot path must be an immutable Hugging Face cache snapshot$",
    ):
        standard_base._repository_from_cache_path(tmp_path / _REVISION)
    with pytest.raises(
        ValueError,
        match="^snapshot path does not identify a Hugging Face repository$",
    ):
        standard_base._repository_from_cache_path(
            tmp_path / "local--tiny-llama" / "snapshots" / _REVISION
        )
    for malformed in ("models--owner", "models----name", "models--owner--name--extra"):
        with pytest.raises(
            ValueError,
            match="^snapshot cache repository identity is malformed$",
        ):
            standard_base._repository_from_cache_path(
                tmp_path / malformed / "snapshots" / _REVISION
            )

    missing = settings.source.config.model_copy(update={"path": "missing.json"})
    with pytest.raises(
        ValueError,
        match="^required snapshot file is missing: missing.json$",
    ):
        standard_base._verify_file(snapshot, missing)
    wrong_size = settings.source.config.model_copy(
        update={"bytes": settings.source.config.bytes + 1}
    )
    with pytest.raises(ValueError) as size_error:
        standard_base._verify_file(snapshot, wrong_size)
    assert str(size_error.value) == (
        "snapshot file size mismatch for config.json: "
        f"expected {wrong_size.bytes}, got {settings.source.config.bytes}"
    )
    wrong_hash = settings.source.config.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError) as hash_error:
        standard_base._verify_file(snapshot, wrong_hash)
    assert str(hash_error.value) == (
        "snapshot SHA-256 mismatch for config.json: "
        f"expected {'0' * 64}, got {_sha256(snapshot / 'config.json')}"
    )

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_bytes(b"\xff")
    with pytest.raises(ValueError) as json_error:
        standard_base._require_json_mapping(invalid_json, "fixture")
    assert str(json_error.value) == "fixture is not valid UTF-8 JSON"
    json_list = tmp_path / "list.json"
    json_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError) as mapping_error:
        standard_base._require_json_mapping(json_list, "fixture")
    assert str(mapping_error.value) == "fixture must contain a JSON object"


def test_architecture_tokenizer_and_tensor_drift_are_all_fail_closed(
    tmp_path: Path,
) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    raw_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )

    remote_config = {**raw_config, "auto_map": {"AutoModel": "remote.Model"}}
    with pytest.raises(ValueError) as remote_error:
        standard_base._verify_architecture(remote_config, settings.architecture)
    assert str(remote_error.value) == (
        "snapshot config must not declare remote-code auto_map entries"
    )
    mismatched_config = {**raw_config, "hidden_size": "16"}
    with pytest.raises(ValueError) as architecture_error:
        standard_base._verify_architecture(mismatched_config, settings.architecture)
    assert str(architecture_error.value) == (
        "architecture fingerprint mismatch for hidden_size: expected 16, got '16'"
    )

    mismatched_pad = {**tokenizer_config, "pad_token": "<eos>"}
    with pytest.raises(ValueError, match="^tokenizer fingerprint mismatch for pad_token"):
        standard_base._verify_tokenizer(
            snapshot,
            mismatched_pad,
            settings.tokenizer,
        )
    missing_decoder = {**tokenizer_config, "added_tokens_decoder": []}
    with pytest.raises(ValueError) as decoder_error:
        standard_base._verify_tokenizer(
            snapshot,
            missing_decoder,
            settings.tokenizer,
        )
    assert str(decoder_error.value) == (
        "tokenizer added_tokens_decoder must be a JSON object"
    )
    malformed_decoder = copy.deepcopy(tokenizer_config)
    malformed_decoder["added_tokens_decoder"]["0"] = "<eos>"
    with pytest.raises(ValueError) as token_error:
        standard_base._verify_tokenizer(
            snapshot,
            malformed_decoder,
            settings.tokenizer,
        )
    assert str(token_error.value) == "tokenizer fingerprint mismatch for added token 0"
    (snapshot / "chat_template.jinja").write_text("unsafe", encoding="utf-8")
    with pytest.raises(ValueError) as template_error:
        standard_base._verify_tokenizer(
            snapshot,
            tokenizer_config,
            settings.tokenizer,
        )
    assert str(template_error.value) == (
        "tokenizer standalone chat-template fingerprint mismatch"
    )
    (snapshot / "chat_template.jinja").unlink()

    wrong_dtype = settings.architecture.model_copy(update={"tensor_dtype": "BF16"})
    with pytest.raises(ValueError, match="^weights dtype mismatch"):
        standard_base._verify_tensor_inventory(
            snapshot / "model.safetensors",
            wrong_dtype,
        )
    wrong_head = settings.architecture.model_copy(
        update={"separate_lm_head_weight": True}
    )
    with pytest.raises(ValueError) as head_error:
        standard_base._verify_tensor_inventory(
            snapshot / "model.safetensors",
            wrong_head,
        )
    assert str(head_error.value) == "weights tied-head fingerprint mismatch"
    wrong_parameters = settings.architecture.model_copy(
        update={
            "unique_parameter_count": settings.architecture.unique_parameter_count + 1
        }
    )
    with pytest.raises(ValueError) as parameter_error:
        standard_base._verify_tensor_inventory(
            snapshot / "model.safetensors",
            wrong_parameters,
        )
    assert str(parameter_error.value).startswith(
        "weights parameter-count fingerprint mismatch: expected "
    )
    wrong_tensors = settings.architecture.model_copy(
        update={"tensor_count": settings.architecture.tensor_count + 1}
    )
    with pytest.raises(ValueError) as tensor_error:
        standard_base._verify_tensor_inventory(
            snapshot / "model.safetensors",
            wrong_tensors,
        )
    assert str(tensor_error.value) == (
        "weights tensor-count fingerprint mismatch: "
        f"expected {wrong_tensors.tensor_count}, got {settings.architecture.tensor_count}"
    )


def test_default_snapshot_resolver_remains_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    calls: list[dict[str, object]] = []

    def local_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(standard_base, "snapshot_download", local_snapshot_download)
    verified = verify_snapshot(settings)

    assert verified.path == snapshot.absolute()
    assert calls == [
        {
            "repo_id": "local/tiny-llama",
            "revision": _REVISION,
            "local_files_only": True,
        }
    ]


def test_snapshot_rejects_repository_and_tokenizer_drift(tmp_path: Path) -> None:
    config_path, snapshot = _tiny_snapshot(tmp_path)
    settings = load_standard_base_settings(config_path)
    wrong_repository = settings.model_copy(
        update={
            "source": settings.source.model_copy(update={"repository": "other/model"})
        }
    )
    with pytest.raises(ValueError, match="repository mismatch"):
        verify_snapshot(wrong_repository, snapshot)

    tokenizer_config = snapshot / "tokenizer_config.json"
    raw = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    raw["chat_template"] = "unsafe"
    tokenizer_config.write_text(json.dumps(raw), encoding="utf-8")
    drifted_asset = next(
        asset for asset in settings.source.tokenizer_assets if asset.path == tokenizer_config.name
    ).model_copy(
        update={
            "bytes": tokenizer_config.stat().st_size,
            "sha256": _sha256(tokenizer_config),
        }
    )
    drifted_settings = settings.model_copy(
        update={
            "source": settings.source.model_copy(
                update={
                    "tokenizer_assets": [
                        drifted_asset if asset.path == tokenizer_config.name else asset
                        for asset in settings.source.tokenizer_assets
                    ]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="chat_template"):
        verify_snapshot(drifted_settings, snapshot)


def test_device_resolution_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_standard_base_device("auto").type == "cpu"
    assert resolve_standard_base_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError) as mps_error:
        resolve_standard_base_device("mps")
    assert str(mps_error.value) == "MPS was requested but is unavailable"
    with pytest.raises(RuntimeError) as cuda_error:
        resolve_standard_base_device("cuda")
    assert str(cuda_error.value) == "CUDA was requested but is unavailable"
    with pytest.raises(ValueError) as device_error:
        resolve_standard_base_device("metal")
    assert str(device_error.value) == "device must be auto, cpu, mps, or cuda"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_standard_base_device("auto").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_standard_base_device("auto").type == "cuda"


def _run_standard_base_probe(
    tmp_path: Path,
    *,
    kind: str,
    device: str,
    runs: int,
) -> dict[str, object]:
    output_path = tmp_path / f"{kind}-{device}.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("standard_base_mps_probe.py")),
            kind,
            str(output_path),
            "--device",
            device,
            "--runs",
            str(runs),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        cwd=_ROOT,
        env=_subprocess_environment(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, object], json.loads(output_path.read_text(encoding="utf-8")))


def _assert_probe_identity(
    result: dict[str, object],
    *,
    kind: str,
    device: str,
    environment: dict[str, object],
) -> None:
    assert result["kind"] == kind
    assert result["cached"] is True
    assert result["device"] == device
    assert result["requested_device"] == device
    assert result["torch_version"] == environment["torch_version"]
    assert result["transformers_version"] == environment["transformers_version"]
    assert result["platform"] == environment["platform"]
    assert result["stock_llama"] is True
    assert result["tied"] is True
    assert result["dtypes"] == ["torch.float32"]


def _prompts_by_case(run: dict[str, object]) -> dict[str, dict[str, object]]:
    prompts = cast(list[dict[str, object]], run["prompts"])
    return {cast(str, prompt["case_id"]): prompt for prompt in prompts}


def _assert_frozen_cpu_reference(
    cpu: dict[str, object],
    fixture: dict[str, object],
) -> None:
    tolerance = cast(dict[str, float], fixture["chosen_tolerance"])
    runs = cast(list[dict[str, object]], cpu["records"])
    assert len(runs) == 1
    prompts = _prompts_by_case(runs[0])
    references = cast(list[dict[str, object]], fixture["cpu_reference"])
    for reference in references:
        current = prompts[cast(str, reference["case_id"])]
        assert current["generated_token_ids"] == reference["generated_token_ids"]
        assert current["top1_token_id"] == reference["top1_token_id"]
        assert cast(float, current["top1_top2_margin"]) == pytest.approx(
            reference["top1_top2_margin"],
            abs=tolerance["cpu_mps_margin_absolute"],
        )
        assert cast(float, current["loss"]) == pytest.approx(
            reference["loss"],
            abs=tolerance["cpu_mps_loss_absolute"],
        )
        decisions = cast(list[dict[str, object]], current["generation_decisions"])
        assert [decision["top1_top2_margin"] for decision in decisions] == pytest.approx(
            reference["generation_top1_top2_margins"],
            abs=tolerance["cpu_mps_margin_absolute"],
        )


def _assert_mps_repeatability(
    mps: dict[str, object],
    fixture: dict[str, object],
) -> None:
    tolerance = cast(dict[str, float], fixture["chosen_tolerance"])
    runs = cast(list[dict[str, object]], mps["records"])
    registered_runs = cast(list[dict[str, object]], fixture["mps_observations"])
    assert len(runs) == 5
    assert len(registered_runs) == len(runs)
    for run, registered_run in zip(runs, registered_runs, strict=True):
        prompts = _prompts_by_case(run)
        registered_prompts = _prompts_by_case(registered_run)
        assert prompts.keys() == registered_prompts.keys()
        for case_id, registered in registered_prompts.items():
            current = prompts[case_id]
            assert current["generated_token_ids"] == registered["generated_token_ids"]
            assert current["top1_token_id"] == registered["top1_token_id"]
            assert cast(float, current["top1_top2_margin"]) == pytest.approx(
                registered["top1_top2_margin"],
                abs=tolerance["mps_repeat_margin_absolute"],
            )
            assert cast(float, current["loss"]) == pytest.approx(
                registered["loss"],
                abs=tolerance["mps_repeat_loss_absolute"],
            )
    reference = _prompts_by_case(runs[0])
    for run in runs[1:]:
        prompts = _prompts_by_case(run)
        assert prompts.keys() == reference.keys()
        for case_id, expected in reference.items():
            current = prompts[case_id]
            assert current["generated_token_ids"] == expected["generated_token_ids"]
            assert current["top1_token_id"] == expected["top1_token_id"]
            assert current["stop_reason"] == expected["stop_reason"]
            assert cast(float, current["top1_top2_margin"]) == pytest.approx(
                expected["top1_top2_margin"],
                abs=tolerance["mps_repeat_margin_absolute"],
            )
            assert cast(float, current["loss"]) == pytest.approx(
                expected["loss"],
                abs=tolerance["mps_repeat_loss_absolute"],
            )
            current_decisions = cast(
                list[dict[str, object]], current["generation_decisions"]
            )
            expected_decisions = cast(
                list[dict[str, object]], expected["generation_decisions"]
            )
            assert [item["top1_token_id"] for item in current_decisions] == [
                item["top1_token_id"] for item in expected_decisions
            ]
            assert [item["top1_top2_margin"] for item in current_decisions] == (
                pytest.approx(
                    [item["top1_top2_margin"] for item in expected_decisions],
                    abs=tolerance["mps_repeat_margin_absolute"],
                )
            )

    floor = cast(dict[str, float], fixture["measured_floor"])
    assert tolerance["mps_repeat_margin_absolute"] == max(
        1e-6,
        2 * floor["mps_margin_max_absolute_delta"],
    )
    assert tolerance["mps_repeat_loss_absolute"] == max(
        1e-6,
        2 * floor["mps_loss_max_absolute_delta"],
    )
    assert tolerance["cpu_mps_margin_absolute"] == max(
        1e-6,
        2 * floor["cpu_mps_margin_max_absolute_delta"],
    )
    assert tolerance["cpu_mps_loss_absolute"] == max(
        1e-6,
        2 * floor["cpu_mps_loss_max_absolute_delta"],
    )
    assert tolerance["cpu_decision_margin_epsilon"] == tolerance[
        "cpu_mps_margin_absolute"
    ]


def _first_difference(left: list[object], right: list[object]) -> int | None:
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
        if left_item != right_item:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _assert_cpu_mps_agreement(
    cpu: dict[str, object],
    mps: dict[str, object],
    fixture: dict[str, object],
) -> None:
    tolerance = cast(dict[str, float], fixture["chosen_tolerance"])
    epsilon = tolerance["cpu_decision_margin_epsilon"]
    cpu_run = cast(list[dict[str, object]], cpu["records"])[0]
    cpu_prompts = _prompts_by_case(cpu_run)
    for run in cast(list[dict[str, object]], mps["records"]):
        mps_prompts = _prompts_by_case(run)
        assert mps_prompts.keys() == cpu_prompts.keys()
        for case_id, cpu_prompt in cpu_prompts.items():
            mps_prompt = mps_prompts[case_id]
            if mps_prompt["top1_token_id"] != cpu_prompt["top1_token_id"]:
                assert cast(float, cpu_prompt["top1_top2_margin"]) <= epsilon
            assert cast(float, mps_prompt["top1_top2_margin"]) == pytest.approx(
                cpu_prompt["top1_top2_margin"],
                abs=tolerance["cpu_mps_margin_absolute"],
            )
            assert cast(float, mps_prompt["loss"]) == pytest.approx(
                cpu_prompt["loss"],
                abs=tolerance["cpu_mps_loss_absolute"],
            )

            cpu_ids = cast(list[object], cpu_prompt["generated_token_ids"])
            mps_ids = cast(list[object], mps_prompt["generated_token_ids"])
            difference = _first_difference(cpu_ids, mps_ids)
            cpu_decisions = cast(
                list[dict[str, object]], cpu_prompt["generation_decisions"]
            )
            if difference is not None:
                assert difference < len(cpu_decisions)
                assert cast(float, cpu_decisions[difference]["top1_top2_margin"]) <= epsilon
            mps_decisions = cast(
                list[dict[str, object]], mps_prompt["generation_decisions"]
            )
            for cpu_decision, mps_decision in zip(
                cpu_decisions,
                mps_decisions,
                strict=False,
            ):
                if mps_decision["top1_token_id"] != cpu_decision["top1_token_id"]:
                    assert cast(float, cpu_decision["top1_top2_margin"]) <= epsilon
                assert cast(float, mps_decision["top1_top2_margin"]) == pytest.approx(
                    cpu_decision["top1_top2_margin"],
                    abs=tolerance["cpu_mps_margin_absolute"],
                )


@pytest.mark.mps
def test_tiny_cpu_and_mps_decisions_meet_measured_tolerances(tmp_path: Path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    registered = cast(
        dict[str, object],
        json.loads(_MPS_TOLERANCE_PATH.read_text(encoding="utf-8")),
    )
    environment = cast(dict[str, object], registered["environment"])
    fixture = cast(
        dict[str, object],
        cast(dict[str, object], registered["fixtures"])["tiny"],
    )
    cpu = _run_standard_base_probe(tmp_path, kind="tiny", device="cpu", runs=1)
    mps = _run_standard_base_probe(tmp_path, kind="tiny", device="mps", runs=5)

    _assert_probe_identity(cpu, kind="tiny", device="cpu", environment=environment)
    _assert_probe_identity(mps, kind="tiny", device="mps", environment=environment)
    assert cast(dict[str, object], mps["fixture"])["weight_seed"] == _TINY_MODEL_SEED
    _assert_frozen_cpu_reference(cpu, fixture)
    _assert_mps_repeatability(mps, fixture)
    _assert_cpu_mps_agreement(cpu, mps, fixture)

    cpu_prompts = _prompts_by_case(cast(list[dict[str, object]], cpu["records"])[0])
    assert len({prompt["top1_token_id"] for prompt in cpu_prompts.values()}) == len(
        cpu_prompts
    )
    assert all(
        prompt["top1_token_id"] != prompt["top2_token_id"]
        for prompt in cpu_prompts.values()
    )
    assert all(
        len(cast(list[object], prompt["generated_token_ids"])) >= 2
        for prompt in cpu_prompts.values()
    )
    assert all(prompt["stop_reason"] == "length" for prompt in cpu_prompts.values())
    narrow = cpu_prompts["narrow-margin"]
    tolerance = cast(dict[str, float], fixture["chosen_tolerance"])
    assert 0.0 < cast(float, narrow["top1_top2_margin"]) <= tolerance[
        "narrow_prompt_margin_maximum"
    ]


@pytest.mark.integration
@pytest.mark.mps
def test_cached_real_smollm2_cpu_and_mps_decisions_agree(tmp_path: Path) -> None:
    if os.environ.get("MUTANT_UNDER_TEST"):
        pytest.skip("real cached model is outside mutation-test scope")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    settings = load_standard_base_settings()
    cached_config = try_to_load_from_cache(
        settings.source.repository,
        "config.json",
        revision=settings.source.revision,
    )
    if not isinstance(cached_config, str):
        pytest.skip("pinned SmolLM2 snapshot is not cached")
    registered = cast(
        dict[str, object],
        json.loads(_MPS_TOLERANCE_PATH.read_text(encoding="utf-8")),
    )
    environment = cast(dict[str, object], registered["environment"])
    fixture = cast(
        dict[str, object],
        cast(dict[str, object], registered["fixtures"])["real_smollm2_360m"],
    )
    cpu = _run_standard_base_probe(tmp_path, kind="real", device="cpu", runs=1)
    mps = _run_standard_base_probe(tmp_path, kind="real", device="mps", runs=5)

    _assert_probe_identity(cpu, kind="real", device="cpu", environment=environment)
    _assert_probe_identity(mps, kind="real", device="mps", environment=environment)
    _assert_frozen_cpu_reference(cpu, fixture)
    _assert_mps_repeatability(mps, fixture)
    _assert_cpu_mps_agreement(cpu, mps, fixture)


def _subprocess_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(_ROOT)
    return environment
