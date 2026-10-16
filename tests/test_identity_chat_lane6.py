"""Tiny-fixture acceptance tests for ADR-0011 lane 6 identity/chat."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import aptivra.inference.causal_lm as standard_causal_lm
import aptivra.inference.chat_canonical as identity_contract
import aptivra.inference.chat_finetune as chat_finetune
from aptivra.experts.standard import (
    load_promoted_expert,
    load_standard_expert_settings,
)
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.inference.chat_canonical import (
    APTIVRA_INTRO,
    CAPABILITIES_SUMMARY,
    CREATOR_BIO_REPLY,
    CREATOR_REPLY,
    IDENTITY_CHAT_TEMPLATES,
    NAME_REPLY,
    IdentityManifest,
    PromptIntent,
    allowed_release_metric,
    canonical_identity_records,
    default_system_message,
    identity_chat_templates,
    identity_manifest_sha256,
    load_configured_identity_contract,
    load_identity_manifest,
    release_record_sha256,
)
from aptivra.inference.chat_finetune import (
    IdentityTrainingResult,
    identity_training_instances,
    materialize_identity_batch,
    train_identity_adapter,
)
from aptivra.inference.chat_prompt import (
    END_OF_TEXT,
    IM_END,
    IM_START,
    format_chat_prompt,
    load_identity_settings,
    validate_chat_tokenizer,
)
from aptivra.inference.chat_quality import default_identity_cases, identity_bundle_ok
from aptivra.inference.standard_base import (
    DecodedTextStoppingCriteria,
    GenerationResult,
    StandardBase,
    VerifiedSnapshot,
    generate_task_answer,
    load_standard_base_settings,
)
from aptivra.resources import aptivra_resource
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.conversational import ConversationalTaskFamily
from aptivra.tasks.identity import ModelIdentityTaskFamily


@pytest.fixture(autouse=True)
def _preserve_torch_rng_state() -> Any:
    state = torch.random.get_rng_state()
    yield
    torch.random.set_rng_state(state)


def _tiny_chat_base(tmp_path: Path) -> StandardBase:
    vocab = {
        END_OF_TEXT: 0,
        IM_START: 1,
        IM_END: 2,
        "<unk>": 3,
        "Aptivra": 4,
        "answer": 5,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        unk_token="<unk>",
        additional_special_tokens=[IM_START, IM_END],
    )
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
    )
    torch.manual_seed(20_261_017)
    model = LlamaForCausalLM(config)
    model.tie_weights()
    settings = load_standard_base_settings()
    architecture = settings.architecture.model_copy(
        update={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": len(tokenizer),
            "max_position_embeddings": 256,
            "unique_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "tensor_count": len(model.state_dict()),
            "tensor_dtype": "F32",
        }
    )
    tokenizer_settings = settings.tokenizer.model_copy(
        update={
            "bos_token": END_OF_TEXT,
            "eos_token": END_OF_TEXT,
            "unk_token": "<unk>",
            "required_added_tokens": {0: END_OF_TEXT, 1: IM_START, 2: IM_END},
        }
    )
    runtime = settings.runtime.model_copy(
        update={
            "max_prompt_tokens": 240,
            "max_sequence_tokens": 256,
            "max_new_tokens": 16,
        }
    )
    tiny_settings = settings.model_copy(
        update={
            "feature": settings.feature.model_copy(update={"enabled": True}),
            "architecture": architecture,
            "tokenizer": tokenizer_settings,
            "runtime": runtime,
        }
    )
    snapshot = VerifiedSnapshot(
        path=tmp_path,
        repository=tiny_settings.source.repository,
        revision=tiny_settings.source.revision,
        unique_parameter_count=architecture.unique_parameter_count,
        tensor_count=architecture.tensor_count,
    )
    return StandardBase(model, tokenizer, tiny_settings, snapshot, torch.device("cpu"))


def _enabled_identity_settings(max_steps: int = 2) -> Any:
    settings = load_identity_settings()
    training = settings.training.model_copy(
        update={"max_optimizer_steps": max_steps, "max_sequence_tokens": 256}
    )
    return settings.model_copy(
        update={
            "feature": settings.feature.model_copy(update={"enabled": True}),
            "training": training,
        }
    )


def _enabled_expert_settings() -> Any:
    settings = load_standard_expert_settings()
    return settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": True})}
    )


def _identity_manifest_body() -> dict[str, Any]:
    raw = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/identity_manifest.yaml").read_text()),
    )
    return cast(dict[str, Any], copy.deepcopy(raw["identity_manifest"]))


def _valid_release_record(body: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "metric": {"name": "exact_match_accuracy", "value": 0.75, "unit": "ratio"},
        "stream_version": body["release_evaluation"]["expected_stream_version"],
        "seed_set": [1, 2, 3, 4, 5],
        "denominator": 100,
        "date": date(2026, 10, 17),
        "base_fingerprint": body["base"]["fingerprint"],
    }
    record["content_sha256"] = release_record_sha256(record)
    return record


def _write_manifest(
    tmp_path: Path,
    body: dict[str, Any],
    *,
    name: str,
) -> Path:
    body["content_sha256"] = identity_manifest_sha256(body)
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump({"identity_manifest": body}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _set_nested(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: object,
) -> None:
    target = payload
    for part in path[:-1]:
        target = cast(dict[str, Any], target[part])
    target[path[-1]] = value


def test_identity_configs_are_disabled_strict_and_byte_identical() -> None:
    settings = load_identity_settings()
    assert settings.feature.enabled is False
    assert settings.template.version == "aptivra-chat-v1"
    assert settings.quality.minimum_canonical_prefix_ratio == 0.65
    assert Path("configs/smollm2_identity.yaml").read_bytes() == aptivra_resource(
        "configs", "smollm2_identity.yaml"
    ).read_bytes()
    assert Path("configs/identity_manifest.yaml").read_bytes() == aptivra_resource(
        "configs", "identity_manifest.yaml"
    ).read_bytes()


def test_template_tokens_are_single_distinct_ids_in_tiny_fixture(tmp_path: Path) -> None:
    base = _tiny_chat_base(tmp_path)
    token_ids = validate_chat_tokenizer(
        base.tokenizer,  # type: ignore[arg-type]
        load_identity_settings().template,
    )
    assert token_ids == {IM_START: 1, IM_END: 2, END_OF_TEXT: 0}


class _BrokenTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [1, 2] if text == IM_START else [0]

    def convert_tokens_to_ids(self, token: str) -> int:
        del token
        return 0

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "broken"


class _ScriptedBatchTokenizer:
    eos_token_id = 0

    def __init__(
        self,
        *,
        prompt_ids: list[int] | None = None,
        target_ids: list[int] | None = None,
    ) -> None:
        self.prompt_ids = [10, 11] if prompt_ids is None else prompt_ids
        self.target_ids = [12, 2, 0] if target_ids is None else target_ids
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        special = {IM_START: [1], IM_END: [2], END_OF_TEXT: [0]}
        if text in special:
            return special[text]
        if text == "prompt":
            return list(self.prompt_ids)
        if text == f"answer{IM_END}{END_OF_TEXT}":
            return list(self.target_ids)
        raise AssertionError(f"unexpected text passed to tokenizer: {text!r}")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {IM_START: 1, IM_END: 2, END_OF_TEXT: 0}[token]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return {1: IM_START, 2: IM_END, 0: END_OF_TEXT}[token_ids[0]]


def _scripted_batch_base(
    tokenizer: _ScriptedBatchTokenizer,
    *,
    runtime_limit: int,
) -> StandardBase:
    return cast(
        StandardBase,
        SimpleNamespace(
            tokenizer=tokenizer,
            settings=SimpleNamespace(
                runtime=SimpleNamespace(max_sequence_tokens=runtime_limit)
            ),
            device=torch.device("cpu"),
        ),
    )


def _batch_instance() -> TaskInstance:
    return TaskInstance("identity-test", "identity", "prompt", "answer")


def test_template_token_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="must map to one registered ID"):
        validate_chat_tokenizer(_BrokenTokenizer(), load_identity_settings().template)


def test_generate_chat_history_and_limit_are_keyword_only() -> None:
    parameters = inspect.signature(AptivraCausalLM.generate_chat).parameters
    assert parameters["history"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["max_new_tokens"].kind is inspect.Parameter.KEYWORD_ONLY


def test_missing_tampered_and_stale_identity_evidence_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_identity_manifest(tmp_path / "missing.yaml")

    raw = yaml.safe_load(Path("configs/identity_manifest.yaml").read_text())
    tampered = copy.deepcopy(raw)
    tampered["identity_manifest"]["research_status"] = "tampered"
    tampered_path = tmp_path / "tampered.yaml"
    tampered_path.write_text(yaml.safe_dump(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content SHA-256 is stale"):
        load_identity_manifest(tampered_path)

    stale_base = copy.deepcopy(raw)
    body = stale_base["identity_manifest"]
    body["base"]["revision"] = "a" * 40
    body["content_sha256"] = identity_manifest_sha256(body)
    stale_base_path = tmp_path / "stale-base.yaml"
    stale_base_path.write_text(yaml.safe_dump(stale_base), encoding="utf-8")
    with pytest.raises(ValueError, match="base fingerprint is stale"):
        load_identity_manifest(stale_base_path)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"unexpected": {}},
        {"identity_manifest": {}, "unexpected": {}},
    ],
)
def test_manifest_root_is_fail_closed(tmp_path: Path, document: object) -> None:
    path = tmp_path / "invalid-root.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="root must contain only 'identity_manifest'",
    ):
        load_identity_manifest(path)


@pytest.mark.parametrize(
    ("field_path", "invalid"),
    [
        (("base", "repository"), "missing-slash"),
        (("base", "repository"), "a/b/c"),
        (("base", "repository"), "prefix a/b suffix"),
        (("base", "revision"), "A" * 40),
        (("base", "config_sha256"), "g" * 64),
        (("base", "weights_sha256"), "g" * 64),
        (("base", "fingerprint"), "g" * 64),
        (("base", "unique_parameter_count"), 0),
        (("base", "unique_parameter_count"), True),
        (("components", "routed_lora", "required_rank"), 15),
        (("research_status",), ""),
        (("research_status",), "r" * 241),
        (("required_limitations",), ["too", "short", "for validation"]),
        (("release_evaluation", "expected_stream_version"), ""),
        (("release_evaluation", "expected_stream_version"), "v" * 41),
    ],
)
def test_manifest_schema_rejects_each_invalid_boundary(
    field_path: tuple[str, ...],
    invalid: object,
) -> None:
    body = _identity_manifest_body()
    _set_nested(body, field_path, invalid)
    body["content_sha256"] = identity_manifest_sha256(body)
    with pytest.raises(ValueError):
        IdentityManifest.model_validate(body)


def test_manifest_schema_accepts_each_exact_boundary() -> None:
    body = _identity_manifest_body()
    body["base"]["repository"] = "a/b"
    body["base"]["unique_parameter_count"] = 1
    body["research_status"] = "r" * 240
    body["release_evaluation"]["expected_stream_version"] = "v" * 40
    body["content_sha256"] = identity_manifest_sha256(body)

    manifest = IdentityManifest.model_validate(body)

    assert manifest.base.repository == "a/b"
    assert manifest.base.unique_parameter_count == 1
    assert len(manifest.research_status) == 240
    assert len(manifest.release_evaluation.expected_stream_version) == 40


@pytest.mark.parametrize(
    ("field_path", "invalid"),
    [
        (("metric", "name"), ""),
        (("metric", "name"), "A_metric"),
        (("metric", "name"), "a" * 81),
        (("metric", "value"), -1.0e12),
        (("metric", "value"), 1.0e12),
        (("metric", "value"), -2.0e12),
        (("metric", "value"), 2.0e12),
        (("metric", "unit"), ""),
        (("metric", "unit"), "u" * 41),
        (("stream_version",), ""),
        (("stream_version",), "v" * 41),
        (("seed_set",), []),
        (("seed_set",), [-1]),
        (("seed_set",), [True]),
        (("seed_set",), [1, 1]),
        (("denominator",), 0),
        (("denominator",), True),
        (("date",), "not-a-date"),
        (("base_fingerprint",), "g" * 64),
        (("content_sha256",), "g" * 64),
    ],
)
def test_release_record_schema_rejects_each_invalid_boundary(
    field_path: tuple[str, ...],
    invalid: object,
) -> None:
    body = _identity_manifest_body()
    record = _valid_release_record(body)
    _set_nested(record, field_path, invalid)
    if field_path != ("content_sha256",):
        record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)

    with pytest.raises(ValueError):
        IdentityManifest.model_validate(body)


def test_release_record_schema_accepts_each_exact_boundary() -> None:
    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["metric"] = {
        "name": "a" * 80,
        "value": -999_999_999_999.5,
        "unit": "u",
    }
    record["stream_version"] = "v"
    record["seed_set"] = [0]
    record["denominator"] = 1
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["expected_stream_version"] = "v"
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)

    manifest = IdentityManifest.model_validate(body)
    bound = manifest.release_evaluation.record

    assert bound is not None
    assert len(bound.metric.name) == 80
    assert len(bound.metric.unit) == 1
    assert bound.metric.value == -999_999_999_999.5
    assert bound.stream_version == "v"
    assert bound.seed_set == (0,)
    assert bound.denominator == 1


def test_release_record_schema_accepts_maximum_text_and_metric_boundaries() -> None:
    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["metric"] = {
        "name": "a" * 80,
        "value": 999_999_999_999.5,
        "unit": "u" * 40,
    }
    record["stream_version"] = "v" * 40
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["expected_stream_version"] = "v" * 40
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)

    manifest = IdentityManifest.model_validate(body)
    bound = manifest.release_evaluation.record

    assert bound is not None
    assert bound.metric.value == 999_999_999_999.5
    assert len(bound.metric.unit) == 40
    assert len(bound.stream_version) == 40


def test_stale_release_record_content_hash_fails_before_manifest_hash() -> None:
    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["metric"]["value"] = 0.5
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)

    with pytest.raises(ValueError, match="release evaluation content SHA-256 is stale"):
        IdentityManifest.model_validate(body)


def test_contract_models_are_strict_and_immutable() -> None:
    manifest = IdentityManifest.model_validate(_identity_manifest_body())
    with pytest.raises(ValidationError):
        manifest.research_status = "changed"

    records = canonical_identity_records(manifest)
    with pytest.raises(FrozenInstanceError):
        records[0].answer = "changed"

    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["date"] = "2026-10-17"
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)
    with pytest.raises(ValidationError):
        IdentityManifest.model_validate(body)


@pytest.mark.parametrize(
    ("field_path", "stale"),
    [
        (("base", "repository"), "other/repository"),
        (("base", "revision"), "0" * 40),
        (("base", "config_sha256"), "0" * 64),
        (("base", "weights_sha256"), "0" * 64),
        (("base", "fingerprint"), "0" * 64),
        (("base", "unique_parameter_count"), 1),
    ],
)
def test_each_stale_base_binding_fails_generation(
    tmp_path: Path,
    field_path: tuple[str, ...],
    stale: object,
) -> None:
    body = _identity_manifest_body()
    _set_nested(body, field_path, stale)
    path = _write_manifest(tmp_path, body, name="stale-base-binding.yaml")

    with pytest.raises(ValueError, match="base fingerprint is stale"):
        load_identity_manifest(path)


def test_stale_release_base_fingerprint_fails_generation(tmp_path: Path) -> None:
    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["base_fingerprint"] = "0" * 64
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["record"] = record
    path = _write_manifest(tmp_path, body, name="stale-release-base.yaml")

    with pytest.raises(ValueError, match="release evaluation base fingerprint is stale"):
        load_identity_manifest(path)


def test_identity_manifest_accepts_a_string_path() -> None:
    manifest = load_identity_manifest("configs/identity_manifest.yaml")
    assert manifest.model.name == "Aptivra"


def test_creator_attribution_records_owner_approval() -> None:
    manifest = load_identity_manifest()
    creator = manifest.model.creator
    assert creator.name == "Raghunath Chava"
    assert creator.attribution_kind == "owner_attribution"
    assert creator.approval == "approved by the owner on 2026-10-17"
    creator_record = next(
        record
        for record in canonical_identity_records(manifest)
        if record.prompt_intent is PromptIntent.CREATOR
    )
    assert creator_record.answer == (
        "The owner-approved creator attribution names Raghunath Chava as "
        "Aptivra's creator."
    )
    assert "must confirm" not in creator_record.answer


def test_bound_release_record_is_the_only_capability_source(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/identity_manifest.yaml").read_text())
    body = raw["identity_manifest"]
    record: dict[str, object] = {
        "metric": {"name": "exact_match_accuracy", "value": 0.75, "unit": "ratio"},
        "stream_version": "v2",
        "seed_set": [1, 2, 3, 4, 5],
        "denominator": 100,
        "date": date(2026, 10, 17),
        "base_fingerprint": body["base"]["fingerprint"],
    }
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)
    path = tmp_path / "bound.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    manifest = load_identity_manifest(path)
    records = canonical_identity_records(manifest)
    capability = next(
        item for item in records if item.prompt_intent is PromptIntent.CAPABILITIES
    )
    assert capability.answer == (
        "Bound release evidence reports exact_match_accuracy=0.75 ratio on stream v2, "
        "across seeds 1, 2, 3, 4, 5, with denominator 100, dated 2026-10-17. "
        "Capability is limited to the named evaluations in bound release evidence."
    )

    bound_record = manifest.release_evaluation.record
    assert bound_record is not None
    assert {item.prompt_intent for item in records} == set(PromptIntent)
    for item in records:
        allowed_metric = allowed_release_metric(item.prompt_intent, bound_record)
        if item.prompt_intent is PromptIntent.CAPABILITIES:
            assert allowed_metric is bound_record.metric
        else:
            assert allowed_metric is None

    quality_cases = default_identity_cases(manifest)
    assert {
        case.user_message for case in quality_cases if case.allowed_metric is not None
    } == {capability.question}
    raw_answers = {item.question: item.answer for item in records}
    assert identity_bundle_ok(raw_answers, manifest)

    stale_stream = copy.deepcopy(raw)
    stale_body = stale_stream["identity_manifest"]
    stale_record = stale_body["release_evaluation"]["record"]
    stale_record["stream_version"] = "v1"
    stale_record["content_sha256"] = release_record_sha256(stale_record)
    stale_body["content_sha256"] = identity_manifest_sha256(stale_body)
    stale_path = tmp_path / "stale-stream.yaml"
    stale_path.write_text(yaml.safe_dump(stale_stream), encoding="utf-8")
    with pytest.raises(ValueError, match="stream version is stale"):
        load_identity_manifest(stale_path)


def test_release_metric_binding_is_exhaustive_and_fail_closed() -> None:
    body = _identity_manifest_body()
    record_payload = _valid_release_record(body)
    body["release_evaluation"]["record"] = record_payload
    body["content_sha256"] = identity_manifest_sha256(body)
    record = IdentityManifest.model_validate(body).release_evaluation.record
    assert record is not None

    assert allowed_release_metric(PromptIntent.CAPABILITIES, record) is record.metric
    for intent in (
        PromptIntent.WHAT_ARE_YOU,
        PromptIntent.ARCHITECTURE,
        PromptIntent.CHAT_LEARNING,
        PromptIntent.CREATOR,
    ):
        assert allowed_release_metric(intent, record) is None
    assert allowed_release_metric(PromptIntent.CAPABILITIES, None) is None
    with pytest.raises(AssertionError):
        allowed_release_metric(cast(PromptIntent, "unknown"), record)


def test_release_record_hash_uses_canonical_iso_date() -> None:
    payload: dict[str, object] = {
        "metric": {"name": "exact_match_accuracy", "value": 0.75, "unit": "ratio"},
        "stream_version": "v2",
        "seed_set": [1, 2, 3, 4, 5],
        "denominator": 100,
        "date": date(2026, 10, 17),
        "base_fingerprint": "7" * 64,
    }
    assert release_record_sha256(payload) == (
        "a9664a0600b4d9f20e8dcdce036a3af0db3e102a1212f1ae76fdcf5f0eac9b82"
    )


def test_manifest_accepts_minimum_research_status_length(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/identity_manifest.yaml").read_text())
    body = raw["identity_manifest"]
    body["research_status"] = "R"
    body["content_sha256"] = identity_manifest_sha256(body)
    path = tmp_path / "one-character-research-status.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert load_identity_manifest(path).research_status == "R"


def test_routed_lora_claim_requires_enabled_release_components() -> None:
    disabled = canonical_identity_records()
    assert all("adapted with separately routed LoRA" not in item.answer for item in disabled)
    enabled_settings = _enabled_expert_settings()
    wrong_rank = enabled_settings.model_copy(
        update={
            "lora": enabled_settings.lora.model_copy(
                update={"rank": enabled_settings.lora.rank + 1}
            )
        }
    )
    missing_router = enabled_settings.model_copy(update={"router": None})
    feature_disabled = enabled_settings.model_copy(
        update={
            "feature": enabled_settings.feature.model_copy(update={"enabled": False})
        }
    )
    for unavailable in (wrong_rank, missing_router, feature_disabled):
        records = canonical_identity_records(expert_settings=unavailable)
        assert "No routed LoRA component is enabled" in records[1].answer
        assert "separately routed LoRA modules" not in records[0].answer

    enabled = canonical_identity_records(expert_settings=enabled_settings)
    assert "adapted with separately routed LoRA modules" in enabled[0].answer
    assert "sequence-level top-1 router" in enabled[1].answer


def test_routed_lora_none_is_fail_closed() -> None:
    assert identity_contract._routed_lora_enabled(load_identity_manifest(), None) is False


def test_canonical_optional_parameters_remain_keyword_only() -> None:
    keyword_only_parameters = (
        (identity_contract.load_identity_manifest, "base_settings"),
        (identity_contract.load_configured_identity_contract, "base_settings"),
        (identity_contract._forbidden_target_reason, "allowed_metric"),
        (identity_contract.assert_permitted_identity_target, "allowed_metric"),
        (identity_contract.canonical_identity_records, "expert_settings"),
        (identity_contract.identity_chat_templates, "expert_settings"),
    )
    for function, parameter_name in keyword_only_parameters:
        parameter = inspect.signature(function).parameters[parameter_name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_canonical_target_runs_through_forbidden_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[tuple[str, object]] = []

    def record_check(
        target: str,
        *,
        allowed_metric: object = None,
    ) -> None:
        checked.append((target, allowed_metric))

    monkeypatch.setattr(
        identity_contract,
        "assert_permitted_identity_target",
        record_check,
    )
    records = canonical_identity_records()

    assert [target for target, _ in checked] == [record.answer for record in records]
    assert [metric for _, metric in checked] == [None] * len(records)


def test_configured_contract_and_default_system_message_are_exact() -> None:
    default_settings, default_manifest, default_experts = (
        load_configured_identity_contract()
    )
    explicit_settings, explicit_manifest, explicit_experts = (
        load_configured_identity_contract("configs/smollm2_identity.yaml")
    )

    assert explicit_settings == default_settings
    assert explicit_manifest == default_manifest
    assert explicit_experts == default_experts
    expected = (
        "You are Aptivra, a research language model. "
        "Answer identity questions only from the bound identity manifest."
    )
    assert default_system_message() == expected
    assert default_system_message(default_manifest) == expected


def test_canonical_records_templates_and_exported_defaults_are_exact() -> None:
    manifest = load_identity_manifest()
    records = canonical_identity_records(manifest)
    expected = (
        (
            PromptIntent.WHAT_ARE_YOU,
            "What are you?",
            "I am Aptivra, a research language model built on "
            "`HuggingFaceTB/SmolLM2-360M` at revision "
            "`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. My measured abilities and "
            "limitations are listed in my model card. Aptivra may be wrong.",
        ),
        (
            PromptIntent.ARCHITECTURE,
            "What is your architecture?",
            "My frozen base is a 361,821,120-parameter Llama causal language model. "
            "No routed LoRA component is enabled for this release.",
        ),
        (
            PromptIntent.CHAT_LEARNING,
            "Do you learn from this chat?",
            "This conversation changes my temporary context, but it does not "
            "persistently update my weights, replay buffer, or episodic memory. "
            "Persistent learning requires the explicit verifier-gated training and "
            "promotion process. Under automatic routing an adapter is active only when "
            "selected; manual adapter selection is an explicit evaluation or runtime "
            "mode.",
        ),
        (
            PromptIntent.CREATOR,
            "Who created Aptivra?",
            "The owner-approved creator attribution names Raghunath Chava as Aptivra's "
            "creator.",
        ),
        (
            PromptIntent.CAPABILITIES,
            "What can you do?",
            "No release evidence is available, so I do not claim measured capabilities. "
            "Capability is limited to the named evaluations in bound release evidence.",
        ),
    )
    assert tuple(
        (record.prompt_intent, record.question, record.answer) for record in records
    ) == expected

    system = default_system_message(manifest)
    templates = identity_chat_templates(manifest)
    assert templates == [
        (
            question,
            format_chat_prompt([], question, system_message=system),
            answer,
        )
        for _, question, answer in expected
    ]
    assert templates == IDENTITY_CHAT_TEMPLATES
    assert expected[0][2] == APTIVRA_INTRO
    assert NAME_REPLY == "My name is Aptivra."
    assert expected[3][2] == CREATOR_REPLY
    assert CREATOR_BIO_REPLY == CREATOR_REPLY
    assert expected[4][2] == CAPABILITIES_SUMMARY


def test_identity_chat_templates_bind_explicit_and_configured_manifests() -> None:
    assert identity_chat_templates() == IDENTITY_CHAT_TEMPLATES

    body = _identity_manifest_body()
    record = _valid_release_record(body)
    record["content_sha256"] = release_record_sha256(record)
    body["release_evaluation"]["record"] = record
    body["content_sha256"] = identity_manifest_sha256(body)
    manifest = IdentityManifest.model_validate(body)

    templates = identity_chat_templates(manifest)
    assert "exact_match_accuracy=0.75 ratio" in templates[-1][2]


def test_manifest_hash_preserves_unicode_bytes() -> None:
    payload: dict[str, object] = {"name": "Aptivra — 研究"}
    expected = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert identity_manifest_sha256(payload) == expected


def test_target_only_batch_includes_message_close_and_eos(tmp_path: Path) -> None:
    base = _tiny_chat_base(tmp_path)
    settings = _enabled_identity_settings()
    instance = ModelIdentityTaskFamily().generate_instances(1, seed=0)[0]
    batch = materialize_identity_batch(base, instance, settings)
    labels = batch.labels[0].tolist()
    assert labels[: batch.prompt_tokens] == [-100] * batch.prompt_tokens
    assert labels[-2:] == [2, 0]
    assert batch.input_ids[0, -2:].tolist() == [2, 0]
    assert batch.target_tokens == len(labels) - batch.prompt_tokens


def test_identity_training_records_cover_each_canonical_intent_twice() -> None:
    instances = identity_training_instances(
        load_identity_manifest(),
        expert_settings=_enabled_expert_settings(),
    )
    assert len(instances) == 2 * len(PromptIntent)
    assert [item.task_id for item in instances[:5]] == [
        f"identity_{index}" for index in range(5)
    ]
    assert [item.task_id for item in instances[5:]] == [
        f"conv_dialogue_{index}" for index in range(5)
    ]
    assert {item.metadata["prompt_intent"] for item in instances[:5]} == set(
        PromptIntent
    )
    assert {item.metadata["prompt_intent"] for item in instances[5:]} == set(
        PromptIntent
    )


def test_identity_task_seed_rotation_and_default_are_exact() -> None:
    family = ModelIdentityTaskFamily()
    records = canonical_identity_records()
    default_instances = family.generate_instances(3)
    wrapped_instances = family.generate_instances(3, seed=4)

    assert [item.target_solution for item in default_instances] == [
        records[2].answer,
        records[3].answer,
        records[4].answer,
    ]
    assert [item.target_solution for item in wrapped_instances] == [
        records[4].answer,
        records[0].answer,
        records[1].answer,
    ]
    assert [item.metadata["seed"] for item in default_instances] == [42, 42, 42]
    assert [item.metadata["index"] for item in wrapped_instances] == [0, 1, 2]
    assert [item.task_id for item in wrapped_instances] == [
        "identity_0",
        "identity_1",
        "identity_2",
    ]


def test_identity_task_expert_settings_are_keyword_only() -> None:
    parameter = inspect.signature(ModelIdentityTaskFamily).parameters["expert_settings"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_conversational_task_seed_rotation_and_turn_pairing_are_exact() -> None:
    manifest = load_identity_manifest()
    family = ConversationalTaskFamily(manifest)
    records = canonical_identity_records(manifest)
    default_instances = family.generate_instances(2)
    wrapped_instances = family.generate_instances(3, seed=4)
    overlapping_bit_instances = family.generate_instances(2, seed=3)

    assert [item.target_solution for item in default_instances] == [
        records[3].answer,
        records[4].answer,
    ]
    assert [item.target_solution for item in wrapped_instances] == [
        records[0].answer,
        records[1].answer,
        records[2].answer,
    ]
    assert wrapped_instances[0].prompt == format_chat_prompt(
        [(records[4].question, records[4].answer)],
        records[0].question,
        system_message=default_system_message(manifest),
    )
    assert wrapped_instances[1].prompt == format_chat_prompt(
        [(records[0].question, records[0].answer)],
        records[1].question,
        system_message=default_system_message(manifest),
    )
    assert overlapping_bit_instances[1].prompt == format_chat_prompt(
        [(records[4].question, records[4].answer)],
        records[0].question,
        system_message=default_system_message(manifest),
    )
    assert [item.metadata["seed"] for item in default_instances] == [42, 42]
    assert [item.metadata["turn"] for item in wrapped_instances] == [0, 1, 2]
    assert [item.task_id for item in wrapped_instances] == [
        "conv_dialogue_0",
        "conv_dialogue_1",
        "conv_dialogue_2",
    ]


def test_conversational_task_expert_settings_are_keyword_only() -> None:
    parameter = inspect.signature(ConversationalTaskFamily).parameters[
        "expert_settings"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_identity_training_instance_config_is_keyword_only() -> None:
    parameter = inspect.signature(identity_training_instances).parameters[
        "expert_settings"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_materialized_batch_is_exact_and_immutable() -> None:
    tokenizer = _ScriptedBatchTokenizer()
    settings = _enabled_identity_settings().model_copy(
        update={
            "training": _enabled_identity_settings().training.model_copy(
                update={"max_sequence_tokens": 5}
            )
        }
    )
    batch = materialize_identity_batch(
        _scripted_batch_base(tokenizer, runtime_limit=5),
        _batch_instance(),
        settings,
    )

    assert batch.input_ids.tolist() == [[10, 11, 12, 2, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert batch.labels.tolist() == [[-100, -100, 12, 2, 0]]
    assert batch.task_id == "identity-test"
    assert batch.prompt_tokens == 2
    assert batch.target_tokens == 3
    assert all(add_special_tokens is False for _, add_special_tokens in tokenizer.encode_calls)
    with pytest.raises(FrozenInstanceError):
        batch.target_tokens = 4


@pytest.mark.parametrize(
    ("prompt_ids", "target_ids", "message"),
    [
        ([], [12, 2, 0], "empty tokenized prompt"),
        ([10], [12, 0, 2], "message delimiter followed by EOS"),
        ([10], [2], "message delimiter followed by EOS"),
    ],
)
def test_materialized_batch_rejects_invalid_token_boundaries(
    prompt_ids: list[int],
    target_ids: list[int],
    message: str,
) -> None:
    tokenizer = _ScriptedBatchTokenizer(
        prompt_ids=prompt_ids,
        target_ids=target_ids,
    )
    with pytest.raises(ValueError, match=message):
        materialize_identity_batch(
            _scripted_batch_base(tokenizer, runtime_limit=20),
            _batch_instance(),
            _enabled_identity_settings(),
        )


@pytest.mark.parametrize(
    ("training_limit", "runtime_limit"),
    [(5, 100), (100, 5)],
)
def test_materialized_batch_enforces_the_smaller_sequence_limit(
    training_limit: int,
    runtime_limit: int,
) -> None:
    tokenizer = _ScriptedBatchTokenizer(target_ids=[12, 13, 2, 0])
    original = _enabled_identity_settings()
    settings = original.model_copy(
        update={
            "training": original.training.model_copy(
                update={"max_sequence_tokens": training_limit}
            )
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "prompt=2, target=4, total=6, limit=5"
        ),
    ):
        materialize_identity_batch(
            _scripted_batch_base(tokenizer, runtime_limit=runtime_limit),
            _batch_instance(),
            settings,
        )


def test_training_result_is_immutable() -> None:
    result = IdentityTrainingResult(
        manifest=cast(Any, object()),
        base_sha256_before="a",
        base_sha256_after="a",
        optimizer_steps=1,
        target_tokens=1,
        final_loss=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.optimizer_steps = 2


class _FakeTrainingModel:
    def __init__(self, parameter: torch.nn.Parameter, loss_mode: str) -> None:
        self.parameter = parameter
        self.loss_mode = loss_mode
        self.train_calls = 0
        self.calls: list[dict[str, object]] = []

    def train(self) -> None:
        self.train_calls += 1

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.loss_mode == "none":
            loss = None
        elif self.loss_mode == "nan":
            loss = self.parameter * torch.tensor(float("nan"))
        elif self.loss_mode == "detached":
            loss = torch.tensor(1.0, requires_grad=True)
        else:
            loss = (self.parameter - 1.0).square()
        return SimpleNamespace(loss=loss)


class _FakeTrainingBank:
    def __init__(self, loss_mode: str, *, base_changes: bool = False) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.5))
        self.model = _FakeTrainingModel(self.parameter, loss_mode)
        self.base_changes = base_changes
        self.hash_calls = 0
        self.promoted_to: Path | str | None = None
        self.manifest = cast(
            Any,
            SimpleNamespace(adapter_name="identity", content_sha256="a" * 64),
        )

    def adapter_parameters(self, adapter_name: str) -> list[torch.nn.Parameter]:
        assert adapter_name == "identity"
        return [self.parameter]

    def base_parameter_sha256(self) -> str:
        self.hash_calls += 1
        if self.base_changes and self.hash_calls > 1:
            return "b" * 64
        return "a" * 64

    def promote_candidate(self, output_dir: Path | str) -> object:
        self.promoted_to = output_dir
        return self.manifest


def _install_fake_identity_training(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_mode: str = "finite",
    instance_count: int = 3,
    base_changes: bool = False,
) -> _FakeTrainingBank:
    bank = _FakeTrainingBank(loss_mode, base_changes=base_changes)
    instances = tuple(
        TaskInstance(f"task-{index}", "identity", "prompt", "answer")
        for index in range(instance_count)
    )
    batch = chat_finetune.IdentityTrainingBatch(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        labels=torch.tensor([[-100, 2]], dtype=torch.long),
        task_id="task",
        prompt_tokens=1,
        target_tokens=7,
    )
    monkeypatch.setattr(
        chat_finetune.PeftExpertBank,
        "start_candidate",
        classmethod(lambda cls, base, settings, adapter_name: bank),
    )
    monkeypatch.setattr(
        chat_finetune,
        "validate_chat_tokenizer",
        lambda tokenizer, template: {IM_START: 1, IM_END: 2, END_OF_TEXT: 0},
    )
    monkeypatch.setattr(
        chat_finetune,
        "identity_training_instances",
        lambda manifest, *, expert_settings: instances,
    )
    monkeypatch.setattr(
        chat_finetune,
        "materialize_identity_batch",
        lambda base, instance, settings: batch,
    )
    return bank


def test_identity_training_feature_gates_fail_before_adapter_construction(
    tmp_path: Path,
) -> None:
    base = _tiny_chat_base(tmp_path)
    manifest = load_identity_manifest()
    with pytest.raises(RuntimeError, match="identity adapter training is disabled"):
        train_identity_adapter(
            base,
            _enabled_expert_settings(),
            load_identity_settings(),
            manifest,
            tmp_path / "disabled-identity",
        )
    with pytest.raises(RuntimeError, match="PEFT expert path is disabled"):
        train_identity_adapter(
            base,
            load_standard_expert_settings(),
            _enabled_identity_settings(),
            manifest,
            tmp_path / "disabled-expert",
        )


@pytest.mark.parametrize("loss_mode", ["none", "nan"])
def test_identity_training_rejects_missing_or_nonfinite_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss_mode: str,
) -> None:
    _install_fake_identity_training(monkeypatch, loss_mode=loss_mode)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1),
            load_identity_manifest(),
            tmp_path / loss_mode,
        )


def test_identity_training_rejects_loss_without_adapter_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_identity_training(monkeypatch, loss_mode="detached")
    with pytest.raises(RuntimeError, match="received no gradients"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1),
            load_identity_manifest(),
            tmp_path / "no-gradient",
        )


def test_identity_training_rejects_empty_generated_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_identity_training(monkeypatch, instance_count=0)
    with pytest.raises(RuntimeError, match="completed no optimizer steps"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1),
            load_identity_manifest(),
            tmp_path / "empty",
        )


def test_identity_training_stops_at_configured_budget_and_records_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = _install_fake_identity_training(monkeypatch)
    settings = _enabled_identity_settings(max_steps=2)
    settings = settings.model_copy(
        update={"training": settings.training.model_copy(update={"epochs": 3})}
    )
    result = train_identity_adapter(
        cast(StandardBase, SimpleNamespace(tokenizer=object())),
        _enabled_expert_settings(),
        settings,
        load_identity_manifest(),
        tmp_path / "bounded",
    )

    assert result.optimizer_steps == 2
    assert result.target_tokens == 14
    assert math.isfinite(result.final_loss)
    assert result.base_sha256_before == "a" * 64
    assert result.base_sha256_after == "a" * 64
    assert result.manifest is bank.manifest
    assert bank.promoted_to == tmp_path / "bounded"
    assert bank.model.train_calls == 1
    assert len(bank.model.calls) == 2
    assert all(call["use_cache"] is False for call in bank.model.calls)
    assert all(call["return_dict"] is True for call in bank.model.calls)


def test_identity_training_rejects_frozen_base_change_after_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_identity_training(monkeypatch, base_changes=True)
    with pytest.raises(RuntimeError, match="modified frozen base bytes"):
        train_identity_adapter(
            cast(StandardBase, SimpleNamespace(tokenizer=object())),
            _enabled_expert_settings(),
            _enabled_identity_settings(max_steps=1),
            load_identity_manifest(),
            tmp_path / "changed-base",
        )


def test_tiny_identity_training_preserves_base_and_reloads_hashed_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _tiny_chat_base(tmp_path / "source")
    identity_settings = _enabled_identity_settings(max_steps=2)
    expert_settings = _enabled_expert_settings()
    artifact = tmp_path / "identity-adapter"
    result = train_identity_adapter(
        base,
        expert_settings,
        identity_settings,
        load_identity_manifest(),
        artifact,
    )
    assert result.optimizer_steps == 2
    assert result.target_tokens > 0
    assert result.base_sha256_after == result.base_sha256_before
    assert result.manifest.adapter_name == "identity"
    assert len(result.manifest.content_sha256) == 64

    reloaded = load_promoted_expert(
        _tiny_chat_base(tmp_path / "reload"),
        expert_settings,
        artifact,
    )
    assert reloaded.adapter_names == ("identity",)
    assert reloaded.trainable_adapter is None

    chat = AptivraCausalLM(
        reloaded.base,
        reloaded,
        identity_settings,
        load_identity_manifest(),
    ).eval()
    assert chat.device == torch.device("cpu")
    output = chat.forward(
        input_ids=torch.tensor([[1, 3, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        use_cache=False,
    )
    assert output.logits.shape == (1, 3, len(chat.base.tokenizer))

    captured: dict[str, object] = {}

    def fake_generate(
        generated_base: StandardBase,
        prompt: str,
        *,
        max_new_tokens: int | None,
        stop_sequences: tuple[str, ...],
    ) -> GenerationResult:
        captured.update(
            base=generated_base,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
        return GenerationResult(f"answer{IM_END}", (5, 2), "terminator", IM_END)

    monkeypatch.setattr(standard_causal_lm, "generate_task_answer", fake_generate)
    generated = chat.generate_chat(
        "What are you?",
        history=(("Who created Aptivra?", "The owner attribution is recorded."),),
        max_new_tokens=4,
    )
    assert generated.text == "answer"
    assert captured["stop_sequences"] == (IM_END,)
    assert cast(str, captured["prompt"]).endswith(f"{IM_START}assistant\n")


def test_message_stop_is_detected_across_decoded_token_boundaries() -> None:
    class SplitTokenizer:
        eos_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del text, add_special_tokens
            return [9]

        def decode(
            self, token_ids: list[int], *, skip_special_tokens: bool
        ) -> str:
            assert skip_special_tokens is False
            return {(10,): "answer<|im_", (10, 11): f"answer{IM_END}"}[
                tuple(token_ids)
            ]

    criterion = DecodedTextStoppingCriteria(
        cast(Any, SplitTokenizer()),
        prompt_length=0,
        stop_sequences=(IM_END,),
    )
    assert criterion(torch.tensor([[10]]), None).tolist() == [False]
    assert criterion(torch.tensor([[10, 11]]), None).tolist() == [True]

    class SplitGenerator:
        def generate(self, **kwargs: object) -> torch.Tensor:
            input_ids = cast(torch.Tensor, kwargs["input_ids"])
            suffix = torch.tensor([[10, 11]], dtype=torch.long)
            return torch.cat((input_ids, suffix), dim=1)

    base = cast(StandardBase, object.__new__(StandardBase))
    real = load_standard_base_settings()
    object.__setattr__(base, "model", SplitGenerator())
    object.__setattr__(base, "tokenizer", SplitTokenizer())
    object.__setattr__(base, "settings", real)
    object.__setattr__(base, "snapshot", None)
    object.__setattr__(base, "device", torch.device("cpu"))
    result = generate_task_answer(
        base,
        "prompt",
        max_new_tokens=2,
        stop_sequences=(IM_END,),
    )
    assert result.text == "answer"
    assert result.stop_reason == "terminator"
    assert result.matched_terminator == IM_END
