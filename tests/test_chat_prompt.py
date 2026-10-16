"""Tests for the versioned Aptivra chat prompt."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from aptivra.inference.chat_prompt import (
    END_OF_TEXT,
    IM_END,
    IM_START,
    ChatTemplateSettings,
    IdentityComponentSettings,
    IdentityFeatureSettings,
    IdentityQualitySettings,
    IdentityTrainingSettings,
    format_chat_prompt,
    format_chat_sample,
    load_identity_settings,
    normalize_gradio_history,
    validate_chat_tokenizer,
)


def _template_settings(manifest_path: str = "identity_manifest.yaml") -> ChatTemplateSettings:
    return ChatTemplateSettings(
        version="aptivra-chat-v1",
        manifest_path=manifest_path,
        start_token=IM_START,
        end_token=IM_END,
        eos_token=END_OF_TEXT,
    )


def _training_settings(**changes: Any) -> IdentityTrainingSettings:
    values: dict[str, Any] = {
        "adapter_name": "identity",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "epochs": 1,
        "max_optimizer_steps": 100,
        "max_sequence_tokens": 8192,
    }
    values.update(changes)
    return IdentityTrainingSettings.model_validate(values)


class _ProbeTokenizer:
    def __init__(self) -> None:
        self.registered = {IM_START: 1000, IM_END: 1001, END_OF_TEXT: 1002}
        self.encoded: dict[str, list[int]] = {}
        self.decoded: dict[str, str] = {}
        self.eos_token_id = int("1002")
        self.encode_options: list[bool] = []
        self.decode_options: list[bool] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_options.append(add_special_tokens)
        if text in self.encoded:
            return self.encoded[text]
        return [int(str(self.registered[text]))]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.registered[token]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decode_options.append(skip_special_tokens)
        token = (IM_START, IM_END, END_OF_TEXT)[len(self.decode_options) - 1]
        if token_ids[0] != self.registered[token]:
            raise AssertionError("unexpected token ID")
        return self.decoded.get(token, token.encode().decode())


def test_format_chat_prompt_multiturn() -> None:
    history = [("Hi", "Hello")]
    prompt = format_chat_prompt(history, "How are you?")
    assert prompt.endswith(f"{IM_START}assistant\n")
    assert f"{IM_START}user\nHi{IM_END}\n" in prompt
    assert f"{IM_START}assistant\nHello{IM_END}\n" in prompt
    assert f"{IM_START}user\nHow are you?{IM_END}\n" in prompt
    assert "User:" not in prompt


def test_normalize_gradio6_message_history() -> None:
    history = [
        {"role": "user", "content": "What is Aptivra?"},
        {"role": "assistant", "content": "I am Aptivra."},
    ]
    pairs = normalize_gradio_history(history)
    assert pairs == [("What is Aptivra?", "I am Aptivra.")]
    prompt = format_chat_prompt(pairs, "Who created you?")
    assert f"{IM_START}user\nWho created you?{IM_END}\n" in prompt
    assert f"{IM_START}assistant\nI am Aptivra.{IM_END}\n" in prompt


def test_complete_sample_closes_message_then_sample_exactly() -> None:
    sample = format_chat_sample([], "Question?", "Answer.")
    assert sample.endswith(f"Answer.{IM_END}{END_OF_TEXT}")
    assert not sample.endswith(f"{IM_END}\n{END_OF_TEXT}")


def test_identity_settings_load_from_string_path_and_reject_malformed_roots(
    tmp_path: Path,
) -> None:
    assert load_identity_settings("configs/smollm2_identity.yaml").feature.enabled is False
    for index, raw in enumerate(
        (
            [],
            {},
            {"smollm2_identity": {}, "unexpected": {}},
        )
    ):
        path = tmp_path / f"invalid-{index}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="identity config root"):
            load_identity_settings(path)


@pytest.mark.parametrize(
    "value",
    ("", ".", "./identity.yaml", "identity.yaml/", "nested/identity.yaml", "/identity.yaml"),
)
def test_adjacent_config_paths_reject_unsafe_shapes(value: str) -> None:
    with pytest.raises(ValidationError):
        _template_settings(value)
    with pytest.raises(ValidationError):
        IdentityComponentSettings(standard_expert_config_path=value)


def test_strict_models_forbid_coercion_mutation_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        IdentityFeatureSettings.model_validate({"enabled": 1})
    with pytest.raises(ValidationError):
        IdentityFeatureSettings.model_validate({"enabled": False, "extra": True})
    settings = IdentityFeatureSettings(enabled=False)
    with pytest.raises(ValidationError):
        settings.enabled = True


@pytest.mark.parametrize("value", ("0.5", 0.0, 1.5))
def test_quality_ratio_is_strict_and_bounded(value: object) -> None:
    with pytest.raises(ValidationError):
        IdentityQualitySettings.model_validate({"minimum_canonical_prefix_ratio": value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", "0.1"),
        ("learning_rate", 0.0),
        ("learning_rate", 1.5),
        ("weight_decay", "0.1"),
        ("weight_decay", -0.5),
        ("weight_decay", 1.5),
        ("epochs", 1.0),
        ("epochs", 0),
        ("epochs", 101),
        ("max_optimizer_steps", 1.0),
        ("max_optimizer_steps", 0),
        ("max_optimizer_steps", 100_001),
        ("max_sequence_tokens", 1.0),
        ("max_sequence_tokens", 0),
        ("max_sequence_tokens", 8193),
    ),
)
def test_training_controls_are_strict_and_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _training_settings(**{field: value})


def test_training_controls_accept_their_exact_upper_boundaries() -> None:
    settings = _training_settings(
        epochs=100,
        max_optimizer_steps=100_000,
        max_sequence_tokens=8192,
    )
    assert settings.epochs == 100
    assert settings.max_optimizer_steps == 100_000
    assert settings.max_sequence_tokens == 8192


def test_chat_tokenizer_validation_uses_exact_raw_token_operations() -> None:
    tokenizer = _ProbeTokenizer()
    token_ids = validate_chat_tokenizer(tokenizer, _template_settings())
    assert token_ids == {IM_START: 1000, IM_END: 1001, END_OF_TEXT: 1002}
    assert token_ids[END_OF_TEXT] is not tokenizer.eos_token_id
    assert tokenizer.encode_options == [False, False, False]
    assert tokenizer.decode_options == [False, False, False]


@pytest.mark.parametrize("encoded", ([], [1000, 1000]))
def test_chat_tokenizer_rejects_non_single_token_encodings(encoded: list[int]) -> None:
    tokenizer = _ProbeTokenizer()
    tokenizer.encoded[IM_START] = encoded
    with pytest.raises(ValueError, match="must map to one registered ID"):
        validate_chat_tokenizer(tokenizer, _template_settings())


@pytest.mark.parametrize(("encoded", "registered"), ((999, 1000), (1001, 1000)))
def test_chat_tokenizer_rejects_registered_id_mismatch(
    encoded: int,
    registered: int,
) -> None:
    tokenizer = _ProbeTokenizer()
    tokenizer.registered[IM_START] = registered
    tokenizer.encoded[IM_START] = [encoded]
    with pytest.raises(ValueError, match="must map to one registered ID"):
        validate_chat_tokenizer(tokenizer, _template_settings())


@pytest.mark.parametrize("decoded", ("!", "~"))
def test_chat_tokenizer_rejects_non_exact_round_trip(decoded: str) -> None:
    tokenizer = _ProbeTokenizer()
    tokenizer.decoded[IM_START] = decoded
    with pytest.raises(ValueError, match="does not round-trip exactly"):
        validate_chat_tokenizer(tokenizer, _template_settings())


def test_chat_tokenizer_rejects_duplicate_special_token_ids() -> None:
    tokenizer = _ProbeTokenizer()
    tokenizer.registered[IM_END] = tokenizer.registered[IM_START]
    with pytest.raises(ValueError, match="distinct IDs"):
        validate_chat_tokenizer(tokenizer, _template_settings())


@pytest.mark.parametrize("eos_token_id", (999, 1003))
def test_chat_tokenizer_rejects_eos_registration_mismatch(eos_token_id: int) -> None:
    tokenizer = _ProbeTokenizer()
    tokenizer.eos_token_id = eos_token_id
    with pytest.raises(ValueError, match="does not match"):
        validate_chat_tokenizer(tokenizer, _template_settings())


def test_normalize_pair_history_and_empty_history() -> None:
    assert normalize_gradio_history([]) == []
    assert normalize_gradio_history([("user", "assistant")]) == [("user", "assistant")]


def test_normalize_message_history_skips_noise_and_accepts_nested_text() -> None:
    user_role = "".join(("us", "er"))
    assistant_role = "".join(("assist", "ant"))
    history = [
        {"role": "assistant", "content": "orphan"},
        {"role": "unknown", "content": "ignored"},
        {"role": user_role, "content": {"text": "Question"}},
        "not a message",
        {"role": assistant_role, "content": {"text": "Answer"}},
    ]
    assert normalize_gradio_history(history) == [("Question", "Answer")]


def test_normalize_message_history_uses_empty_text_for_null_nested_content() -> None:
    history = [
        {"role": "user", "content": {"text": None}},
        {"role": "assistant", "content": "ignored without a pending user"},
    ]
    assert normalize_gradio_history(history) == []


def test_prompt_and_sample_system_message_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        format_chat_prompt([], "Question", "System")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        format_chat_sample([], "Question", "Answer", "System")  # type: ignore[call-arg]


def test_explicit_system_message_is_formatted_exactly() -> None:
    prompt = format_chat_prompt([], "Question", system_message="System")
    assert prompt == (
        f"{IM_START}system\nSystem{IM_END}\n"
        f"{IM_START}user\nQuestion{IM_END}\n"
        f"{IM_START}assistant\n"
    )


@pytest.mark.parametrize("content", ("", IM_START, IM_END, END_OF_TEXT))
def test_prompt_rejects_empty_or_delimiter_bearing_messages(content: str) -> None:
    with pytest.raises(ValueError):
        format_chat_prompt([], content, system_message="System")


@pytest.mark.parametrize("answer", ("", IM_START, IM_END, END_OF_TEXT))
def test_sample_rejects_empty_or_delimiter_bearing_answers(answer: str) -> None:
    with pytest.raises(ValueError):
        format_chat_sample([], "Question", answer, system_message="System")
