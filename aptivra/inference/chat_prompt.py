"""Versioned Aptivra chat formatting on the pinned SmolLM2 tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Protocol, Union

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from aptivra.resources import aptivra_resource

DEFAULT_IDENTITY_CONFIG = aptivra_resource("configs", "smollm2_identity.yaml")
CHAT_TEMPLATE_VERSION = "aptivra-chat-v1"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"

PositiveInt = Annotated[int, Field(strict=True, gt=0)]

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  The classic typing
# spellings keep the public types precise without generating inert mutants.


def _safe_adjacent_filename(value: str) -> str:
    path = Path(value)
    if not value or path.name != value:
        raise ValueError("path must be one safe adjacent filename")
    return value


AdjacentFilename = Annotated[str, AfterValidator(_safe_adjacent_filename)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IdentityFeatureSettings(_StrictModel):
    """Fail-closed switch for the unaccepted identity adapter path."""

    enabled: bool


class ChatTemplateSettings(_StrictModel):
    """Exact versioned message/sample delimiter contract."""

    version: Literal["aptivra-chat-v1"]
    manifest_path: AdjacentFilename
    start_token: Literal["<|im_start|>"]
    end_token: Literal["<|im_end|>"]
    eos_token: Literal["<|endoftext|>"]

class IdentityComponentSettings(_StrictModel):
    """Adjacent standard expert/router configuration binding."""

    standard_expert_config_path: AdjacentFilename


class IdentityQualitySettings(_StrictModel):
    """Configurable unsanitized identity-evaluation thresholds."""

    minimum_canonical_prefix_ratio: float = Field(strict=True, gt=0.0, le=1.0)


class IdentityTrainingSettings(_StrictModel):
    """Bounded target-only identity-adapter training controls."""

    adapter_name: Literal["identity"]
    learning_rate: float = Field(strict=True, gt=0.0, le=1.0)
    weight_decay: float = Field(strict=True, ge=0.0, le=1.0)
    epochs: PositiveInt = Field(le=100)
    max_optimizer_steps: PositiveInt = Field(le=100_000)
    max_sequence_tokens: PositiveInt = Field(le=8192)


class IdentitySettings(_StrictModel):
    """Complete lane-6 runtime configuration."""

    schema_version: Literal["aptivra.identity_config.v1"]
    feature: IdentityFeatureSettings
    template: ChatTemplateSettings
    components: IdentityComponentSettings
    quality: IdentityQualitySettings
    training: IdentityTrainingSettings


class ChatTokenizer(Protocol):
    eos_token_id: Optional[int]  # noqa: UP045

    def encode(self, text: str, **options: bool) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...

    def decode(self, token_ids: list[int], **options: bool) -> str: ...


def load_identity_settings(
    path: Union[Path, str, Traversable] = DEFAULT_IDENTITY_CONFIG,  # noqa: UP007
) -> IdentitySettings:
    """Load the strict disabled-by-default lane-6 configuration."""
    source = Path(path) if isinstance(path, str) else path
    raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"smollm2_identity"}:
        raise ValueError("identity config root must contain only 'smollm2_identity'")
    return IdentitySettings.model_validate(raw["smollm2_identity"])


def validate_chat_tokenizer(
    tokenizer: ChatTokenizer,
    template: ChatTemplateSettings,
) -> dict[str, int]:
    """Prove all three configured protocol tokens are distinct single IDs."""
    token_ids: dict[str, int] = {}
    for token in (template.start_token, template.end_token, template.eos_token):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        registered = tokenizer.convert_tokens_to_ids(token)
        try:
            (encoded_id,) = encoded
        except ValueError as error:
            raise ValueError(
                f"chat template token {token!r} must map to one registered ID"
            ) from error
        if encoded_id != registered:
            raise ValueError(f"chat template token {token!r} must map to one registered ID")
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        if decoded != token:
            raise ValueError(f"chat template token {token!r} does not round-trip exactly")
        token_ids[token] = registered
    if len(set(token_ids.values())) == 3:
        if tokenizer.eos_token_id != token_ids[template.eos_token]:
            raise ValueError("chat template EOS token does not match tokenizer.eos_token_id")
        return token_ids
    raise ValueError("chat template special tokens must map to distinct IDs")


def normalize_gradio_history(history: Sequence[Any]) -> list[tuple[str, str]]:
    """Convert Gradio message dictionaries or pair histories into turn pairs."""
    if not history:
        return []
    first_message = next(iter(history))
    if not isinstance(first_message, dict):
        return [(str(user), str(assistant)) for user, assistant in history]

    pairs: list[tuple[str, str]] = []
    pending_user = ""
    for message in history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if isinstance(content, dict):
            content = content.get("text", "") or ""
        text = str(content)
        match role:
            case "user":
                pending_user = text
            case "assistant" if pending_user:
                pairs.append((pending_user, text))
                pending_user = ""
            case _:
                continue
    return pairs


def _message(role: Literal["system", "user", "assistant"], content: str) -> str:
    if not content:
        raise ValueError(f"chat {role} content must be non-empty")
    if IM_START in content or IM_END in content or END_OF_TEXT in content:
        raise ValueError(f"chat {role} content must not contain protocol delimiters")
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def format_chat_prompt(
    history: Sequence[tuple[str, str]],
    user_message: str,
    *,
    system_message: Optional[str] = None,  # noqa: UP045
) -> str:
    """Format closed prior messages and leave exactly one assistant turn open."""
    if system_message is None:
        from aptivra.inference.chat_canonical import default_system_message

        system_message = default_system_message()
    parts = [_message("system", system_message)]
    for user_turn, assistant_turn in history:
        parts.extend((_message("user", user_turn), _message("assistant", assistant_turn)))
    parts.append(_message("user", user_message))
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def format_chat_sample(
    history: Sequence[tuple[str, str]],
    user_message: str,
    answer: str,
    *,
    system_message: Optional[str] = None,  # noqa: UP045
) -> str:
    """Format one complete training sample with message close followed by EOS."""
    prompt = format_chat_prompt(
        history,
        user_message,
        system_message=system_message,
    )
    if not answer or IM_START in answer or IM_END in answer or END_OF_TEXT in answer:
        raise ValueError("assistant answer must be non-empty and delimiter-free")
    return f"{prompt}{answer}{IM_END}{END_OF_TEXT}"


def strip_assistant_turn_delimiter(text: str) -> str:
    """Remove protocol delimiters only; never insert, replace, or truncate facts."""
    return text.replace(IM_END, "").replace(END_OF_TEXT, "")
