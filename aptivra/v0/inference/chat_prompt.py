"""Chat prompt formatting for byte-level Aptivra models."""

from collections.abc import Sequence
from typing import Any


def normalize_gradio_history(history: Sequence[Any]) -> list[tuple[str, str]]:
    """Converts Gradio 6 OpenAI-style messages to (user, assistant) pairs."""
    if not history:
        return []
    first = history[0]
    if not isinstance(first, dict):
        return [(str(u), str(a)) for u, a in history]

    pairs: list[tuple[str, str]] = []
    pending_user = ""
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("text", "") or ""
        text = str(content)
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user:
            pairs.append((pending_user, text))
            pending_user = ""
    return pairs


def format_chat_prompt(
    history: Sequence[tuple[str, str]],
    user_message: str,
) -> str:
    """Builds a User/Assistant transcript ending with an open Assistant turn."""
    parts: list[str] = []
    for user_turn, assistant_turn in history:
        parts.append(f"User: {user_turn.strip()}\nAssistant: {assistant_turn.strip()}\n")
    parts.append(f"User: {user_message.strip()}\nAssistant:")
    return "".join(parts)
