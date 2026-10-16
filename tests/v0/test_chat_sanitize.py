"""Tests for chat reply sanitization."""

from aptivra.v0.inference.chat_sanitize import sanitize_chat_reply


def test_sanitize_strips_sql_drift() -> None:
    raw = "My name is Aptivra. SELECT (high + low) WHERE age 30;"
    assert "SELECT" not in sanitize_chat_reply(raw)


def test_sanitize_identity_name() -> None:
    raw = "My name is Aptivra, an autonomous cuntinual learning language model.-d as 2tE"
    out = sanitize_chat_reply(raw)
    assert out.startswith("My name is Aptivra")
    assert "2tE" not in out


def test_sanitize_first_sentence() -> None:
    raw = (
        "I was created by Raghunath Chava. "
        "def binary_search(arr, x): return mid"
    )
    out = sanitize_chat_reply(raw)
    assert "Raghunath" in out
    assert "def " not in out
