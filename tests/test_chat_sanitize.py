"""Delimiter-only output cleanup tests for the standard chat path."""

from aptivra.inference.chat_prompt import strip_assistant_turn_delimiter


def test_sanitize_identity_name() -> None:
    raw = "My name is Aptivra, with an unsupported claim.<|im_end|><|endoftext|>"
    out = strip_assistant_turn_delimiter(raw)
    assert out == "My name is Aptivra, with an unsupported claim."
    assert "unsupported claim" in out


def test_delimiter_cleanup_does_not_rescue_sql_or_replace_identity_facts() -> None:
    raw = "Wrong identity. SELECT * FROM users<|im_end|>"
    assert strip_assistant_turn_delimiter(raw) == "Wrong identity. SELECT * FROM users"
