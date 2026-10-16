"""Tests for chat prompt formatting."""

from aptivra.v0.inference.chat_prompt import format_chat_prompt, normalize_gradio_history


def test_format_chat_prompt_multiturn() -> None:
    history = [("Hi", "Hello")]
    prompt = format_chat_prompt(history, "How are you?")
    assert prompt.endswith("User: How are you?\nAssistant:")
    assert "User: Hi\nAssistant: Hello" in prompt


def test_normalize_gradio6_message_history() -> None:
    history = [
        {"role": "user", "content": "What is Aptivra?"},
        {"role": "assistant", "content": "I am Aptivra."},
    ]
    pairs = normalize_gradio_history(history)
    assert pairs == [("What is Aptivra?", "I am Aptivra.")]
    prompt = format_chat_prompt(pairs, "Who created you?")
    assert "User: Who created you?\nAssistant:" in prompt
    assert "What is Aptivra?" in prompt
