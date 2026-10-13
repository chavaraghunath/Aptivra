"""Regression tests for Gradio checkpoint layout selection."""

from pathlib import Path

import pytest

from scripts.gradio_chat import checkpoint_weights_path


def test_checkpoint_weights_path_prefers_safetensors_and_supports_legacy(tmp_path: Path) -> None:
    legacy = tmp_path / "dense" / "backbone.pt"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")
    assert checkpoint_weights_path(tmp_path) == legacy

    safetensors = tmp_path / "model.safetensors"
    safetensors.write_bytes(b"current")
    assert checkpoint_weights_path(tmp_path) == safetensors


def test_checkpoint_weights_path_rejects_directory_without_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors.*dense/backbone.pt"):
        checkpoint_weights_path(tmp_path)
