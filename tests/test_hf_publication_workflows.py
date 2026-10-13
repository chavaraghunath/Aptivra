"""Publication workflow guards for scripts that can contact Hugging Face."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from scripts.train_multidomain_conversational import train_multidomain_conversational


def test_multidomain_training_refuses_push_without_results_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hf_results_record is required"):
        train_multidomain_conversational(
            output_jsonl=str(tmp_path / "must-not-be-created.jsonl"),
            push_hf=True,
            hf_results_record=None,
        )

    assert not (tmp_path / "must-not-be-created.jsonl").exists()


def test_phase_pipeline_requires_results_record_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase_a_then_b.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--skip-restore", "--skip-a"])

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")

    assert error.value.code == 2
    assert "--results-record is required unless --skip-b is set" in capsys.readouterr().err
