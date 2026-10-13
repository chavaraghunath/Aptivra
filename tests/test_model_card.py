"""Tests for measured-only model-card rendering and release gating."""

from __future__ import annotations

import re
from pathlib import Path

from aptivra.export.model_card import generate_model_card_readme
from aptivra.model.config import ModelConfig

DENIED_CARD_TEXT = (
    "89.4% average accuracy",
    "+0.042 BWT",
    "Dense 64.2%",
    "EWC 71.8%",
    "Standard MoE 82.1%",
    "AID-Bench 4-Phase Developmental Benchmark Results",
    "100.0% / PASSED",
    "200M parameter",
    "RoPE",
    "5,716+ MCP skills",
    "zero catastrophic forgetting",
    "C execution engine",
    "tool-use",
    "function-calling",
    "embodied-ai",
)


def test_generated_card_denies_unsupported_claims_and_marks_absent_fields() -> None:
    card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=8,
        results_record={"evaluation": {"baselines": {}}},
    )

    for denied in DENIED_CARD_TEXT:
        assert denied not in card
    assert "not measured" in card
    assert "Not a usable language model" in card


def test_release_gate_notice_present_below_gate_and_absent_above_gate() -> None:
    base_record = {
        "evaluation": {
            "heldout_prompt_answer_bpb": {"mean": 6.0, "min": 6.0, "max": 6.0, "samples": 1},
            "baselines": {"unigram_bpb": 4.0},
        }
    }
    below_card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=base_record,
    )
    assert "Not a usable language model" in below_card

    above_record = {
        "evaluation": {
            "heldout_prompt_answer_bpb": {"mean": 3.0, "min": 3.0, "max": 3.0, "samples": 1},
            "baselines": {"unigram_bpb": 4.0},
        }
    }
    above_card = generate_model_card_readme(
        repo_id="example/aptivra",
        model_config=ModelConfig(),
        num_experts=1,
        results_record=above_record,
    )
    assert "Not a usable language model" not in above_card


def test_model_card_source_has_no_result_literal_regressions() -> None:
    source = Path("aptivra/export/model_card.py").read_text(encoding="utf-8")
    forbidden_patterns = (
        r"\b\d+(?:\.\d+)?\s*%\b",
        r"[+-]\d+\.\d+\s*BWT",
        r"\baccuracy\b.*\d+\.\d+",
        r"\bBWT\b.*\d+\.\d+",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None
