"""Tests for model-card evaluation metrics on tiny local exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aptivra.eval.model_card_eval import (
    PromptAnswer,
    answer_span_bits_per_byte,
    evaluate_export_dir,
    uniform_bits_per_byte,
    unigram_bits_per_byte,
)
from aptivra.experts.expert import ExpertModule
from aptivra.experts.router import GatingRouter
from aptivra.export.multitier import MultitierExporter
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer


def test_baselines_on_known_strings() -> None:
    assert uniform_bits_per_byte(256) == pytest.approx(8.0)
    assert unigram_bits_per_byte("aaaa") == pytest.approx(0.0)
    assert unigram_bits_per_byte("aabb") == pytest.approx(1.0)


def test_tiny_export_memorized_string_scores_near_zero_and_untrained_near_uniform(
    tmp_path: Path,
) -> None:
    cfg = ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=0,
        n_heads=1,
        d_ff=16,
        max_seq_len=32,
        dropout=0.0,
        device="cpu",
    )
    prompt_answer = [PromptAnswer(prompt="a", answer="aaaa")]
    heldout_text = "aabb"

    untrained_dir = _export_tiny(tmp_path / "untrained", cfg, memorized=False)
    untrained_record = evaluate_export_dir(
        untrained_dir,
        trained_prompts=prompt_answer,
        heldout_prompts=prompt_answer,
        heldout_text=heldout_text,
    )
    untrained_bpb = _heldout_mean(untrained_record)
    assert untrained_bpb == pytest.approx(8.0, abs=0.5)

    memorized_dir = _export_tiny(tmp_path / "memorized", cfg, memorized=True)
    memorized_record = evaluate_export_dir(
        memorized_dir,
        trained_prompts=prompt_answer,
        heldout_prompts=prompt_answer,
        heldout_text=heldout_text,
    )
    memorized_bpb = _heldout_mean(memorized_record)
    assert memorized_bpb < 0.01
    assert memorized_record["model"]["total_parameters"] > 0  # type: ignore[index]


def test_answer_span_scoring_rejects_empty_answer() -> None:
    cfg = ModelConfig(d_model=16, n_layers=0, n_heads=1, d_ff=16, device="cpu")
    model = BackboneTransformer(cfg)
    model.device = torch.device("cpu")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="answer must not be empty"):
        answer_span_bits_per_byte(model, "prompt", "")


def _export_tiny(export_dir: Path, cfg: ModelConfig, *, memorized: bool) -> Path:
    model = BackboneTransformer(cfg)
    if memorized:
        with torch.no_grad():
            assert isinstance(model.token_embedding, torch.nn.Embedding)
            model.token_embedding.weight.zero_()
            model.pos_embedding.weight.zero_()
            model.token_embedding.weight[ord("a"), 0] = 50.0

    router = GatingRouter(d_model=cfg.d_model, initial_num_experts=1)
    experts = [ExpertModule(expert_id=0, d_model=cfg.d_model)]
    MultitierExporter(export_dir).export_model(model, router, experts, cfg)
    config_json = {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
        "dropout": cfg.dropout,
    }
    (export_dir / "config.json").write_text(json.dumps(config_json), encoding="utf-8")
    return export_dir


def _heldout_mean(record: dict[str, object]) -> float:
    evaluation = record["evaluation"]
    assert isinstance(evaluation, dict)
    summary = evaluation["heldout_prompt_answer_bpb"]
    assert isinstance(summary, dict)
    mean = summary["mean"]
    assert isinstance(mean, float)
    return mean
