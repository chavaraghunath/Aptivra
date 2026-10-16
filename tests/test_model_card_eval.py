"""Tests for model-card evaluation metrics on tiny local exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aptivra.eval import model_card_eval
from aptivra.eval.model_card_eval import (
    DEFAULT_HELDOUT_PROMPTS_PATH,
    DEFAULT_HELDOUT_TEXT_PATH,
    PromptAnswer,
    answer_span_bits_per_byte,
    count_export_parameters,
    count_state_dict_parameters,
    evaluate_export_dir,
    load_prompt_answers,
    sha256_file,
    summarize_bits_per_byte,
    uniform_bits_per_byte,
    unigram_bits_per_byte,
    unigram_bits_per_bytes,
    whole_text_bits_per_byte,
)
from aptivra.export.hf_hub_artifacts import export_huggingface_hub_layout
from aptivra.inference.chat_canonical import IDENTITY_CHAT_TEMPLATES
from aptivra.model.config import ModelConfig
from aptivra.model.transformer import BackboneTransformer
from aptivra.tasks.identity import ModelIdentityTaskFamily
from aptivra.v0.experts.expert import ExpertModule
from aptivra.v0.experts.router import GatingRouter


def test_baselines_on_known_strings() -> None:
    assert uniform_bits_per_byte(256) == pytest.approx(8.0)
    assert unigram_bits_per_byte("aaaa") == pytest.approx(0.0)
    assert unigram_bits_per_byte("aabb") == pytest.approx(1.0)


def test_metric_helpers_reject_invalid_or_empty_inputs() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        uniform_bits_per_byte(1)
    with pytest.raises(ValueError, match="must not be empty"):
        unigram_bits_per_byte("")
    with pytest.raises(ValueError, match="empty measurement"):
        summarize_bits_per_byte([])

    cfg = _tiny_config()
    model = BackboneTransformer(cfg)
    model.device = torch.device("cpu")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="at least two bytes"):
        whole_text_bits_per_byte(model, "x")


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
    untrained_checkpoint = untrained_record["checkpoint"]
    assert isinstance(untrained_checkpoint, dict)
    assert untrained_checkpoint["model_safetensors_sha256"] == sha256_file(
        untrained_dir / "model.safetensors"
    )

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


def test_versioned_heldout_sets_have_required_coverage_and_no_training_strings() -> None:
    heldout = load_prompt_answers(DEFAULT_HELDOUT_PROMPTS_PATH)
    assert len(heldout) >= 24
    assert {item.category for item in heldout} == {
        "general_knowledge",
        "arithmetic",
        "short_code",
        "plain_conversation",
    }

    identity_answers = {
        item.target_solution
        for item in ModelIdentityTaskFamily().generate_instances(count=7, seed=0)
    }
    chat_answers = {answer for _, _, answer in IDENTITY_CHAT_TEMPLATES}
    heldout_answers = {item.answer for item in heldout}
    assert len(heldout_answers) == len(heldout)
    assert heldout_answers.isdisjoint(identity_answers | chat_answers)

    heldout_text = DEFAULT_HELDOUT_TEXT_PATH.read_text(encoding="utf-8")
    assert len(heldout_text.encode("utf-8")) >= 8192
    training_strings: set[str] = set()
    for instance in ModelIdentityTaskFamily().generate_instances(count=7, seed=42):
        training_strings.update((instance.prompt, instance.target_solution))
    for question, prompt, answer in IDENTITY_CHAT_TEMPLATES:
        training_strings.update((question, prompt, answer))
    assert all(value not in heldout_text for value in training_strings)


def test_legacy_parameter_count_deduplicates_tied_embedding_and_output_head(
    tmp_path: Path,
) -> None:
    cfg = _tiny_config()
    model = BackboneTransformer(cfg)
    checkpoint = tmp_path / "tied.pt"
    torch.save(model.state_dict(), checkpoint)

    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert count_state_dict_parameters(checkpoint) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_hub_parameter_count_matches_packaging_metadata_on_tied_model(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=cfg.d_model, initial_num_experts=1)
    experts = [ExpertModule(expert_id=0, d_model=cfg.d_model)]
    export_huggingface_hub_layout(tmp_path, model, router, experts, cfg)

    counts = count_export_parameters(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert counts["backbone_parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert counts["total_parameters"] == config["parameter_count"]


def test_hub_parameter_count_rejects_incorrect_packaging_metadata(tmp_path: Path) -> None:
    cfg = _tiny_config()
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=cfg.d_model, initial_num_experts=1)
    experts = [ExpertModule(expert_id=0, d_model=cfg.d_model)]
    export_huggingface_hub_layout(tmp_path, model, router, experts, cfg)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["parameter_count"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="parameter_count does not match"):
        count_export_parameters(tmp_path)


def test_hub_evaluation_records_v4_weights_corpora_and_provenance(tmp_path: Path) -> None:
    cfg = _tiny_config(max_seq_len=192)
    model = BackboneTransformer(cfg)
    router = GatingRouter(d_model=cfg.d_model, initial_num_experts=1)
    experts = [ExpertModule(expert_id=0, d_model=cfg.d_model)]
    export_huggingface_hub_layout(tmp_path, model, router, experts, cfg)
    record = evaluate_export_dir(
        tmp_path,
        trained_prompts=[PromptAnswer(prompt="a", answer="b")],
    )

    assert record["schema_version"] == "aptivra.model_card_eval.v4"
    checkpoint = record["checkpoint"]
    assert isinstance(checkpoint, dict)
    assert checkpoint["model_safetensors_sha256"] == sha256_file(
        tmp_path / "model.safetensors"
    )
    assert checkpoint["source_hub_revision"] is None
    assert checkpoint["backbone_sha256"] is None
    assert checkpoint["provenance"]["source_layout"] == "native_huggingface"  # type: ignore[index]
    evaluation = record["evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["heldout_prompt_set"] == {
        "name": DEFAULT_HELDOUT_PROMPTS_PATH.name,
        "sha256": sha256_file(DEFAULT_HELDOUT_PROMPTS_PATH),
        "samples": 24,
    }
    assert evaluation["heldout_text_set"] == {
        "name": DEFAULT_HELDOUT_TEXT_PATH.name,
        "sha256": sha256_file(DEFAULT_HELDOUT_TEXT_PATH),
        "bytes": DEFAULT_HELDOUT_TEXT_PATH.stat().st_size,
    }
    tokenizer = record["tokenizer"]
    assert tokenizer == {
        "padding_token_id": 0,
        "end_of_sequence_token_ids": [0, 10],
        "newline_stop_token_id": 10,
    }

    with pytest.raises(ValueError, match="must be recorded during conversion"):
        evaluate_export_dir(
            tmp_path,
            trained_prompts=[PromptAnswer(prompt="a", answer="b")],
            hub_revision="different-revision",
        )


def test_each_metric_uses_baselines_from_exactly_its_scored_bytes(tmp_path: Path) -> None:
    cfg = _tiny_config(max_seq_len=64)
    answers = [PromptAnswer("p", "aaaa"), PromptAnswer("q", "ab")]
    export_dir = _export_tiny(tmp_path, cfg, memorized=False)

    record = evaluate_export_dir(
        export_dir,
        trained_prompts=answers,
        heldout_prompts=answers,
        heldout_text="zaabb",
    )

    evaluation = record["evaluation"]
    assert isinstance(evaluation, dict)
    answer_metric = evaluation["heldout_prompt_answer_bpb"]
    assert isinstance(answer_metric, dict)
    baselines = answer_metric["baselines"]
    assert isinstance(baselines, dict)
    unigram = baselines["unigram_bpb"]
    assert isinstance(unigram, dict)
    assert unigram["mean"] == pytest.approx(
        (unigram_bits_per_byte("aaaa") + unigram_bits_per_byte("ab")) / 2
    )
    assert answer_metric["evaluated_bytes"] == 6

    text_metric = evaluation["heldout_text_bpb"]
    assert isinstance(text_metric, dict)
    text_baselines = text_metric["baselines"]
    assert isinstance(text_baselines, dict)
    assert text_baselines["unigram_bpb"] == pytest.approx(
        unigram_bits_per_bytes(b"aabb")
    )
    assert text_metric["evaluated_bytes"] == 4


def test_evaluation_rejects_ambiguous_heldout_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        evaluate_export_dir(
            tmp_path,
            heldout_prompts=[PromptAnswer("a", "b")],
            heldout_prompts_path=tmp_path / "pairs.json",
        )


def test_load_prompt_answers_rejects_non_string_category(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps([{"prompt": "a", "answer": "b", "category": 7}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category must be a string"):
        load_prompt_answers(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"prompt": "a", "answer": "b"}, "must be a JSON array"),
        (["not-an-object"], "item must be an object"),
        ([{"prompt": "a", "answer": 1}], "string prompt and answer"),
    ],
)
def test_load_prompt_answers_rejects_malformed_data(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_prompt_answers(path)


def test_parameter_counter_rejects_non_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save([torch.ones(1)], checkpoint)
    with pytest.raises(ValueError, match="not a state dict"):
        count_state_dict_parameters(checkpoint)


def test_parameter_counter_ignores_non_tensor_state_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metadata.pt"
    torch.save({"weight": torch.ones(3), "format": "legacy"}, checkpoint)
    assert count_state_dict_parameters(checkpoint) == 3


def test_hub_parameter_counter_rejects_unknown_parameter_group(tmp_path: Path) -> None:
    save_file({"unknown.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="unrecognized model.safetensors parameter"):
        count_export_parameters(tmp_path)


def test_hub_parameter_counter_requires_valid_config_metadata(tmp_path: Path) -> None:
    save_file({"backbone.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    with pytest.raises(FileNotFoundError, match="config.json is required"):
        count_export_parameters(tmp_path)

    config_path = tmp_path / "config.json"
    config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        count_export_parameters(tmp_path)

    config_path.write_text(json.dumps({"parameter_count": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter_count must be an integer"):
        count_export_parameters(tmp_path)


def test_legacy_counts_and_private_evaluation_guards(tmp_path: Path) -> None:
    dense_dir = tmp_path / "legacy" / "dense"
    experts_dir = tmp_path / "legacy" / "experts"
    dense_dir.mkdir(parents=True)
    experts_dir.mkdir(parents=True)
    torch.save({"weight": torch.ones(2)}, dense_dir / "backbone.pt")
    torch.save({"weight": torch.ones(3)}, dense_dir / "router.pt")
    torch.save({"weight": torch.ones(4)}, experts_dir / "expert_0.pt")

    assert count_export_parameters(tmp_path / "legacy") == {
        "backbone_parameters": 2,
        "router_parameters": 3,
        "expert_parameters": 4,
        "total_parameters": 9,
    }
    assert len(model_card_eval.default_trained_prompt_answers()) == 7
    with pytest.raises(ValueError, match="answers must not be empty"):
        model_card_eval._answer_metric(
            [1.0], [PromptAnswer(prompt="prompt", answer="")], 256
        )

    missing = tmp_path / "missing.json"
    assert model_card_eval._load_json_object(missing) == {}
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        model_card_eval._load_json_object(non_object)

    output = tmp_path / "nested" / "record.json"
    model_card_eval.write_evaluation_record({"verified": True}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"verified": True}


def test_sequence_metric_rejects_invalid_target_and_model_output() -> None:
    cfg = _tiny_config()
    model = BackboneTransformer(cfg)
    model.device = torch.device("cpu")  # type: ignore[attr-defined]
    ids = torch.tensor([1, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="target_start must be positive"):
        model_card_eval._sequence_bits_per_byte(model, ids, target_start=0)
    with pytest.raises(ValueError, match="at least one target byte"):
        model_card_eval._sequence_bits_per_byte(model, ids, target_start=2)

    class InvalidOutputModel:
        config = cfg
        device = torch.device("cpu")

        def eval(self) -> object:
            return self

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return "not logits"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="must return logits tensor"):
        model_card_eval._sequence_bits_per_byte(
            InvalidOutputModel(), ids, target_start=1
        )


def test_sequence_metric_moves_targets_to_the_logits_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()

    class RelocatedLogitsModel:
        config = cfg
        device = torch.device("cpu")

        def eval(self) -> object:
            return self

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.empty((1, input_ids.shape[1], 256), device="meta")

    def fake_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        reduction: str,
    ) -> torch.Tensor:
        assert logits.device.type == "meta"
        assert targets.device == logits.device
        assert reduction == "sum"
        return torch.tensor(1.0)

    monkeypatch.setattr(model_card_eval.F, "cross_entropy", fake_cross_entropy)
    score = model_card_eval._sequence_bits_per_byte(
        RelocatedLogitsModel(),
        torch.tensor([1, 2], dtype=torch.long),
        target_start=1,
    )
    assert score == pytest.approx(1.0 / model_card_eval.math.log(2))


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
    export_huggingface_hub_layout(export_dir, model, router, experts, cfg)
    return export_dir


def _tiny_config(*, max_seq_len: int = 32) -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=0,
        n_heads=1,
        d_ff=16,
        max_seq_len=max_seq_len,
        dropout=0.0,
        device="cpu",
    )


def _heldout_mean(record: dict[str, object]) -> float:
    evaluation = record["evaluation"]
    assert isinstance(evaluation, dict)
    summary = evaluation["heldout_prompt_answer_bpb"]
    assert isinstance(summary, dict)
    mean = summary["mean"]
    assert isinstance(mean, float)
    return mean
