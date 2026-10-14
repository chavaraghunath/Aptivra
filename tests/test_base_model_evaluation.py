"""Offline proof tests for pretrained-base measurement and selection."""

from __future__ import annotations

import copy
import math
import shutil
import socket
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

import aptivra.eval.base_model_evaluation as base_eval
from aptivra.eval.base_model_evaluation import (
    BaseModelEvaluationSettings,
    CandidateSpec,
    DecisionSettings,
    LoadedCandidate,
    MeasurementSettings,
    apply_decision_rule,
    compute_task_stream_properties,
    derive_existing_records_analysis,
    evaluate_all_candidates,
    evaluate_candidate,
    inspect_expert_attachment_structure,
    load_candidate,
    paired_family_comparison,
    run_learnability_probe,
    whole_text_bits_per_byte,
    wilson_interval,
)
from aptivra.tasks.base import TaskInstance
from aptivra.tasks.stream import ContinualTaskStream, TaskStreamSplit


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental network access a deterministic test failure."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access is forbidden in base-model evaluator tests")

    monkeypatch.setattr(socket, "create_connection", fail_network)


def _settings(tmp_path: Path) -> MeasurementSettings:
    text_path = tmp_path / "text.txt"
    prompts_path = tmp_path / "prompts.json"
    text_path.write_text("hello", encoding="utf-8")
    prompts_path.write_text('[{"prompt":"P","answer":"A"}]', encoding="utf-8")
    return MeasurementSettings(
        device="cpu",
        local_files_only=True,
        torch_dtype="float32",
        deterministic_algorithms=True,
        deterministic_warn_only=False,
        seed=7,
        variance_seeds=(7, 8, 9),
        variance_top_candidates=2,
        train_items_per_family=1,
        test_items_per_family=1,
        validation_items_per_family=1,
        max_generation_attempts=100,
        max_new_tokens=2,
        probe_steps=30,
        probe_batch_size=1,
        probe_learning_rates=(0.2,),
        probe_warmup_steps=2,
        probe_weight_decay=0.0,
        evaluation_batch_size=2,
        confidence_level=0.95,
        generation_benchmark_prompts=1,
        generation_benchmark_new_tokens=1,
        generation_benchmark_warmup_runs=0,
        generation_benchmark_repetitions=1,
        scoring_stride_tokens=2,
        load_retries=1,
        load_retry_backoff_seconds=0.0,
        memory_sample_interval_seconds=0.01,
        candidate_timeout_seconds=10.0,
        expected_wall_time_seconds=60.0,
        utf8_roundtrip_samples=("é",),
        heldout_text_path=text_path,
        heldout_prompts_path=prompts_path,
    )


def _save_tokenizer(path: Path, vocab_size: int) -> None:
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "P": 3, "A": 4}
    for index in range(5, vocab_size):
        vocab[f"t{index}"] = index
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)


def test_local_auto_loading_and_structure_for_gpt2_and_llama(tmp_path: Path) -> None:
    """The standard Auto path handles two local architectures with no network."""
    settings = _settings(tmp_path)
    model_specs = [
        (
            "gpt2",
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=16,
                    n_positions=16,
                    n_embd=8,
                    n_layer=2,
                    n_head=2,
                    bos_token_id=1,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            ),
        ),
        (
            "llama",
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=16,
                    max_position_embeddings=16,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    bos_token_id=1,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            ),
        ),
    ]
    for model_type, model in model_specs:
        model_dir = tmp_path / model_type
        model.save_pretrained(model_dir)
        _save_tokenizer(model_dir, 16)
        loaded = load_candidate(
            CandidateSpec(model_type, str(model_dir), None, "mit", model_type, False),
            settings,
        )
        structure = inspect_expert_attachment_structure(loaded.model, 8)
        assert structure["transformer_blocks"] == 2
        assert structure["all_blocks_expose_feed_forward"] is True


class _FixedDistributionModel(nn.Module):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(probabilities).log())
        self.config = SimpleNamespace(vocab_size=len(probabilities), model_type="fixed")

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> Any:
        del kwargs
        logits = self.fixed_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


class _MultiByteTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 for character in text if character == "é"]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join("é" for token_id in ids if token_id == 1)


class _BytePairTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def __len__(self) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {195: 0, 169: 1}
        return [mapping[value] for value in text.encode("utf-8")]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        del kwargs
        reverse = {0: 195, 1: 169}
        return bytes(reverse[value] for value in ids).decode("utf-8")


def test_bits_per_byte_is_analytic_windowed_and_cross_tokenizer_comparable() -> None:
    """Multi-byte tokens and byte tokens produce analytically comparable BPB."""
    device = torch.device("cpu")
    token_model = cast(Any, _FixedDistributionModel([0.75, 0.25]))
    token_score = whole_text_bits_per_byte(
        token_model,
        cast(Any, _MultiByteTokenizer()),
        "éééé",
        context_length=3,
        device=device,
        stride_tokens=2,
    )
    assert token_score.total_bits == pytest.approx(8.0)
    assert token_score.bits_per_byte == pytest.approx(1.0)
    assert token_score.scored_tokens == 4
    assert token_score.scored_bytes == 8
    assert token_score.windows == 2

    byte_model = cast(Any, _FixedDistributionModel([0.5, 0.5]))
    byte_score = whole_text_bits_per_byte(
        byte_model,
        cast(Any, _BytePairTokenizer()),
        "éééé",
        context_length=3,
        device=device,
        stride_tokens=2,
    )
    assert byte_score.scored_tokens == 8
    assert byte_score.scored_bytes == 8
    assert byte_score.bits_per_byte == pytest.approx(token_score.bits_per_byte)


def test_wilson_interval_is_measured_and_rejects_invalid_counts() -> None:
    """A 100-item accuracy carries the expected 95% Wilson uncertainty."""
    interval = wilson_interval(50, 100, 0.95)
    assert interval["lower"] == pytest.approx(0.4038, abs=0.0001)
    assert interval["upper"] == pytest.approx(0.5962, abs=0.0001)
    with pytest.raises(ValueError, match="positive denominator"):
        wilson_interval(0, 0, 0.95)
    with pytest.raises(ValueError, match="between zero and total"):
        wilson_interval(2, 1, 0.95)


class _ToyTokenizer:
    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 0
    eos_token = "<eos>"
    model_max_length = 16

    def __len__(self) -> int:
        return 5

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {"P": 1, "A": 2, "\n": 3}
        return [mapping[character] for character in text]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        mapping = {0: "", 1: "P", 2: "A", 3: "\n", 4: "<eos>"}
        return "".join(
            "" if skip_special_tokens and value == 4 else mapping[value] for value in ids
        )


class _ToyLearnableLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.0, 4.0, 0.0, 0.0, 0.0]))
        self.config = SimpleNamespace(
            vocab_size=5,
            model_type="toy",
            max_position_embeddings=16,
            eos_token_id=4,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        del kwargs
        logits = self.bias.expand(input_ids.shape[0], input_ids.shape[1], -1)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, 5),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class _CountingToyLM(_ToyLearnableLM):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        self.batch_sizes.append(int(input_ids.shape[0]))
        return super().forward(input_ids, labels=labels, **kwargs)


def test_generation_batches_prompts_without_changing_prompt_positions() -> None:
    """Powered evaluation sends multiple prompts through each autoregressive step."""
    model = _CountingToyLM()
    predictor = base_eval.TransformersPredictor(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        device=torch.device("cpu"),
        context_length=16,
        max_new_tokens=2,
    )
    results = predictor.generate_batch(["P", "PP"], stop_token_ids=(10,))
    assert [item.text for item in results] == ["PP", "PP"]
    assert model.batch_sizes == [2, 2]


class _ToyStream:
    def __init__(self) -> None:
        self.families = [SimpleNamespace(family_id="toy", stop_token_ids=(10,))]
        instance = TaskInstance("toy-1", "toy", "P", "A")
        self.split = TaskStreamSplit([instance], [instance], [instance])
        self.num_train_per_task = 1
        self.num_val_per_task = 1
        self.num_test_per_task = 1

    def get_split(self, family_id: str) -> TaskStreamSplit:
        assert family_id == "toy"
        return self.split


def test_stream_properties_measure_targets_templates_and_empty_splits() -> None:
    """Target reuse and renamed-template reuse are measured from generated rows."""
    family = SimpleNamespace(family_id="code")
    train = [
        TaskInstance(
            "train-1",
            "code",
            "def alpha(x):",
            "def alpha(x): return 1",
            {"fn_name": "alpha"},
        ),
        TaskInstance("train-2", "code", "choose", "NO"),
    ]
    validation = [TaskInstance("val-1", "code", "new", "YES")]
    test = [
        TaskInstance(
            "test-1",
            "code",
            "def beta(x):",
            "def beta(x): return 1",
            {"fn_name": "beta"},
        ),
        TaskInstance("test-2", "code", "choose again", "NO"),
    ]
    stream = SimpleNamespace(
        families=[family],
        get_split=lambda family_id: TaskStreamSplit(train, validation, test),
    )
    result = compute_task_stream_properties(cast(Any, stream))
    row = cast(list[dict[str, object]], result["families"])[0]
    assert row["distinct_targets"] == {"train": 2, "validation": 1, "test": 2}
    assert row["uniform_over_distinct_test_targets_chance_level"] == 0.5
    assert row["test_target_overlap_with_train"] == {
        "distinct_targets": 1,
        "distinct_target_fraction": 0.5,
        "items": 1,
        "item_fraction": 0.5,
    }
    assert row["test_template_overlap_with_train"] == {
        "distinct_templates": 1,
        "distinct_template_fraction": 0.5,
        "items": 1,
        "item_fraction": 0.5,
    }

    empty = SimpleNamespace(
        families=[family],
        get_split=lambda family_id: TaskStreamSplit(train, validation, []),
    )
    with pytest.raises(ValueError, match="empty split"):
        compute_task_stream_properties(cast(Any, empty))


def test_powered_stream_properties_are_computed_at_recorded_split_sizes() -> None:
    """The powered stream exposes small target sets and train-target reuse."""
    stream = ContinualTaskStream(
        num_train_per_task=256,
        num_val_per_task=100,
        num_test_per_task=100,
        seed=20261014,
        max_generation_attempts=1000,
    )
    result = compute_task_stream_properties(stream)
    by_id = {
        cast(str, row["family_id"]): row
        for row in cast(list[dict[str, object]], result["families"])
    }
    assert cast(dict[str, int], by_id["T5_boolean_logic"]["distinct_targets"])[
        "test"
    ] == 2
    assert by_id["T5_boolean_logic"][
        "uniform_over_distinct_test_targets_chance_level"
    ] == 0.5
    assert cast(dict[str, int], by_id["T7_state_machines"]["distinct_targets"])[
        "test"
    ] == 4
    assert by_id["T7_state_machines"][
        "uniform_over_distinct_test_targets_chance_level"
    ] == 0.25
    assert [
        cast(
            dict[str, int], by_id[family_id]["test_target_overlap_with_train"]
        )["items"]
        for family_id in (
            "T1_modular_arithmetic",
            "T6_sequence_completion",
            "T8_python_microsynth",
        )
    ] == [97, 73, 95]
    assert [
        cast(
            dict[str, int], by_id[family_id]["distinct_template_signatures"]
        )["test"]
        for family_id in (
            "T9_algo_synth",
            "T10_docstring_code",
            "T11_bug_fixer",
            "T12_type_inferrer",
        )
    ] == [3, 2, 2, 2]
    assert result["all_prompt_and_target_text_ascii"] is True


def test_learnability_probe_learns_without_mutating_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real optimizer update changes held-out accuracy only on the copied model."""
    settings = _settings(tmp_path)
    model = _ToyLearnableLM()
    original = copy.deepcopy(model.state_dict())

    def evaluate_accuracy(
        predictor: Any, *args: object, **kwargs: object
    ) -> dict[str, object]:
        del args, kwargs
        passed = int(torch.argmax(predictor.model.bias).item()) == 2
        return {
            "accuracy": float(passed),
            "successes": int(passed),
            "total": 1,
            "wilson_interval": wilson_interval(int(passed), 1, 0.95),
        }

    monkeypatch.setattr(base_eval, "evaluate_batched_accuracy", evaluate_accuracy)
    result = run_learnability_probe(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        0,
        settings,
        16,
        torch.device("cpu"),
    )
    assert result["zero_shot_accuracy"] == 0.0
    assert result["learned_accuracy"] == 1.0
    assert result["accuracy_gain"] == 1.0
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_learning_rate_grid_never_evaluates_test_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the selected LR retrain may access test rows in future runs."""
    settings = replace(_settings(tmp_path), probe_learning_rates=(0.1, 0.2))
    instance = TaskInstance("item", "toy", "P", "A")
    stream = SimpleNamespace(
        families=[
            SimpleNamespace(family_id="toy-a", stop_token_ids=(10,)),
            SimpleNamespace(family_id="toy-b", stop_token_ids=(10,)),
        ],
        get_split=lambda family_id: TaskStreamSplit(
            [instance], [instance], [instance]
        ),
    )
    events: list[tuple[str, float | None, bool | None]] = []

    def fake_probe(*args: object, **kwargs: object) -> dict[str, object]:
        family_index = cast(int, args[3])
        learning_rate = cast(float, kwargs["learning_rate"])
        evaluate_test = cast(bool, kwargs["evaluate_test"])
        events.append(("probe", learning_rate, evaluate_test))
        validation_accuracy = 0.4 if learning_rate == 0.1 else 0.6
        result: dict[str, object] = {
            "family_id": f"toy-{family_index}",
            "validation": {
                "accuracy": validation_accuracy,
                "successes": round(validation_accuracy * 10),
                "total": 10,
                "wilson_interval": wilson_interval(
                    round(validation_accuracy * 10), 10, 0.95
                ),
            },
        }
        if evaluate_test:
            result.update(
                {
                    "zero_shot": kwargs["zero_shot"],
                    "learned": {
                        "accuracy": 0.7,
                        "successes": 7,
                        "total": 10,
                        "wilson_interval": wilson_interval(7, 10, 0.95),
                    },
                    "zero_shot_accuracy": 0.0,
                    "learned_accuracy": 0.7,
                    "accuracy_gain": 0.7,
                }
            )
        return result

    def fake_accuracy(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        events.append(("zero-shot-test", None, None))
        return {
            "accuracy": 0.0,
            "successes": 0,
            "total": 10,
            "wilson_interval": wilson_interval(0, 10, 0.95),
        }

    monkeypatch.setattr(base_eval, "run_learnability_probe", fake_probe)
    monkeypatch.setattr(base_eval, "evaluate_batched_accuracy", fake_accuracy)
    result = base_eval.evaluate_task_readiness(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        cast(Any, stream),
        settings,
        16,
        torch.device("cpu"),
    )
    assert events == [
        ("probe", 0.1, False),
        ("probe", 0.1, False),
        ("probe", 0.2, False),
        ("probe", 0.2, False),
        ("zero-shot-test", None, None),
        ("zero-shot-test", None, None),
        ("probe", 0.2, True),
        ("probe", 0.2, True),
    ]
    selection = cast(dict[str, object], result["learning_rate_selection"])
    assert selection["selected_learning_rate"] == 0.2
    assert "not evaluated" in cast(str, selection["test_evaluation_timing"])


def _completed_measurements() -> dict[str, object]:
    families = [
        {
            "family_id": f"T{index}",
            "training_tokens_per_second": 10.0,
            "training_bytes_per_second": 8.0,
            "learned_accuracy": 0.5,
            "accuracy_gain": 0.5,
            "learned": {"successes": 50, "total": 100},
        }
        for index in range(12)
    ]
    return {
        "tokenizer_fitness": {
            "exact_utf8_roundtrip": True,
            "eos_usable_as_answer_terminator": True,
        },
        "structural_fitness": {"all_blocks_expose_feed_forward": True},
        "language_modeling": {"combined": {"bits_per_byte": 2.0}},
        "task_stream_readiness": {
            "families": families,
            "mean_zero_shot_accuracy": 0.0,
            "mean_learned_accuracy": 0.5,
            "mean_accuracy_gain": 0.5,
            "mean_learned": {
                "wilson_interval": {
                    "confidence_level": 0.95,
                    "lower": 0.45,
                    "upper": 0.55,
                }
            },
            "mean_accuracy_gain_interval": {"lower": 0.4, "upper": 0.6},
        },
        "cost": {
            "median_probe_training_tokens_per_second": 10.0,
            "median_probe_training_bytes_per_second": 8.0,
        },
    }


def test_failures_continue_and_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load and mid-measurement OOM failures persist while another candidate completes."""
    settings = _settings(tmp_path)
    candidates = tuple(
        CandidateSpec(name, name, "a" * 40, "mit", "fixed", False)
        for name in ("load-fail", "oom", "good")
    )
    calls: list[str] = []

    def loader(candidate: CandidateSpec, _: MeasurementSettings) -> LoadedCandidate:
        calls.append(candidate.candidate_id)
        if candidate.candidate_id == "load-fail":
            raise RuntimeError("load failed")
        model = _FixedDistributionModel([0.5, 0.5])
        model.marker = candidate.candidate_id
        return LoadedCandidate(cast(Any, model), cast(Any, _BytePairTokenizer()), "a" * 40)

    monkeypatch.setattr(base_eval, "model_context_length", lambda model, tokenizer: 8)
    monkeypatch.setattr(base_eval, "_hidden_size", lambda model: 2)
    monkeypatch.setattr(base_eval, "_architecture_type", lambda model: "fixed")
    monkeypatch.setattr(
        base_eval,
        "tokenizer_fitness",
        lambda *args: {
            "exact_utf8_roundtrip": True,
            "eos_usable_as_answer_terminator": True,
        },
    )
    monkeypatch.setattr(
        base_eval,
        "inspect_expert_attachment_structure",
        lambda *args: {"all_blocks_expose_feed_forward": True},
    )

    def language(model: Any, *args: object) -> dict[str, object]:
        if model.marker == "oom":
            raise RuntimeError("MPS backend out of memory")
        return {"combined": {"bits_per_byte": 2.0}}

    monkeypatch.setattr(base_eval, "evaluate_language_quality", language)
    monkeypatch.setattr(
        base_eval,
        "evaluate_task_readiness",
        lambda *args: _completed_measurements()["task_stream_readiness"],
    )
    monkeypatch.setattr(base_eval, "benchmark_generation", lambda *args: {})
    all_settings = BaseModelEvaluationSettings(
        candidates,
        settings,
        DecisionSettings(("mit",), 10**12, 0.1, 0.0, 0.0, 0.02, True),
    )
    records = evaluate_all_candidates(
        all_settings,
        tmp_path / "out",
        resume=True,
        loader=loader,
    )
    assert [record["status"] for record in records] == [
        "failed",
        "failed",
        "completed",
    ], records
    assert cast(dict[str, object], records[1]["failure"])["out_of_memory"] is True
    calls.clear()
    evaluate_all_candidates(
        all_settings,
        tmp_path / "out",
        resume=True,
        loader=loader,
    )
    assert calls == ["load-fail", "oom"]


def _decision_record(
    candidate_id: str,
    *,
    learned: float = 0.5,
    gain: float = 0.2,
    bpb: float = 2.0,
    throughput: float = 10.0,
    parameters: int = 100,
    family_learned: list[float] | None = None,
    family_gains: list[float] | None = None,
) -> dict[str, object]:
    learned_values = family_learned or [learned] * 12
    gain_values = family_gains or [gain] * 12
    if len(learned_values) != len(gain_values):
        raise ValueError("family metric lengths must match")
    return {
        "status": "completed",
        "candidate_id": candidate_id,
        "reference_only": False,
        "identity": {
            "license": "mit",
            "trust_remote_code": False,
            "unique_parameters": parameters,
        },
        "tokenizer_fitness": {
            "exact_utf8_roundtrip": True,
            "eos_usable_as_answer_terminator": True,
        },
        "structural_fitness": {"all_blocks_expose_feed_forward": True},
        "language_modeling": {"combined": {"bits_per_byte": bpb}},
        "task_stream_readiness": {
            "families": [
                {
                    "family_id": f"T{index + 1}",
                    "learned_accuracy": family_learned_value,
                    "accuracy_gain": family_gain,
                    "learned": {
                        "successes": round(family_learned_value * 100),
                        "total": 100,
                    },
                }
                for index, (family_learned_value, family_gain) in enumerate(
                    zip(learned_values, gain_values, strict=True)
                )
            ],
            "mean_learned_accuracy": learned,
            "mean_accuracy_gain": gain,
            "mean_learned": {
                "wilson_interval": {
                    "confidence_level": 0.95,
                    "lower": max(0.0, learned - 0.05),
                    "upper": min(1.0, learned + 0.05),
                }
            },
            "mean_accuracy_gain_interval": {
                "lower": max(-1.0, gain - 0.05),
                "upper": min(1.0, gain + 0.05),
            },
        },
        "cost": {
            "memory": {"hard_constraint_peak_bytes": 100},
            "median_probe_training_tokens_per_second": throughput,
        },
    }


def test_paired_family_comparison_matches_review_and_fails_closed() -> None:
    """Reviewer deltas reproduce the clustered interval and exact sign test."""
    differences = [0.05, 0.0, 0.05, 0.06, 0.09, 0.34, 0.03, 0.18, 0, 0, 0, 0]
    second = _decision_record(
        "second",
        family_learned=[0.5] * 12,
        family_gains=[0.0] * 12,
    )
    first = _decision_record(
        "first",
        family_learned=[0.5 + value for value in differences],
        family_gains=differences,
    )
    comparison = paired_family_comparison(first, second, "learned_accuracy", 0.95)
    interval = cast(dict[str, float | int], comparison["mean_difference_interval"])
    assert interval["mean"] == pytest.approx(0.0666666667)
    assert interval["lower"] == pytest.approx(0.0024784126)
    assert interval["upper"] == pytest.approx(0.1308549207)
    assert comparison["family_differences"] == pytest.approx(differences)
    assert comparison["sign_test"] == {
        "positive": 7,
        "negative": 0,
        "tied": 5,
        "non_tied": 7,
        "two_sided_p_value": 0.015625,
        "method": "exact two-sided binomial sign test with ties excluded",
    }

    malformed = copy.deepcopy(second)
    families = cast(
        list[dict[str, object]],
        cast(dict[str, object], malformed["task_stream_readiness"])["families"],
    )
    families[0]["family_id"] = "other"
    with pytest.raises(ValueError, match="paired family IDs differ"):
        paired_family_comparison(first, malformed, "learned_accuracy", 0.95)

    one_family = copy.deepcopy(first)
    cast(
        dict[str, object], one_family["task_stream_readiness"]
    )["families"] = cast(
        list[dict[str, object]],
        cast(dict[str, object], one_family["task_stream_readiness"])["families"],
    )[:1]
    with pytest.raises(ValueError, match="at least two families"):
        paired_family_comparison(one_family, one_family, "learned_accuracy", 0.95)


def test_decision_uses_clustered_practical_ties_and_rejects_hard_failures() -> None:
    """Paired family uncertainty, not pooled Wilson overlap, controls fallthrough."""
    settings = DecisionSettings(("mit",), 1000, 0.1, 0.1, 0.0, 0.02, True)
    differences = [0.05, 0.0, 0.05, 0.06, 0.09, 0.34, 0.03, 0.18, 0, 0, 0, 0]
    baseline = [0.5] * 12
    noisy_leader = _decision_record(
        "noisy-leader",
        learned=statistics.mean(
            [
                base + diff
                for base, diff in zip(baseline, differences, strict=True)
            ]
        ),
        gain=statistics.mean(differences),
        bpb=3.0,
        family_learned=[
            base + diff
            for base, diff in zip(baseline, differences, strict=True)
        ],
        family_gains=differences,
    )
    better_bpb = _decision_record(
        "better-bpb",
        learned=0.5,
        gain=0.0,
        bpb=1.0,
        family_learned=baseline,
        family_gains=[0.0] * 12,
    )
    separated = _decision_record(
        "separated",
        learned=0.75,
        gain=0.25,
        bpb=4.0,
        family_learned=[0.75] * 12,
        family_gains=[0.25] * 12,
    )
    hard_fail = _decision_record("hard-fail")
    cast(dict[str, object], hard_fail["identity"])["license"] = "proprietary"
    cast(dict[str, object], hard_fail["tokenizer_fitness"])[
        "exact_utf8_roundtrip"
    ] = False
    result = apply_decision_rule([better_bpb, hard_fail, noisy_leader], settings)
    assert result["selected_candidate"] == "better-bpb"
    comparisons = cast(
        list[dict[str, object]], result["pairwise_family_clustered_comparisons"]
    )
    learned_comparison = next(
        item for item in comparisons if item["metric"] == "learned_accuracy"
    )
    interval = cast(dict[str, float], learned_comparison["mean_difference_interval"])
    assert interval["lower"] <= settings.learnability_tie_margin
    assert interval["upper"] >= -settings.learnability_tie_margin
    rejected = cast(list[dict[str, object]], result["rejected_candidates"])
    reasons = cast(list[str], rejected[0]["reasons"])
    assert any("licence" in reason for reason in reasons)
    assert any("round-trip" in reason for reason in reasons)

    separated_result = apply_decision_rule([better_bpb, separated], settings)
    assert separated_result["selected_candidate"] == "separated"

    stricter = DecisionSettings(("mit",), 1000, 0.1, 0.9, 0.0, 0.02, True)
    strict_result = apply_decision_rule([better_bpb, noisy_leader], stricter)
    assert strict_result["decision_status"] == "no_eligible_candidate"
    assert strict_result["selected_candidate"] is None


def test_settings_reject_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("candidates: []\nmeasurement: {}\ndecision: {}\nunknown: true\n")
    with pytest.raises(ValueError, match="unknown root keys"):
        base_eval.load_base_model_evaluation_settings(config)


def test_repository_config_and_manifest_are_strict_stable_and_offline(
    tmp_path: Path,
) -> None:
    """The registered sweep config resolves to one immutable, reusable manifest."""
    config_path = Path("configs/base_model_candidates.yaml")
    settings = base_eval.load_base_model_evaluation_settings(config_path)
    assert len(settings.candidates) == 6
    assert settings.candidates[-1].loader == "aptivra_legacy"
    assert settings.measurement.seed == 20261014
    assert settings.measurement.test_items_per_family == 100
    assert settings.measurement.probe_steps == 200
    assert settings.measurement.probe_batch_size == 8
    assert settings.measurement.probe_learning_rates == (1e-5, 3e-5, 1e-4)
    assert settings.measurement.variance_seeds == (20261014, 20261015, 20261016)
    assert settings.decision.learnability_tie_margin == 0.02
    assert settings.decision.clustered_confidence_interval_intersects_tie_region is True
    assert settings.execution.resume is True

    manifest, manifest_sha256 = base_eval.prepare_run_manifest(
        config_path, tmp_path, settings
    )
    repeated, repeated_sha256 = base_eval.prepare_run_manifest(
        config_path, tmp_path, settings
    )
    assert repeated == manifest
    assert repeated_sha256 == manifest_sha256
    assert len(manifest_sha256) == 64
    candidates = cast(list[dict[str, object]], manifest["candidates"])
    assert all("resolution_failure" not in item for item in candidates)
    assert all(len(cast(str, item["resolved_revision"])) == 40 for item in candidates)

    manifest["evaluator_sha256"] = "0" * 64
    base_eval.write_json(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="frozen evidence is not overwritten"):
        base_eval.prepare_run_manifest(config_path, tmp_path, settings)


def test_derived_analysis_is_model_free_idempotent_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment analysis rewrites only derived artifacts, never measurements."""
    source = Path("experiments/base_model_candidates")
    destination = tmp_path / "evidence"
    shutil.copytree(source, destination)
    source_paths = [
        destination / "run_manifest.json",
        *sorted((destination / "candidates").glob("*.json")),
        *sorted((destination / "seed_runs").glob("*.json")),
    ]
    before = {path: base_eval.sha256_file(path) for path in source_paths}

    def no_model_work(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("derived analysis must not load or evaluate a model")

    monkeypatch.setattr(base_eval, "load_candidate", no_model_work)
    first_analysis, first_ranking = derive_existing_records_analysis(
        Path("configs/base_model_candidates.yaml"), destination
    )
    second_analysis, second_ranking = derive_existing_records_analysis(
        Path("configs/base_model_candidates.yaml"), destination
    )
    assert second_analysis == first_analysis
    assert second_ranking == first_ranking
    assert first_ranking["selected_candidate"] == "smollm2-360m"
    audit = cast(dict[str, object], first_analysis["amendment_2_decision_audit"])
    assert audit["selection_changed"] is False
    path_audit = cast(dict[str, object], audit["selected_vs_runner_up"])
    assert path_audit["learned_accuracy_tied"] is True
    assert path_audit["accuracy_gain_tied"] is True
    assert path_audit["deciding_criterion"] == "combined held-out bits per byte"
    saturation = cast(dict[str, object], first_analysis["saturated_family_analysis"])
    assert set(cast(list[str], saturation["family_ids"])) == {
        "T9_algo_synth",
        "T10_docstring_code",
        "T11_bug_fixer",
        "T12_type_inferrer",
    }
    aggregates = cast(list[dict[str, object]], saturation["candidate_aggregates"])
    candidate_360 = next(
        item for item in aggregates if item["candidate_id"] == "smollm2-360m"
    )
    all_pooled = cast(
        dict[str, object],
        cast(dict[str, object], candidate_360["all_families"])[
            "pooled_item_analysis"
        ],
    )
    unsaturated_pooled = cast(
        dict[str, object],
        cast(dict[str, object], candidate_360["excluding_saturated_families"])[
            "pooled_item_analysis"
        ],
    )
    assert (all_pooled["successes"], all_pooled["total"]) == (788, 1200)
    assert (unsaturated_pooled["successes"], unsaturated_pooled["total"]) == (
        388,
        800,
    )
    assert "optimistic" in cast(str, all_pooled["precision_note"])
    assert {path: base_eval.sha256_file(path) for path in source_paths} == before


def test_language_tokenizer_and_generation_metrics_share_real_inputs(
    tmp_path: Path,
) -> None:
    """Metric groups consume the same lossless tokenizer and stream contract."""
    settings = _settings(tmp_path)
    settings.heldout_text_path.write_text("A\n", encoding="utf-8")
    tokenizer = cast(Any, _ToyTokenizer())
    model = cast(Any, _ToyLearnableLM())
    stream = cast(Any, _ToyStream())

    quality = base_eval.evaluate_language_quality(
        model, tokenizer, settings, 16, torch.device("cpu")
    )
    assert cast(dict[str, object], quality["heldout_text"])["scored_bytes"] == 2
    assert cast(dict[str, object], quality["heldout_prompt_answers"])[
        "scored_bytes"
    ] == 1
    assert cast(dict[str, object], quality["combined"])["scored_bytes"] == 3

    fitness = base_eval.tokenizer_fitness(model, tokenizer, stream, ("PA",))
    assert fitness["exact_utf8_roundtrip"] is True
    assert fitness["eos_usable_as_answer_terminator"] is True
    assert fitness["mean_bytes_per_token"] == pytest.approx(1.0)

    benchmark = base_eval.benchmark_generation(
        model, tokenizer, stream, settings, 16, torch.device("cpu")
    )
    assert benchmark["prompts"] == 1
    repetitions = cast(list[dict[str, float | int]], benchmark["repetitions"])
    assert repetitions[0]["generated_tokens"] == 1
    assert cast(float, benchmark["median_tokens_per_second"]) > 0.0


class _NfcNormalizingTokenizer(_ToyTokenizer):
    backend_tokenizer = SimpleNamespace(normalizer="NFC()")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del ids, skip_special_tokens, clean_up_tokenization_spaces
        return "é"


def test_tokenizer_roundtrip_records_unicode_normalization_diagnosis() -> None:
    """A genuine NFC loss is distinguished from cleanup and special-token effects."""
    result = base_eval.tokenizer_fitness(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _NfcNormalizingTokenizer()),
        cast(Any, _ToyStream()),
        ("e\u0301",),
    )
    failure = cast(list[dict[str, object]], result["roundtrip_failures"])[0]
    assert failure["diagnosed_cause"] == "tokenizer_unicode_normalization_nfc"
    assert failure["add_special_tokens"] is False
    assert failure["decoded_with_cleanup_enabled"] == "é"
    assert failure["decoded_with_cleanup_disabled"] == "é"
    assert failure["input_code_points"] == ["U+0065", "U+0301"]


def test_hard_constraint_failure_skips_language_and_probe_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tokenizer rejection is persisted before any expensive candidate measurement."""
    settings = _settings(tmp_path)
    candidate = CandidateSpec("bad-tokenizer", "local", "a" * 40, "mit", "toy", False)
    model = _ToyLearnableLM()

    monkeypatch.setattr(base_eval, "model_context_length", lambda *args: 16)
    monkeypatch.setattr(base_eval, "_hidden_size", lambda *args: 5)
    monkeypatch.setattr(base_eval, "_architecture_type", lambda *args: "toy")
    monkeypatch.setattr(base_eval, "ContinualTaskStream", lambda **kwargs: _ToyStream())
    monkeypatch.setattr(
        base_eval,
        "tokenizer_fitness",
        lambda *args: {
            "exact_utf8_roundtrip": False,
            "eos_usable_as_answer_terminator": True,
            "roundtrip_failures": [{"diagnosed_cause": "tokenizer_unicode_normalization_nfc"}],
        },
    )
    monkeypatch.setattr(
        base_eval,
        "inspect_expert_attachment_structure",
        lambda *args: {"all_blocks_expose_feed_forward": True},
    )

    def expensive_work(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise AssertionError("expensive work must not run after a hard rejection")

    monkeypatch.setattr(base_eval, "evaluate_language_quality", expensive_work)
    monkeypatch.setattr(base_eval, "evaluate_task_readiness", expensive_work)
    record = evaluate_candidate(
        candidate,
        settings,
        loader=lambda *args: LoadedCandidate(
            cast(Any, model), cast(Any, _ToyTokenizer()), "a" * 40
        ),
        decision_settings=DecisionSettings(
            ("mit",), 10**12, 0.1, 0.0, 0.0, 0.02, True
        ),
    )
    assert record["status"] == "hard_constraint_rejected"
    assert record["probe_skipped"] is True


def test_public_orchestration_ranks_existing_records_and_child_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-level orchestration preserves evidence across isolated child outcomes."""
    config_path = Path("configs/base_model_candidates.yaml")
    output_dir = tmp_path / "evidence"
    completed = _decision_record("smollm2-135m")

    def fake_evaluate(
        settings: BaseModelEvaluationSettings,
        destination: Path,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        del destination
        assert [item.candidate_id for item in settings.candidates] == ["smollm2-135m"]
        record = copy.deepcopy(completed)
        record["manifest_sha256"] = kwargs["manifest_sha256"]
        record["correlation_id"] = kwargs["correlation_id"]
        return [record]

    monkeypatch.setattr(base_eval, "evaluate_all_candidates", fake_evaluate)
    records, ranking = base_eval.run_evaluation(
        config_path,
        output_dir,
        candidate_ids={"smollm2-135m"},
    )
    assert records[0]["status"] == "completed"
    assert ranking["selected_candidate"] == "smollm2-135m"

    base_eval.write_json(
        base_eval._candidate_record_path(output_dir, "smollm2-135m"), records[0]
    )
    existing = base_eval.rank_existing_candidate_records(config_path, output_dir)
    assert existing["selected_candidate"] == "smollm2-135m"
    assert len(cast(list[object], existing["rejected_candidates"])) == 4
    assert len(cast(list[object], existing["reference_records"])) == 1

    failure = base_eval.write_isolated_process_failure(
        config_path,
        output_dir,
        "qwen2.5-0.5b",
        returncode=137,
        message="worker terminated",
    )
    assert cast(dict[str, object], failure["failure"])["out_of_memory"] is True


def test_config_validation_and_device_fail_closed(tmp_path: Path) -> None:
    """Invalid loader, device, and offline mutable revisions fail before model work."""
    config = Path("configs/base_model_candidates.yaml").read_text(encoding="utf-8")
    bad_loader = tmp_path / "bad-loader.yaml"
    bad_loader.write_text(config.replace("loader: transformers", "loader: custom", 1))
    with pytest.raises(ValueError, match="loader must be"):
        base_eval.load_base_model_evaluation_settings(bad_loader)

    with pytest.raises(ValueError, match="must be auto"):
        base_eval.resolve_device("metal")

    settings = replace(_settings(tmp_path), local_files_only=True)
    candidate = CandidateSpec("mutable", "owner/model", None, "mit", None, False)
    with pytest.raises(ValueError, match="immutable revision"):
        base_eval._resolve_preflight_candidate(candidate, settings)


def test_analytic_expectation_is_not_rounded() -> None:
    assert math.log2(4.0) / len("é".encode()) == 1.0
