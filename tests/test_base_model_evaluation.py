"""Offline proof tests for pretrained-base measurement and selection."""

from __future__ import annotations

import copy
import json
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
import scripts.evaluate_base_model_candidates as evaluator_cli
import scripts.run_amendment_4e as amendment_4e_cli
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
from scripts.evaluate_base_model_candidates import _equivalence_allows_sweep


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
        return SimpleNamespace(logits=logits, loss=loss, last_hidden_state=logits)

    def get_decoder(self) -> nn.Module:
        return self

    def get_output_embeddings(self) -> nn.Module:
        return nn.Identity()


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


def test_batched_accuracy_persists_per_item_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = [
        TaskInstance("item-1", "toy", "P", "A"),
        TaskInstance("item-2", "toy", "P", "A"),
    ]
    predictor = SimpleNamespace(
        generate_batch=lambda *args, **kwargs: [
            SimpleNamespace(text="pass"),
            SimpleNamespace(text="fail"),
        ]
    )
    monkeypatch.setattr(
        base_eval.ContinualEvalHarness,
        "verify_generated_output",
        lambda family, instance, text: SimpleNamespace(passed=text == "pass"),
    )
    record = base_eval.evaluate_batched_accuracy(
        cast(Any, predictor),
        cast(Any, SimpleNamespace(stop_token_ids=(10,))),
        instances,
        batch_size=2,
        confidence_level=0.95,
    )
    assert record["successes"] == 1
    assert record["item_outcomes"] == [
        {"task_id": "item-1", "passed": True},
        {"task_id": "item-2", "passed": False},
    ]


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


def test_target_only_loss_and_accumulation_match_legacy_batch() -> None:
    """Target slicing happens before the head and preserves an uneven-token update."""
    torch.manual_seed(11)
    config = GPT2Config(
        vocab_size=5,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=4,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    legacy = cast(Any, GPT2LMHeadModel(config))
    amended = copy.deepcopy(legacy)
    legacy.train()
    amended.train()
    tokenizer = cast(Any, _ToyTokenizer())
    instances = [
        TaskInstance(f"item-{index}", "toy", "P" * (index + 1), "A" * (index % 2 + 1))
        for index in range(4)
    ]
    legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=1e-3)
    amended_optimizer = torch.optim.AdamW(amended.parameters(), lr=1e-3)
    legacy_before = base_eval._parameter_snapshots(legacy)
    amended_before = base_eval._parameter_snapshots(amended)
    head_input_rows: list[int] = []
    hook = amended.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: head_input_rows.append(int(args[0].shape[0]))
    )
    legacy_loss = base_eval._optimizer_step_legacy(
        legacy,
        tokenizer,
        instances,
        16,
        torch.device("cpu"),
        legacy_optimizer,
    )
    amended_step = base_eval._optimizer_step_target_only(
        amended,
        tokenizer,
        instances,
        16,
        torch.device("cpu"),
        amended_optimizer,
        base_eval.TrainingRecipe(2, False),
    )
    hook.remove()
    difference = base_eval._parameter_update_difference(
        legacy,
        amended,
        legacy_before,
        amended_before,
    )
    assert legacy_loss == pytest.approx(cast(float, amended_step["loss"]), abs=5e-6)
    assert difference["maximum_absolute"] <= 1e-6
    assert difference["relative_l2"] <= 1e-4
    assert len(head_input_rows) == 2
    assert sum(head_input_rows) == amended_step["supervised_tokens"]
    full_micro_batch_positions = 2 * max(
        len(item.prompt) + len(item.training_target()) + 1 for item in instances
    )
    assert all(rows < full_micro_batch_positions for rows in head_input_rows)


def test_target_only_loss_rejects_noncontiguous_supervision() -> None:
    """Slice-based target selection fails closed if the target-layout invariant breaks."""
    input_ids = torch.tensor([[1, 2, 3, 2, 4]])
    labels = torch.tensor([[-100, 2, -100, 2, -100]])
    attention_mask = torch.ones_like(input_ids)
    with pytest.raises(ValueError, match="contiguous"):
        base_eval._target_only_loss_mean(
            cast(Any, _ToyLearnableLM()), input_ids, labels, attention_mask
        )


def test_recipe_selection_descends_then_uses_checkpoint_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OOM attempts descend deterministically and checkpoint only after size one."""
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=4,
        probe_micro_batch_sizes=(4, 2, 1),
    )
    instances = [TaskInstance(f"item-{index}", "toy", "P", "A") for index in range(4)]
    stream = cast(
        Any,
        SimpleNamespace(
            families=[SimpleNamespace(family_id="toy")],
            get_split=lambda family_id: TaskStreamSplit(instances, instances, instances),
        ),
    )
    attempts: list[tuple[int, bool]] = []

    def fail_until_checkpoint(*args: object, **kwargs: object) -> dict[str, float | int]:
        recipe = cast(base_eval.TrainingRecipe, args[6])
        attempts.append((recipe.micro_batch_size, recipe.gradient_checkpointing))
        if not recipe.gradient_checkpointing:
            raise RuntimeError("MPS backend out of memory")
        return {
            "loss": 1.0,
            "processed_tokens": 1,
            "supervised_tokens": 1,
            "trained_bytes": 1,
            "accumulation_steps": 4,
        }

    monkeypatch.setattr(base_eval, "_optimizer_step_target_only", fail_until_checkpoint)
    monkeypatch.setattr(base_eval, "_set_gradient_checkpointing", lambda *args: None)
    decision = DecisionSettings(("mit",), 10**12, 0.1, 0.0, 0.0, 0.02, True)
    with base_eval.MemoryPeakSampler(torch.device("cpu"), 0.01) as guard:
        selected, evidence = base_eval.select_training_recipe(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            stream,
            settings,
            decision,
            16,
            torch.device("cpu"),
            resource_guard=guard,
        )
    assert attempts == [(4, False), (2, False), (1, False), (1, True)]
    assert selected == base_eval.TrainingRecipe(1, True)
    rows = cast(list[dict[str, object]], evidence["attempts"])
    assert [row["feasible"] for row in rows] == [False, False, False, True]


def test_recipe_selection_honestly_rejects_checkpointed_micro_batch_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path), probe_batch_size=1, probe_micro_batch_sizes=(1,)
    )
    instance = TaskInstance("item", "toy", "P", "A")
    stream = cast(
        Any,
        SimpleNamespace(
            families=[SimpleNamespace(family_id="toy")],
            get_split=lambda family_id: TaskStreamSplit(
                [instance], [instance], [instance]
            ),
        ),
    )
    monkeypatch.setattr(
        base_eval,
        "_optimizer_step_target_only",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("MPS backend out of memory")
        ),
    )
    monkeypatch.setattr(base_eval, "_set_gradient_checkpointing", lambda *args: None)
    decision = DecisionSettings(("mit",), 10**12, 0.1, 0.0, 0.0, 0.02, True)
    with base_eval.MemoryPeakSampler(torch.device("cpu"), 0.01) as guard:
        selected, evidence = base_eval.select_training_recipe(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            stream,
            settings,
            decision,
            16,
            torch.device("cpu"),
            resource_guard=guard,
        )
    assert selected is None
    assert len(cast(list[object], evidence["attempts"])) == 2


def test_swap_sampler_records_environment_signal_without_rejecting_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter((100, 151, 151))
    monkeypatch.setattr(base_eval, "system_swap_used_bytes", lambda: next(values))
    sampler = base_eval.MemoryPeakSampler(
        torch.device("cpu"), 10.0, maximum_swap_growth_bytes=50
    )
    sampler.raise_if_swap_limit_exceeded()
    record = sampler.record()
    assert record["system_swap_before_bytes"] == 100
    assert record["system_swap_after_bytes"] == 151
    assert record["system_swap_peak_growth_bytes"] == 51


def test_system_free_memory_percentage_parses_darwin_and_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment pressure telemetry is deterministic on both supported hosts."""
    monkeypatch.setattr(base_eval.sys, "platform", "darwin")
    monkeypatch.setattr(
        base_eval.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="System-wide memory free percentage: 37%\n"
        ),
    )
    assert base_eval.system_free_memory_percentage() == 37.0

    class FakeMeminfo:
        def is_file(self) -> bool:
            return True

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return (
                "MemTotal: 1000 kB\nMemAvailable: 125 kB\n"
                "SwapTotal: 500 kB\nSwapFree: 125 kB\n"
            )

    monkeypatch.setattr(base_eval.sys, "platform", "linux")
    monkeypatch.setattr(base_eval, "Path", lambda _: FakeMeminfo())
    assert base_eval.system_free_memory_percentage() == 12.5
    assert base_eval.system_swap_used_bytes() == 375 * 1024


def test_system_memory_telemetry_fails_closed_on_malformed_or_unsupported_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_eval.sys, "platform", "darwin")
    monkeypatch.setattr(
        base_eval.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="unparseable"),
    )
    with pytest.raises(RuntimeError, match="vm.swapusage"):
        base_eval.system_swap_used_bytes()
    with pytest.raises(RuntimeError, match="memory_pressure"):
        base_eval.system_free_memory_percentage()

    monkeypatch.setattr(base_eval.sys, "platform", "unsupported")
    monkeypatch.setattr(
        base_eval,
        "Path",
        lambda _: SimpleNamespace(is_file=lambda: False),
    )
    with pytest.raises(RuntimeError, match="swap measurement is unsupported"):
        base_eval.system_swap_used_bytes()
    with pytest.raises(RuntimeError, match="memory-pressure measurement is unsupported"):
        base_eval.system_free_memory_percentage()


def test_mps_sampler_records_driver_peak_and_degrades_on_driver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter((10, 20))
    monkeypatch.setattr(
        base_eval.torch.mps, "driver_allocated_memory", lambda: next(values)
    )
    sampler = base_eval.MemoryPeakSampler(
        torch.device("mps"), 10.0, swap_baseline_bytes=0
    )
    monkeypatch.setattr(base_eval, "system_swap_used_bytes", lambda: 0)
    sampler.raise_if_swap_limit_exceeded()
    assert sampler.record()["mps_driver_peak_bytes"] == 20

    monkeypatch.setattr(
        base_eval.torch.mps,
        "driver_allocated_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("driver unavailable")),
    )
    unavailable = base_eval.MemoryPeakSampler(
        torch.device("mps"), 10.0, swap_baseline_bytes=0
    )
    assert unavailable.record()["mps_driver_peak_bytes"] is None


def test_cuda_sampler_uses_allocated_and_reserved_peaks_for_fairness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA feasibility excludes host RSS and uses PyTorch's device peak counters."""
    resets: list[torch.device] = []
    monkeypatch.setattr(base_eval, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(base_eval, "_rss_high_water_bytes", lambda: 10_000)
    monkeypatch.setattr(base_eval.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        base_eval.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: resets.append(device),
    )
    monkeypatch.setattr(base_eval.torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(base_eval.torch.cuda, "memory_reserved", lambda device: 200)
    monkeypatch.setattr(
        base_eval.torch.cuda, "max_memory_allocated", lambda device: 300
    )
    monkeypatch.setattr(
        base_eval.torch.cuda, "max_memory_reserved", lambda device: 400
    )
    device = torch.device("cuda")
    with base_eval.MemoryPeakSampler(device, 10.0) as sampler:
        sampler.raise_if_swap_limit_exceeded()
    record = sampler.record()
    assert resets == [device]
    assert record["cuda_max_memory_allocated_bytes"] == 300
    assert record["cuda_max_memory_reserved_bytes"] == 400
    assert record["hard_constraint_peak_bytes"] == 400
    assert record["hard_constraint_metric"] == (
        "max(CUDA max_memory_allocated, CUDA max_memory_reserved)"
    )


def test_cuda_sampler_degrades_to_rss_when_driver_metrics_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_eval, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(base_eval, "_rss_high_water_bytes", lambda: 123)
    monkeypatch.setattr(base_eval.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(base_eval.torch.cuda, "reset_peak_memory_stats", lambda device: None)

    def unavailable(device: torch.device) -> int:
        del device
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(base_eval.torch.cuda, "memory_allocated", unavailable)
    monkeypatch.setattr(base_eval.torch.cuda, "memory_reserved", unavailable)
    monkeypatch.setattr(base_eval.torch.cuda, "max_memory_allocated", unavailable)
    monkeypatch.setattr(base_eval.torch.cuda, "max_memory_reserved", unavailable)
    with base_eval.MemoryPeakSampler(torch.device("cuda"), 10.0) as sampler:
        pass
    assert sampler.record()["hard_constraint_peak_bytes"] == 123


def test_device_resolution_prefers_real_accelerators_and_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_eval.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(base_eval.torch.cuda, "is_available", lambda: True)
    assert base_eval.resolve_device("auto").type == "cuda"
    assert base_eval.resolve_device("cuda").type == "cuda"
    monkeypatch.setattr(base_eval.torch.cuda, "is_available", lambda: False)
    assert base_eval.resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        base_eval.resolve_device("cuda")


def test_host_pristine_placement_does_not_remove_a_unified_memory_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        pristine_model_device="host",
        target_only_training_loss=False,
        probe_steps=1,
        probe_warmup_steps=0,
    )
    model = _ToyLearnableLM()
    digest_before = base_eval.model_state_digest(cast(Any, model))
    monkeypatch.setattr(
        base_eval,
        "evaluate_batched_accuracy",
        lambda *args, **kwargs: _accuracy_record_for_test(),
    )
    monkeypatch.setattr(
        base_eval,
        "student_t_interval",
        lambda values, confidence: {
            "mean": statistics.mean(values),
            "lower": 0.0,
            "upper": 1.0,
            "confidence_level": confidence,
            "clusters": len(values),
            "method": "test",
        },
    )
    run_learnability_probe(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        0,
        settings,
        16,
        torch.device("cpu"),
    )
    assert base_eval.model_state_digest(cast(Any, model)) == digest_before
    memory = base_eval.full_finetuning_memory_feasibility(
        10, settings, DecisionSettings(("mit",), 10_000, 0.25, 0.0, 0.0, 0.02, True)
    )
    assert cast(dict[str, int], memory["components_bytes"])[
        "pristine_base_weights_on_candidate_device"
    ] == 40
    assert memory["bytes_per_parameter"] == 20
    released = base_eval.full_finetuning_memory_feasibility(
        10,
        replace(settings, release_pristine_before_training=True),
        DecisionSettings(("mit",), 10_000, 0.25, 0.0, 0.0, 0.02, True),
    )
    assert cast(dict[str, int], released["components_bytes"])[
        "pristine_base_weights_on_candidate_device"
    ] == 0
    assert released["bytes_per_parameter"] == 16


def test_disk_snapshot_restores_one_reusable_training_model(tmp_path: Path) -> None:
    model = _ToyLearnableLM()
    expected = base_eval.model_state_digest(cast(Any, model))
    snapshot = tmp_path / "pristine.pt"
    assert (
        base_eval._write_pristine_snapshot(
            cast(Any, model), snapshot, torch.device("cpu")
        )
        == expected
    )
    with torch.no_grad():
        model.bias.add_(9.0)
    assert base_eval.model_state_digest(cast(Any, model)) != expected
    base_eval._restore_pristine_snapshot(
        cast(Any, model), snapshot, torch.device("cpu"), expected
    )
    assert base_eval.model_state_digest(cast(Any, model)) == expected


def test_task_readiness_reuses_one_model_and_restores_pristine_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        release_pristine_before_training=True,
        probe_steps=1,
        probe_warmup_steps=0,
        probe_learning_rates=(0.01,),
    )
    model = _ToyLearnableLM()
    expected = base_eval.model_state_digest(cast(Any, model))
    snapshot = tmp_path / "pristine.pt"
    base_eval._write_pristine_snapshot(
        cast(Any, model), snapshot, torch.device("cpu")
    )
    monkeypatch.setattr(
        base_eval,
        "evaluate_batched_accuracy",
        lambda *args, **kwargs: _accuracy_record_for_test(),
    )
    monkeypatch.setattr(
        base_eval,
        "student_t_interval",
        lambda values, confidence: {
            "mean": statistics.mean(values),
            "lower": 0.0,
            "upper": 1.0,
            "confidence_level": confidence,
            "clusters": len(values),
            "method": "test",
        },
    )
    zero_shot = {
        "families": [
            {"family_id": "toy", "zero_shot": _accuracy_record_for_test()}
        ],
        "mean_zero_shot_accuracy": 0.0,
    }
    result = base_eval.evaluate_task_readiness(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        settings,
        16,
        torch.device("cpu"),
        zero_shot_readiness=zero_shot,
        recipe=base_eval.TrainingRecipe(1, False),
        pristine_snapshot_path=snapshot,
        pristine_snapshot_digest=expected,
    )
    assert result["mean_zero_shot_accuracy"] == 0.0
    assert base_eval.model_state_digest(cast(Any, model)) == expected


def test_recipe_selection_restores_snapshot_backed_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        probe_steps=1,
        probe_warmup_steps=0,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    stream = cast(
        Any,
        SimpleNamespace(
            families=[SimpleNamespace(family_id="toy")],
            get_split=lambda family_id: TaskStreamSplit(
                [instance], [instance], [instance]
            ),
        ),
    )
    model = _ToyLearnableLM()
    expected = base_eval.model_state_digest(cast(Any, model))
    snapshot = tmp_path / "pristine.pt"
    base_eval._write_pristine_snapshot(
        cast(Any, model), snapshot, torch.device("cpu")
    )
    monkeypatch.setattr(base_eval, "_set_gradient_checkpointing", lambda *args: None)
    with base_eval.MemoryPeakSampler(torch.device("cpu"), 10.0) as guard:
        selected, _ = base_eval.select_training_recipe(
            cast(Any, model),
            cast(Any, _ToyTokenizer()),
            stream,
            settings,
            DecisionSettings(("mit",), 10**12, 0.1, 0.0, 0.0, 0.02, True),
            16,
            torch.device("cpu"),
            resource_guard=guard,
            pristine_snapshot_path=snapshot,
            pristine_snapshot_digest=expected,
        )
    assert selected == base_eval.TrainingRecipe(1, False)
    assert base_eval.model_state_digest(cast(Any, model)) == expected


def test_amendment_4e_comparison_requires_exact_items_and_bounds_losses() -> None:
    reference = {
        "status": "completed",
        "candidate_id": "toy",
        "task_stream_readiness": {
            "families": [
                {
                    "family_id": "toy",
                    "initial_loss": 1.0,
                    "final_loss": 0.5,
                    "learned": {
                        "accuracy": 1.0,
                        "successes": 1,
                        "total": 1,
                        "item_outcomes": [{"task_id": "toy-1", "passed": True}],
                    },
                }
            ]
        },
        "cost": {"memory": {"hard_constraint_peak_bytes": 1}},
    }
    candidate = copy.deepcopy(reference)
    rows = cast(
        list[dict[str, object]],
        cast(dict[str, object], candidate["task_stream_readiness"])["families"],
    )
    rows[0]["final_loss"] = 0.500005
    comparison = base_eval.compare_amendment_4e_records(
        reference, candidate, include_item_outcomes=True
    )
    assert comparison["passed"] is True
    learned = cast(dict[str, object], rows[0]["learned"])
    learned["item_outcomes"] = [{"task_id": "toy-1", "passed": False}]
    mismatch = base_eval.compare_amendment_4e_records(
        reference, candidate, include_item_outcomes=True
    )
    assert mismatch["passed"] is False
    incomplete = base_eval.compare_amendment_4e_records(
        reference,
        {"status": "failed", "candidate_id": "toy"},
        include_item_outcomes=False,
    )
    assert incomplete["passed"] is False
    assert incomplete["record_statuses"] == {
        "reference": "completed",
        "candidate": "failed",
    }


def test_amendment_4e_foreground_gates_fail_closed() -> None:
    amendment_4e_cli._require_passed({"status": "passed"})
    with pytest.raises(SystemExit) as failure:
        amendment_4e_cli._require_passed({"status": "failed"})
    assert failure.value.code == 2


def test_environment_status_records_are_durable_and_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retryable and exhausted environment outcomes persist without an OOM verdict."""
    config = Path("configs/base_model_candidates_amendment_4c.yaml")
    manifest = {"correlation_id": "environment-test"}
    manifest_sha256 = "a" * 64
    monkeypatch.setattr(
        base_eval,
        "prepare_run_manifest",
        lambda *args, **kwargs: (manifest, manifest_sha256),
    )
    attempt = {"attempt": 1, "status": "environment_invalid"}
    retryable = base_eval.write_environment_status(
        config,
        tmp_path,
        "gpt2",
        status="environment_invalid",
        attempts=[attempt],
    )
    assert retryable["status"] == "environment_invalid"
    assert cast(dict[str, object], retryable["failure"])["out_of_memory"] is False
    final = base_eval.write_environment_status(
        config,
        tmp_path,
        "gpt2",
        status="not_measured_environment",
        attempts=[attempt, {"attempt": 2, "status": "environment_invalid"}],
        variance_seed=20261015,
    )
    assert final["status"] == "not_measured_environment"
    assert base_eval._candidate_record_path(tmp_path, "gpt2").is_file()
    assert base_eval._seed_record_path(tmp_path, "gpt2", 20261015).is_file()
    with pytest.raises(ValueError, match="unsupported environment status"):
        base_eval.write_environment_status(
            config,
            tmp_path,
            "gpt2",
            status="rejected",
            attempts=[],
        )


class _FakeCandidateProcess:
    def __init__(
        self, *, completed: bool = False, polls_before_complete: int | None = None
    ) -> None:
        self.returncode: int | None = 0 if completed else None
        self.polls_before_complete = polls_before_complete

    def poll(self) -> int | None:
        if self.returncode is None and self.polls_before_complete is not None:
            if self.polls_before_complete == 0:
                self.returncode = 0
            else:
                self.polls_before_complete -= 1
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout: float) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def communicate(self) -> tuple[str, str]:
        return "", ""


def test_swap_growth_keeps_valid_result_and_compromises_only_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whole-system swap preserves arithmetic evidence but excludes throughput."""
    process = _FakeCandidateProcess(polls_before_complete=1)
    monkeypatch.setattr(
        evaluator_cli,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": True, "samples": []},
    )
    monkeypatch.setattr(
        evaluator_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    swap_values = iter((100, 200, 200))
    free_values = iter((50.0, 50.0, 50.0))
    monkeypatch.setattr(
        evaluator_cli, "system_swap_used_bytes", lambda: next(swap_values)
    )
    monkeypatch.setattr(
        evaluator_cli, "system_free_memory_percentage", lambda: next(free_values)
    )
    monitors: list[dict[str, object]] = []
    attached: list[dict[str, object]] = []
    monkeypatch.setattr(
        evaluator_cli,
        "write_json",
        lambda path, payload: monitors.append(dict(payload)),
    )
    monkeypatch.setattr(evaluator_cli, "_other_processes_above_rss", lambda threshold: [])
    monkeypatch.setattr(
        evaluator_cli,
        "attach_environment_attempts",
        lambda output, candidate, attempts, **kwargs: attached.extend(
            dict(item) for item in attempts
        ),
    )

    evaluator_cli._run_child(
        tmp_path / "config.yaml",
        tmp_path / "evidence",
        "candidate",
        resume=False,
        timeout_seconds=10.0,
        maximum_swap_growth_bytes=50,
        swap_poll_interval_seconds=0.01,
        minimum_free_memory_percent=10.0,
        environment_recovery_timeout_seconds=1.0,
        environment_max_attempts=2,
    )

    assert [item["status"] for item in monitors] == ["completed"]
    assert attached[0]["throughput_compromised"] is True
    assert process.returncode == 0


def test_quiet_machine_check_waits_for_heavy_process_and_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator_cli, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(evaluator_cli, "system_free_memory_percentage", lambda: 80.0)
    heavy = iter(
        ([{"pid": 99, "rss_bytes": 3_000_000_000, "executable": "other"}], [])
    )
    monkeypatch.setattr(
        evaluator_cli, "_other_processes_above_rss", lambda threshold: next(heavy)
    )
    monkeypatch.setattr(evaluator_cli.time, "sleep", lambda seconds: None)
    recovered = evaluator_cli._wait_until_environment_normal(
        minimum_free_memory_percent=70.0,
        timeout_seconds=10.0,
        poll_seconds=0.1,
        maximum_other_process_rss_bytes=2_147_483_648,
    )
    assert recovered["normal"] is True
    assert len(cast(list[object], recovered["samples"])) == 2

    monkeypatch.setattr(
        evaluator_cli,
        "_other_processes_above_rss",
        lambda threshold: [
            {"pid": 99, "rss_bytes": threshold + 1, "executable": "other"}
        ],
    )
    timed_out = evaluator_cli._wait_until_environment_normal(
        minimum_free_memory_percent=70.0,
        timeout_seconds=0.0,
        poll_seconds=0.1,
        maximum_other_process_rss_bytes=2_147_483_648,
    )
    assert timed_out["normal"] is False


def test_quiet_machine_timeout_never_starts_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator_cli,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": False, "samples": [{"free_memory_percent": 60.0}]},
    )
    monkeypatch.setattr(
        evaluator_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("candidate process must not start"),
    )
    statuses: list[str] = []
    monkeypatch.setattr(evaluator_cli, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluator_cli,
        "write_environment_status",
        lambda *args, **kwargs: statuses.append(cast(str, kwargs["status"])),
    )
    evaluator_cli._run_child(
        tmp_path / "config.yaml",
        tmp_path / "evidence",
        "candidate",
        resume=False,
        timeout_seconds=10.0,
        maximum_swap_growth_bytes=50,
        swap_poll_interval_seconds=0.01,
        minimum_free_memory_percent=10.0,
        environment_recovery_timeout_seconds=1.0,
        environment_max_attempts=2,
        prestart_minimum_free_memory_percent=70.0,
        prestart_max_other_process_rss_bytes=2_147_483_648,
        prestart_timeout_seconds=0.0,
        prestart_poll_interval_seconds=0.1,
    )
    assert statuses == ["not_measured_environment"]


def test_critical_memory_floor_aborts_as_environment_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical host pressure stops the active child and invalidates the run."""
    process = _FakeCandidateProcess()
    monkeypatch.setattr(
        evaluator_cli,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": True, "samples": []},
    )
    monkeypatch.setattr(
        evaluator_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    swap_values = iter((100, 100, 100))
    monkeypatch.setattr(
        evaluator_cli, "system_swap_used_bytes", lambda: next(swap_values)
    )
    monkeypatch.setattr(
        evaluator_cli, "system_free_memory_percentage", lambda: 4.0
    )
    monkeypatch.setattr(evaluator_cli, "_other_processes_above_rss", lambda threshold: [])
    statuses: list[str] = []
    monkeypatch.setattr(evaluator_cli, "write_json", lambda *args: None)
    monkeypatch.setattr(
        evaluator_cli,
        "write_environment_status",
        lambda *args, **kwargs: statuses.append(cast(str, kwargs["status"])),
    )

    evaluator_cli._run_child(
        tmp_path / "config.yaml",
        tmp_path / "evidence",
        "candidate",
        resume=False,
        timeout_seconds=10.0,
        maximum_swap_growth_bytes=50,
        swap_poll_interval_seconds=0.01,
        minimum_free_memory_percent=10.0,
        environment_recovery_timeout_seconds=1.0,
        environment_max_attempts=2,
    )

    assert statuses == ["environment_invalid"]
    assert process.returncode == -15


@pytest.mark.parametrize("abort_kind", ["wall_clock", "foreign_process"])
def test_runtime_safety_stop_aborts_as_environment_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abort_kind: str
) -> None:
    process = _FakeCandidateProcess()
    monkeypatch.setattr(
        evaluator_cli,
        "_wait_until_environment_normal",
        lambda **kwargs: {"normal": True, "samples": []},
    )
    monkeypatch.setattr(evaluator_cli.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(evaluator_cli, "system_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(evaluator_cli, "system_free_memory_percentage", lambda: 80.0)
    monkeypatch.setattr(
        evaluator_cli,
        "_other_processes_above_rss",
        lambda threshold: ([{"pid": 99}] if abort_kind == "foreign_process" else []),
    )
    monkeypatch.setattr(evaluator_cli, "write_json", lambda *args: None)
    statuses: list[str] = []
    monkeypatch.setattr(
        evaluator_cli,
        "write_environment_status",
        lambda *args, **kwargs: statuses.append(cast(str, kwargs["status"])),
    )
    evaluator_cli._run_child(
        tmp_path / "config.yaml",
        tmp_path / "evidence",
        "candidate",
        resume=False,
        timeout_seconds=-1.0 if abort_kind == "wall_clock" else 10.0,
        maximum_swap_growth_bytes=50,
        swap_poll_interval_seconds=0.01,
        minimum_free_memory_percent=5.0,
        environment_recovery_timeout_seconds=1.0,
        environment_max_attempts=2,
    )
    assert statuses == ["environment_invalid"]
    assert process.returncode == -15


def test_tie_breaking_ignores_swap_compromised_throughput() -> None:
    compromised = _decision_record("compromised", throughput=1000.0, parameters=200)
    compromised["throughput_compromised"] = True
    valid = _decision_record("valid", throughput=1.0, parameters=300)
    ranking = base_eval.apply_decision_rule(
        [compromised, valid],
        DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
    )
    rows = cast(list[dict[str, object]], ranking["ranked_candidates"])
    assert rows[0]["candidate_id"] == "valid"
    compromised_row = next(row for row in rows if row["candidate_id"] == "compromised")
    assert compromised_row["training_tokens_per_second"] is None
    assert compromised_row["throughput_tie_breaker_status"] == "excluded_swap_compromised"


def test_equivalence_proof_compares_losses_and_parameter_updates(
    tmp_path: Path,
) -> None:
    """The pre-sweep proof measures each registered tolerance in train mode."""
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=4,
        probe_micro_batch_sizes=(4, 2, 1),
        probe_learning_rates=(1e-3,),
        equivalence_steps=2,
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=4,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    with base_eval.MemoryPeakSampler(torch.device("cpu"), 0.01) as guard:
        proof = base_eval.prove_training_recipe_equivalence(
            cast(Any, model),
            cast(Any, _ToyTokenizer()),
            cast(Any, _ToyStream()),
            settings,
            16,
            torch.device("cpu"),
            base_eval.TrainingRecipe(2, False),
            resource_guard=guard,
        )
    assert proof["passed"] is True
    assert proof["training_mode"] is True
    assert proof["dropout_disabled_for_proof"] is False
    rows = cast(list[dict[str, object]], proof["steps"])
    assert len(rows) == 2
    assert all(row["passed"] is True for row in rows)


def test_amendment_4a_float64_gradient_equivalence_exercises_accumulation(
    tmp_path: Path,
) -> None:
    """Exact proof compares the changed quantity at selected and size-one batches."""
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=4,
        probe_micro_batch_sizes=(4, 2, 1),
        equivalence_exact_loss_absolute_tolerance=1e-12,
        equivalence_exact_gradient_relative_l2_tolerance=1e-10,
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=4,
            pad_token_id=0,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
        )
    )
    instances = [
        TaskInstance(f"item-{index}", "toy", "P" * (index + 1), "A")
        for index in range(4)
    ]
    proof = base_eval.prove_exact_gradient_equivalence(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        instances,
        settings,
        16,
        base_eval.TrainingRecipe(2, False),
    )
    assert proof["passed"] is True
    assert proof["device"] == "cpu"
    assert proof["dtype"] == "float64"
    assert proof["dropout_disabled"] is True
    rows = cast(list[dict[str, object]], proof["comparisons"])
    assert [row["micro_batch_size"] for row in rows] == [2, 1]
    assert all(cast(float, row["absolute_loss_difference"]) <= 1e-12 for row in rows)
    assert all(
        cast(float, row["gradient_relative_l2_difference"]) <= 1e-10
        for row in rows
    )


def test_amendment_4a_moves_off_mps_before_float64_cast() -> None:
    """The exact proof must never request an unsupported float64 MPS tensor."""

    class MoveRecorder:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def to(self, **kwargs: object) -> MoveRecorder:
            self.calls.append(kwargs)
            return self

    model = MoveRecorder()
    base_eval._move_model_for_measurement(
        cast(Any, model), torch.device("cpu"), torch.float64
    )
    assert model.calls == [{"device": torch.device("cpu")}, {"dtype": torch.float64}]


def test_amendment_4a_noise_floor_is_batchwise_and_zero_is_strict() -> None:
    rows = [
        {
            "control_gradient_relative_l2": 1e-5,
            "target_gradient_relative_l2": 2e-5,
            "control_update_relative_l2": 3e-5,
            "target_update_relative_l2": 6e-5,
        },
        {
            "control_gradient_relative_l2": 0.0,
            "target_gradient_relative_l2": 1e-16,
            "control_update_relative_l2": 1e-4,
            "target_update_relative_l2": 1e-4,
        },
    ]
    assessed = base_eval.assess_noise_floor(rows, multiplier=2.0)
    assert assessed["passed"] is False
    assessed_rows = cast(list[dict[str, object]], assessed["batches"])
    assert assessed_rows[0]["passed"] is True
    assert assessed_rows[1]["gradient_passed"] is False
    assert assessed_rows[1]["passed"] is False


def test_amendment_4a_trajectory_gate_checks_noise_and_each_family() -> None:
    passed = base_eval.assess_training_trajectory(
        legacy_losses=[2.0, 1.5, 1.0],
        control_losses=[2.01, 1.49, 1.01],
        target_losses=[2.02, 1.5, 0.99],
        legacy_accuracy={"family-a": 0.5, "family-b": 0.75},
        target_accuracy={"family-a": 0.52, "family-b": 0.74},
        noise_multiplier=2.0,
        accuracy_tie_margin=0.02,
    )
    assert passed["passed"] is True
    failed = base_eval.assess_training_trajectory(
        legacy_losses=[2.0, 1.5, 1.0],
        control_losses=[2.01, 1.49, 1.01],
        target_losses=[2.0, 1.5, 1.0],
        legacy_accuracy={"family-a": 0.5, "family-b": 0.75},
        target_accuracy={"family-a": 0.53, "family-b": 0.75},
        noise_multiplier=2.0,
        accuracy_tie_margin=0.02,
    )
    assert failed["trajectory_passed"] is True
    assert failed["accuracy_passed"] is False
    assert failed["passed"] is False


def test_amendment_4a_noise_measurement_compares_all_three_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_noise_floor_batches=3,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    measurements: list[base_eval._VectorMeasurement] = []
    for _ in range(3):
        measurements.extend(
            [
                base_eval._VectorMeasurement(
                    1.0, [torch.tensor([1.0])], [torch.tensor([0.1])], 0
                ),
                base_eval._VectorMeasurement(
                    1.0,
                    [torch.tensor([1.0001])],
                    [torch.tensor([0.10001])],
                    0,
                ),
                base_eval._VectorMeasurement(
                    1.0,
                    [torch.tensor([1.0001])],
                    [torch.tensor([0.10001])],
                    0,
                ),
            ]
        )
    iterator = iter(measurements)
    monkeypatch.setattr(
        base_eval, "_measure_gradient_and_update", lambda *args, **kwargs: next(iterator)
    )
    proof = base_eval.measure_noise_floor(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        [[instance], [instance], [instance]],
        settings,
        16,
        torch.device("mps"),
        base_eval.TrainingRecipe(1, False),
    )
    assert proof["passed"] is True
    rows = cast(list[dict[str, object]], proof["batches"])
    assert len(rows) == 3
    assert all(row["control_instance_ids"] == ["item"] for row in rows)


def test_amendment_4a_measurement_validates_inputs_and_accuracy_evidence(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_noise_floor_batches=3,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    with pytest.raises(ValueError, match="requires an accelerator"):
        base_eval.measure_noise_floor(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            [[instance]] * 3,
            settings,
            16,
            torch.device("cpu"),
            base_eval.TrainingRecipe(1, False),
        )
    with pytest.raises(ValueError, match="too few"):
        base_eval.measure_noise_floor(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            [[instance]],
            settings,
            16,
            torch.device("mps"),
            base_eval.TrainingRecipe(1, False),
        )
    with pytest.raises(ValueError, match="trajectory measurement requires an accelerator"):
        base_eval.measure_training_trajectory(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            cast(Any, _ToyStream()),
            settings,
            DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
            16,
            torch.device("cpu"),
            base_eval.TrainingRecipe(1, False),
            resource_guard=cast(Any, SimpleNamespace()),
        )
    with pytest.raises(ValueError, match="multiplier"):
        base_eval.assess_noise_floor([], multiplier=0.5)
    with pytest.raises(ValueError, match="equal length"):
        base_eval.assess_training_trajectory(
            legacy_losses=[],
            control_losses=[],
            target_losses=[],
            legacy_accuracy={"toy": 0.0},
            target_accuracy={"toy": 0.0},
            noise_multiplier=2.0,
            accuracy_tie_margin=0.02,
        )
    with pytest.raises(ValueError, match="accuracy maps"):
        base_eval.assess_training_trajectory(
            legacy_losses=[1.0],
            control_losses=[1.0],
            target_losses=[1.0],
            legacy_accuracy={"toy": 0.0},
            target_accuracy={"other": 0.0},
            noise_multiplier=2.0,
            accuracy_tie_margin=0.02,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        base_eval._vector_relative_l2_difference(
            [torch.tensor([1.0])], [torch.tensor([math.nan])]
        )
    empty_stream = SimpleNamespace(
        families=[SimpleNamespace(family_id="empty")],
        get_split=lambda _: TaskStreamSplit([], [instance], [instance]),
    )
    with pytest.raises(ValueError, match="non-empty"):
        base_eval._equivalence_batches(cast(Any, empty_stream), count=1, batch_size=1)
    assert base_eval._probe_accuracy_map(
        {"families": [{"family_id": "toy", "zero_shot": {"accuracy": 0.5}}]}
    ) == {"toy": 0.5}
    with pytest.raises(TypeError, match="no family"):
        base_eval._probe_accuracy_map({})
    with pytest.raises(TypeError, match="family row"):
        base_eval._probe_accuracy_map({"families": [None]})
    with pytest.raises(TypeError, match="result"):
        base_eval._probe_accuracy_map(
            {"families": [{"family_id": "toy", "zero_shot": {}}]}
        )


def test_amendment_4a_trajectory_runs_legacy_control_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_trajectory_steps=2,
    )
    variants = iter(
        [
            {
                "losses": [1.0, 0.9],
                "accuracy": {"toy": 0.5},
                "dropout_modules_disabled": 0,
            },
            {
                "losses": [1.01, 0.9],
                "accuracy": None,
                "dropout_modules_disabled": 0,
            },
            {
                "losses": [1.02, 0.9],
                "accuracy": {"toy": 0.51},
                "dropout_modules_disabled": 0,
            },
        ]
    )
    monkeypatch.setattr(
        base_eval, "_run_trajectory_variant", lambda *args, **kwargs: next(variants)
    )
    decision = DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True)
    proof = base_eval.measure_training_trajectory(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        settings,
        decision,
        16,
        torch.device("mps"),
        base_eval.TrainingRecipe(1, False),
        resource_guard=cast(Any, SimpleNamespace()),
    )
    assert proof["passed"] is True
    assert proof["steps"] == 2
    assert proof["legacy_losses"] == [1.0, 0.9]


def test_amendment_4a_vector_and_trajectory_workers_run_real_tiny_model(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        probe_learning_rates=(1e-3,),
        probe_steps=2,
        probe_warmup_steps=1,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    recipe = base_eval.TrainingRecipe(1, False)
    legacy = base_eval._measure_gradient_and_update(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        [instance],
        settings,
        16,
        torch.device("cpu"),
        dtype=torch.float32,
        target_only=False,
        recipe=recipe,
        include_update=True,
    )
    target = base_eval._measure_gradient_and_update(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        [instance],
        settings,
        16,
        torch.device("cpu"),
        dtype=torch.float32,
        target_only=True,
        recipe=recipe,
        include_update=True,
    )
    assert legacy.loss == pytest.approx(target.loss)
    assert len(legacy.gradients) == len(target.gradients) == 1
    assert len(legacy.updates) == len(target.updates) == 1

    guard = SimpleNamespace(raise_if_swap_limit_exceeded=lambda: None)
    legacy_trajectory = base_eval._run_trajectory_variant(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        [[instance], [instance]],
        settings,
        16,
        torch.device("cpu"),
        recipe,
        target_only=False,
        permute_within_batch=True,
        evaluate_accuracy=False,
        resource_guard=cast(Any, guard),
    )
    target_trajectory = base_eval._run_trajectory_variant(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        [[instance], [instance]],
        settings,
        16,
        torch.device("cpu"),
        recipe,
        target_only=True,
        permute_within_batch=False,
        evaluate_accuracy=False,
        resource_guard=cast(Any, guard),
    )
    assert len(cast(list[float], legacy_trajectory["losses"])) == 2
    assert len(cast(list[float], target_trajectory["losses"])) == 2
    assert legacy_trajectory["accuracy"] is None


def test_amendment_4b_chunked_head_changes_shapes_but_preserves_objective(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=2,
        probe_micro_batch_sizes=(2, 1),
        probe_learning_rates=(1e-3,),
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=4,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    instances = [
        TaskInstance("one", "toy", "P", "A"),
        TaskInstance("two", "toy", "PP", "A"),
    ]
    legacy = base_eval._measure_gradient_and_update(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        instances,
        settings,
        16,
        torch.device("cpu"),
        dtype=torch.float32,
        target_only=False,
        recipe=base_eval.TrainingRecipe(2, False),
        include_update=True,
    )
    head_shapes: list[tuple[int, ...]] = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: head_shapes.append(tuple(args[0].shape))
    )
    chunked = base_eval._measure_gradient_and_update(
        cast(Any, model),
        cast(Any, _ToyTokenizer()),
        instances,
        settings,
        16,
        torch.device("cpu"),
        dtype=torch.float32,
        target_only=False,
        recipe=base_eval.TrainingRecipe(2, False),
        include_update=True,
        legacy_sequence_chunk_size=1,
    )
    hook.remove()
    assert head_shapes
    assert all(shape[1] == 1 for shape in head_shapes)
    assert chunked.loss == pytest.approx(legacy.loss, abs=1e-6)
    gradient = base_eval._vector_relative_l2_difference(
        legacy.gradients, chunked.gradients
    )
    assert gradient["relative_l2"] < 1e-5
    with pytest.raises(ValueError, match="mutually exclusive"):
        base_eval._measure_gradient_and_update(
            cast(Any, model),
            cast(Any, _ToyTokenizer()),
            instances,
            settings,
            16,
            torch.device("cpu"),
            dtype=torch.float32,
            target_only=True,
            recipe=base_eval.TrainingRecipe(2, False),
            include_update=False,
            legacy_sequence_chunk_size=1,
        )


def test_amendment_4b_shape_control_is_informational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        equivalence_shape_control_chunk_size=2,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    measurements = iter(
        [
            base_eval._VectorMeasurement(
                1.0, [torch.tensor([1.0])], [torch.tensor([0.1])], 0
            ),
            base_eval._VectorMeasurement(
                1.0, [torch.tensor([1.01])], [torch.tensor([0.101])], 0
            ),
            base_eval._VectorMeasurement(
                1.0, [torch.tensor([1.02])], [torch.tensor([0.102])], 0
            ),
        ]
    )
    calls: list[dict[str, object]] = []

    def measure(*args: object, **kwargs: object) -> base_eval._VectorMeasurement:
        del args
        calls.append(kwargs)
        return next(measurements)

    monkeypatch.setattr(base_eval, "_measure_gradient_and_update", measure)
    result = base_eval.measure_shape_changing_control(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        [[instance]],
        settings,
        16,
        torch.device("mps"),
        base_eval.TrainingRecipe(1, False),
    )
    assert result["gating"] is False
    assert calls[1]["legacy_sequence_chunk_size"] == 2
    rows = cast(list[dict[str, object]], result["batches"])
    assert rows[0]["target_gradient_within_shape_control"] is False


def test_amendment_4b_outcome_gate_uses_seed_variation_and_larger_accuracy_bound() -> None:
    passed = base_eval.assess_outcome_gate(
        old_seed_0_losses=[2.0, 1.0],
        old_seed_1_losses=[2.1, 0.95],
        new_seed_0_losses=[2.05, 1.02],
        old_seed_0_accuracy={"a": 0.50, "b": 0.60},
        old_seed_1_accuracy={"a": 0.55, "b": 0.61},
        new_seed_0_accuracy={"a": 0.54, "b": 0.60},
        accuracy_tie_margin=0.02,
    )
    assert passed["passed"] is True
    assert passed["accuracy_limit"] == pytest.approx(0.03)
    failed = base_eval.assess_outcome_gate(
        old_seed_0_losses=[2.0, 1.0],
        old_seed_1_losses=[2.01, 1.0],
        new_seed_0_losses=[2.02, 1.0],
        old_seed_0_accuracy={"a": 0.5},
        old_seed_1_accuracy={"a": 0.5},
        new_seed_0_accuracy={"a": 0.53},
        accuracy_tie_margin=0.02,
    )
    assert failed["trajectory_passed"] is False
    assert failed["accuracy_passed"] is False
    assert failed["passed"] is False


def test_amendment_4b_outcome_runs_old_seeds_then_new_seed_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_trajectory_steps=2,
        equivalence_outcome_seeds=(0, 1),
    )
    variants = iter(
        [
            {"seed": 0, "losses": [1.0, 0.9], "accuracy": {"toy": 0.5}},
            {"seed": 1, "losses": [1.1, 0.9], "accuracy": {"toy": 0.52}},
            {"seed": 0, "losses": [1.05, 0.9], "accuracy": {"toy": 0.51}},
        ]
    )
    calls: list[tuple[int, bool]] = []

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        calls.append((cast(int, kwargs["seed"]), cast(bool, kwargs["target_only"])))
        return next(variants)

    monkeypatch.setattr(base_eval, "_run_outcome_variant", run)
    result = base_eval.measure_outcome_gate(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _ToyTokenizer()),
        cast(Any, _ToyStream()),
        settings,
        DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
        16,
        torch.device("mps"),
        base_eval.TrainingRecipe(1, False),
        resource_guard=cast(Any, SimpleNamespace()),
    )
    assert result["passed"] is True
    assert calls == [(0, False), (1, False), (0, True)]


def test_amendment_4b_rejects_invalid_measurement_inputs(tmp_path: Path) -> None:
    settings = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_trajectory_steps=1,
    )
    instance = TaskInstance("item", "toy", "P", "A")
    with pytest.raises(ValueError, match="shape-changing control requires an accelerator"):
        base_eval.measure_shape_changing_control(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            [[instance]],
            settings,
            16,
            torch.device("cpu"),
            base_eval.TrainingRecipe(1, False),
        )
    with pytest.raises(ValueError, match="at least one batch"):
        base_eval.measure_shape_changing_control(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            [],
            settings,
            16,
            torch.device("mps"),
            base_eval.TrainingRecipe(1, False),
        )
    with pytest.raises(ValueError, match="outcome gate requires an accelerator"):
        base_eval.measure_outcome_gate(
            cast(Any, _ToyLearnableLM()),
            cast(Any, _ToyTokenizer()),
            cast(Any, _ToyStream()),
            settings,
            DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
            16,
            torch.device("cpu"),
            base_eval.TrainingRecipe(1, False),
            resource_guard=cast(Any, SimpleNamespace()),
        )
    with pytest.raises(ValueError, match="equal length"):
        base_eval.assess_outcome_gate(
            old_seed_0_losses=[],
            old_seed_1_losses=[],
            new_seed_0_losses=[],
            old_seed_0_accuracy={"toy": 0.0},
            old_seed_1_accuracy={"toy": 0.0},
            new_seed_0_accuracy={"toy": 0.0},
            accuracy_tie_margin=0.02,
        )
    with pytest.raises(ValueError, match="accuracy maps"):
        base_eval.assess_outcome_gate(
            old_seed_0_losses=[1.0],
            old_seed_1_losses=[1.0],
            new_seed_0_losses=[1.0],
            old_seed_0_accuracy={"toy": 0.0},
            old_seed_1_accuracy={"other": 0.0},
            new_seed_0_accuracy={"toy": 0.0},
            accuracy_tie_margin=0.02,
        )


class _FakeMemorySampler:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _FakeMemorySampler:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def record(self) -> dict[str, object]:
        return {"test_sampler": True}

    def raise_if_swap_limit_exceeded(self) -> None:
        return None


def _patch_amendment_4a_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_stage: str | None,
) -> BaseModelEvaluationSettings:
    measurement = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_candidate_ids=("smollm2-135m", "gpt2"),
        equivalence_noise_floor_batches=3,
        equivalence_trajectory_steps=2,
    )
    candidates = (
        CandidateSpec("smollm2-135m", "local", "a" * 40, "mit", "toy", False),
        CandidateSpec("gpt2", "local", "b" * 40, "mit", "toy", False),
    )
    settings = BaseModelEvaluationSettings(
        candidates,
        measurement,
        DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
    )
    monkeypatch.setattr(base_eval, "load_base_model_evaluation_settings", lambda _: settings)
    monkeypatch.setattr(
        base_eval,
        "prepare_run_manifest",
        lambda *args: ({"correlation_id": "test-correlation"}, "f" * 64),
    )
    monkeypatch.setattr(base_eval, "resolve_device", lambda _: torch.device("mps"))
    monkeypatch.setattr(base_eval, "MemoryPeakSampler", _FakeMemorySampler)
    monkeypatch.setattr(
        base_eval,
        "_load_with_retries",
        lambda candidate, *args: (
            LoadedCandidate(
                cast(Any, _ToyLearnableLM()),
                cast(Any, _ToyTokenizer()),
                cast(str, candidate.revision),
            ),
            1,
        ),
    )
    monkeypatch.setattr(base_eval, "model_context_length", lambda *args: 16)
    monkeypatch.setattr(base_eval, "_new_equivalence_stream", lambda _: _ToyStream())
    monkeypatch.setattr(
        base_eval,
        "select_training_recipe",
        lambda *args, **kwargs: (base_eval.TrainingRecipe(1, False), {"selected": 1}),
    )
    monkeypatch.setattr(
        base_eval,
        "prove_exact_gradient_equivalence",
        lambda *args, **kwargs: {"passed": failed_stage != "exact_math"},
    )
    monkeypatch.setattr(
        base_eval,
        "measure_noise_floor",
        lambda *args, **kwargs: {"passed": failed_stage != "noise_floor"},
    )
    monkeypatch.setattr(
        base_eval,
        "measure_training_trajectory",
        lambda *args, **kwargs: {"passed": failed_stage != "training_trajectory"},
    )
    monkeypatch.setattr(base_eval, "cleanup_device", lambda _: None)
    return settings


@pytest.mark.parametrize(
    ("failed_stage", "expected_status"),
    [
        (None, "passed"),
        ("exact_math", "failed"),
        ("noise_floor", "failed"),
        ("training_trajectory", "failed"),
    ],
)
def test_amendment_4a_orchestration_is_fail_closed_at_each_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str | None,
    expected_status: str,
) -> None:
    _patch_amendment_4a_orchestration(
        tmp_path, monkeypatch, failed_stage=failed_stage
    )
    record = base_eval.run_amendment_4a_equivalence(
        tmp_path / "config.yaml", tmp_path / "evidence"
    )
    assert record["status"] == expected_status
    if failed_stage is None:
        assert cast(dict[str, object], record["exact_math"])["status"] == "passed"
        assert cast(dict[str, object], record["noise_floor"])["status"] == "passed"
        assert cast(dict[str, object], record["training_trajectory"])["status"] == "passed"
    else:
        assert record["failed_stage"] == failed_stage


@pytest.mark.parametrize(
    "exception_stage", ["exact_math", "noise_floor", "training_trajectory"]
)
def test_amendment_4a_orchestration_persists_structured_stage_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_stage: str,
) -> None:
    _patch_amendment_4a_orchestration(tmp_path, monkeypatch, failed_stage=None)

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("measured failure")

    target = {
        "exact_math": "prove_exact_gradient_equivalence",
        "noise_floor": "measure_noise_floor",
        "training_trajectory": "measure_training_trajectory",
    }[exception_stage]
    monkeypatch.setattr(base_eval, target, fail)
    record = base_eval.run_amendment_4a_equivalence(
        tmp_path / "config.yaml", tmp_path / "evidence"
    )
    assert record["status"] == "failed"
    assert record["failed_stage"] == exception_stage
    section = cast(dict[str, object], record[exception_stage])
    if exception_stage == "training_trajectory":
        failure = cast(dict[str, object], section["failure"])
    else:
        rows = cast(list[dict[str, object]], section["candidate_results"])
        failure = cast(dict[str, object], rows[-1]["failure"])
    assert failure["type"] == "RuntimeError"
    assert failure["message"] == "measured failure"


def _patch_amendment_4b_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exact_passed: bool = True,
    outcome_passed: bool = True,
    diagnostic_raises: bool = False,
) -> None:
    measurement = replace(
        _settings(tmp_path),
        probe_batch_size=1,
        probe_micro_batch_sizes=(1,),
        equivalence_candidate_ids=("smollm2-135m", "gpt2"),
        equivalence_noise_floor_batches=3,
        equivalence_trajectory_steps=2,
        equivalence_protocol="amendment_4b",
        equivalence_outcome_seeds=(0, 1),
    )
    settings = BaseModelEvaluationSettings(
        (
            CandidateSpec("smollm2-135m", "local", "a" * 40, "mit", "toy", False),
            CandidateSpec("gpt2", "local", "b" * 40, "mit", "toy", False),
        ),
        measurement,
        DecisionSettings(("mit",), 1000, 0.1, 0.0, 0.0, 0.02, True),
    )
    monkeypatch.setattr(base_eval, "load_base_model_evaluation_settings", lambda _: settings)
    monkeypatch.setattr(
        base_eval,
        "prepare_run_manifest",
        lambda *args: ({"correlation_id": "test-correlation"}, "f" * 64),
    )
    monkeypatch.setattr(base_eval, "resolve_device", lambda _: torch.device("mps"))
    monkeypatch.setattr(base_eval, "MemoryPeakSampler", _FakeMemorySampler)
    monkeypatch.setattr(
        base_eval,
        "_load_with_retries",
        lambda candidate, *args: (
            LoadedCandidate(
                cast(Any, _ToyLearnableLM()),
                cast(Any, _ToyTokenizer()),
                cast(str, candidate.revision),
            ),
            1,
        ),
    )
    monkeypatch.setattr(base_eval, "model_context_length", lambda *args: 16)
    monkeypatch.setattr(base_eval, "_new_equivalence_stream", lambda _: _ToyStream())
    monkeypatch.setattr(
        base_eval,
        "select_training_recipe",
        lambda *args, **kwargs: (base_eval.TrainingRecipe(1, False), {"selected": 1}),
    )
    monkeypatch.setattr(
        base_eval,
        "prove_exact_gradient_equivalence",
        lambda *args, **kwargs: {"passed": exact_passed},
    )

    def diagnostic(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        if diagnostic_raises:
            raise RuntimeError("diagnostic only")
        return {"status": "completed", "gating": False}

    monkeypatch.setattr(base_eval, "measure_shape_changing_control", diagnostic)
    monkeypatch.setattr(
        base_eval,
        "measure_outcome_gate",
        lambda *args, **kwargs: {"passed": outcome_passed},
    )
    monkeypatch.setattr(base_eval, "cleanup_device", lambda _: None)


def test_amendment_4b_diagnostic_cannot_halt_outcome_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_amendment_4b_orchestration(
        tmp_path, monkeypatch, diagnostic_raises=True
    )
    record = base_eval.run_amendment_4b_equivalence(
        tmp_path / "config.yaml", tmp_path / "evidence"
    )
    assert record["status"] == "passed"
    diagnostic = cast(dict[str, object], record["shape_changing_control"])
    assert diagnostic["gating"] is False
    assert diagnostic["status"] == "completed_with_diagnostic_failures"
    assert cast(dict[str, object], record["outcome_gate"])["status"] == "passed"


@pytest.mark.parametrize("failed_stage", ["exact_math", "outcome_gate"])
def test_amendment_4b_gates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    _patch_amendment_4b_orchestration(
        tmp_path,
        monkeypatch,
        exact_passed=failed_stage != "exact_math",
        outcome_passed=failed_stage != "outcome_gate",
    )
    record = base_eval.run_amendment_4b_equivalence(
        tmp_path / "config.yaml", tmp_path / "evidence"
    )
    assert record["status"] == "failed"
    assert record["failed_stage"] == failed_stage


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
        lambda *args, **kwargs: _completed_measurements()["task_stream_readiness"],
    )
    monkeypatch.setattr(
        base_eval,
        "evaluate_zero_shot_readiness",
        lambda *args: {"families": [], "mean_zero_shot_accuracy": 0.0},
    )
    monkeypatch.setattr(base_eval, "benchmark_generation", lambda *args: {})
    monkeypatch.setattr(
        base_eval,
        "select_training_recipe",
        lambda *args, **kwargs: (
            base_eval.TrainingRecipe(settings.probe_batch_size, False),
            {
                "selected_micro_batch_size": settings.probe_batch_size,
                "selected_gradient_checkpointing": False,
            },
        ),
    )
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


def _add_probe_losses(record: dict[str, object]) -> dict[str, object]:
    families = cast(
        list[dict[str, object]],
        cast(dict[str, object], record["task_stream_readiness"])["families"],
    )
    for row in families:
        row["initial_loss"] = 1.0
        row["final_loss"] = 0.5
    return record


def test_amendment_4e_foreground_check_and_mixed_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = (
        CandidateSpec("gpt2", "gpt2", "a" * 40, "mit", None, False),
        CandidateSpec("qwen2.5-0.5b", "qwen", "b" * 40, "mit", None, False),
    )
    settings = BaseModelEvaluationSettings(
        candidates=candidates,
        measurement=_settings(tmp_path),
        decision=DecisionSettings(
            ("mit",), 1_000_000, 0.25, 0.0, 0.0, 0.02, True, 100, True
        ),
    )
    manifest = {
        "correlation_id": "4e-test",
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "resolved_revision": candidate.revision,
                "license": candidate.license,
            }
            for candidate in candidates
        ],
    }
    monkeypatch.setattr(
        base_eval, "load_base_model_evaluation_settings", lambda path: settings
    )
    monkeypatch.setattr(
        base_eval,
        "prepare_run_manifest",
        lambda *args: (manifest, "4e-manifest"),
    )
    prior = tmp_path / "prior"
    output = tmp_path / "output"
    gpt_record = _add_probe_losses(_decision_record("gpt2"))
    gpt_record["manifest_sha256"] = "prior-manifest"
    base_eval.write_json(base_eval._candidate_record_path(prior, "gpt2"), gpt_record)
    monkeypatch.setattr(
        base_eval,
        "evaluate_candidate",
        lambda *args, **kwargs: copy.deepcopy(gpt_record),
    )
    check = base_eval.run_amendment_4e_candidate_check(
        tmp_path / "config.yaml",
        output,
        prior,
        "gpt2",
        placement_equivalence=True,
    )
    assert check["status"] == "passed"
    assert (
        cast(dict[str, object], check["comparisons"])[
            "released_vs_current_legacy_per_item"
        ]
        is not None
    )

    prior_manifest = {"correlation_id": "prior"}
    base_eval.write_json(prior / "run_manifest.json", prior_manifest)
    prior_hash = base_eval.hashlib.sha256(
        json.dumps(prior_manifest, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    gpt_record["manifest_sha256"] = prior_hash
    base_eval.write_json(base_eval._candidate_record_path(prior, "gpt2"), gpt_record)
    qwen_record = _decision_record("qwen2.5-0.5b", learned=0.6)
    qwen_record["manifest_sha256"] = "4e-manifest"
    base_eval.write_json(
        base_eval._candidate_record_path(output, "qwen2.5-0.5b"), qwen_record
    )
    ranking = base_eval.rank_amendment_4e_records(
        tmp_path / "config.yaml",
        output,
        prior,
        rerun_completed_4c_candidates=False,
    )
    assert ranking["decision_status"] != "measurement_incomplete"
    assert len(cast(list[object], ranking["record_provenance"])) == 2
    agreement = cast(dict[str, object], ranking["top_two_seed_repeat_agreement"])
    assert agreement["complete"] is False
    assert agreement["agrees"] is None


def test_top_two_seed_repeat_agreement_reports_order_stability() -> None:
    summaries = [
        {
            "candidate_id": "leader",
            "missing_seeds": [],
            "runs": [
                {
                    "seed": seed,
                    "mean_learned_accuracy": accuracy,
                    "mean_accuracy_gain": accuracy,
                }
                for seed, accuracy in ((1, 0.7), (2, 0.6), (3, 0.8))
            ],
        },
        {
            "candidate_id": "runner-up",
            "missing_seeds": [],
            "runs": [
                {
                    "seed": seed,
                    "mean_learned_accuracy": accuracy,
                    "mean_accuracy_gain": accuracy,
                }
                for seed, accuracy in ((1, 0.6), (2, 0.5), (3, 0.7))
            ],
        },
    ]
    stable = base_eval._top_two_seed_repeat_agreement(summaries, (1, 2, 3))
    assert stable["complete"] is True
    assert stable["agrees"] is True

    cast(list[dict[str, object]], summaries[1]["runs"])[1][
        "mean_learned_accuracy"
    ] = 0.9
    flipped = base_eval._top_two_seed_repeat_agreement(summaries, (1, 2, 3))
    assert flipped["complete"] is True
    assert flipped["agrees"] is False


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


def test_decision_rejects_measured_footprint_over_cap() -> None:
    """The candidate's own measured RSS/MPS peak remains a hard constraint."""
    settings = DecisionSettings(
        ("mit",), 1_000, 0.1, 0.0, 0.0, 0.02, True, 1_073_741_824, True
    )
    record = _decision_record("too-large")
    memory = cast(dict[str, object], cast(dict[str, object], record["cost"])["memory"])
    memory["hard_constraint_peak_bytes"] = 901
    result = apply_decision_rule([record], settings)
    assert result["selected_candidate"] is None
    assert result["decision_status"] == "no_eligible_candidate"
    rejected = cast(list[dict[str, object]], result["rejected_candidates"])
    assert "peak memory" in cast(list[str], rejected[0]["reasons"])[0]


def test_not_measured_environment_blocks_selection_and_names_candidate() -> None:
    """A candidate cannot win by default while a competitor lacks valid evidence."""
    completed = _decision_record("completed")
    missing = {
        "candidate_id": "environment-noise",
        "status": "not_measured_environment",
        "reference_only": False,
    }
    settings = DecisionSettings(
        ("mit",), 1_000, 0.1, 0.0, 0.0, 0.02, True, 1_073_741_824, True
    )
    result = apply_decision_rule([completed, missing], settings)
    assert result["decision_status"] == "measurement_incomplete"
    assert result["selected_candidate"] is None
    assert result["missing_candidates"] == [
        {
            "candidate_id": "environment-noise",
            "status": "not_measured_environment",
        }
    ]


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
    assert len(settings.candidates) == 8
    assert settings.candidates[-1].loader == "aptivra_legacy"
    assert settings.measurement.seed == 20261014
    assert settings.measurement.test_items_per_family == 100
    assert settings.measurement.probe_steps == 200
    assert settings.measurement.probe_batch_size == 8
    assert settings.measurement.probe_micro_batch_sizes == (8, 4, 2, 1)
    assert settings.measurement.target_only_training_loss is True
    assert settings.measurement.gradient_checkpointing_fallback is True
    assert settings.measurement.equivalence_candidate_ids == ("smollm2-135m", "gpt2")
    assert settings.measurement.equivalence_steps == 3
    assert settings.measurement.probe_learning_rates == (1e-5, 3e-5, 1e-4)
    assert settings.measurement.variance_seeds == (20261014, 20261015, 20261016)
    assert settings.decision.learnability_tie_margin == 0.02
    assert settings.decision.clustered_confidence_interval_intersects_tie_region is True
    assert settings.decision.maximum_swap_growth_bytes == 1_073_741_824
    assert settings.execution.resume is True
    assert settings.execution.output_dir.name == "amendment-4"

    manifest, manifest_sha256 = base_eval.prepare_run_manifest(
        config_path, tmp_path, settings
    )
    repeated, repeated_sha256 = base_eval.prepare_run_manifest(
        config_path, tmp_path, settings
    )
    assert repeated == manifest
    assert repeated_sha256 == manifest_sha256
    assert len(manifest_sha256) == 64
    assert len(cast(str, manifest["task_sources_sha256"])) == 64
    candidates = cast(list[dict[str, object]], manifest["candidates"])
    assert all("resolution_failure" not in item for item in candidates)
    assert all(len(cast(str, item["resolved_revision"])) == 40 for item in candidates)

    amendment_4a = base_eval.load_base_model_evaluation_settings(
        Path("configs/base_model_candidates_amendment_4a.yaml")
    )
    assert amendment_4a.measurement.equivalence_exact_loss_absolute_tolerance == 1e-12
    assert (
        amendment_4a.measurement.equivalence_exact_gradient_relative_l2_tolerance
        == 1e-10
    )
    assert amendment_4a.measurement.equivalence_noise_floor_batches == 3
    assert amendment_4a.measurement.equivalence_noise_floor_multiplier == 2.0
    assert amendment_4a.measurement.equivalence_trajectory_steps == 50
    assert amendment_4a.execution is not None
    assert amendment_4a.execution.output_dir.name == "attempt-2"
    assert amendment_4a.execution.output_dir.parent.name == "amendment-4a"
    assert [
        candidate.candidate_id
        for candidate in amendment_4a.candidates
        if not candidate.reference_only
    ] == [
        "gpt2",
        "smollm2-135m",
        "pythia-160m",
        "smollm2-360m",
        "qwen2.5-0.5b",
        "qwen2.5-1.5b",
        "smollm2-1.7b",
    ]

    amendment_4b = base_eval.load_base_model_evaluation_settings(
        Path("configs/base_model_candidates_amendment_4b.yaml")
    )
    assert amendment_4b.measurement.equivalence_protocol == "amendment_4b"
    assert amendment_4b.measurement.equivalence_shape_control_chunk_size == 1
    assert amendment_4b.measurement.equivalence_outcome_seeds == (0, 1)
    assert amendment_4b.measurement.equivalence_trajectory_steps == 50
    assert amendment_4b.decision.learnability_tie_margin == 0.02
    assert amendment_4b.execution is not None
    assert amendment_4b.execution.output_dir.name == "amendment-4b-override"

    amendment_4c = base_eval.load_base_model_evaluation_settings(
        Path("configs/base_model_candidates_amendment_4c.yaml")
    )
    assert amendment_4c.measurement.environment_minimum_free_memory_percent == 10.0
    assert amendment_4c.measurement.environment_recovery_timeout_seconds == 900.0
    assert amendment_4c.measurement.environment_max_attempts == 2
    assert amendment_4c.decision.require_complete_candidate_measurements is True
    assert amendment_4c.execution is not None
    assert amendment_4c.execution.output_dir.name == "amendment-4c"
    assert amendment_4c.execution.resume is False

    amendment_4e = base_eval.load_base_model_evaluation_settings(
        Path("configs/base_model_candidates_amendment_4e.yaml")
    )
    assert amendment_4e.measurement.release_pristine_before_training is True
    assert amendment_4e.measurement.prestart_minimum_free_memory_percent == 70.0
    assert amendment_4e.measurement.prestart_max_other_process_rss_bytes == 2**31
    assert amendment_4e.measurement.prestart_timeout_seconds == 900.0
    assert amendment_4e.measurement.prestart_poll_interval_seconds == 5.0
    assert amendment_4e.execution is not None
    assert amendment_4e.execution.output_dir.name == "measurement-after-ruling"
    assert amendment_4e.execution.output_dir.parent.name == "amendment-4e"

    manifest["evaluator_sha256"] = "0" * 64
    base_eval.write_json(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="frozen evidence is not overwritten"):
        base_eval.prepare_run_manifest(config_path, tmp_path, settings)


def test_failed_equivalence_gate_requires_manifested_owner_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed gate stays closed unless its exact owner override is manifested."""
    config_path = Path("configs/base_model_candidates_amendment_4b.yaml")
    settings = base_eval.load_base_model_evaluation_settings(config_path)
    expected = {
        "by": "raghunath",
        "date": "2026-10-14",
        "adr_section": "Owner override of Amendment 4b (2026-10-14)",
    }
    failed = {"status": "failed"}
    without_override = replace(settings, equivalence_gate_owner_override=None)
    assert _equivalence_allows_sweep(failed, without_override) is False
    assert _equivalence_allows_sweep(failed, settings) is True

    monkeypatch.setattr(
        base_eval,
        "_resolve_preflight_candidate",
        lambda candidate, measurement: {
            "candidate_id": candidate.candidate_id,
            "source": candidate.source,
            "resolved_revision": candidate.revision,
            "license": candidate.license,
            "reference_only": candidate.reference_only,
        },
    )
    manifest, _ = base_eval.prepare_run_manifest(config_path, tmp_path, settings)
    assert manifest["equivalence_gate_owner_override"] == expected


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


def test_amendment_3_analysis_combines_provenance_without_mutating_prior(
    tmp_path: Path,
) -> None:
    """Mixed-manifest ranking prefers new records and preserves prior evidence bytes."""
    prior = tmp_path / "prior"
    current = tmp_path / "amendment-3"
    shutil.copytree(Path("experiments/base_model_candidates"), prior)
    prior_paths = [
        prior / "run_manifest.json",
        *sorted((prior / "candidates").glob("*.json")),
        *sorted((prior / "seed_runs").glob("*.json")),
    ]
    before = {path: base_eval.sha256_file(path) for path in prior_paths}
    settings = base_eval.load_base_model_evaluation_settings(
        Path("configs/base_model_candidates.yaml")
    )
    _, manifest_sha256 = base_eval.prepare_run_manifest(
        Path("configs/base_model_candidates.yaml"), current, settings
    )
    prior_smol = json.loads(
        base_eval._candidate_record_path(prior, "smollm2-360m").read_text(
            encoding="utf-8"
        )
    )
    family_ids = [
        row["family_id"] for row in prior_smol["task_stream_readiness"]["families"]
    ]
    for candidate_id in ("pythia-160m", "qwen2.5-0.5b"):
        record = _decision_record(candidate_id, learned=0.5, gain=0.5, bpb=1.5)
        record["schema_version"] = base_eval.EVALUATION_SCHEMA_VERSION
        record["manifest_sha256"] = manifest_sha256
        record["correlation_id"] = "test"
        cast(dict[str, object], record["tokenizer_fitness"]).update(
            {
                "exact_roundtrip_after_nfc_normalization": True,
                "exact_task_stream_roundtrip": True,
            }
        )
        families = cast(
            list[dict[str, object]],
            cast(dict[str, object], record["task_stream_readiness"])["families"],
        )
        for row, family_id in zip(families, family_ids, strict=True):
            row["family_id"] = family_id
        base_eval.write_json(
            base_eval._candidate_record_path(current, candidate_id), record
        )
    for candidate_id in ("smollm2-1.7b", "qwen2.5-1.5b"):
        base_eval.write_json(
            base_eval._candidate_record_path(current, candidate_id),
            {
                "schema_version": base_eval.EVALUATION_SCHEMA_VERSION,
                "manifest_sha256": manifest_sha256,
                "candidate_id": candidate_id,
                "status": "hard_constraint_rejected",
                "reference_only": False,
                "hard_constraint_reasons": ["training memory headroom failure"],
            },
        )

    analysis, ranking = base_eval.derive_amendment_3_analysis(
        Path("configs/base_model_candidates.yaml"), current, prior
    )
    assert ranking["selected_candidate"] == "smollm2-360m"
    sources = cast(list[dict[str, object]], analysis["source_records"])
    assert len(sources) == 8
    pythia = next(item for item in sources if item["candidate_id"] == "pythia-160m")
    smol = next(item for item in sources if item["candidate_id"] == "smollm2-360m")
    assert pythia["protocol"] == "amendment_3"
    assert smol["protocol"] == "prior_powered"
    assert {path: base_eval.sha256_file(path) for path in prior_paths} == before


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

    def __len__(self) -> int:
        return 6

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        import unicodedata

        del add_special_tokens
        mapping = {"P": 1, "A": 2, "\n": 3, "é": 5}
        return [mapping[character] for character in unicodedata.normalize("NFC", text)]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        mapping = {0: "", 1: "P", 2: "A", 3: "\n", 4: "<eos>", 5: "é"}
        return "".join(
            "" if skip_special_tokens and value == 4 else mapping[value] for value in ids
        )


def test_nfc_normalizing_tokenizer_passes_when_task_stream_is_exact() -> None:
    """Amendment 3 permits NFC normalization outside an exact ASCII task stream."""
    result = base_eval.tokenizer_fitness(
        cast(Any, _ToyLearnableLM()),
        cast(Any, _NfcNormalizingTokenizer()),
        cast(Any, _ToyStream()),
        ("e\u0301",),
    )
    failure = cast(list[dict[str, object]], result["roundtrip_failures"])[0]
    assert result["exact_utf8_roundtrip"] is False
    assert result["exact_roundtrip_after_nfc_normalization"] is True
    assert result["exact_task_stream_roundtrip"] is True
    assert failure["diagnosed_cause"] == "tokenizer_unicode_normalization_nfc"
    assert failure["add_special_tokens"] is False
    assert failure["decoded_with_cleanup_enabled"] == "é"
    assert failure["decoded_with_cleanup_disabled"] == "é"
    assert failure["input_code_points"] == ["U+0065", "U+0301"]


def test_nfc_normalizing_tokenizer_fails_when_task_text_changes() -> None:
    """NFC relaxation never permits a tokenizer to alter a stored task string."""
    tokenizer = cast(Any, _NfcNormalizingTokenizer())
    changed = TaskInstance("changed", "toy", "e\u0301", "A")
    stream = SimpleNamespace(
        families=[SimpleNamespace(family_id="toy")],
        get_split=lambda family_id: TaskStreamSplit([changed], [changed], [changed]),
    )
    result = base_eval.tokenizer_fitness(
        cast(Any, _ToyLearnableLM()), tokenizer, cast(Any, stream), ("e\u0301",)
    )
    assert result["exact_roundtrip_after_nfc_normalization"] is True
    assert result["exact_task_stream_roundtrip"] is False
    failures = cast(list[dict[str, object]], result["task_stream_roundtrip_failures"])
    assert failures[0]["diagnosed_cause"] == "tokenizer_unicode_normalization_nfc"


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
            "exact_roundtrip_after_nfc_normalization": True,
            "exact_task_stream_roundtrip": False,
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


def test_memory_lower_bound_is_informational_and_normal_completion_is_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only measured peak can reject; an analytic lower bound cannot skip the probe."""
    settings = _settings(tmp_path)
    candidate = CandidateSpec("large", "local", "a" * 40, "mit", "toy", False)
    model = _ToyLearnableLM()
    zero_shot = {
        "families": [{"family_id": "toy", "zero_shot": _accuracy_record_for_test()}],
        "mean_zero_shot_accuracy": 0.0,
    }
    monkeypatch.setattr(base_eval, "unique_parameter_count", lambda _: 2_000_000_000)
    monkeypatch.setattr(base_eval, "model_context_length", lambda *args: 16)
    monkeypatch.setattr(base_eval, "_hidden_size", lambda *args: 5)
    monkeypatch.setattr(base_eval, "_architecture_type", lambda *args: "toy")
    monkeypatch.setattr(base_eval, "ContinualTaskStream", lambda **kwargs: _ToyStream())
    monkeypatch.setattr(
        base_eval,
        "tokenizer_fitness",
        lambda *args: {
            "exact_roundtrip_after_nfc_normalization": True,
            "exact_task_stream_roundtrip": True,
            "eos_usable_as_answer_terminator": True,
        },
    )
    monkeypatch.setattr(
        base_eval,
        "inspect_expert_attachment_structure",
        lambda *args: {"all_blocks_expose_feed_forward": True},
    )
    monkeypatch.setattr(
        base_eval,
        "evaluate_language_quality",
        lambda *args: {"combined": {"bits_per_byte": 1.25}},
    )
    monkeypatch.setattr(base_eval, "evaluate_zero_shot_readiness", lambda *args: zero_shot)

    monkeypatch.setattr(
        base_eval,
        "select_training_recipe",
        lambda *args, **kwargs: (
            base_eval.TrainingRecipe(1, False),
            {
                "selected_micro_batch_size": 1,
                "selected_gradient_checkpointing": False,
            },
        ),
    )
    monkeypatch.setattr(
        base_eval,
        "evaluate_task_readiness",
        lambda *args, **kwargs: _completed_measurements()["task_stream_readiness"],
    )
    monkeypatch.setattr(base_eval, "benchmark_generation", lambda *args: {})
    record = evaluate_candidate(
        candidate,
        settings,
        loader=lambda *args: LoadedCandidate(
            cast(Any, model), cast(Any, _ToyTokenizer()), "a" * 40
        ),
        decision_settings=DecisionSettings(
            ("mit",), 38_654_705_664, 0.25, 0.0, 0.0, 0.02, True
        ),
    )
    assert record["status"] == "completed"
    assert record["language_modeling"] == {"combined": {"bits_per_byte": 1.25}}
    assert record["zero_shot_readiness"] == zero_shot
    feasibility = cast(dict[str, object], record["full_finetuning_memory_feasibility"])
    assert feasibility["fits_parameter_state_lower_bound"] is False
    assert feasibility["bytes_per_parameter"] == 20
    assert "informational lower bound" in cast(str, feasibility["verdict_role"])


def test_candidate_releases_pristine_copy_before_training_and_persists_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 4e order benchmarks first, then trains from one disk-backed object."""
    settings = replace(_settings(tmp_path), release_pristine_before_training=True)
    candidate = CandidateSpec("released", "local", "a" * 40, "mit", "toy", False)
    model = _ToyLearnableLM()
    expected_digest = base_eval.model_state_digest(cast(Any, model))
    zero_shot = {
        "families": [{"family_id": "toy", "zero_shot": _accuracy_record_for_test()}],
        "mean_zero_shot_accuracy": 0.0,
    }
    generation_calls: list[str] = []
    snapshot_paths: list[Path] = []

    monkeypatch.setattr(base_eval, "model_context_length", lambda *args: 16)
    monkeypatch.setattr(base_eval, "_hidden_size", lambda *args: 5)
    monkeypatch.setattr(base_eval, "_architecture_type", lambda *args: "toy")
    monkeypatch.setattr(base_eval, "ContinualTaskStream", lambda **kwargs: _ToyStream())
    monkeypatch.setattr(
        base_eval,
        "tokenizer_fitness",
        lambda *args: {
            "exact_roundtrip_after_nfc_normalization": True,
            "exact_task_stream_roundtrip": True,
            "eos_usable_as_answer_terminator": True,
        },
    )
    monkeypatch.setattr(
        base_eval,
        "inspect_expert_attachment_structure",
        lambda *args: {"all_blocks_expose_feed_forward": True},
    )
    monkeypatch.setattr(
        base_eval,
        "evaluate_language_quality",
        lambda *args: {"combined": {"bits_per_byte": 1.25}},
    )
    monkeypatch.setattr(base_eval, "evaluate_zero_shot_readiness", lambda *args: zero_shot)

    def benchmark(*args: object) -> dict[str, object]:
        del args
        generation_calls.append("before-training")
        return {"median_tokens_per_second": 1.0}

    def select_recipe(*args: object, **kwargs: object) -> tuple[object, object]:
        del args
        snapshot = cast(Path, kwargs["pristine_snapshot_path"])
        snapshot_paths.append(snapshot)
        assert snapshot.is_file()
        assert kwargs["pristine_snapshot_digest"] == expected_digest
        return base_eval.TrainingRecipe(1, False), {
            "selected_micro_batch_size": 1,
            "selected_gradient_checkpointing": False,
        }

    def task_readiness(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        assert kwargs["pristine_snapshot_path"] == snapshot_paths[0]
        assert kwargs["pristine_snapshot_digest"] == expected_digest
        return cast(dict[str, object], _completed_measurements()["task_stream_readiness"])

    monkeypatch.setattr(base_eval, "benchmark_generation", benchmark)
    monkeypatch.setattr(base_eval, "select_training_recipe", select_recipe)
    monkeypatch.setattr(base_eval, "evaluate_task_readiness", task_readiness)
    progress_path = tmp_path / "released-progress.json"
    record = evaluate_candidate(
        candidate,
        settings,
        loader=lambda *args: LoadedCandidate(
            cast(Any, model), cast(Any, _ToyTokenizer()), "a" * 40
        ),
        decision_settings=DecisionSettings(
            ("mit",), 38_654_705_664, 0.25, 0.0, 0.0, 0.02, True
        ),
        progress_path=progress_path,
    )

    assert record["status"] == "completed"
    assert generation_calls == ["before-training"]
    placement = cast(dict[str, object], record["pristine_model_placement"])
    assert placement["released_before_training"] is True
    assert placement["snapshot_sha256"] == expected_digest
    feasibility = cast(dict[str, object], record["full_finetuning_memory_feasibility"])
    assert feasibility["bytes_per_parameter"] == 16
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "pristine_evaluation_persisted"
    assert progress["generation_benchmark"] == {"median_tokens_per_second": 1.0}
    assert snapshot_paths and not snapshot_paths[0].exists()


def _accuracy_record_for_test() -> dict[str, object]:
    return {
        "accuracy": 0.0,
        "successes": 0,
        "total": 1,
        "wilson_interval": wilson_interval(0, 1, 0.95),
    }


def test_public_orchestration_refuses_ranking_with_missing_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-level orchestration preserves evidence across isolated child outcomes."""
    config_path = Path("configs/base_model_candidates_amendment_4c.yaml")
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
    assert existing["selected_candidate"] is None
    assert existing["decision_status"] == "measurement_incomplete"
    assert len(cast(list[object], existing["missing_candidates"])) == 6
    assert len(cast(list[object], existing["rejected_candidates"])) == 6
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
