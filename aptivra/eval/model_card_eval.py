"""Evaluation helpers that produce machine-readable inputs for the model card."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from aptivra.eval.model_card_schema import (
    EVALUATION_SCHEMA_VERSION,
    PROVENANCE_FILENAME,
    ConversionProvenance,
)
from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.inference.chat_canonical import IDENTITY_CHAT_TEMPLATES
from aptivra.inference.hub_config import model_config_from_export_dir
from aptivra.model.config import ModelConfig
from aptivra.tasks.identity import ModelIdentityTaskFamily


@dataclass(frozen=True)
class PromptAnswer:
    """Prompt plus answer span to score."""

    prompt: str
    answer: str
    category: str | None = None


class ByteCausalModel(Protocol):
    """Small protocol shared by AptivraCausalLM and test doubles."""

    config: ModelConfig
    device: torch.device

    def eval(self) -> object:
        """Switches to evaluation mode."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns next-byte logits."""


DEFAULT_HELDOUT_PROMPTS_PATH = Path(__file__).with_name("model_card_heldout_v1.json")
DEFAULT_HELDOUT_TEXT_PATH = Path(__file__).with_name("model_card_heldout_text_v1.txt")
DEFAULT_HELDOUT_TEXT = DEFAULT_HELDOUT_TEXT_PATH.read_text(encoding="utf-8")


def default_trained_prompt_answers() -> list[PromptAnswer]:
    """Returns identity prompts used by the existing training task family."""
    family = ModelIdentityTaskFamily()
    return [
        PromptAnswer(prompt=instance.prompt, answer=instance.target_solution)
        for instance in family.generate_instances(count=7, seed=42)
    ]


def sha256_file(path: Path) -> str:
    """Computes a SHA-256 digest for a checkpoint file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Computes a SHA-256 digest for an in-memory byte sequence."""
    return hashlib.sha256(data).hexdigest()


def _unique_tensor_count(state: dict[str, object]) -> int:
    """Counts exact tensor aliases once while preserving distinct storage views."""
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    total = 0
    for value in state.values():
        if not isinstance(value, torch.Tensor):
            continue
        identity = (
            value.untyped_storage().data_ptr(),
            int(value.storage_offset()),
            tuple(value.size()),
            tuple(value.stride()),
        )
        if identity in seen:
            continue
        seen.add(identity)
        total += value.numel()
    return total


def count_state_dict_parameters(path: Path) -> int:
    """Counts unique tensor parameters in a saved PyTorch state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dict: {path}")
    return _unique_tensor_count(state)


def _count_hub_parameters(export_dir: Path) -> dict[str, int]:
    weights_path = export_dir / "model.safetensors"
    state = load_file(weights_path, device="cpu")
    groups = {"backbone": 0, "router": 0, "experts": 0}
    for name, tensor in state.items():
        prefix = name.split(".", 1)[0]
        if prefix not in groups:
            raise ValueError(f"unrecognized model.safetensors parameter: {name}")
        groups[prefix] += tensor.numel()
    total = sum(groups.values())

    config_path = export_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json is required with model.safetensors: {config_path}")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError("config.json must contain a JSON object")
    declared_count = config_data.get("parameter_count")
    if declared_count is not None:
        if isinstance(declared_count, bool) or not isinstance(declared_count, int):
            raise ValueError("config.json parameter_count must be an integer")
        if declared_count != total:
            raise ValueError(
                "config.json parameter_count does not match model.safetensors: "
                f"declared={declared_count}, measured={total}"
            )
    return {
        "backbone_parameters": groups["backbone"],
        "router_parameters": groups["router"],
        "expert_parameters": groups["experts"],
        "total_parameters": total,
    }


def count_export_parameters(export_dir: Path) -> dict[str, int]:
    """Counts backbone, router, expert, and total parameters for an export directory."""
    if (export_dir / "model.safetensors").is_file():
        return _count_hub_parameters(export_dir)
    dense_dir = export_dir / "dense"
    backbone = count_state_dict_parameters(dense_dir / "backbone.pt")
    router_path = dense_dir / "router.pt"
    router = count_state_dict_parameters(router_path) if router_path.is_file() else 0
    expert_total = 0
    for expert_path in sorted((export_dir / "experts").glob("expert_*.pt")):
        expert_total += count_state_dict_parameters(expert_path)
    return {
        "backbone_parameters": backbone,
        "router_parameters": router,
        "expert_parameters": expert_total,
        "total_parameters": backbone + router + expert_total,
    }


def uniform_bits_per_byte(vocab_size: int) -> float:
    """Computes the uniform byte baseline in bits per byte."""
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    return math.log2(vocab_size)


def unigram_bits_per_byte(text: str) -> float:
    """Computes empirical unigram entropy for a byte string."""
    return unigram_bits_per_bytes(text.encode("utf-8"))


def unigram_bits_per_bytes(data: bytes) -> float:
    """Computes empirical unigram entropy for the exact evaluated bytes."""
    if not data:
        raise ValueError("unigram baseline bytes must not be empty")
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _answer_metric(
    scores: Sequence[float],
    answers: Sequence[PromptAnswer],
    vocab_size: int,
) -> dict[str, object]:
    answer_bytes = [item.answer.encode("utf-8") for item in answers]
    if any(not data for data in answer_bytes):
        raise ValueError("answers must not be empty")
    uniform_scores = [uniform_bits_per_byte(vocab_size) for _ in answer_bytes]
    unigram_scores = [unigram_bits_per_bytes(data) for data in answer_bytes]
    return {
        **summarize_bits_per_byte(scores),
        "evaluated_bytes": sum(len(data) for data in answer_bytes),
        "baselines": {
            "uniform_bpb": summarize_bits_per_byte(uniform_scores),
            "unigram_bpb": summarize_bits_per_byte(unigram_scores),
            "unigram_fit": "evaluated_bytes",
        },
    }


def answer_span_bits_per_byte(model: ByteCausalModel, prompt: str, answer: str) -> float:
    """Scores only the answer bytes conditioned on prompt plus prior answer bytes."""
    prompt_ids = _byte_ids(prompt)
    answer_ids = _byte_ids(answer)
    if answer_ids.numel() == 0:
        raise ValueError("answer must not be empty")
    full_ids = torch.cat([prompt_ids, answer_ids], dim=0)
    return _bounded_sequence_bits_per_byte(
        model, full_ids, target_start=int(prompt_ids.numel())
    )


def whole_text_bits_per_byte(model: ByteCausalModel, text: str) -> float:
    """Scores each byte after the first using bounded, overlapping context windows."""
    ids = _byte_ids(text)
    if ids.numel() < 2:
        raise ValueError("held-out text must contain at least two bytes")
    return _bounded_sequence_bits_per_byte(model, ids, target_start=1)


def summarize_bits_per_byte(values: Sequence[float]) -> dict[str, float | int]:
    """Summarizes a non-empty set of bits-per-byte measurements."""
    if not values:
        raise ValueError("cannot summarize an empty measurement set")
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def evaluate_export_dir(
    export_dir: Path,
    *,
    trained_prompts: Sequence[PromptAnswer] | None = None,
    heldout_prompts: Sequence[PromptAnswer] | None = None,
    heldout_prompts_path: Path | None = None,
    heldout_text: str | None = None,
    heldout_text_path: Path | None = None,
    hub_revision: str | None = None,
    model_loader: Callable[[Path], ByteCausalModel] | None = None,
) -> dict[str, object]:
    """Evaluates an exported checkpoint and returns one JSON-serializable record."""
    export_path = Path(export_dir)
    if heldout_prompts is not None and heldout_prompts_path is not None:
        raise ValueError("provide heldout_prompts or heldout_prompts_path, not both")
    if heldout_text is not None and heldout_text_path is not None:
        raise ValueError("provide heldout_text or heldout_text_path, not both")
    prompt_set_path = heldout_prompts_path or DEFAULT_HELDOUT_PROMPTS_PATH
    if heldout_prompts is None:
        heldout = load_prompt_answers(prompt_set_path)
        prompt_set_name = prompt_set_path.name
        prompt_set_sha256 = sha256_file(prompt_set_path)
    else:
        heldout = list(heldout_prompts)
        prompt_set_name = "inline"
        prompt_set_sha256 = _prompt_answers_sha256(heldout)

    if heldout_text is not None:
        evaluated_text = heldout_text
        text_set_name = "inline"
        text_set_bytes = evaluated_text.encode("utf-8")
        text_set_sha256 = sha256_bytes(text_set_bytes)
    else:
        text_set_path = heldout_text_path or DEFAULT_HELDOUT_TEXT_PATH
        evaluated_text = text_set_path.read_text(encoding="utf-8")
        text_set_name = text_set_path.name
        text_set_bytes = evaluated_text.encode("utf-8")
        text_set_sha256 = sha256_file(text_set_path)
        if text_set_path == DEFAULT_HELDOUT_TEXT_PATH:
            _assert_default_heldout_text_is_disjoint(evaluated_text)

    cfg = model_config_from_export_dir(export_path, device="cpu")
    loader = model_loader or _load_aptivra_model
    model = loader(export_path)
    model.eval()

    trained = (
        list(trained_prompts)
        if trained_prompts is not None
        else default_trained_prompt_answers()
    )
    trained_scores = [
        answer_span_bits_per_byte(model, item.prompt, item.answer) for item in trained
    ]
    heldout_scores = [
        answer_span_bits_per_byte(model, item.prompt, item.answer) for item in heldout
    ]
    param_counts = count_export_parameters(export_path)

    weights_path = export_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            "model-card evaluation requires a converted model.safetensors Hub layout"
        )

    provenance_path = export_path / PROVENANCE_FILENAME
    provenance = ConversionProvenance.model_validate(_load_json_object(provenance_path))
    if hub_revision is not None and hub_revision != provenance.source_hub_revision:
        raise ValueError(
            "hub_revision must be recorded during conversion and match export provenance"
        )
    backbone_sha256 = provenance.source_shards_sha256.get("dense/backbone.pt")
    checkpoint: dict[str, object] = {
        "model_safetensors_sha256": sha256_file(weights_path),
        "config_sha256": sha256_file(export_path / "config.json"),
        "provenance_sha256": sha256_file(provenance_path),
        "backbone_sha256": backbone_sha256,
        "source_hub_revision": provenance.source_hub_revision,
        "provenance": provenance.model_dump(mode="json"),
    }

    config_data = _load_json_object(export_path / "config.json")
    eos_token_ids = config_data.get("eos_token_id")
    if isinstance(eos_token_ids, int) and not isinstance(eos_token_ids, bool):
        eos_token_ids = [eos_token_ids]
    if not (
        isinstance(eos_token_ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in eos_token_ids)
    ):
        eos_token_ids = None
    pad_token_id = config_data.get("pad_token_id")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        pad_token_id = None

    declared_parameter_count = config_data.get("parameter_count")
    if isinstance(declared_parameter_count, bool) or not isinstance(
        declared_parameter_count, int
    ):
        raise ValueError("config.json parameter_count must be an integer")

    heldout_text_bytes = evaluated_text.encode("utf-8")
    if len(heldout_text_bytes) < 2:
        raise ValueError("held-out text must contain at least two bytes")
    scored_text_bytes = heldout_text_bytes[1:]
    heldout_text_score = whole_text_bits_per_byte(model, evaluated_text)

    record: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "checkpoint": checkpoint,
        "model": {
            **param_counts,
            "config_parameter_count": declared_parameter_count,
            "context_length": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "position_embedding": "learned_absolute",
        },
        "evaluation": {
            "moe_enabled": False,
            "heldout_prompt_set": {
                "name": prompt_set_name,
                "sha256": prompt_set_sha256,
                "samples": len(heldout),
            },
            "heldout_text_set": {
                "name": text_set_name,
                "sha256": text_set_sha256,
                "bytes": len(text_set_bytes),
            },
            "trained_prompt_answer_bpb": _answer_metric(
                trained_scores, trained, cfg.vocab_size
            ),
            "heldout_prompt_answer_bpb": _answer_metric(
                heldout_scores, heldout, cfg.vocab_size
            ),
            "heldout_text_bpb": {
                "value": heldout_text_score,
                "evaluated_bytes": len(scored_text_bytes),
                "baselines": {
                    "uniform_bpb": uniform_bits_per_byte(cfg.vocab_size),
                    "unigram_bpb": unigram_bits_per_bytes(scored_text_bytes),
                    "unigram_fit": "evaluated_bytes",
                },
            },
        },
        "tokenizer": {
            "padding_token_id": pad_token_id,
            "end_of_sequence_token_ids": eos_token_ids,
            "newline_stop_token_id": (
                ord("\n") if eos_token_ids is not None and ord("\n") in eos_token_ids else None
            ),
        },
    }
    return record


def write_evaluation_record(record: dict[str, object], output_path: Path) -> None:
    """Writes a single JSON evaluation record."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_prompt_answers(path: Path) -> list[PromptAnswer]:
    """Loads prompt-answer pairs from a JSON array."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("prompt-answer file must be a JSON array")
    pairs: list[PromptAnswer] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each prompt-answer item must be an object")
        prompt = item.get("prompt")
        answer = item.get("answer")
        category = item.get("category")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("each prompt-answer item must contain string prompt and answer")
        if category is not None and not isinstance(category, str):
            raise ValueError("prompt-answer category must be a string when present")
        pairs.append(PromptAnswer(prompt=prompt, answer=answer, category=category))
    return pairs


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return cast(dict[str, object], raw)


def _prompt_answers_sha256(pairs: Sequence[PromptAnswer]) -> str:
    payload = [
        {"prompt": pair.prompt, "answer": pair.answer, "category": pair.category}
        for pair in pairs
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_default_heldout_text_is_disjoint(text: str) -> None:
    """Assert the bundled prose/code corpus excludes known identity-training strings."""
    training_strings: set[str] = set()
    for instance in ModelIdentityTaskFamily().generate_instances(count=7, seed=42):
        training_strings.add(instance.prompt)
        training_strings.add(instance.target_solution)
    for question, prompt, answer in IDENTITY_CHAT_TEMPLATES:
        training_strings.update((question, prompt, answer))
    overlaps = sorted(value for value in training_strings if value and value in text)
    if overlaps:
        raise AssertionError(
            "bundled held-out text contains identity-training strings: "
            + ", ".join(repr(value) for value in overlaps)
        )


def _load_aptivra_model(export_dir: Path) -> ByteCausalModel:
    cfg = model_config_from_export_dir(export_dir, device="cpu")
    lm = AptivraCausalLM.from_export_dir(export_dir, model_config=cfg)
    lm.use_moe_at_inference = False
    return cast(ByteCausalModel, lm)


def _byte_ids(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def _sequence_bits_per_byte(
    model: ByteCausalModel,
    ids: torch.Tensor,
    *,
    target_start: int,
) -> float:
    if target_start <= 0:
        raise ValueError("target_start must be positive for next-byte scoring")
    if target_start >= ids.numel():
        raise ValueError("target_start must point to at least one target byte")

    input_ids = ids.unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model.forward(input_ids)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model.forward must return logits tensor")

    start_position = target_start - 1
    prediction_logits = logits[0, start_position:-1, :]
    targets = input_ids[0, target_start:].to(prediction_logits.device)
    loss = F.cross_entropy(prediction_logits, targets, reduction="sum")
    return float(loss.item() / targets.numel() / math.log(2))


def _bounded_sequence_bits_per_byte(
    model: ByteCausalModel,
    ids: torch.Tensor,
    *,
    target_start: int,
) -> float:
    """Score a target suffix once using windows bounded by the model context length."""
    if target_start <= 0:
        raise ValueError("target_start must be positive for next-byte scoring")
    if target_start >= ids.numel():
        raise ValueError("target_start must point to at least one target byte")
    context_length = model.config.max_seq_len
    if context_length < 2:
        raise ValueError("model context length must be at least two bytes")

    target_cursor = target_start
    total_bits = 0.0
    total_targets = 0
    while target_cursor < ids.numel():
        prior_context = context_length - 1 if target_cursor == target_start else 1
        window_start = max(0, target_cursor - prior_context)
        window_end = min(ids.numel(), window_start + context_length)
        local_target_start = target_cursor - window_start
        window = ids[window_start:window_end]
        target_count = window_end - target_cursor
        score = _sequence_bits_per_byte(
            model,
            window,
            target_start=local_target_start,
        )
        total_bits += score * target_count
        total_targets += target_count
        target_cursor = window_end
    return total_bits / total_targets
