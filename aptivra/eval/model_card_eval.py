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

from aptivra.inference.causal_lm import AptivraCausalLM
from aptivra.inference.hub_config import model_config_from_export_dir
from aptivra.model.config import ModelConfig
from aptivra.tasks.identity import ModelIdentityTaskFamily


@dataclass(frozen=True)
class PromptAnswer:
    """Prompt plus answer span to score."""

    prompt: str
    answer: str


class ByteCausalModel(Protocol):
    """Small protocol shared by AptivraCausalLM and test doubles."""

    config: ModelConfig
    device: torch.device

    def eval(self) -> object:
        """Switches to evaluation mode."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns next-byte logits."""


DEFAULT_HELDOUT_PROMPTS: tuple[PromptAnswer, ...] = (
    PromptAnswer(
        prompt="User: State the checkpoint evaluation policy.\nAssistant:",
        answer="Report only measurements produced by the evaluation record.",
    ),
    PromptAnswer(
        prompt="User: Is this checkpoint a general assistant?\nAssistant:",
        answer="No. It is a research prototype unless held-out byte prediction beats baseline.",
    ),
)

DEFAULT_HELDOUT_TEXT = (
    "A model card is an audit document. It should separate measured facts from hypotheses, "
    "mark absent evidence as not measured, and avoid unsupported capability claims."
)


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


def count_state_dict_parameters(path: Path) -> int:
    """Counts tensor parameters in a saved PyTorch state dict."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dict: {path}")
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            total += value.numel()
    return total


def count_export_parameters(export_dir: Path) -> dict[str, int]:
    """Counts backbone, router, expert, and total parameters for an export directory."""
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
    data = text.encode("utf-8")
    if not data:
        raise ValueError("unigram baseline text must not be empty")
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def answer_span_bits_per_byte(model: ByteCausalModel, prompt: str, answer: str) -> float:
    """Scores only the answer bytes conditioned on prompt plus prior answer bytes."""
    prompt_ids = _byte_ids(prompt)
    answer_ids = _byte_ids(answer)
    if answer_ids.numel() == 0:
        raise ValueError("answer must not be empty")
    full_ids = torch.cat([prompt_ids, answer_ids], dim=0)
    return _sequence_bits_per_byte(model, full_ids, target_start=int(prompt_ids.numel()))


def whole_text_bits_per_byte(model: ByteCausalModel, text: str) -> float:
    """Scores each byte after the first byte in a held-out text."""
    ids = _byte_ids(text)
    if ids.numel() < 2:
        raise ValueError("held-out text must contain at least two bytes")
    return _sequence_bits_per_byte(model, ids, target_start=1)


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
    heldout_prompts: Sequence[PromptAnswer] = DEFAULT_HELDOUT_PROMPTS,
    heldout_text: str = DEFAULT_HELDOUT_TEXT,
    hub_revision: str | None = None,
    model_loader: Callable[[Path], ByteCausalModel] | None = None,
) -> dict[str, object]:
    """Evaluates an exported checkpoint and returns one JSON-serializable record."""
    export_path = Path(export_dir)
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
        answer_span_bits_per_byte(model, item.prompt, item.answer) for item in heldout_prompts
    ]
    param_counts = count_export_parameters(export_path)

    record: dict[str, object] = {
        "schema_version": "aptivra.model_card_eval.v1",
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "checkpoint": {
            "hub_revision": hub_revision,
            "backbone_sha256": sha256_file(export_path / "dense" / "backbone.pt"),
        },
        "model": {
            **param_counts,
            "context_length": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "position_embedding": "learned_absolute",
        },
        "evaluation": {
            "moe_enabled": False,
            "trained_prompt_answer_bpb": summarize_bits_per_byte(trained_scores),
            "heldout_prompt_answer_bpb": summarize_bits_per_byte(heldout_scores),
            "heldout_text_bpb": whole_text_bits_per_byte(model, heldout_text),
            "baselines": {
                "uniform_bpb": uniform_bits_per_byte(cfg.vocab_size),
                "unigram_bpb": unigram_bits_per_byte(heldout_text),
            },
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
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError("each prompt-answer item must contain string prompt and answer")
        pairs.append(PromptAnswer(prompt=prompt, answer=answer))
    return pairs


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
    targets = input_ids[0, target_start:]
    loss = F.cross_entropy(prediction_logits, targets, reduction="sum")
    return float(loss.item() / targets.numel() / math.log(2))
