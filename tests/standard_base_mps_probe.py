"""Isolated CPU/MPS decision probe used by the ADR-0011 integration tests."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import LlamaForCausalLM

from aptivra.inference.standard_base import (
    StandardBase,
    collate_task_instances,
    forward_task_batch,
    generate_task_answer,
    load_standard_base,
    load_standard_base_settings,
)
from tests.test_standard_base import _TINY_MODEL_SEED, _task, _tiny_snapshot


@dataclass(frozen=True)
class _ProbeCase:
    case_id: str
    prompt: str
    target: str
    max_new_tokens: int


_TINY_CASES = (
    _ProbeCase("ordinary-short", "short prompt", "answer", 6),
    _ProbeCase("ordinary-numeric", "one two", "target", 6),
    _ProbeCase("narrow-margin", "<unk> <unk> y <unk>", "answer", 6),
)
_REAL_CASES = (
    _ProbeCase("capital", "The capital of France is", "Paris.", 4),
    _ProbeCase("arithmetic", "2 + 2 =", "4", 4),
    _ProbeCase("greeting", "Write a short greeting:", "Hello!", 4),
)


def _decision(logits: torch.Tensor) -> dict[str, object]:
    values, indices = torch.topk(logits.float(), k=2)
    return {
        "top1_token_id": int(indices[0].item()),
        "top2_token_id": int(indices[1].item()),
        "top1_top2_margin": float((values[0] - values[1]).detach().cpu()),
    }


def _generation_decisions(
    base: StandardBase,
    prompt: str,
    generated_token_ids: tuple[int, ...],
) -> list[dict[str, object]]:
    prompt_ids = [
        int(token_id)
        for token_id in base.tokenizer.encode(prompt, add_special_tokens=False)
    ]
    trace_ids = [*prompt_ids, *generated_token_ids[:-1]]
    input_ids = torch.tensor([trace_ids], dtype=torch.long, device=base.device)
    with torch.inference_mode():
        output = base.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            return_dict=True,
        )
    records: list[dict[str, object]] = []
    for index, generated_token_id in enumerate(generated_token_ids):
        decision = _decision(output.logits[0, len(prompt_ids) - 1 + index])
        records.append(
            {
                "index": index,
                "generated_token_id": generated_token_id,
                **decision,
            }
        )
    return records


def _case_record(base: StandardBase, case: _ProbeCase) -> dict[str, object]:
    prompt_ids = base.tokenizer.encode(case.prompt, add_special_tokens=False)
    batch = collate_task_instances(
        base,
        [_task(case.case_id, case.prompt, case.target)],
    )
    with torch.inference_mode():
        output = forward_task_batch(base, batch)
    generated = generate_task_answer(
        base,
        case.prompt,
        max_new_tokens=case.max_new_tokens,
    )
    generation_decisions = _generation_decisions(
        base,
        case.prompt,
        generated.generated_token_ids,
    )
    if base.device.type == "mps":
        torch.mps.synchronize()
    loss = cast(torch.Tensor, output.loss)
    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "generated_token_ids": list(generated.generated_token_ids),
        "stop_reason": generated.stop_reason,
        **_decision(output.logits[0, len(prompt_ids) - 1]),
        "generation_decisions": generation_decisions,
        "loss": float(loss.detach().cpu()),
    }


def _records(
    base: StandardBase,
    cases: tuple[_ProbeCase, ...],
    runs: int,
) -> list[dict[str, object]]:
    return [
        {
            "run": run,
            "prompts": [_case_record(base, case) for case in cases],
        }
        for run in range(1, runs + 1)
    ]


def _result(
    *,
    kind: str,
    device: str,
    base: StandardBase,
    cases: tuple[_ProbeCase, ...],
    runs: int,
) -> dict[str, object]:
    return {
        "kind": kind,
        "cached": True,
        "device": base.device.type,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "stock_llama": isinstance(base.model, LlamaForCausalLM),
        "tied": (
            base.model.get_input_embeddings().weight
            is base.model.get_output_embeddings().weight
        ),
        "dtypes": sorted({str(parameter.dtype) for parameter in base.model.parameters()}),
        "requested_device": device,
        "records": _records(base, cases, runs),
    }


def _tiny_records(device: str, runs: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        config_path, snapshot = _tiny_snapshot(Path(directory))
        base = load_standard_base(
            load_standard_base_settings(config_path),
            snapshot_path=snapshot,
            device=device,
        )
        result = _result(
            kind="tiny",
            device=device,
            base=base,
            cases=_TINY_CASES,
            runs=runs,
        )
        result["fixture"] = {
            "model": "stock LlamaForCausalLM",
            "weight_seed": _TINY_MODEL_SEED,
            "weight_initializer": "normal(mean=0, std=config.initializer_range)",
            "normalization_weights": 1.0,
        }
        return result


def _real_records(device: str, runs: int) -> dict[str, object]:
    settings = load_standard_base_settings()
    cached_config = try_to_load_from_cache(
        settings.source.repository,
        "config.json",
        revision=settings.source.revision,
    )
    if not isinstance(cached_config, str):
        return {"kind": "real", "cached": False}
    enabled = settings.model_copy(
        update={"feature": settings.feature.model_copy(update={"enabled": True})}
    )
    base = load_standard_base(
        enabled,
        snapshot_path=Path(cached_config).parent,
        device=device,
    )
    return _result(
        kind="real",
        device=device,
        base=base,
        cases=_REAL_CASES,
        runs=runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("tiny", "real"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    result = (
        _tiny_records(args.device, args.runs)
        if args.kind == "tiny"
        else _real_records(args.device, args.runs)
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
