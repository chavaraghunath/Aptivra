"""Output-identical batched generation for standard task instances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from transformers import StoppingCriteriaList

from aptivra.inference.standard_base import (
    DecodedTextStoppingCriteria,
    GenerationResult,
    StandardBase,
    _encode,
    _first_decoded_terminator,
    _GreedyGenerator,
    _validated_stop_sequences,
)
from aptivra.tasks.base import TaskInstance


def generate_task_instance_answers(
    base: StandardBase,
    instances: Sequence[TaskInstance],
    *,
    max_new_tokens: int | None = None,
) -> tuple[GenerationResult, ...]:
    """Generate a greedy batch with results identical to scalar task generation."""
    if not instances:
        return ()
    requested_tokens = (
        base.settings.runtime.max_new_tokens
        if max_new_tokens is None
        else max_new_tokens
    )
    if isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= requested_tokens <= base.settings.runtime.max_new_tokens:
        raise ValueError(
            f"max_new_tokens must be between 1 and {base.settings.runtime.max_new_tokens}"
        )
    eos_id = base.tokenizer.eos_token_id
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise ValueError("standard-base tokenizer must define an integer EOS ID")
    results: list[GenerationResult | None] = [None] * len(instances)
    grouped: dict[tuple[str, ...], list[tuple[int, TaskInstance]]] = {}
    for index, instance in enumerate(instances):
        stops = _validated_stop_sequences(instance.answer_stop_characters)
        grouped.setdefault(stops, []).append((index, instance))
    for stops, members in grouped.items():
        encoded = [_encode(base.tokenizer, instance.prompt) for _, instance in members]
        if any(not row for row in encoded):
            raise ValueError("generation prompt tokenized to zero tokens")
        maximum = max(map(len, encoded))
        if (
            maximum > base.settings.runtime.max_prompt_tokens
            or maximum + requested_tokens > base.settings.runtime.max_sequence_tokens
        ):
            raise ValueError(
                "generation input exceeds configured token limits: "
                f"prompt={maximum}, requested_new={requested_tokens}, "
                f"max_prompt={base.settings.runtime.max_prompt_tokens}, "
                f"max_sequence={base.settings.runtime.max_sequence_tokens}"
            )
        input_ids = torch.tensor(
            [[eos_id] * (maximum - len(row)) + row for row in encoded],
            dtype=torch.long,
            device=base.device,
        )
        attention_mask = torch.tensor(
            [[0] * (maximum - len(row)) + [1] * len(row) for row in encoded],
            dtype=torch.long,
            device=base.device,
        )
        stopping_criteria = (
            StoppingCriteriaList(
                [
                    DecodedTextStoppingCriteria(
                        base.tokenizer,
                        prompt_length=maximum,
                        stop_sequences=stops,
                    )
                ]
            )
            if stops
            else None
        )
        options: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            "max_new_tokens": requested_tokens,
            "eos_token_id": eos_id,
            "pad_token_id": eos_id,
            "use_cache": True,
        }
        if stopping_criteria is not None:
            options["stopping_criteria"] = stopping_criteria
        with torch.inference_mode():
            output = cast(_GreedyGenerator, base.model).generate(**options)
        for row_index, (result_index, _) in enumerate(members):
            raw_generated = [
                int(token_id) for token_id in output[row_index, maximum:].tolist()
            ]
            first_eos = next(
                (index for index, token_id in enumerate(raw_generated) if token_id == eos_id),
                None,
            )
            generated = (
                tuple(raw_generated)
                if first_eos is None
                else tuple(raw_generated[: first_eos + 1])
            )
            for end in range(1, len(generated) + 1):
                decoded_prefix = cast(
                    str,
                    base.tokenizer.decode(
                        list(generated[:end]), skip_special_tokens=False
                    ),
                )
                if _first_decoded_terminator(decoded_prefix, stops) is not None:
                    generated = generated[:end]
                    break
            ended_with_eos = bool(generated and generated[-1] == eos_id)
            text_ids = generated[:-1] if ended_with_eos else generated
            decoded = cast(
                str,
                base.tokenizer.decode(list(text_ids), skip_special_tokens=False),
            )
            terminator = _first_decoded_terminator(decoded, stops)
            if terminator is None:
                text = decoded
                matched = None
                reason: Literal["eos", "terminator", "length"] = (
                    "eos" if ended_with_eos else "length"
                )
            else:
                byte_offset, matched = terminator
                text = decoded.encode("utf-8")[:byte_offset].decode("utf-8")
                reason = "terminator"
            results[result_index] = GenerationResult(
                text=text,
                generated_token_ids=generated,
                stop_reason=reason,
                matched_terminator=matched,
            )
    if any(result is None for result in results):
        raise RuntimeError("batched generation did not produce every requested result")
    return tuple(cast(GenerationResult, result) for result in results)
