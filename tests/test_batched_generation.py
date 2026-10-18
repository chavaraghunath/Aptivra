"""Behavioral and mutation tests for output-identical batched generation."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
import torch

from aptivra.inference.batched_generation import generate_task_instance_answers
from aptivra.inference.standard_base import GenerationResult, StandardBase
from aptivra.tasks.base import TaskInstance


class _Tokenizer:
    eos_token_id: object = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {
            "a": [2],
            "b": [3, 4],
            "c": [7, 8, 9],
            "too-long": [1, 2, 3, 4],
            "empty": [],
        }[text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        pieces = {
            0: "<eos>",
            5: "A",
            6: "B",
            9: "\n",
            10: "é",
            11: "\x00",
            1000: "<eos>",
        }
        return "".join(pieces[token] for token in token_ids)


class _Generator:
    def __init__(self, suffixes: Sequence[Sequence[Sequence[int]]]) -> None:
        self._suffixes = iter(suffixes)
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        input_ids = cast(torch.Tensor, kwargs["input_ids"])
        suffixes = torch.tensor(
            next(self._suffixes), dtype=torch.long, device=input_ids.device
        )
        assert suffixes.shape[0] == input_ids.shape[0]
        return torch.cat((input_ids, suffixes), dim=1)


def _base(
    generator: _Generator,
    *,
    tokenizer: _Tokenizer | None = None,
    max_prompt_tokens: int = 3,
    max_sequence_tokens: int = 6,
    max_new_tokens: int = 3,
) -> StandardBase:
    runtime = SimpleNamespace(
        max_prompt_tokens=max_prompt_tokens,
        max_sequence_tokens=max_sequence_tokens,
        max_new_tokens=max_new_tokens,
    )
    return cast(
        StandardBase,
        SimpleNamespace(
            model=generator,
            tokenizer=tokenizer or _Tokenizer(),
            settings=SimpleNamespace(runtime=runtime),
            device=torch.device("cpu"),
        ),
    )


def _instance(prompt: str, stops: tuple[str, ...] = ()) -> TaskInstance:
    return TaskInstance("task", "tiny", prompt, "A", answer_stop_characters=stops)


def test_empty_batch_does_not_access_base() -> None:
    assert generate_task_instance_answers(cast(StandardBase, None), ()) == ()


def test_public_annotations_preserve_the_batch_contract() -> None:
    hints = get_type_hints(generate_task_instance_answers)

    assert hints == {
        "base": StandardBase,
        "instances": Sequence[TaskInstance],
        "max_new_tokens": int | None,
        "return": tuple[GenerationResult, ...],
    }


@pytest.mark.parametrize("value", [True, 1.0, 0, -1, 4])
def test_new_token_bound_rejects_invalid_values(value: object) -> None:
    base = _base(_Generator([[[5]]]))

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_task_instance_answers(base, (_instance("a"),), max_new_tokens=cast(Any, value))


def test_default_token_bound_and_generation_options_are_exact() -> None:
    generator = _Generator([[[5, 0, 0], [6, 5, 0]]])
    base = _base(generator)

    results = generate_task_instance_answers(base, (_instance("a"), _instance("b")))

    assert results == (
        GenerationResult("A", (5, 0), "eos"),
        GenerationResult("BA", (6, 5, 0), "eos"),
    )
    assert len(generator.calls) == 1
    options = generator.calls[0]
    assert cast(torch.Tensor, options["input_ids"]).tolist() == [[0, 2], [3, 4]]
    assert cast(torch.Tensor, options["attention_mask"]).tolist() == [[0, 1], [1, 1]]
    assert options["do_sample"] is False
    assert options["max_new_tokens"] == 3
    assert options["eos_token_id"] == 0
    assert options["pad_token_id"] == 0
    assert options["use_cache"] is True
    assert "stopping_criteria" not in options


def test_explicit_minimum_token_bound_and_length_stop() -> None:
    generator = _Generator([[[5]]])

    result = generate_task_instance_answers(
        _base(generator), (_instance("a"),), max_new_tokens=1
    )

    assert result == (GenerationResult("A", (5,), "length"),)
    assert generator.calls[0]["max_new_tokens"] == 1


def test_eos_comparisons_do_not_depend_on_identity_or_numeric_order() -> None:
    tokenizer = _Tokenizer()
    tokenizer.eos_token_id = 1000
    generator = _Generator([[[5, 1000, 6], [5, 6, 6]]])

    results = generate_task_instance_answers(
        _base(generator, tokenizer=tokenizer),
        (_instance("a"), _instance("b")),
    )

    assert results == (
        GenerationResult("A", (5, 1000), "eos"),
        GenerationResult("ABB", (5, 6, 6), "length"),
    )


def test_tokenizer_requires_a_non_boolean_integer_eos() -> None:
    for eos in (None, True, "0"):
        tokenizer = _Tokenizer()
        tokenizer.eos_token_id = eos
        with pytest.raises(ValueError, match="integer EOS ID"):
            generate_task_instance_answers(
                _base(_Generator([[[5]]]), tokenizer=tokenizer), (_instance("a"),)
            )


def test_zero_token_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero tokens"):
        generate_task_instance_answers(
            _base(_Generator([[[5]]])), (_instance("empty"),)
        )


@pytest.mark.parametrize(
    ("prompt", "max_prompt", "max_sequence", "requested"),
    [("too-long", 3, 8, 1), ("c", 3, 5, 3)],
)
def test_prompt_and_sequence_limits_are_independently_enforced(
    prompt: str,
    max_prompt: int,
    max_sequence: int,
    requested: int,
) -> None:
    base = _base(
        _Generator([[[5]]]),
        max_prompt_tokens=max_prompt,
        max_sequence_tokens=max_sequence,
    )

    with pytest.raises(ValueError, match="exceeds configured token limits"):
        generate_task_instance_answers(
            base, (_instance(prompt),), max_new_tokens=requested
        )


def test_stop_groups_preserve_input_order_and_first_complete_answer() -> None:
    generator = _Generator(
        [
            [[5, 9, 6], [10, 11, 6]],
            [[6, 0]],
        ]
    )
    instances = (
        _instance("a", ("\n",)),
        _instance("b"),
        _instance("c", ("\n",)),
    )

    results = generate_task_instance_answers(_base(generator), instances)

    assert results == (
        GenerationResult("A", (5, 9), "terminator", "\n"),
        GenerationResult("B", (6, 0), "eos"),
        GenerationResult("é\x00B", (10, 11, 6), "length"),
    )
    assert len(generator.calls) == 2
    first_options = generator.calls[0]
    assert cast(torch.Tensor, first_options["input_ids"]).tolist() == [
        [0, 0, 2],
        [7, 8, 9],
    ]
    assert cast(torch.Tensor, first_options["attention_mask"]).tolist() == [
        [0, 0, 1],
        [1, 1, 1],
    ]
    assert "stopping_criteria" in first_options
    assert "stopping_criteria" not in generator.calls[1]
