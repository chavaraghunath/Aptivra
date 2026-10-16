"""Tests for unsanitized identity quality evaluation."""

from dataclasses import FrozenInstanceError

import pytest

from aptivra.inference.chat_canonical import canonical_identity_records
from aptivra.inference.chat_quality import (
    ChatQualityCase,
    canonical_prefix_ok,
    default_identity_cases,
    identity_bundle_ok,
    passes_case,
)


def test_passes_case_blocks_sql_on_raw_output() -> None:
    case = ChatQualityCase("What are you?", ("aptivra",))
    assert not passes_case("I am Aptivra. SELECT * FROM users", case)


def test_default_identity_cases_cover_exact_canonical_records() -> None:
    records = canonical_identity_records()
    cases = default_identity_cases()
    assert [case.user_message for case in cases] == [
        record.question for record in records
    ]
    assert cases[0].required_substrings == ("aptivra",)
    assert [case.required_substrings for case in cases[1:]] == [
        (record.answer.split()[0].lower(),) for record in records[1:]
    ]
    assert all(case.allowed_metric is None for case in cases)


def test_quality_case_defaults_are_exact_and_immutable() -> None:
    case = ChatQualityCase("question", ("required",))
    assert case.forbidden_substrings == (
        "SELECT ",
        "def ",
        "elif ",
        "binary_search",
    )
    assert case.allowed_metric is None
    with pytest.raises(FrozenInstanceError):
        case.user_message = "changed"


@pytest.mark.parametrize(
    "raw_reply",
    [
        "",
        "   ",
        "required SELECT value",
        "required def function",
        "required elif branch",
        "required binary_search helper",
        "required autonomous behavior",
        "unrelated safe text",
    ],
)
def test_passes_case_rejects_each_raw_failure_mode(raw_reply: str) -> None:
    assert not passes_case(raw_reply, ChatQualityCase("question", ("required",)))


def test_passes_case_requires_all_substrings_case_insensitively() -> None:
    case = ChatQualityCase("question", ("first", "SECOND"), forbidden_substrings=())
    assert passes_case("  FIRST and second are present  ", case)
    assert not passes_case("first only", case)


def test_canonical_prefix_ok() -> None:
    canonical = canonical_identity_records()[0].answer
    assert canonical_prefix_ok(canonical, canonical)
    assert canonical_prefix_ok(canonical, canonical, min_ratio=1.0)
    assert not canonical_prefix_ok("I am Aptivra, an autonomous model.", canonical)


def test_canonical_prefix_ratio_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        canonical_prefix_ok("abcdefghijkl", "abcdefghijkl", 1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("ratio", [-1.0, 0.0, 1.000001, 2.0])
def test_canonical_prefix_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        canonical_prefix_ok("abcdefghijkl", "abcdefghijklmnop", min_ratio=ratio)


def test_canonical_prefix_enforces_floor_prefix_and_conjunction() -> None:
    canonical = "abcdefghijklmnopqrst"
    assert canonical_prefix_ok("abcdefghijkl", canonical, min_ratio=0.5)
    assert not canonical_prefix_ok("abcdefghijk", canonical, min_ratio=0.5)
    assert canonical_prefix_ok("ABCDEFGHIJKLMNO", canonical, min_ratio=0.75)
    assert not canonical_prefix_ok("xbcdefghijklmno", canonical, min_ratio=0.75)
    assert not canonical_prefix_ok("", canonical, min_ratio=0.5)


def test_canonical_prefix_uses_configured_default_ratio() -> None:
    canonical = "abcdefghijklmnopqrst"
    assert canonical_prefix_ok("abcdefghijklm", canonical)
    assert not canonical_prefix_ok("abcdefghijkl", canonical)


def test_identity_bundle_ok_scores_raw_exact_answers() -> None:
    records = canonical_identity_records()
    raw = {record.question: record.answer for record in records}
    assert identity_bundle_ok(raw)
    raw[records[0].question] = f"{records[0].answer} SELECT * FROM users"
    assert not identity_bundle_ok(raw)


def test_identity_bundle_requires_every_exact_unsanitized_answer() -> None:
    records = canonical_identity_records()
    exact = {record.question: record.answer for record in records}
    missing = dict(exact)
    missing.pop(records[-1].question)
    assert not identity_bundle_ok(missing)
    changed = dict(exact)
    changed[records[-1].question] = records[-1].answer.lower()
    assert not identity_bundle_ok(changed)
    lexically_lower = dict(exact)
    upper_answer = records[-1].answer.upper()
    assert upper_answer < records[-1].answer
    lexically_lower[records[-1].question] = upper_answer
    assert not identity_bundle_ok(lexically_lower)
    padded = dict(exact)
    padded[records[-1].question] = f"  {records[-1].answer}  "
    assert identity_bundle_ok(padded)


def test_identity_bundle_propagates_case_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = canonical_identity_records()
    exact = {record.question: record.answer for record in records}
    monkeypatch.setattr(
        "aptivra.inference.chat_quality.passes_case",
        lambda raw_reply, case: False,
    )
    assert not identity_bundle_ok(exact)
