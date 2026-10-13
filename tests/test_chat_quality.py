"""Unit tests for chat quality helpers."""

from aptivra.inference.chat_canonical import APTIVRA_INTRO, CREATOR_REPLY, NAME_REPLY
from aptivra.inference.chat_quality import (
    canonical_prefix_ok,
    default_identity_cases,
    identity_bundle_ok,
    passes_case,
)


def test_passes_case_blocks_sql() -> None:
    case = default_identity_cases()[0]
    assert not passes_case("My name is Aptivra. SELECT * FROM users", case)


def test_canonical_prefix_ok() -> None:
    assert canonical_prefix_ok(NAME_REPLY, NAME_REPLY)
    assert canonical_prefix_ok("My name is Aptivra, an autonomous", NAME_REPLY)


def test_identity_bundle_ok() -> None:
    raw = {
        "What is Aptivra?": APTIVRA_INTRO,
        "What is your name?": NAME_REPLY,
        "Who created you?": CREATOR_REPLY,
    }
    assert identity_bundle_ok(raw)
