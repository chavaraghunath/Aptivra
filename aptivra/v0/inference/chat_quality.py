"""Acceptance checks for identity chat (raw model output, no UI sanitization)."""

from __future__ import annotations

from dataclasses import dataclass

from aptivra.v0.inference.chat_canonical import APTIVRA_INTRO, CREATOR_REPLY, NAME_REPLY


@dataclass(frozen=True)
class ChatQualityCase:
    user_message: str
    required_substrings: tuple[str, ...]
    forbidden_substrings: tuple[str, ...] = ("SELECT ", "def ", "elif ", "binary_search")


def default_identity_cases() -> list[ChatQualityCase]:
    return [
        ChatQualityCase("What is Aptivra?", ("aptivra", "raghunath chava")),
        ChatQualityCase("What is your name?", ("my name is aptivra",)),
        ChatQualityCase("Who created you?", ("raghunath chava",)),
        ChatQualityCase("Who is Raghunath Chava?", ("raghunath chava", "creator")),
        ChatQualityCase("What do you know so far?", ("aptivra", "continual learning")),
    ]


def passes_case(raw_reply: str, case: ChatQualityCase) -> bool:
    text = raw_reply.strip().lower()
    if not text:
        return False
    for bad in case.forbidden_substrings:
        if bad.lower() in text:
            return False
    return all(req in text for req in case.required_substrings)


def canonical_prefix_ok(raw_reply: str, canonical: str, *, min_ratio: float = 0.65) -> bool:
    raw = raw_reply.strip()
    if not raw:
        return False
    raw_l = raw.lower()
    canon_l = canonical.lower()
    if canon_l.startswith(raw_l) and len(raw_l) >= 12:
        return True
    n = max(12, int(len(canonical) * min_ratio))
    needle = canonical[:n].lower()
    return raw_l.startswith(needle) or needle in raw_l


def identity_bundle_ok(raw_by_question: dict[str, str]) -> bool:
    checks = [
        ("What is Aptivra?", APTIVRA_INTRO, ("aptivra", "raghunath chava")),
        ("What is your name?", NAME_REPLY, ("my name is aptivra",)),
        ("Who created you?", CREATOR_REPLY, ("raghunath chava",)),
    ]
    for question, canonical, reqs in checks:
        raw = raw_by_question.get(question, "")
        case = ChatQualityCase(question, reqs)
        if passes_case(raw, case) or canonical_prefix_ok(raw, canonical):
            continue
        return False
    return True
