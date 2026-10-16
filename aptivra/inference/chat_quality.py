"""Unsanitized acceptance checks for evidence-bound identity chat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, assert_never

from aptivra.inference.chat_canonical import (
    IdentityManifest,
    PromptIntent,
    ReleaseMetric,
    allowed_release_metric,
    assert_permitted_identity_target,
    canonical_identity_records,
    load_identity_manifest,
)
from aptivra.inference.chat_prompt import load_identity_settings

# Cosmic Ray treats postponed PEP 604 annotations as executable binary
# expressions, creating equivalent arithmetic mutants.  Classic Optional
# preserves the public types while keeping the mutation inventory behavioral.


@dataclass(frozen=True)
class ChatQualityCase:
    user_message: str
    required_substrings: tuple[str, ...]
    forbidden_substrings: tuple[str, ...] = (
        "SELECT ",
        "def ",
        "elif ",
        "binary_search",
    )
    allowed_metric: Optional[ReleaseMetric] = None  # noqa: UP045


def _required_substrings(
    prompt_intent: PromptIntent,
    answer: str,
    model_name: str,
) -> tuple[str, ...]:
    match prompt_intent:
        case PromptIntent.WHAT_ARE_YOU:
            return (model_name.lower(),)
        case (
            PromptIntent.ARCHITECTURE
            | PromptIntent.CHAT_LEARNING
            | PromptIntent.CREATOR
            | PromptIntent.CAPABILITIES
        ):
            return (answer.split()[0].lower(),)
        case unreachable:
            assert_never(unreachable)


def default_identity_cases(
    manifest: Optional[IdentityManifest] = None,  # noqa: UP045
) -> list[ChatQualityCase]:
    """Build checks from the same manifest-generated records used for training."""
    bound = manifest or load_identity_manifest()
    release_record = bound.release_evaluation.record
    cases: list[ChatQualityCase] = []
    for record in canonical_identity_records(bound):
        cases.append(
            ChatQualityCase(
                record.question,
                _required_substrings(
                    record.prompt_intent,
                    record.answer,
                    bound.model.name,
                ),
                allowed_metric=allowed_release_metric(
                    record.prompt_intent,
                    release_record,
                ),
            )
        )
    return cases


def passes_case(raw_reply: str, case: ChatQualityCase) -> bool:
    """Score raw model text directly; no sanitizer is called on this path."""
    text = raw_reply.strip()
    if not text:
        return False
    try:
        assert_permitted_identity_target(text, allowed_metric=case.allowed_metric)
    except ValueError:
        return False
    lowered = text.lower()
    if any(forbidden.lower() in lowered for forbidden in case.forbidden_substrings):
        return False
    return all(required.lower() in lowered for required in case.required_substrings)


def canonical_prefix_ok(
    raw_reply: str,
    canonical: str,
    *,
    min_ratio: Optional[float] = None,  # noqa: UP045
) -> bool:
    """Accept a substantial exact prefix only when it contains no forbidden claim."""
    if min_ratio is None:
        min_ratio = load_identity_settings().quality.minimum_canonical_prefix_ratio
    if not 0.0 < min_ratio <= 1.0:
        raise ValueError("canonical prefix ratio must be in (0, 1]")
    raw = raw_reply.strip()
    if not raw:
        return False
    try:
        assert_permitted_identity_target(raw)
    except ValueError:
        return False
    canonical_lower = canonical.lower()
    raw_lower = raw.lower()
    minimum = max(12, int(len(canonical) * min_ratio))
    return len(raw) >= minimum and canonical_lower.startswith(raw_lower)


def identity_bundle_ok(
    raw_by_question: dict[str, str],
    manifest: Optional[IdentityManifest] = None,  # noqa: UP045
) -> bool:
    """Require every manifest-generated canonical answer from raw model outputs."""
    bound = manifest or load_identity_manifest()
    release_record = bound.release_evaluation.record
    for record in canonical_identity_records(bound):
        raw = raw_by_question.get(record.question, "")
        if raw.strip() != record.answer:
            return False
        if not passes_case(
            raw,
            ChatQualityCase(
                record.question,
                (),
                allowed_metric=allowed_release_metric(
                    record.prompt_intent,
                    release_record,
                ),
            ),
        ):
            return False
    return True
