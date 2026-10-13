"""Post-process byte-LM chat output for readable UI replies."""

import re

from aptivra.inference.chat_canonical import (
    APTIVRA_INTRO,
    CAPABILITIES_SUMMARY,
    CREATOR_BIO_REPLY,
)

_CODE_MARKERS = ("SELECT ", "def ", "elif ", "return ", "import ", "class ")

_IDENTITY_PATTERNS = (
    r"I was created by Raghunath Chava\.?",
    r"My name is Aptivra(?:, an autonomous continual learning language model)?\.?",
    r"I am Aptivra(?:, an autonomous continual learning language model[^.!?]*)[.!?]?",
    r"Raghunath Chava is my creator[^.!?]*[.!?]?",
    r"So far I know that I am Aptivra[^.!?]*[.!?]?",
)


def sanitize_chat_reply(text: str, *, max_chars: int = 280) -> str:
    """Keeps the first coherent answer line; drops code-SQL drift and runaway text."""
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        return cleaned

    for pattern in _IDENTITY_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = match.group(0).strip()
            return cleaned[:max_chars]

    if re.search(r"created by Raghunath Chava", cleaned, flags=re.IGNORECASE) and not re.match(
        r"^I was created", cleaned, flags=re.IGNORECASE
    ):
        return APTIVRA_INTRO[:max_chars]

    if re.search(r"raghunath chava", cleaned, flags=re.IGNORECASE) and re.search(
        r"creator|engineer", cleaned, flags=re.IGNORECASE
    ):
        return CREATOR_BIO_REPLY[:max_chars]

    if re.search(r"what do you know|so far i know", cleaned, flags=re.IGNORECASE):
        return CAPABILITIES_SUMMARY[:max_chars]

    if "User:" in cleaned:
        cleaned = cleaned.split("User:", 1)[0].strip()

    for marker in _CODE_MARKERS:
        idx = cleaned.find(marker)
        if idx > 0:
            cleaned = cleaned[:idx].strip()

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        cleaned = lines[0]

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "…"

    # Prefer first sentence when model runs on
    m = re.match(r"^(.{20,200}?[.!?])(\s|$)", cleaned)
    if m:
        cleaned = m.group(1).strip()

    return cleaned
