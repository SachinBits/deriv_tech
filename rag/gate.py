"""Deterministic pre-generation evidence gate."""

import re

from . import config
from .obs import log

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def key_terms(question: str) -> list[str]:
    """Acronyms (ALL CAPS, 2+ chars), CamelCase/mixed case (GraphQL) and tokens with digits (P1).

    Terms keep their original casing for display; matching is case-insensitive.
    """
    shouting = not re.search(r"[a-z]", question)  # an all-caps question has no usable acronyms
    terms = []
    for tok in _TOKEN.findall(question):
        is_acronym = not shouting and len(tok) >= 2 and tok.isupper()
        is_mixed = bool(re.search(r"[a-z][A-Z]", tok))
        has_digit = any(ch.isdigit() for ch in tok)
        if (is_acronym or is_mixed or has_digit) and tok.lower() not in {t.lower() for t in terms}:
            terms.append(tok)
    return terms


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def evidence_gate(question: str, hits: list[dict], threshold_enabled: bool = True,
                  threshold: float = config.MIN_SCORE) -> tuple[bool, str | None, list[str]]:
    top = hits[0]["score"] if hits else 0.0
    ok, reason, missing = True, None, []

    # Rule A (optional): confidence threshold on the best retrieval score.
    if threshold_enabled and top < threshold:
        ok, reason = False, "low_retrieval_score"
    else:
        # Rule B (always on): every key term must appear in the retrieved text.
        combined = " ".join(h["text"] for h in hits).lower()
        missing = [t for t in key_terms(question) if not _contains(combined, t.lower())]
        if missing:
            ok, reason = False, "key_term_not_in_docs"

    log("gate", ok=ok, reason=reason, missing=missing, top_score=top,
        threshold_enabled=threshold_enabled, threshold=threshold)
    return ok, reason, missing
