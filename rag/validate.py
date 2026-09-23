"""Deterministic, fail-closed post-generation validator (V1-V6)."""

import re

from . import config
from .controls import apply_length, within_length
from .obs import log

_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate(record: dict, hits: list[dict], length: str | None = None,
             question: str = "") -> tuple[dict, list[dict]]:
    answer = (record.get("answer") or "").strip()
    supported = bool(record.get("supported"))
    cited = list(dict.fromkeys(record.get("cited_chunk_ids") or []))
    text_by_id = {h["chunk_id"]: h["text"] for h in hits}
    checks = []

    # V1: the answer is not blank.
    checks.append(_check("non_empty", bool(answer), "ok" if answer else "answer is blank"))

    # V2: a supported answer must cite something.
    v2 = not supported or bool(cited)
    checks.append(_check("supported_has_citation", v2,
                         f"{len(cited)} citation(s)" if supported else "not applicable (refusal)"))

    # V3: every citation must be a retrieved chunk; invalid ones are dropped.
    valid = [c for c in cited if c in text_by_id]
    dropped = [c for c in cited if c not in text_by_id]
    v3 = bool(valid) if supported else True
    detail = f"dropped {dropped}" if dropped else "all citations retrieved"
    checks.append(_check("citations_in_retrieved", v3, detail if cited else "no citations"))

    # V4: an unsupported answer must start with the refusal sentence.
    v4 = supported or answer.startswith(config.REFUSAL)
    checks.append(_check("unsupported_marked", v4,
                         "ok" if v4 else "refusal prefix added"))

    # V5: every number must be grounded. Supported answers: in the cited chunks.
    # Refusals may only repeat numbers from the question or the retrieved text.
    if supported:
        allowed = set().union(*(numbers_in(text_by_id[c]) for c in valid)) if valid else set()
    else:
        allowed = numbers_in(question).union(*(numbers_in(h["text"]) for h in hits))
    ungrounded = sorted(numbers_in(answer) - allowed)
    v5 = not ungrounded
    checks.append(_check("numbers_grounded", v5,
                         f"ungrounded: {ungrounded}" if ungrounded else "ok"))

    # V6 (only when a length mode is set): over-long answers are trimmed, not refused.
    was_trimmed = False
    if length:
        if not supported:
            checks.append(_check("length_within_limit", True, "refusal (not length-limited)"))
        elif within_length(answer, length):
            checks.append(_check("length_within_limit", True, f"within {length}"))
        else:
            answer, was_trimmed = apply_length(answer, length)
            checks.append(_check("length_within_limit", True, "trimmed"))

    refusal_reason = None if supported else record.get("reason")
    if supported and not (checks[0]["passed"] and v2 and v3 and v5):
        answer, valid, supported, refusal_reason = config.REFUSAL, [], False, "validation_failed"
    elif not supported and not v4:
        answer = f"{config.REFUSAL} {answer}".strip()
    elif not supported and not v5:
        answer = config.REFUSAL  # never let a refusal carry ungrounded numbers

    final = {
        "answer": answer,
        "cited_chunk_ids": valid,
        "supported": supported,
        "refusal_reason": refusal_reason,
        "pre_validation_supported": bool(record.get("supported")),
        "validation_passed": all(c["passed"] for c in checks),
        "was_trimmed": was_trimmed,
    }
    log("validate", passed=final["validation_passed"], supported=supported,
        failed=[c["name"] for c in checks if not c["passed"]], refusal_reason=refusal_reason)
    return final, checks
