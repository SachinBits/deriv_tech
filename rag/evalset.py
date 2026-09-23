"""Load the evaluation questions. Only `id` and `question` are required per item."""

import json
import os
import sys


def expected_supported(value) -> bool | None:
    """answerable -> True; unanswerable / partially_answerable / anything else -> False; missing -> None."""
    if value is None or not str(value).strip():
        return None
    return str(value).strip().lower() == "answerable"


def load_questions(path: str) -> list[dict]:
    """Return [] if the file is missing; raise ValueError if an item lacks id or question."""
    if not os.path.exists(path):
        print(f"warning: {path} not found; continuing with no questions", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    questions = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not str(item.get("id", "")).strip() \
                or not str(item.get("question", "")).strip():
            raise ValueError(f"{path}[{i}] needs non-empty 'id' and 'question'")
        questions.append({
            "id": str(item["id"]),
            "question": str(item["question"]).strip(),
            "expected_behavior": item.get("expected_behavior"),
            "expected_doc": item.get("expected_doc") or None,
        })
    return questions
