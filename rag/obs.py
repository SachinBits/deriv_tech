"""Structured JSONL logging: one line per pipeline stage."""

import contextvars
import json
import os
import threading
import time
import uuid

from . import config

_run_id = contextvars.ContextVar("run_id", default=uuid.uuid4().hex[:12])
_lock = threading.Lock()


def new_run_id() -> str:
    """Start a new run (a batch, a CLI call or an API request) and return its id."""
    run_id = uuid.uuid4().hex[:12]
    _run_id.set(run_id)
    return run_id


def current_run_id() -> str:
    return _run_id.get()


def log(stage: str, **fields) -> None:
    record = {"ts": round(time.time(), 3), "run_id": _run_id.get(), "stage": stage, **fields}
    path = os.environ.get("RAG_LOG_PATH") or config.root_path(config.LOG_PATH)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(record, default=str)
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # logging must never break answering
