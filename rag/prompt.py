"""Build the grounded-answer prompt from prompts/answer.txt."""

from functools import lru_cache

from . import config
from .controls import length_instruction


@lru_cache(maxsize=4)
def load_template(path: str = config.PROMPT_PATH) -> str:
    with open(config.root_path(path), encoding="utf-8") as f:
        return f.read()


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)


def build_prompt(question: str, hits: list[dict], length: str | None = None) -> str:
    # str.replace rather than str.format: the template contains literal JSON braces.
    return (load_template()
            .replace("{length_instruction}", length_instruction(length))
            .replace("{context}", format_context(hits))
            .replace("{question}", question.strip()))
