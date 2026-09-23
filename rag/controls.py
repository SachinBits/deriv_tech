"""Optional, user-controllable answer controls: answer length and confidence threshold."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import config
from .ingest import split_sentences

LENGTH_PRESETS = {
    "short": {"max_sentences": 1, "max_words": 40, "max_tokens": 120},
    "medium": {"max_sentences": 2, "max_words": 80, "max_tokens": 250},
    "detailed": {"max_sentences": 4, "max_words": 160, "max_tokens": 500},
}

LengthMode = Literal["short", "medium", "detailed"]


class AskOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_threshold_enabled: bool = True
    confidence_threshold: float = Field(config.MIN_SCORE, ge=0.0, le=0.5)
    length: Optional[LengthMode] = "medium"  # None = off
    generator: Literal["auto", "extractive", "claude"] = "auto"

    def applied(self) -> dict:
        return {
            "confidence_threshold": {
                "enabled": self.confidence_threshold_enabled,
                "value": self.confidence_threshold,
            },
            "length": self.length,
        }


def word_count(text: str) -> int:
    return len(text.split())


def within_length(answer: str, length: Optional[str]) -> bool:
    if not length:
        return True
    preset = LENGTH_PRESETS[length]
    return (len(split_sentences(answer)) <= preset["max_sentences"]
            and word_count(answer) <= preset["max_words"])


def apply_length(answer: str, length: Optional[str]) -> tuple[str, bool]:
    """Trim deterministically at a sentence boundary. Returns (text, was_trimmed).

    Only cuts inside a sentence when that single sentence alone exceeds max_words.
    """
    if not length or within_length(answer, length):
        return answer, False
    preset = LENGTH_PRESETS[length]
    kept = split_sentences(answer)[: preset["max_sentences"]]
    while len(kept) > 1 and word_count(" ".join(kept)) > preset["max_words"]:
        kept.pop()
    text = " ".join(kept)
    if word_count(text) > preset["max_words"]:
        text = " ".join(text.split()[: preset["max_words"]]).rstrip(",;:") + "…"
    return text, True


def length_instruction(length: Optional[str]) -> str:
    if not length:
        return ""
    preset = LENGTH_PRESETS[length]
    unit = "sentence" if preset["max_sentences"] == 1 else "sentences"
    return f"Answer in at most {preset['max_sentences']} {unit} / {preset['max_words']} words."


def max_tokens_for(length: Optional[str]) -> int:
    return LENGTH_PRESETS[length]["max_tokens"] if length else config.CLAUDE_MAX_TOKENS
