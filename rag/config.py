"""Pipeline constants. Thresholds were calibrated with run_pipeline.py (see README)."""

DOCS_DIR = "docs"
QUESTIONS_PATH = "questions.json"
PROMPT_PATH = "prompts/answer.txt"
LOG_PATH = "logs/pipeline.jsonl"

TOP_K = 4
CHUNK_CHARS = 400

# Evidence gate: minimum cosine similarity of the best retrieved chunk.
MIN_SCORE = 0.10
# Extractive generator: fraction of the question's content tokens a sentence must contain.
MIN_COVERAGE = 0.6
# Extra sentences for longer answers need at least this coverage.
EXTRA_SENTENCE_COVERAGE = 0.3

REFUSAL = "I can't answer this from the available documentation."

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 400


def root_path(path: str) -> str:
    """Resolve a relative path against the repo root so scripts work from any cwd."""
    import os

    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
