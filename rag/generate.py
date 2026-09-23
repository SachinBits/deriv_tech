"""Answer generators. Both return {answer, cited_chunk_ids, supported} and fail closed."""

import json
import os
import re

from sklearn.metrics.pairwise import linear_kernel

from . import config
from .controls import LENGTH_PRESETS, max_tokens_for, word_count
from .ingest import split_sentences
from .obs import log
from .prompt import build_prompt
from .retrieve import make_vectorizer

# Auxiliary verbs that sklearn's English stop list keeps but that carry no content.
_EXTRA_STOP = {"does", "did", "doing"}


def stem(token: str) -> str:
    """Light stemmer: strip ing / ed / es / s from tokens longer than 4 chars."""
    if len(token) <= 4:
        return token
    if token.endswith("ing"):
        return token[:-3]
    if token.endswith("ed") and not token.endswith("eed"):  # keep exceed, proceed
        return token[:-2]
    if re.search(r"(?:ss|x|ch|sh)es$", token):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class ExtractiveGenerator:
    """Offline generator: returns verbatim sentences from the retrieved chunks."""

    name = "extractive"

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer if vectorizer is not None else make_vectorizer()
        self._analyzer = self.vectorizer.build_analyzer()

    def content_tokens(self, text: str) -> list[str]:
        out = []
        for tok in self._analyzer(text):
            if " " in tok or tok in _EXTRA_STOP:  # unigrams only
                continue
            s = stem(tok)
            if s not in out:
                out.append(s)
        return out

    def _similarities(self, question: str, sentences: list[str]) -> list[float]:
        vec = self.vectorizer
        if not hasattr(vec, "vocabulary_"):  # not fitted: fit a local model on the evidence
            vec = make_vectorizer()
            try:
                vec.fit(sentences + [question])
            except ValueError:  # empty vocabulary
                return [0.0] * len(sentences)
        return linear_kernel(vec.transform([question]), vec.transform(sentences)).ravel().tolist()

    def generate(self, question: str, hits: list[dict], prompt: str | None = None,
                 length: str | None = None) -> dict:
        # `prompt` is accepted for interface parity with ClaudeGenerator and ignored here.
        q_tokens = self.content_tokens(question)
        q_numbers = set(re.findall(r"\d+", question))
        candidates, seen = [], set()
        for hit in hits:
            for sentence in split_sentences(hit["text"]):
                tokens = set(self.content_tokens(sentence))
                # New information = a content word or a number the question doesn't already have.
                # (Numbers are checked separately: the analyzer drops 1-char tokens like "3".)
                adds_info = bool(tokens - set(q_tokens)) or bool(set(re.findall(r"\d+", sentence)) - q_numbers)
                informative = word_count(sentence) >= config.MIN_ANSWER_WORDS and adds_info
                if sentence not in seen and informative:
                    seen.add(sentence)
                    candidates.append({"text": sentence, "chunk_id": hit["chunk_id"], "tokens": tokens})

        if not q_tokens or not candidates:
            return self._refuse(q_tokens, 0.0)

        sims = self._similarities(question, [c["text"] for c in candidates])
        for c, sim in zip(candidates, sims):
            c["coverage"] = sum(t in c["tokens"] for t in q_tokens) / len(q_tokens)
            c["sim"] = sim
        ranked = sorted(candidates, key=lambda c: (c["coverage"], c["sim"]), reverse=True)
        best = ranked[0]

        if best["coverage"] < config.MIN_COVERAGE:
            evidence = set().union(*(c["tokens"] for c in candidates))
            missing = [t for t in q_tokens if t not in evidence]
            missing = missing or [t for t in q_tokens if t not in best["tokens"]]
            return self._refuse(missing, best["coverage"])

        kept = [best]
        if length is None:
            # Base behaviour: add the next-best sentence from a different chunk.
            for c in ranked[1:]:
                if c["coverage"] >= config.MIN_COVERAGE and c["chunk_id"] != best["chunk_id"]:
                    kept.append(c)
                    break
        else:
            preset = LENGTH_PRESETS[length]
            extras = sorted((c for c in ranked[1:] if c["coverage"] >= config.EXTRA_SENTENCE_COVERAGE),
                            key=lambda c: c["sim"], reverse=True)
            words = word_count(best["text"])
            for c in extras:
                if len(kept) >= preset["max_sentences"]:
                    break
                if words + word_count(c["text"]) <= preset["max_words"]:
                    kept.append(c)
                    words += word_count(c["text"])

        return {
            "answer": " ".join(c["text"] for c in kept),
            "cited_chunk_ids": list(dict.fromkeys(c["chunk_id"] for c in kept)),
            "supported": True,
            "coverage": round(best["coverage"], 4),
        }

    @staticmethod
    def _refuse(missing: list[str], coverage: float) -> dict:
        note = f" Not covered: {', '.join(missing)}." if missing else ""
        return {"answer": config.REFUSAL + note, "cited_chunk_ids": [], "supported": False,
                "reason": "insufficient_coverage", "coverage": round(coverage, 4)}


def parse_model_json(text: str) -> dict:
    """Parse {"answer", "cited_chunk_ids", "supported"} from model output; raise ValueError if invalid."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start:end + 1])
    answer, cited, supported = data.get("answer"), data.get("cited_chunk_ids"), data.get("supported")
    if not isinstance(answer, str) or not isinstance(supported, bool) or not isinstance(cited, list) \
            or not all(isinstance(c, str) for c in cited):
        raise ValueError("model JSON does not match the required schema")
    return {"answer": answer.strip(), "cited_chunk_ids": cited, "supported": supported}


class ClaudeGenerator:
    """Claude-backed generator. `anthropic` is imported lazily so offline mode never needs it."""

    name = "claude"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("CLAUDE_MODEL", config.DEFAULT_CLAUDE_MODEL)
        self._client = None

    def _fail(self, error: str) -> dict:
        log("generate_error", generator=self.name, model=self.model, error=error)
        return {"answer": config.REFUSAL, "cited_chunk_ids": [], "supported": False,
                "reason": "model_unsupported", "error": error}

    def generate(self, question: str, hits: list[dict], prompt: str | None = None,
                 length: str | None = None) -> dict:
        prompt = prompt or build_prompt(question, hits, length)
        try:
            import anthropic
        except ImportError:
            return self._fail("anthropic package not installed")
        try:
            if self._client is None:
                self._client = anthropic.Anthropic()
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens_for(length),
                messages=[{"role": "user", "content": prompt}],
                # anthropic 1.x removed the `temperature` kwarg, but Haiku 4.5 still honours it
                # in the request body. Models that reject it return a 400, which fails closed.
                extra_body={"temperature": 0},
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            result = parse_model_json(text)
        except anthropic.RateLimitError as e:
            return self._fail(f"rate_limited: {e}")
        except anthropic.APIStatusError as e:
            return self._fail(f"api_status_{e.status_code}: {e}")
        except anthropic.APIConnectionError as e:
            return self._fail(f"connection: {e}")
        except ValueError as e:  # includes json.JSONDecodeError and schema mismatches
            return self._fail(f"bad_json: {e}")
        except Exception as e:  # e.g. missing credentials; always fail closed
            return self._fail(f"{type(e).__name__}: {e}")
        if not result["supported"]:
            result["reason"] = "model_unsupported"
        return result


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_generator(name: str = "auto", vectorizer=None):
    if name == "auto":
        name = "claude" if claude_available() else "extractive"
    if name == "claude":
        gen = ClaudeGenerator()
    elif name == "extractive":
        gen = ExtractiveGenerator(vectorizer)
    else:
        raise ValueError(f"unknown generator: {name!r}")
    return gen
