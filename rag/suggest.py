"""Self-verified suggested questions for an uploaded doc.

Candidates come from Claude (when available) or from deterministic templates. Every candidate is
run through the full pipeline, scoped to the doc, and only questions that come back supported and
validated are kept, so the UI never suggests a question the system would refuse.
"""

import contextvars
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from collections import Counter

from .controls import AskOptions
from .gate import key_terms
from .generate import claude_available, get_generator
from .ingest import split_sentences
from .obs import log
from .pipeline import answer_question

MAX_CANDIDATES = 12
KEEP = 3
CLAUDE_DOC_CHARS = 12_000

# "<The> <noun phrase of 1-5 words> <verb>" at the start of a sentence.
_SUBJECT_VERB = re.compile(
    r"^(?:(the|a|an)\s+)?((?:[\w'-]+\s+){0,4}?[\w'-]+)\s+"
    r"(is|are|costs?|takes?|lasts?|expires?|includes?|allows?|requires?|receives?|gets?|locks?)\b",
    re.I)
_NOT_A_SUBJECT = {"it", "this", "that", "these", "those", "they", "we", "you", "he", "she", "there",
                  "because", "and", "but", "or", "so", "if", "when", "which", "what", "who"}
_QUESTION_FORM = {  # verb base -> question template ({aux} is does/do)
    "cost": "How much {aux} {np} cost?",
    "take": "How long {aux} {np} take?",
    "last": "How long {aux} {np} last?",
    "expire": "When {aux} {np} expire?",
    "lock": "When {aux} {np} lock?",
}


def _title(doc_id: str) -> str:
    return re.sub(r"[_-]+", " ", os.path.splitext(doc_id)[0]).strip()


def _lower_first(text: str) -> str:
    """Lower-case the first word unless it looks like an acronym (KYC, 2FA)."""
    first, _, rest = text.partition(" ")
    if not (first.isupper() or any(ch.isdigit() for ch in first)):
        first = first.lower()
    return f"{first} {rest}".strip()


def _plural(np: str) -> bool:
    last = np.split()[-1].lower()
    return last.endswith("s") and not last.endswith("ss")


def sentence_question(sentence: str) -> str | None:
    """Turn a factual sentence into a question about its subject, e.g.
    "The Starter plan costs $29 per month." -> "How much does the Starter plan cost?"."""
    m = _SUBJECT_VERB.match(sentence)
    if not m:
        return None
    article, np, verb = m.group(1), m.group(2).strip(), m.group(3).lower()
    if np.split()[0].lower() in _NOT_A_SUBJECT:
        return None
    np = f"the {np}" if article else _lower_first(np)
    if verb in ("is", "are"):
        return f"What {verb} {np}?"
    base = verb[:-1] if verb.endswith("s") else verb  # costs -> cost; every listed verb is regular
    aux = "do" if _plural(np) else "does"
    template = _QUESTION_FORM.get(base, "What {aux} {np} " + base + "?")
    return template.format(aux=aux, np=np)


def template_candidates(doc_id: str, chunks) -> list[str]:
    """Deterministic candidates: sentences containing a number first, then headings."""
    out = []
    for c in chunks:
        for sentence in split_sentences(c.text):
            if re.search(r"\d", sentence):
                q = sentence_question(sentence)
                if q:
                    out.append(q)
    # ID-like terms the doc mentions repeatedly (AITF-14, GraphQL): "What is AITF-14?".
    # All-caps words alone are skipped: in code samples they are keywords (SELECT, ORDER).
    terms = Counter(t for c in chunks for t in key_terms(c.text)
                    if len(t) >= 3 and not t.isdigit()
                    and (any(ch.isdigit() for ch in t) or re.search(r"[a-z][A-Z]", t)))
    out.extend(f"What is {t}?" for t, n in terms.most_common(5) if n >= 2)
    title = _title(doc_id)
    for heading in dict.fromkeys(c.heading for c in chunks if c.heading):
        h = _lower_first(heading.strip())
        if not _plural(h):
            out.append(f"What is the {h}?")
        out.append(f"What does {title} say about {h}?")
    return out


def claude_candidates(doc_id: str, chunks, n: int = 5) -> list[str]:
    """Ask Claude for questions answerable only from this doc. Returns [] on any failure."""
    try:
        import anthropic

        text = "\n\n".join(c.text for c in chunks)[:CLAUDE_DOC_CHARS]
        prompt = (
            f"Write {n} short, specific support questions that can be answered ONLY from the document "
            "below. Each answer must be stated explicitly in the document. Do not ask about anything "
            'the document does not state. Output JSON only: {"questions": ["...", "..."]}\n\n'
            f"Document ({doc_id}):\n{text}"
        )
        response = anthropic.Anthropic().messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"temperature": 0},
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return [q.strip() for q in data.get("questions", []) if isinstance(q, str) and q.strip()]
    except Exception as e:  # suggestions are a convenience: fall back to templates
        log("suggest_error", doc_id=doc_id, error=f"{type(e).__name__}: {e}")
        return []


def suggest_questions(doc_id: str, index, generator_name: str = "auto") -> dict:
    """Return {"questions": [...verified...], "tried": n, "method": "claude" | "template"}."""
    chunks = [c for c in index.chunks if c.doc_id == doc_id]
    use_claude = generator_name != "extractive" and claude_available()
    candidates = claude_candidates(doc_id, chunks) if use_claude else []
    method = "claude" if candidates else "template"
    if not candidates:
        candidates = template_candidates(doc_id, chunks)
    unique, seen = [], set()
    for q in candidates:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    candidates = unique[:MAX_CANDIDATES]

    generator = get_generator(generator_name, index.vectorizer)
    options = AskOptions(generator=generator_name, doc_ids=[doc_id])

    def verify(q: str) -> bool:
        r = answer_question(q, index, generator, options)
        return r["supported"] and r["validation_passed"] and doc_id in r["citations"]

    contexts = [contextvars.copy_context() for _ in candidates]  # keep the request's run_id in logs
    with ThreadPoolExecutor(max_workers=5) as pool:
        verdicts = list(pool.map(lambda ctx, q: ctx.run(verify, q), contexts, candidates))
    kept = [q for q, ok in zip(candidates, verdicts) if ok][:KEEP]
    return {"questions": kept, "tried": len(candidates), "method": method}
