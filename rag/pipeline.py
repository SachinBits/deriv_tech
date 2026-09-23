"""Orchestrator: retrieve -> gate -> prompt -> generate -> validate."""

import time

from . import config
from .controls import AskOptions, apply_length
from .gate import evidence_gate
from .generate import get_generator
from .obs import log
from .prompt import build_prompt
from .retrieve import Index
from .validate import validate


def _refusal_text(missing: list[str]) -> str:
    return config.REFUSAL + (f" Not covered: {', '.join(missing)}." if missing else "")


def answer_question(question: str, index: Index, generator=None,
                    options: AskOptions | None = None, k: int = config.TOP_K) -> dict:
    started = time.perf_counter()
    options = options or AskOptions()
    generator = generator or get_generator(options.generator, index.vectorizer)

    # 1. Retrieve
    hits = index.search(question, k)
    top_score = hits[0]["score"] if hits else 0.0

    # 2. Gate (threshold is optional; the key-term check always runs)
    ok, gate_reason, missing = evidence_gate(
        question, hits,
        threshold_enabled=options.confidence_threshold_enabled,
        threshold=options.confidence_threshold,
    )

    prompt = None
    was_length_trimmed = False
    if not ok:
        generated = {"answer": _refusal_text(missing), "cited_chunk_ids": [],
                     "supported": False, "reason": gate_reason}
    else:
        # 3. Prompt (built for both generators; the extractive one ignores it)
        prompt = build_prompt(question, hits, options.length)
        log("prompt", prompt_chars=len(prompt), n_context_chunks=len(hits))

        # 4. Generate, then shape length so numbers are validated on the final text
        generated = generator.generate(question, hits, prompt=prompt, length=options.length)
        log("generate", generator=generator.name, supported=generated["supported"],
            cited=generated.get("cited_chunk_ids", []), coverage=generated.get("coverage"))
        if generated["supported"]:
            generated["answer"], was_length_trimmed = apply_length(generated["answer"], options.length)

    # 5. Validate (fail closed)
    final, checks = validate(generated, hits, length=options.length, question=question)

    doc_of = {h["chunk_id"]: h["doc_id"] for h in hits}
    cited_chunks = final["cited_chunk_ids"]
    return {
        "question": question,
        "answer": final["answer"],
        "citations": list(dict.fromkeys(doc_of[c] for c in cited_chunks)),
        "cited_chunks": cited_chunks,
        "supported": final["supported"],
        "refusal_reason": final["refusal_reason"],
        "missing_terms": missing,
        "retrieved_sources": [h["chunk_id"] for h in hits],
        "retrieved_chunks": hits,
        "generator": generator.name,
        "checks": checks,
        "validation_passed": final["validation_passed"],
        "pre_validation_supported": final["pre_validation_supported"],
        "top_score": top_score,
        "coverage": generated.get("coverage"),
        "options_applied": options.applied(),
        "was_trimmed": was_length_trimmed or final["was_trimmed"],
        "prompt": prompt,
        "error": generated.get("error"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
