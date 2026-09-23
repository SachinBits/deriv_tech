"""Run the evaluation set and write retrieval_results.json, answers.json, validation_report.json."""

import argparse
import json
import os
from collections import Counter

from rag import config
from rag.controls import LENGTH_PRESETS, AskOptions, word_count
from rag.evalset import expected_supported, load_questions
from rag.generate import get_generator
from rag.obs import log, new_run_id
from rag.pipeline import answer_question
from rag.retrieve import build_index


def add_option_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared with app.py."""
    parser.add_argument("--generator", choices=["auto", "extractive", "claude"], default="auto")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--length", choices=["short", "medium", "detailed", "off"], default="medium")
    parser.add_argument("--no-threshold", action="store_true", help="disable the confidence threshold")
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE,
                        help="confidence threshold (0.0-0.5)")
    parser.add_argument("--docs", default=config.DOCS_DIR)


def options_from_args(args) -> AskOptions:
    return AskOptions(
        confidence_threshold_enabled=not args.no_threshold,
        confidence_threshold=args.min_score,
        length=None if args.length == "off" else args.length,
        generator=args.generator,
    )


def run_eval(questions, index, generator, options, k) -> list[dict]:
    return [answer_question(q["question"], index, generator, options, k) for q in questions]


def refusal_accuracy(questions, records) -> tuple[float | None, int, int]:
    pairs = [(expected_supported(q["expected_behavior"]), r["supported"])
             for q, r in zip(questions, records)]
    pairs = [(e, s) for e, s in pairs if e is not None]
    correct = sum(e == s for e, s in pairs)
    return (round(correct / len(pairs), 4) if pairs else None), correct, len(pairs)


def pass_rate(records) -> float | None:
    return round(sum(r["validation_passed"] for r in records) / len(records), 4) if records else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_option_flags(parser)
    parser.add_argument("--questions", default=config.QUESTIONS_PATH)
    parser.add_argument("--out-dir", default=".", help="where to write the 3 JSON artifacts")
    args = parser.parse_args()

    run_id = new_run_id()
    options = options_from_args(args)
    index = build_index(config.root_path(args.docs))
    questions = load_questions(config.root_path(args.questions))
    generator = get_generator(options.generator, index.vectorizer)
    log("eval_run", label="default", generator=generator.name, options=options.applied(),
        questions=len(questions))

    records = run_eval(questions, index, generator, options, args.k)

    # Ablation: threshold on/off, and each length preset. Retrieval is instant.
    threshold_on = options.model_copy(update={"confidence_threshold_enabled": True})
    threshold_off = options.model_copy(update={"confidence_threshold_enabled": False})
    ablation = {"threshold_on": {}, "threshold_off": {}, "length": {}}
    for label, opts in (("threshold_on", threshold_on), ("threshold_off", threshold_off)):
        recs = records if opts == options else run_eval(questions, index, generator, opts, args.k)
        acc, correct, n = refusal_accuracy(questions, recs)
        reasons = Counter(r["refusal_reason"] for r in recs if not r["supported"])
        ablation[label] = {"threshold": opts.confidence_threshold, "refusal_accuracy": acc,
                           "refusal_correct": correct, "evaluated": n,
                           "refusal_reasons": dict(sorted(reasons.items()))}
    for length in LENGTH_PRESETS:
        opts = threshold_on.model_copy(update={"length": length})
        recs = records if opts == options else run_eval(questions, index, generator, opts, args.k)
        answered = [r for r in recs if r["supported"]]
        ablation["length"][length] = {
            "avg_words": round(sum(word_count(r["answer"]) for r in answered) / len(answered), 1)
            if answered else None,
            "answered": len(answered),
            "validation_pass_rate": pass_rate(recs),
        }

    retrieval = [{"question_id": q["id"], "question": q["question"],
                  "retrieved_chunks": r["retrieved_chunks"]} for q, r in zip(questions, records)]
    answers = [{"question_id": q["id"], **{key: r[key] for key in (
        "answer", "citations", "cited_chunks", "supported", "refusal_reason",
        "retrieved_sources", "generator")}} for q, r in zip(questions, records)]

    acc, correct, n = refusal_accuracy(questions, records)
    with_doc = [(q, r) for q, r in zip(questions, records) if q["expected_doc"]]
    hits_at_k = sum(q["expected_doc"] in {c["doc_id"] for c in r["retrieved_chunks"]}
                    for q, r in with_doc)
    summary = {
        "total": len(records),
        "validation_pass_rate": pass_rate(records),
        "supported_count": sum(r["supported"] for r in records),
        "refused_count": sum(not r["supported"] for r in records),
        "refusal_accuracy": acc,
        "refusal_correct": correct,
        "refusal_evaluated": n,
        "retrieval_hit_at_k": round(hits_at_k / len(with_doc), 4) if with_doc else None,
        "retrieval_k": args.k,
        "retrieval_evaluated": len(with_doc),
        "generator": generator.name,
        "options_applied": options.applied(),
        "thresholds": {"min_score": config.MIN_SCORE, "min_coverage": config.MIN_COVERAGE},
        "ablation": ablation,
    }
    results = [{
        "question_id": q["id"],
        "expected_behavior": q["expected_behavior"],
        "supported": r["supported"],
        "correct_behavior": (None if expected_supported(q["expected_behavior"]) is None
                             else expected_supported(q["expected_behavior"]) == r["supported"]),
        "refusal_reason": r["refusal_reason"],
        "top_score": r["top_score"],
        "coverage": r["coverage"],
        "checks": r["checks"],
    } for q, r in zip(questions, records)]

    out_dir = config.root_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in (("retrieval_results.json", retrieval), ("answers.json", answers),
                          ("validation_report.json", {"summary": summary, "results": results})):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    log("eval_summary", **{k: v for k, v in summary.items() if k != "ablation"})
    print(f"run {run_id}: {summary['total']} questions | generator={generator.name} | "
          f"validation_pass_rate={summary['validation_pass_rate']} | "
          f"refusal_accuracy={correct}/{n} | hit@{args.k}={summary['retrieval_hit_at_k']} | "
          f"supported={summary['supported_count']} refused={summary['refused_count']} | "
          f"threshold_off={ablation['threshold_off']['refusal_accuracy']}")


if __name__ == "__main__":
    main()
