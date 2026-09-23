"""CLI: answer one question from the local docs."""

import argparse
import json

from rag import config
from rag.obs import new_run_id
from rag.pipeline import answer_question
from rag.retrieve import build_index
from run_pipeline import add_option_flags, options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q", required=True)
    add_option_flags(parser)
    parser.add_argument("--show-prompt", action="store_true", help="print the filled prompt first")
    parser.add_argument("--verbose", action="store_true", help="print the full pipeline record")
    args = parser.parse_args()

    new_run_id()
    index = build_index(config.root_path(args.docs))
    record = answer_question(args.question, index, None, options_from_args(args), args.k)

    if args.show_prompt:
        print("----- prompt -----")
        print(record["prompt"] or "(not built: the evidence gate refused before generation)")
        print("------------------")
    if args.verbose:
        out = {k: v for k, v in record.items() if k != "prompt"}
    else:
        out = {k: record[k] for k in ("answer", "citations", "supported",
                                      "retrieved_sources", "refusal_reason")}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
