# Grounded Support QA (local RAG)

**Live demo:** https://deriv-tech.vercel.app. It serves the 6 base docs, with Claude enabled. Document upload is local-only, so it is hidden there (see [why](#try-your-own-docs-local)).

Answers support questions **only** from a local knowledge base, cites the chunks it used, and refuses when the docs don't support an answer. It runs fully offline by default (TF-IDF retrieval plus an extractive generator). Claude is an optional, swappable generator behind the same interface.

- **Knowledge base:** the 6 product docs in `docs/`. When running locally, the web UI also accepts your own `.md`, `.txt` and `.pdf` files (see [Try your own docs](#try-your-own-docs-local)).
- **Eval:** `run_pipeline.py` scores 10 questions (6 answerable, 4 not) and writes 3 JSON artifacts: 10/10 refusal accuracy and a 1.0 validation pass rate in both extractive and Claude mode.
- **Interfaces:** a CLI (`app.py`), a batch eval (`run_pipeline.py`), a FastAPI server (`server.py`) and a single-file chat UI (`web/index.html`).

## Architecture

```
 docs/ (+ uploads/ in the local server)
     │
     ▼
   ingest ──► retrieve ──► gate ──► prompt ──► generate ──► validate ──► record
  (chunk)    (TF-IDF,     (score +   (fills     (extractive  (V1–V6,
              top-k)       key-term)  template)  or Claude)   fail-closed)
                             │ refuse                            │ refuse / trim
                             └──────────► canonical refusal ◄────┘
```

`rag/pipeline.answer_question()` runs five explicit steps: **retrieve → gate → prompt → generate → validate**. If the gate refuses, the prompt and generate steps are skipped.

```
docs/                 6 product docs (knowledge base)
questions.json        eval set: [{id, question, expected_behavior?, expected_doc?}]
prompts/answer.txt    Claude prompt template ({question}, {context}, {length_instruction})
rag/config.py         constants and calibrated thresholds
rag/obs.py            JSONL stage logging → logs/pipeline.jsonl
rag/ingest.py         load_docs(), chunk()  (.md/.txt/.pdf; split on headings, pack to ≤400 chars)
rag/retrieve.py       Index.build() / Index.search(question, k, doc_ids=None)
rag/gate.py           evidence_gate(): confidence threshold (optional) + key-term check
rag/prompt.py         build_prompt()
rag/generate.py       ExtractiveGenerator, ClaudeGenerator, get_generator()
rag/validate.py       validate(): V1–V6
rag/controls.py       AskOptions, LENGTH_PRESETS, apply_length()
rag/evalset.py        load_questions() (a missing file is tolerated), expected_supported()
rag/pipeline.py       answer_question(): the orchestrator
rag/kb.py             server knowledge base: docs/ + uploads/, thread-safe rebuild, upload checks
rag/suggest.py        self-verified suggested questions for uploaded docs
rag/sinks.py          optional Supabase sink (query_logs, eval_runs, chunks)
run_pipeline.py       batch eval → retrieval_results.json, answers.json, validation_report.json
runs/claude/          the same 3 artifacts from a live Claude (Haiku 4.5) run
app.py                one-question CLI
server.py             FastAPI: /api/config, /api/ask (+ /ask alias), /api/health, /api/docs*, UI at /
web/index.html        single-file chat UI (vanilla JS, no build step)
tests/test_pipeline.py  offline test suite (see Tests)
uploads/              local uploads (git-ignored, created on first upload)
.env.example          environment variable template
```

## Setup

Python 3.10+ (tested on 3.13).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --generator extractive   # regenerates the 3 JSON artifacts (offline)
python app.py --question "How long do bank transfer withdrawals take?"
pytest -q
```

Flags shared by `run_pipeline.py` and `app.py`:
- `--generator auto|extractive|claude`
- `--length short|medium|detailed|off`
- `--no-threshold`
- `--min-score 0.10`
- `--k 4`
- `--docs DIR`

`run_pipeline.py` also takes `--questions PATH`. `app.py` also takes `--show-prompt` and `--verbose`.

### Environment variables

`.env` is **not** auto-loaded. Export the variables yourself, for example with `set -a; source .env; set +a` (see `.env.example`).

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | When set, `--generator auto` (the default) uses **Claude**. For a fully offline, reproducible run, pass `--generator extractive`. |
| `CLAUDE_MODEL` | Claude model id (default `claude-haiku-4-5-20251001`). |
| `SUPABASE_URL` | Enables the Supabase sink, together with a key. |
| `SUPABASE_SECRET_KEY` **or** `SUPABASE_SERVICE_KEY` | Either name is accepted (the secret key name is checked second). |
| `PORT` / `HOST` | Server bind address (default `127.0.0.1:8000`). |
| `VERCEL` | When set, all upload endpoints return 403 and the UI hides upload. |
| `RAG_DOCS_DIR`, `RAG_UPLOADS_DIR`, `RAG_QUESTIONS`, `RAG_LOG_PATH` | Override the server's docs dir, uploads dir, questions file and log file. |

The committed artifacts (`retrieval_results.json`, `answers.json`, `validation_report.json`) were generated offline with `python run_pipeline.py --generator extractive`.

### Run the UI

```bash
python server.py        # then open http://localhost:8000
PORT=8001 python server.py   # if port 8000 is taken ("address already in use")
```

Or use the hosted version: https://deriv-tech.vercel.app.

The server doesn't auto-reload. Restart it after pulling changes, then hard-reload the page (Cmd+Shift+R).

> _Screenshot placeholder: add `screenshot.png` of the chat UI (a grounded q1 answer and a q7 refusal)._

## Retrieval

- Each doc is split on markdown headings. Paragraphs, then sentences, are packed into chunks of at most 400 characters, and every chunk keeps its `doc_id`, `chunk_id` and `heading`.
- A `TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)` is fitted on the chunk text.
- A question is vectorised the same way. Cosine similarity (`linear_kernel` on L2-normalised TF-IDF vectors) ranks the chunks, and the top-k (default 4) are returned with scores.
- Building the index takes milliseconds, so every entry point rebuilds it at startup.

## Grounding and refusal

There are three independent layers. None of them can turn a refusal into an answer.

1. **Evidence gate**, before generation and deterministic:
   - **Rule A (optional):** refuse if the top retrieval score is below the confidence threshold (`low_retrieval_score`).
   - **Rule B (always on):** refuse if a key term in the question is missing from the retrieved text (`key_term_not_in_docs`). Key terms are acronyms (VIP, KYC), mixed-case tokens (GraphQL) and tokens containing digits (P1).
2. **Generator support:**
   - The extractive generator refuses when no sentence covers at least `MIN_COVERAGE` of the question's content tokens (`insufficient_coverage`). A sentence only counts as a candidate if it has ≥ 4 words and adds a content word or number the question lacks, so a fragment that merely repeats the question is never an answer.
   - Claude is instructed to set `supported=false` itself (`model_unsupported`), and any API or JSON error fails closed.
3. **Validator** (`rag/validate.py`), deterministic and fail-closed:

| Rule | Check | On failure |
|---|---|---|
| V1 `non_empty` | the answer is not blank | refuse (`validation_failed`) |
| V2 `supported_has_citation` | a supported answer has ≥1 citation | refuse |
| V3 `citations_in_retrieved` | every cited chunk was retrieved; invalid ones are dropped | refuse if none remain |
| V4 `unsupported_marked` | an unsupported answer starts with the refusal sentence | the refusal sentence is prefixed |
| V5 `numbers_grounded` | every number in the answer appears in the cited chunks | refuse |
| V6 `length_within_limit` | only when a length mode is set | trim at a sentence boundary (`detail="trimmed"`), not a refusal |

### Calibration

I ran `python run_pipeline.py --generator extractive` and it scored 10/10 refusal accuracy at the spec defaults, so I kept **`MIN_SCORE = 0.10`** and **`MIN_COVERAGE = 0.6`**. Per-question signals (from `validation_report.json`):

| id | expected | top score | best coverage | outcome |
|---|---|---|---|---|
| q1–q6 | answerable | 0.351–0.593 | 0.75–1.00 | answered, cited |
| q7 | unanswerable | 0.399 | – | refused: key term `VIP` |
| q8 | partial | 0.422 | – | refused: key term `GraphQL` |
| q9 | partial | 0.164 | 0.33 | refused: coverage (`crypto` not in docs) |
| q10 | unanswerable | 0.166 | 0.50 | refused: coverage (`service credits` not in docs) |

Reasoning:
- **Coverage.** The coverage threshold sits between the highest unanswerable coverage (0.50) and the lowest answerable coverage (0.75), so 0.6 has margin on both sides.
- **Threshold on this set.** The retrieval threshold never decides anything here. Every question scores above 0.10, and q7/q8 score as high as the answerable questions because they share words with real docs ("identity verification", "rate limit"). That is exactly why the key-term gate exists. A threshold of about 0.2 would also separate the classes (answerable ≥ 0.35; q9/q10 ≈ 0.165), but I didn't raise it because the default already reaches 10/10.
- **One extra stop list.** sklearn's English stop list omits "does", so the coverage tokeniser drops `does/did/doing`. This is generic and not tied to any question.

## Stretch item: user-controllable answer controls

This is one improvement with two knobs. Both are optional per request (`AskOptions` in `rag/controls.py`), exposed through the CLI flags, the API body and the UI's "+" menu.

**Knob 1: answer length** (`short` 1 sentence/40 words · `medium` 2/80 · `detailed` 4/160 · off)
- **Why:** support users want terse answers and reviewers want detail.
- **Extractive generator:** the primary sentence must still meet `MIN_COVERAGE`. Extra sentences need coverage ≥ 0.3 and are ranked by TF-IDF similarity. Citations are recomputed from the sentences kept.
- **Claude:** gets a `{length_instruction}` and a preset `max_tokens`.
- `apply_length()` trims deterministically at sentence boundaries before validation, so V5 checks numbers on the final text.
- Sentences are verbatim and trimming is deterministic, so answers stay grounded at every length.

**Knob 2: confidence threshold** (on/off, 0.00–0.50)
- **Why:** it's cheap and deterministic, and it stops the system answering from weak matches.

**Why both are optional:**
- Different consumers need different trade-offs. The UI user may want a lower threshold or longer answers; the batch eval pins the defaults.
- The key-term check and the coverage check **stay on regardless**, so turning the threshold off never removes grounding. The ablation below shows this.

**Ablation** (`summary.ablation` in `validation_report.json`, extractive generator, 10 questions):

| Setting | Refusal accuracy | Refusals by reason |
|---|---|---|
| threshold on (0.10) | **10/10** | 2 key-term, 2 coverage |
| threshold off | **10/10** | 2 key-term, 2 coverage |

| Length | Avg words (6 answered) | Validation pass rate |
|---|---|---|
| short | 11.3 | 1.0 |
| medium (default) | 24.2 | 1.0 |
| detailed | 46.3 | 1.0 |

On this eval set the threshold adds no accuracy, because the always-on safeguards already catch all four unanswerable questions. It is defence in depth for corpora where a question shares no key term with the docs and its words happen to cover a single sentence.

## Optional Claude mode

```bash
# .env is not auto-loaded: set -a; source .env; set +a
export ANTHROPIC_API_KEY=sk-ant-...
export CLAUDE_MODEL=claude-haiku-4-5-20251001   # optional; this is the default
python run_pipeline.py                          # auto → Claude when the key is set
python app.py --generator claude --question "What is the daily withdrawal limit?"
```

- `anthropic` is imported lazily inside `ClaudeGenerator`, so offline mode works without the package.
- The model must return JSON only (`{"answer", "cited_chunk_ids", "supported"}`). Temperature 0 is sent via `extra_body`, because `anthropic` 1.x removed the `temperature` keyword. Haiku 4.5 honours it; a model that rejects it returns a 400, which fails closed.
- The same gate and validator apply to Claude's output.
- **Fail-closed behaviour:** if the LLM call fails mid-run, the answer becomes a refusal rather than silently switching to extractive. An outage produces refusals, never unchecked answers. API errors, timeouts, non-JSON output and schema mismatches all end as `supported=false` with the canonical refusal (`refusal_reason="model_unsupported"`), and the cause is logged as `stage=generate_error`.
- **"The validator makes the LLM safe"** is tested offline with a stubbed client (`tests/test_pipeline.py`, "Claude path"):
  - a grounded answer passes;
  - an **invented number is refused by V5**;
  - prose or non-JSON, a wrong schema, truncated JSON and API errors are refused;
  - a citation outside the retrieved set is refused by V3;
  - a refusal without the canonical prefix gets it added.

  A further test checks the request's keyword arguments against the installed SDK's `messages.create` signature, because a stub would accept anything.

## Extractive vs Claude

Both modes run the same eval (`python run_pipeline.py --generator extractive` for the root artifacts; `python run_pipeline.py --generator claude --out-dir runs/claude` for [`runs/claude/`](runs/claude/)). Claude mode uses `claude-haiku-4-5-20251001`. Both run with threshold on, length `medium`, and the same gate and validator.

| | Extractive (offline) | Claude (Haiku 4.5) |
|---|---|---|
| Refusal accuracy | 10/10 | 10/10 |
| Validation pass rate | 1.0 | 1.0 |
| Avg words (supported answers) | 24.2 | 22.2 |
| Avg latency per question | ~1 ms | ~980 ms |
| How q9/q10 are refused | coverage < 0.6 (`insufficient_coverage`) | model sets `supported=false` (`model_unsupported`) |

Both modes refuse q7/q8 at the key-term gate, before any generation. Claude's refusals for the partial questions say what *is* covered, for example "The context only covers bank transfer withdrawals, which take 1 to 3 business days…". Its numbers still pass V5 because they come from the retrieved chunks.

**Where the modes differ: a reworded question** (not in the eval set; run with `app.py`):

| Question | Extractive | Claude |
|---|---|---|
| "How long until a password reset link stops working?" | refused: `insufficient_coverage` ("long, stop, work" aren't in any sentence) | ✅ "Reset links expire after 30 minutes. If the link has expired, the user must request a new reset…" citing `auth_password_reset.md` |
| "When can I log in again after too many wrong passwords?" | refused: `low_retrieval_score` | refused: `low_retrieval_score` |

The first row is the paraphrase gap in the extractive generator: retrieval found the right chunk, but word-overlap coverage can't see that "stops working" means "expire". Claude can, and the validator still checks its citation and its "30". The second row is the paraphrase gap in *retrieval*: "log in again" and "wrong passwords" barely overlap with "account locks … failed login attempts". Neither generator ever sees the right evidence, so both refuse. That fails safe, and it's the case hybrid or vector search (see Production path) would fix.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the chat UI |
| `GET` | `/api/health` | `{"status": "ok"}` |
| `GET` | `/api/config` | defaults, length presets, generator availability, KB counts, sample questions, `upload_enabled`, `upload_limits` |
| `POST` | `/api/ask` (alias `/ask`) | body `{question (1–500 chars), options?: AskOptions}`; returns the pipeline record plus `retrieved_chunks` (with `cited`), `checks`, `citation_sources`; 422 on invalid input |
| `GET` | `/api/docs` | every doc with `source`, `chunks`, `chars`, `added_at` (local only) |
| `POST` | `/api/docs` | multipart upload, up to 5 files (local only) |
| `DELETE` | `/api/docs/{doc_id}` | delete an upload; base docs → 403 (local only) |
| `GET` | `/api/docs/{doc_id}/chunks` | a doc's chunks (local only) |

`AskOptions` has these fields:
- `confidence_threshold_enabled` (default `true`);
- `confidence_threshold` (0.0–0.5, default 0.10);
- `length` (`short` · `medium` · `detailed` · `null`, default `medium`);
- `generator` (`auto` · `extractive` · `claude`);
- `doc_ids` (scope retrieval to these docs; `null` means all).

The "local only" endpoints return 403 when `VERCEL` is set.

## Tests

`pytest -q` runs 65 offline tests: no network and no API key. They cover:
- **Base spec:**
  - chunking;
  - retrieval: the expected doc is in the top 3;
  - all unanswerable and partial questions are refused;
  - the validator fails closed (no citation, a citation that wasn't retrieved, an invented number).
- **Controls:**
  - length presets;
  - threshold toggle;
  - safeguards stay on with the threshold off;
  - numbers in answers are verbatim.
- **API:** config, ask and the `/ask` alias, invalid input → 422, Claude unavailable → 400.
- **Supabase sink** (stubbed network): `apikey` header with no Bearer, chunk upsert, failures ignored.
- **Claude path** (stubbed client):
  - grounded answer passes;
  - invented number → V5 refusal;
  - bad JSON, API error or a foreign citation → refusal;
  - the request arguments match the installed SDK's signature.
- **Uploads:**
  - `.md` and `.pdf` ingest;
  - 415 / 413 / 409 / 422 errors;
  - path traversal is sanitised;
  - 403 on Vercel;
  - scoped search;
  - delete;
  - suggested questions pass when re-asked;
  - the eval is identical with files in `uploads/`.
- **Regressions:**
  - fragmented PDF text is reflowed;
  - a question-echo fragment is never an answer;
  - a sentence whose only new information is a number is kept.

## Observability

Every stage appends a JSON line to `logs/pipeline.jsonl`, and each line has `ts`, `run_id` and `stage`:
- `ingest`, `retrieve`, `gate`, `prompt`, `generate`, `validate`;
- `generate_error` (a Claude call that failed closed);
- `eval_run` and `eval_summary` from `run_pipeline.py`;
- `http`, `api_ask` and `server_start` from the server;
- `upload` (with `suggestions_kept/tried`), `upload_rejected`, `upload_delete` and `suggest_error`;
- `sink` (every Supabase write, with its status).

A batch run shares one `run_id`, and each API request gets its own.

## Optional Supabase sink

If `SUPABASE_URL` and a key (`SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_KEY`) are set, results are persisted over PostgREST to the tables created by migration `001_init_support_qa`. The code never runs migrations. The migration SQL isn't in this repo yet.

The tables are:
- `query_logs`: question, answer, supported, refusal reason, citations, retrieved sources, top score, validation result, generator, options, latency;
- `eval_runs`: run id, generator, options, summary;
- `chunks`: `chunk_id` primary key, doc, heading, index, content, content hash, plus `embedding vector(384)` and `fts tsvector` for the production path.

| Where | Table | What |
|---|---|---|
| `server.py` (after the response is sent) | `query_logs` | one row per `/api/ask`, `client="api"` |
| `app.py` | `query_logs` | `client="cli"` |
| `run_pipeline.py` | `query_logs`, `eval_runs` | default-run answers (`client="eval"`) and the summary incl. ablation |
| `server.py` startup, `run_pipeline.py` | `chunks` | upsert on `chunk_id` with `content_hash`; `embedding`/`fts` are left to the DB |
| `POST /api/docs` / `DELETE /api/docs/{id}` | `chunks` | upsert the new doc's chunks / `DELETE chunks?doc_id=eq.<id>` |

- **Headers:** `apikey: <key>`, `Content-Type: application/json` and `Prefer: return=minimal`. There is **no** `Authorization: Bearer` header, because new `sb_secret_…` keys are not JWTs. The chunk upsert adds `resolution=merge-duplicates`.
- **Failures:** writes are best-effort with a 5 s timeout. A failure is logged as `stage=sink` and never changes an answer.
- **Loading `.env`:** it isn't auto-loaded; see [Environment variables](#environment-variables). Sourcing it also exports `ANTHROPIC_API_KEY`, which makes `--generator auto` use Claude.

## Try your own docs (local)

Run `python server.py` and add your own `.md`, `.txt` or `.pdf` files. You can drag them anywhere onto the page, use **+ → Add documents**, or paste text with **+ → New note**. Each file gets an ingestion card that steps through **Reading → Chunking → Indexing → Verifying questions** with server-measured times. It also shows how the file was chunked and up to three **✓ verified** suggested questions. Clicking one asks it, scoped to that file.

> _Screenshot placeholder: add `screenshot-upload.png` (ingestion card with verified chips, and an answer citing the uploaded file)._

- **Knowledge base drawer** (click the "N docs · M chunks" pill):
  - lists every doc with a *Base* / *Uploaded* / *New* badge and its chunk count;
  - *View chunks*;
  - delete, for uploads only, with an inline confirmation;
  - a checkbox that scopes search to the ticked docs, shown as a "Searching: …" chip in the composer.
- **Citations:** an answer drawn from an upload shows its citation chip in the accent colour with a dot, so you can see at a glance whether it came from your file or the base docs.
- **Grounding is unchanged:** uploads go through the same gate, prompt, generator and validator. Scoping only filters retrieval (`Index.search(..., doc_ids=[...])`).
- **Limits:**
  - `.md`, `.txt` and `.pdf` only. PDF text is extracted page by page with pypdf, and pages without text are skipped. Some PDFs, such as Google Docs exports, extract as one word per line; those fall back to pypdf's layout mode. Lines are then reflowed into paragraphs so a wrapped sentence stays one sentence;
  - ≤ 2 MB per file, ≤ 5 files per request, ≤ 20 uploaded docs;
  - filenames are sanitised to `[a-z0-9_-]` plus the extension, so path traversal is impossible;
  - a name that clashes with a base doc gets 409;
  - a file with no extractable text gets 422.
- **Storage and rebuilds:** files go to `uploads/`, which is git-ignored. The server rebuilds the whole index on each change and swaps it in under a lock, so in-flight requests keep the old index.
- **The eval is isolated:** `run_pipeline.py` and the committed artifacts read `docs/` only. A test runs the eval with and without a file in `uploads/` and asserts identical output.

**How suggested questions are verified.**
1. Candidates:
   - With Claude available, Claude proposes 5 questions answerable only from the file.
   - Otherwise, deterministic templates:
     - turn factual sentences that contain numbers into questions ("The Starter plan costs $29" → "How much does the Starter plan cost?");
     - ask about ID-like terms the doc repeats ("What is AITF-14?");
     - add heading-based questions.
2. **Each candidate is run through the full pipeline, scoped to that file.**
3. Only questions that come back `supported=true` with `validation_passed` and a citation of that file are kept, up to 3.

So the UI never suggests a question the system would refuse. The upload is logged as `stage=upload` with `suggestions_kept/tried`. For example, a small pricing FAQ kept 3 of 7 candidates ("How much does the Starter plan cost?", …). If nothing passes, the card says so rather than suggesting an unverified question.

**Why upload is disabled on Vercel:** serverless instances are stateless and short-lived, so a file saved on one instance is gone on the next request, which could land on another instance. With `VERCEL` set, every `/api/docs` endpoint returns 403, `upload_enabled` is false, and the UI hides all upload entry points. The production path is to persist uploads to the Supabase `chunks` table and retrieve from there behind the same `Index.search()` interface. The sink already writes uploaded chunks there when configured; retrieving from it is the remaining step.

## Limitations

- Lexical retrieval misses paraphrases ("sign-in" vs "login", "cash out" vs "withdraw").
- Thresholds are tuned on 10 questions, which is too few to be a reliable estimate.
- V5 only catches fabricated numbers. A wrong but number-free claim passes the validator if its citations are valid.
- Extractive answers are verbatim sentences: grounded, but sometimes stilted or missing context from the neighbouring sentence. A candidate sentence must have ≥ 4 words and add a content word or number the question lacks. Without that rule, a fragment such as a lone "AITF-14" line in a PDF would "cover" *What is AITF-14?* completely and pass every check while saying nothing.
- Uploads rebuild the whole TF-IDF index, and IDF shifts with each new doc. Fine for dozens of docs, not thousands.
- Deterministic suggestion templates only cover sentences like "X is/costs/takes …" and repeated IDs ("AITF-14"). Other docs may get fewer suggestions, or none, but never unverified ones.
- For messy sources such as slide decks and code-heavy PDFs, extractive answers can include header text around the relevant sentence. Claude mode writes cleaner answers from the same evidence, under the same validator.
- The lexical key-term gate may refuse valid paraphrased questions on new corpora (e.g. an acronym that the docs spell out). This fails safe: the result is a refusal, never a fabrication.

## Production path

Put Supabase Postgres with **pgvector** (embeddings) plus **full-text search** (`tsvector`) behind the same `Index.search()` interface, and fuse the two ranked lists with Reciprocal Rank Fusion. Hybrid search fixes the paraphrase gap while keeping exact-term recall for acronyms, IDs and numbers. The gate, prompt, generators and validator stay unchanged.
