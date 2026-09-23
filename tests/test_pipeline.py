"""Offline tests. The extractive generator is forced everywhere."""

import os
import re

import pytest

from rag import config
from rag.controls import AskOptions
from rag.evalset import expected_supported, load_questions
from rag.generate import ExtractiveGenerator
from rag.ingest import chunk, load_docs, split_sentences
from rag.pipeline import answer_question
from rag.retrieve import Index
from rag.validate import validate

DOCS = config.root_path(config.DOCS_DIR)
QUESTIONS = load_questions(config.root_path(config.QUESTIONS_PATH))
ANSWERABLE = [q for q in QUESTIONS if expected_supported(q["expected_behavior"]) is True]
NOT_ANSWERABLE = [q for q in QUESTIONS if expected_supported(q["expected_behavior"]) is False]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for var in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RAG_LOG_PATH", str(tmp_path / "pipeline.jsonl"))


@pytest.fixture(scope="module")
def docs():
    return load_docs(DOCS)


@pytest.fixture(scope="module")
def index(docs):
    return Index().build(chunk(docs))


@pytest.fixture(scope="module")
def gen(index):
    return ExtractiveGenerator(index.vectorizer)


def ask(q, index, gen, **opts):
    return answer_question(q, index, gen, AskOptions(generator="extractive", **opts))


# ---------- base spec ----------

def test_chunking(docs, index):
    chunks = index.chunks
    assert all(c.doc_id and c.chunk_id and c.text.strip() for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {d["doc_id"] for d in docs} == {c.doc_id for c in chunks}
    assert all(len(c.text) <= config.CHUNK_CHARS for c in chunks)


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["expected_doc"]], ids=lambda q: q["id"])
def test_retrieval_expected_doc_in_top3(index, q):
    assert q["expected_doc"] in [h["doc_id"] for h in index.search(q["question"], 3)]


@pytest.mark.parametrize("q", NOT_ANSWERABLE, ids=lambda q: q["id"])
def test_refuses_unanswerable(index, gen, q):
    r = ask(q["question"], index, gen)
    assert r["supported"] is False
    assert r["answer"].startswith(config.REFUSAL)
    assert r["citations"] == []


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q["id"])
def test_answers_answerable_with_citations(index, gen, q):
    r = ask(q["question"], index, gen)
    assert r["supported"] is True and r["validation_passed"] is True
    assert r["citations"] and set(r["cited_chunks"]) <= set(r["retrieved_sources"])


HITS = [{"doc_id": "x.md", "chunk_id": "x_0", "score": 0.5,
         "text": "Users can request up to 3 password resets per hour."}]


@pytest.mark.parametrize("record", [
    {"answer": "Users can request up to 3 password resets per hour.", "cited_chunk_ids": [], "supported": True},
    {"answer": "Users can request up to 3 password resets per hour.", "cited_chunk_ids": ["nope_9"], "supported": True},
    {"answer": "Users can request up to 7 resets per hour.", "cited_chunk_ids": ["x_0"], "supported": True},
], ids=["no_citation", "citation_not_retrieved", "ungrounded_number"])
def test_validator_fails_closed(record):
    final, checks = validate(record, HITS)
    assert final["supported"] is False
    assert final["answer"] == config.REFUSAL
    assert final["refusal_reason"] == "validation_failed"
    assert final["validation_passed"] is False


def test_validator_marks_unmarked_refusal():
    final, _ = validate({"answer": "Not sure.", "cited_chunk_ids": [], "supported": False}, HITS)
    assert final["answer"].startswith(config.REFUSAL)


# ---------- add-on: controls ----------

def test_length_short_vs_detailed(index, gen):
    q = ANSWERABLE[0]["question"]
    short = ask(q, index, gen, length="short")
    detailed = ask(q, index, gen, length="detailed")
    assert short["supported"] and detailed["supported"]
    assert len(split_sentences(short["answer"])) <= 1
    assert len(short["answer"].split()) <= 40
    assert len(detailed["answer"].split()) >= len(short["answer"].split())


def test_threshold_toggle(index, gen):
    q = ANSWERABLE[0]["question"]
    # 0.99 is outside the API range (0-0.5), so bypass validation to test the gate itself.
    opts = AskOptions.model_construct(confidence_threshold_enabled=True, confidence_threshold=0.99,
                                      length="medium", generator="extractive")
    refused = answer_question(q, index, gen, opts)
    assert refused["supported"] is False and refused["refusal_reason"] == "low_retrieval_score"
    opts_off = AskOptions.model_construct(confidence_threshold_enabled=False, confidence_threshold=0.99,
                                          length="medium", generator="extractive")
    assert answer_question(q, index, gen, opts_off)["supported"] is True


@pytest.mark.parametrize("q", NOT_ANSWERABLE, ids=lambda q: q["id"])
def test_safeguards_stay_on_without_threshold(index, gen, q):
    r = ask(q["question"], index, gen, confidence_threshold_enabled=False)
    assert r["supported"] is False and r["answer"].startswith(config.REFUSAL)
    assert r["refusal_reason"] in {"key_term_not_in_docs", "insufficient_coverage"}


def test_numbers_in_answers_are_verbatim(index, gen):
    for q in ANSWERABLE:
        r = ask(q["question"], index, gen, length="detailed")
        cited = " ".join(h["text"] for h in r["retrieved_chunks"] if h["chunk_id"] in r["cited_chunks"])
        for n in re.findall(r"\d+(?:[.,:]\d+)*", r["answer"]):
            assert n in cited


def test_missing_questions_file_is_tolerated(tmp_path):
    assert load_questions(str(tmp_path / "missing.json")) == []


def test_expected_behavior_normalised():
    assert expected_supported(" Answerable ") is True
    assert expected_supported("partially_answerable") is False
    assert expected_supported(None) is None


# ---------- add-on: API ----------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    import server

    mp = pytest.MonkeyPatch()
    mp.setenv("RAG_UPLOADS_DIR", str(tmp_path_factory.mktemp("uploads")))
    mp.delenv("VERCEL", raising=False)
    with TestClient(server.app) as c:
        yield c
    mp.undo()


def test_api_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["kb"]["docs"] >= 1 and body["kb"]["chunks"] >= 1
    assert body["generators"]["extractive"] is True
    assert client.get("/api/health").json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/api/ask", "/ask"])
def test_api_ask(client, path):
    q = ANSWERABLE[0]["question"]
    r = client.post(path, json={"question": q, "options": {"generator": "extractive"}})
    assert r.status_code == 200
    body = r.json()
    for key in ("answer", "citations", "supported", "retrieved_sources", "options_applied",
                "retrieved_chunks", "checks", "top_score", "latency_ms", "was_trimmed"):
        assert key in body
    assert body["supported"] is True
    assert any(c["cited"] for c in body["retrieved_chunks"])


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 501},
    {"question": "ok?", "options": {"confidence_threshold": 0.9}},
    {"question": "ok?", "options": {"length": "huge"}},
])
def test_api_rejects_invalid_input(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_api_claude_unavailable(client):
    r = client.post("/api/ask", json={"question": "hi", "options": {"generator": "claude"}})
    assert r.status_code == 400


# ---------- Supabase sink (no network) ----------

class _FakeResponse:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured(monkeypatch):
    import rag.sinks

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _FakeResponse()

    monkeypatch.setattr(rag.sinks.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_sink_disabled_without_env(captured):
    from rag.sinks import SupabaseSink

    assert SupabaseSink().enabled is False
    assert SupabaseSink().log_eval_run("r", "extractive", {}, {}) is False
    assert captured == []


def test_sink_headers_use_apikey_not_bearer(index, gen, captured):
    from rag.sinks import SupabaseSink

    sink = SupabaseSink("https://example.supabase.co/", "sb_secret_test")
    record = ask(ANSWERABLE[0]["question"], index, gen)
    assert sink.log_query(record, client="test") is True
    req = captured[0]
    headers = {k.lower(): v for k, v in req.header_items()}
    assert req.full_url == "https://example.supabase.co/rest/v1/query_logs"
    assert headers["apikey"] == "sb_secret_test"
    assert headers["content-type"] == "application/json"
    assert headers["prefer"] == "return=minimal"
    assert "authorization" not in headers


def test_sink_chunk_upsert(index, captured):
    import json

    from rag.sinks import SupabaseSink

    SupabaseSink("https://example.supabase.co", "sb_secret_test").upsert_chunks(index.chunks)
    req = captured[0]
    headers = {k.lower(): v for k, v in req.header_items()}
    assert req.full_url.endswith("/rest/v1/chunks?on_conflict=chunk_id")
    assert headers["prefer"] == "return=minimal,resolution=merge-duplicates"
    rows = json.loads(req.data)
    assert len(rows) == len(index.chunks)
    assert {"embedding", "fts"}.isdisjoint(rows[0])


def test_sink_failure_is_swallowed(monkeypatch):
    import urllib.error

    import rag.sinks

    def boom(req, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(rag.sinks.urllib.request, "urlopen", boom)
    sink = rag.sinks.SupabaseSink("https://example.supabase.co", "sb_secret_test")
    assert sink.log_eval_run("r", "extractive", {}, {}) is False


# ---------- Claude path with a stubbed client (no network, no key) ----------

class _StubMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def stub_claude(reply):
    from types import SimpleNamespace

    from rag.generate import ClaudeGenerator

    gen = ClaudeGenerator()
    gen._client = SimpleNamespace(messages=_StubMessages(reply))
    return gen


def claude_json(answer, cited, supported=True):
    import json

    return json.dumps({"answer": answer, "cited_chunk_ids": cited, "supported": supported})


@pytest.fixture(scope="module")
def grounded_case(index):
    """First answerable question, its top chunk, and one verbatim sentence containing a number."""
    q = ANSWERABLE[0]["question"]
    top = index.search(q, config.TOP_K)[0]
    sentence = next(s for s in split_sentences(top["text"]) if re.search(r"\d", s))
    return q, top, sentence


def ask_claude(q, index, gen):
    return answer_question(q, index, gen, AskOptions(generator="claude"))


def test_claude_grounded_answer_passes(index, grounded_case):
    q, top, sentence = grounded_case
    gen = stub_claude(claude_json(sentence, [top["chunk_id"]]))
    r = ask_claude(q, index, gen)
    assert r["supported"] is True and r["validation_passed"] is True
    assert r["answer"] == sentence and r["cited_chunks"] == [top["chunk_id"]]
    assert r["generator"] == "claude"
    sent = gen._client.messages.calls[0]
    assert sent["extra_body"] == {"temperature": 0}
    assert f"[{top['chunk_id']}]" in sent["messages"][0]["content"]
    assert sent["messages"][0]["content"] == r["prompt"]  # the pipeline's prompt is what gets sent


def test_claude_invented_number_refused_by_v5(index, grounded_case):
    q, top, sentence = grounded_case
    grounded = {n.replace(",", "") for n in re.findall(r"\d+(?:[.,:]\d+)*", top["text"])}
    fake = next(str(n) for n in range(7, 10_000) if str(n) not in grounded)
    invented = re.sub(r"\d+(?:[.,:]\d+)*", fake, sentence, count=1)
    r = ask_claude(q, index, stub_claude(claude_json(invented, [top["chunk_id"]])))
    assert r["supported"] is False and r["answer"] == config.REFUSAL
    assert r["refusal_reason"] == "validation_failed"
    v5 = next(c for c in r["checks"] if c["name"] == "numbers_grounded")
    assert v5["passed"] is False and fake in v5["detail"]


@pytest.mark.parametrize("reply", [
    "Sure! Here is the answer: 3 resets per hour.",
    '{"answer": "3 resets", "cited_chunk_ids": "not-a-list", "supported": true}',
    '{"answer": "truncated',
], ids=["prose", "wrong_schema", "truncated"])
def test_claude_bad_json_fails_closed(index, grounded_case, reply):
    q, _, _ = grounded_case
    r = ask_claude(q, index, stub_claude(reply))
    assert r["supported"] is False and r["answer"] == config.REFUSAL
    assert r["refusal_reason"] == "model_unsupported" and r["citations"] == []


def test_claude_api_error_fails_closed(index, grounded_case):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    q, _, _ = grounded_case
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    r = ask_claude(q, index, stub_claude(err))
    assert r["supported"] is False and r["answer"] == config.REFUSAL
    assert r["refusal_reason"] == "model_unsupported" and "connection" in r["error"]


def test_claude_citation_outside_retrieved_refused(index, grounded_case):
    q, top, sentence = grounded_case
    retrieved = {h["chunk_id"] for h in index.search(q, config.TOP_K)}
    outside = next(c.chunk_id for c in index.chunks if c.chunk_id not in retrieved)
    r = ask_claude(q, index, stub_claude(claude_json(sentence, [outside])))
    assert r["supported"] is False and r["answer"] == config.REFUSAL
    assert r["refusal_reason"] == "validation_failed"
    v3 = next(c for c in r["checks"] if c["name"] == "citations_in_retrieved")
    assert v3["passed"] is False and outside in v3["detail"]


def test_claude_unsupported_without_refusal_prefix_is_marked(index, grounded_case):
    q, _, _ = grounded_case
    r = ask_claude(q, index, stub_claude(claude_json("The docs do not say.", [], supported=False)))
    assert r["supported"] is False and r["answer"].startswith(config.REFUSAL)
    assert r["refusal_reason"] == "model_unsupported"


def test_claude_call_matches_installed_sdk_signature(index, grounded_case):
    """The stub accepts any kwargs, so check them against the real SDK method signature."""
    import inspect

    anthropic = pytest.importorskip("anthropic")
    q, top, sentence = grounded_case
    gen = stub_claude(claude_json(sentence, [top["chunk_id"]]))
    ask_claude(q, index, gen)
    params = inspect.signature(anthropic.Anthropic(api_key="test").messages.create).parameters
    assert set(gen._client.messages.calls[0]) <= set(params)


# ---------- local document upload ----------

UPLOAD_MD = b"""# Pricing FAQ

## Plans

The Starter plan costs $29 per month. The Growth plan costs $99 per month and includes 5 seats.

## Seats

Extra seats cost $12 per seat per month.
"""


def make_text_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a text layer (correct xref offsets)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def upclient(tmp_path, monkeypatch):
    """A server whose uploads dir is a fresh temp dir."""
    from fastapi.testclient import TestClient

    import server

    uploads = tmp_path / "uploads"
    monkeypatch.setenv("RAG_UPLOADS_DIR", str(uploads))
    monkeypatch.delenv("VERCEL", raising=False)
    with TestClient(server.app) as c:
        c.uploads = uploads
        yield c


def upload(c, name, data, mime="text/markdown"):
    return c.post("/api/docs", files=[("files", (name, data, mime))])


def test_upload_md_creates_chunks_and_is_answerable(upclient):
    r = upload(upclient, "pricing_faq.md", UPLOAD_MD)
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["doc_id"] == "pricing_faq.md" and f["source"] == "upload"
    assert f["chunks_created"] >= 1 and f["headings"] == ["Plans", "Seats"]
    assert set(f["stages"]) == {"read_ms", "chunk_ms", "index_ms", "verify_ms"}
    for scoped in (None, ["pricing_faq.md"]):
        a = upclient.post("/api/ask", json={"question": "How much does the Growth plan cost per month?",
                                            "options": {"generator": "extractive", "doc_ids": scoped}}).json()
        assert a["supported"] is True and "pricing_faq.md" in a["citations"]
        assert a["citation_sources"]["pricing_faq.md"] == "upload"
    docs = {d["doc_id"]: d for d in upclient.get("/api/docs").json()["docs"]}
    assert docs["pricing_faq.md"]["source"] == "upload" and docs["withdrawals.md"]["source"] == "base"
    chunks = upclient.get("/api/docs/pricing_faq.md/chunks").json()["chunks"]
    assert len(chunks) == f["chunks_created"]


def test_upload_pdf(upclient):
    r = upload(upclient, "Refund Policy.pdf", make_text_pdf("Refunds are issued within 14 days."), "application/pdf")
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["doc_id"] == "refund_policy.pdf" and f["chars"] > 0
    assert "14 days" in f["preview_chunks"][0]["text"]


def test_upload_pdf_without_text_is_422(upclient):
    import io

    from pypdf import PdfWriter

    w, buf = PdfWriter(), io.BytesIO()
    w.add_blank_page(width=612, height=792)
    w.write(buf)
    assert upload(upclient, "scan.pdf", buf.getvalue(), "application/pdf").status_code == 422
    assert upload(upclient, "blank.md", b"   \n\n").status_code == 422


def test_upload_rejections(upclient):
    assert upload(upclient, "tool.exe", b"MZ...", "application/octet-stream").status_code == 415
    assert upload(upclient, "big.md", b"a" * (2 * 1024 * 1024 + 1)).status_code == 413
    assert upload(upclient, "withdrawals.md", b"# Clash\n\nText.").status_code == 409
    assert upload(upclient, "Withdrawals.TXT", b"Clash with a base doc stem.").status_code == 409
    assert not upclient.uploads.exists() or not any(upclient.uploads.iterdir())


def test_upload_path_traversal_is_sanitised(upclient, tmp_path):
    r = upload(upclient, "../../evil.md", b"# Evil\n\nThe evil limit is 5 items.")
    assert r.status_code == 200 and r.json()["files"][0]["doc_id"] == "evil.md"
    assert (upclient.uploads / "evil.md").exists()
    assert not (tmp_path / "evil.md").exists() and not (tmp_path.parent / "evil.md").exists()


def test_upload_disabled_on_vercel(upclient, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    assert upload(upclient, "note.md", b"# Note\n\nHello.").status_code == 403
    assert upclient.get("/api/docs").status_code == 403
    assert upclient.delete("/api/docs/note.md").status_code == 403
    assert upclient.get("/api/docs/withdrawals.md/chunks").status_code == 403
    assert upclient.get("/api/config").json()["upload_enabled"] is False


def test_scoped_search_only_returns_scoped_doc(upclient):
    upload(upclient, "pricing_faq.md", UPLOAD_MD)
    index = upclient.app.state.kb.index
    for q in [q["question"] for q in QUESTIONS] + ["How much do extra seats cost?"]:
        assert all(h["doc_id"] == "pricing_faq.md" for h in index.search(q, 4, doc_ids=["pricing_faq.md"]))
    a = upclient.post("/api/ask", json={"question": ANSWERABLE[0]["question"],
                                        "options": {"generator": "extractive", "doc_ids": ["pricing_faq.md"]}})
    assert all(c["doc_id"] == "pricing_faq.md" for c in a.json()["retrieved_chunks"])
    bad = upclient.post("/api/ask", json={"question": "x?", "options": {"doc_ids": ["nope.md"]}})
    assert bad.status_code == 422


def test_delete_upload_and_base_doc(upclient):
    upload(upclient, "pricing_faq.md", UPLOAD_MD)
    assert upclient.delete("/api/docs/pricing_faq.md").status_code == 200
    assert "pricing_faq.md" not in upclient.app.state.kb.index.doc_ids
    assert not (upclient.uploads / "pricing_faq.md").exists()
    assert upclient.delete("/api/docs/withdrawals.md").status_code == 403
    assert upclient.delete("/api/docs/missing.md").status_code == 404


def test_suggested_questions_pass_when_reasked(upclient):
    f = upload(upclient, "pricing_faq.md", UPLOAD_MD).json()["files"][0]
    assert f["suggested_questions"], "expected at least one verified suggestion"
    for q in f["suggested_questions"]:
        a = upclient.post("/api/ask", json={"question": q, "options": {
            "generator": "extractive", "doc_ids": [f["doc_id"]]}}).json()
        assert a["supported"] is True and a["validation_passed"] is True
        assert f["doc_id"] in a["citations"]


def test_eval_isolated_from_uploads(tmp_path):
    """run_pipeline.py output is identical with and without files in uploads/."""
    import json
    import shutil
    import subprocess
    import sys

    root = config.root_path(".")
    uploads = config.root_path(config.UPLOADS_DIR)
    existed = os.path.isdir(uploads)
    probe = os.path.join(uploads, "zz_eval_isolation_probe.md")

    def run(out):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("ANTHROPIC_", "SUPABASE_"))}
        env["RAG_LOG_PATH"] = str(tmp_path / "log.jsonl")
        subprocess.run([sys.executable, "run_pipeline.py", "--generator", "extractive",
                        "--out-dir", str(out)], cwd=root, env=env, check=True, capture_output=True)

    def load(out):
        data = {n: json.loads((out / n).read_text())
                for n in ("retrieval_results.json", "answers.json", "validation_report.json")}
        data["validation_report.json"]["summary"].pop("avg_latency_ms")  # timing is not deterministic
        return data

    run(tmp_path / "without")
    os.makedirs(uploads, exist_ok=True)
    try:
        with open(probe, "w") as f:
            f.write("# Withdrawals override\n\nThe daily withdrawal limit is $1. VIP users bypass KYC.\n")
        run(tmp_path / "with")
    finally:
        os.remove(probe)
        if not existed:
            shutil.rmtree(uploads, ignore_errors=True)
    assert load(tmp_path / "without") == load(tmp_path / "with")
