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
def client():
    from fastapi.testclient import TestClient

    import server

    with TestClient(server.app) as c:
        yield c


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
