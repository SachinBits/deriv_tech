"""FastAPI server: JSON API plus the single-page UI in web/."""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag import config
from rag.controls import LENGTH_PRESETS, AskOptions
from rag.evalset import load_questions
from rag.generate import claude_available, get_generator
from rag.kb import UPLOAD_LIMITS, KnowledgeBase, UploadError, default_uploads_dir
from rag.obs import log, new_run_id
from rag.pipeline import answer_question
from rag.sinks import get_sink
from rag.suggest import suggest_questions

WEB_DIR = config.root_path("web")


def on_vercel() -> bool:
    return bool(os.environ.get("VERCEL"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = config.root_path(os.environ.get("RAG_DOCS_DIR", config.DOCS_DIR))
    # Vercel instances are stateless, so the index there is built from docs/ only.
    app.state.kb = KnowledgeBase(docs_dir, None if on_vercel() else default_uploads_dir())
    app.state.questions = load_questions(
        config.root_path(os.environ.get("RAG_QUESTIONS", config.QUESTIONS_PATH)))
    app.state.sink = get_sink()
    if app.state.sink.enabled:
        await asyncio.to_thread(app.state.sink.upsert_chunks, app.state.kb.index.chunks)
    index = app.state.kb.index
    log("server_start", chunks=len(index.chunks), docs=len(index.doc_ids),
        supabase=app.state.sink.enabled, uploads=bool(app.state.kb.uploads_dir))
    yield


app = FastAPI(title="Support Assistant", lifespan=lifespan)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    log("http", method=request.method, path=request.url.path, status=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 1))
    return response


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    options: AskOptions = Field(default_factory=AskOptions)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config(request: Request):
    index = request.app.state.kb.index
    return {
        "defaults": AskOptions().model_dump(),
        "threshold_range": [0.0, 0.5],
        "length_presets": LENGTH_PRESETS,
        "generators": {"extractive": True, "claude": claude_available()},
        "kb": {"docs": len(index.doc_ids), "chunks": len(index.chunks), "doc_ids": index.doc_ids},
        "sample_questions": [q["question"] for q in request.app.state.questions[:6]],
        "upload_enabled": not on_vercel(),
        "upload_limits": UPLOAD_LIMITS,
    }


@app.post("/api/ask")
@app.post("/ask")
def ask(body: AskRequest, request: Request, background: BackgroundTasks):
    run_id = new_run_id()
    opts = body.options
    if opts.generator == "claude" and not claude_available():
        raise HTTPException(400, "Claude generator unavailable: set ANTHROPIC_API_KEY")
    index = request.app.state.kb.index  # one snapshot for the whole request
    if opts.doc_ids is not None:
        unknown = sorted(set(opts.doc_ids) - set(index.doc_ids))
        if unknown:
            raise HTTPException(422, f"Unknown doc_ids: {unknown}")
    record = answer_question(body.question, index, get_generator(opts.generator, index.vectorizer), opts)
    cited = set(record["cited_chunks"])
    record["retrieved_chunks"] = [{**h, "cited": h["chunk_id"] in cited}
                                  for h in record["retrieved_chunks"]]
    sources = index.doc_sources
    record["citation_sources"] = {d: sources.get(d, "base") for d in record["citations"]}
    record.pop("prompt", None)
    record["run_id"] = run_id
    log("api_ask", question_chars=len(body.question), supported=record["supported"],
        refusal_reason=record["refusal_reason"], generator=record["generator"],
        options=record["options_applied"], latency_ms=record["latency_ms"])
    if request.app.state.sink.enabled:  # after the response is sent, so latency is unaffected
        background.add_task(request.app.state.sink.log_query, record, "api")
    return record


# ---------- local document upload (403 on Vercel) ----------

def require_local():
    if on_vercel():
        raise HTTPException(403, "Document upload is disabled on this deployment")


def _raise(e: UploadError):
    raise HTTPException(e.status, e.message)


@app.get("/api/docs", dependencies=[Depends(require_local)])
def list_docs(request: Request):
    return {"docs": request.app.state.kb.list_docs(), "limits": UPLOAD_LIMITS}


@app.post("/api/docs", dependencies=[Depends(require_local)])
async def upload_docs(request: Request, background: BackgroundTasks,
                      files: list[UploadFile] = File(...)):
    new_run_id()
    kb, sink = request.app.state.kb, request.app.state.sink
    if len(files) > UPLOAD_LIMITS["max_files_per_request"]:
        raise HTTPException(400, f"At most {UPLOAD_LIMITS['max_files_per_request']} files per request")
    payload = []
    for f in files:
        # Read at most one byte past the limit so an oversized file is rejected without buffering it all.
        data = await f.read(UPLOAD_LIMITS["max_file_bytes"] + 1)
        payload.append((f.filename or "", data))
    try:
        prepared, index_ms = await asyncio.to_thread(kb.add_uploads, payload)
    except UploadError as e:
        log("upload_rejected", status=e.status, error=e.message)
        _raise(e)

    index = kb.index
    results = []
    for p in prepared:
        doc_id = p["doc_id"]
        chunks = kb.chunks_of(doc_id)
        t0 = time.perf_counter()
        suggestions = await asyncio.to_thread(suggest_questions, doc_id, index, "auto")
        verify_ms = (time.perf_counter() - t0) * 1000
        stages = {"read_ms": round(p["read_ms"], 1), "chunk_ms": round(p["chunk_ms"], 1),
                  "index_ms": round(index_ms, 1), "verify_ms": round(verify_ms, 1)}
        ingest_ms = round(p["read_ms"] + p["chunk_ms"] + index_ms, 1)
        results.append({
            "doc_id": doc_id,
            "source": "upload",
            "chunks_created": len(chunks),
            "chars": p["chars"],
            "headings": list(dict.fromkeys(c.heading for c in chunks if c.heading)),
            "preview_chunks": [{"chunk_id": c.chunk_id, "heading": c.heading, "text": c.text}
                               for c in chunks[:3]],
            "suggested_questions": suggestions["questions"],
            "suggestions_method": suggestions["method"],
            "ingest_ms": ingest_ms,
            "stages": stages,
        })
        log("upload", doc_id=doc_id, chunks_created=len(chunks), ingest_ms=ingest_ms,
            suggestions_kept=len(suggestions["questions"]), suggestions_tried=suggestions["tried"],
            suggestions_method=suggestions["method"])
        if sink.enabled:
            background.add_task(sink.upsert_chunks, chunks)
    return {"files": results, "kb": {"docs": len(index.doc_ids), "chunks": len(index.chunks)}}


@app.delete("/api/docs/{doc_id}", dependencies=[Depends(require_local)])
def delete_doc(doc_id: str, request: Request, background: BackgroundTasks):
    new_run_id()
    kb, sink = request.app.state.kb, request.app.state.sink
    try:
        index_ms = kb.delete_upload(doc_id)
    except UploadError as e:
        _raise(e)
    log("upload_delete", doc_id=doc_id, index_ms=round(index_ms, 1))
    if sink.enabled:
        background.add_task(sink.delete_chunks, doc_id)
    index = kb.index
    return {"deleted": doc_id, "kb": {"docs": len(index.doc_ids), "chunks": len(index.chunks)}}


@app.get("/api/docs/{doc_id}/chunks", dependencies=[Depends(require_local)])
def doc_chunks(doc_id: str, request: Request):
    kb = request.app.state.kb
    if doc_id not in kb.docs:
        raise HTTPException(404, "Unknown document")
    return {"doc_id": doc_id, "source": kb.docs[doc_id]["source"],
            "chunks": [{"chunk_id": c.chunk_id, "heading": c.heading, "text": c.text}
                       for c in kb.chunks_of(doc_id)]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
