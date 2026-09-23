"""FastAPI server: JSON API plus the single-page UI in web/."""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag import config
from rag.controls import LENGTH_PRESETS, AskOptions
from rag.evalset import load_questions
from rag.generate import claude_available, get_generator
from rag.obs import log, new_run_id
from rag.pipeline import answer_question
from rag.retrieve import build_index

WEB_DIR = config.root_path("web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = config.root_path(os.environ.get("RAG_DOCS_DIR", config.DOCS_DIR))
    app.state.index = build_index(docs_dir)
    app.state.questions = load_questions(
        config.root_path(os.environ.get("RAG_QUESTIONS", config.QUESTIONS_PATH)))
    log("server_start", chunks=len(app.state.index.chunks), docs=len(app.state.index.doc_ids))
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
    index = request.app.state.index
    return {
        "defaults": AskOptions().model_dump(),
        "threshold_range": [0.0, 0.5],
        "length_presets": LENGTH_PRESETS,
        "generators": {"extractive": True, "claude": claude_available()},
        "kb": {"docs": len(index.doc_ids), "chunks": len(index.chunks), "doc_ids": index.doc_ids},
        "sample_questions": [q["question"] for q in request.app.state.questions[:6]],
    }


@app.post("/api/ask")
@app.post("/ask")
def ask(body: AskRequest, request: Request):
    run_id = new_run_id()
    opts = body.options
    if opts.generator == "claude" and not claude_available():
        raise HTTPException(400, "Claude generator unavailable: set ANTHROPIC_API_KEY")
    index = request.app.state.index
    record = answer_question(body.question, index, get_generator(opts.generator, index.vectorizer), opts)
    cited = set(record["cited_chunks"])
    record["retrieved_chunks"] = [{**h, "cited": h["chunk_id"] in cited}
                                  for h in record["retrieved_chunks"]]
    record.pop("prompt", None)
    record["run_id"] = run_id
    log("api_ask", question_chars=len(body.question), supported=record["supported"],
        refusal_reason=record["refusal_reason"], generator=record["generator"],
        options=record["options_applied"], latency_ms=record["latency_ms"])
    return record


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
