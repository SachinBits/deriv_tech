"""Live knowledge base for the server: base docs plus local uploads, with thread-safe rebuilds."""

import os
import re
import threading
import time

from . import config
from .ingest import DOC_EXTENSIONS, chunk, load_docs, read_text
from .retrieve import Index

UPLOAD_LIMITS = {
    "extensions": list(DOC_EXTENSIONS),
    "max_file_bytes": 2 * 1024 * 1024,
    "max_files_per_request": 5,
    "max_uploaded_docs": 20,
}


class UploadError(Exception):
    """An upload was rejected. `status` is the HTTP status the API should return."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def sanitize_filename(name: str) -> str:
    """Keep only the basename, lower-cased, as [a-z0-9_-] plus an allowed extension."""
    base = re.split(r"[\\/]", name or "")[-1].strip().lower()
    stem, ext = os.path.splitext(base)
    if ext not in DOC_EXTENSIONS:
        raise UploadError(415, f"Unsupported file type {ext or '(none)'}; use .md, .txt or .pdf")
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem).strip("_-")[:80] or "upload"
    return stem + ext


class KnowledgeBase:
    def __init__(self, docs_dir: str, uploads_dir: str | None):
        self.docs_dir = docs_dir
        self.uploads_dir = uploads_dir  # None disables uploads (e.g. on Vercel)
        self._lock = threading.Lock()  # serialises mutations; readers just take self.index
        self.index: Index = Index()
        self.docs: dict[str, dict] = {}
        self.rebuild()

    def _dirs(self) -> list[tuple[str, str]]:
        dirs = [(self.docs_dir, "base")]
        if self.uploads_dir:
            dirs.append((self.uploads_dir, "upload"))
        return dirs

    def rebuild(self) -> float:
        """Build a new Index from disk, then swap it in. Returns the build time in ms."""
        started = time.perf_counter()
        docs = load_docs(self._dirs())
        chunks = chunk(docs)
        index = Index().build(chunks)
        meta = {}
        for d in docs:
            n = sum(c.doc_id == d["doc_id"] for c in chunks)
            meta[d["doc_id"]] = {"doc_id": d["doc_id"], "source": d["source"], "chunks": n,
                                 "chars": len(d["text"]), "added_at": d["added_at"]}
        self.index, self.docs = index, meta  # swap: in-flight requests keep the old index
        return (time.perf_counter() - started) * 1000

    def list_docs(self) -> list[dict]:
        return sorted(self.docs.values(), key=lambda d: (d["source"] != "base", d["doc_id"]))

    def chunks_of(self, doc_id: str) -> list:
        return [c for c in self.index.chunks if c.doc_id == doc_id]

    def add_uploads(self, files: list[tuple[str, bytes]]) -> tuple[list[dict], float]:
        """Validate every file first (all or nothing), then save and rebuild once.

        Returns per-file info and the rebuild time in ms. Raises UploadError.
        """
        if not self.uploads_dir:
            raise UploadError(403, "Uploads are disabled")
        if not files:
            raise UploadError(400, "No files provided")
        if len(files) > UPLOAD_LIMITS["max_files_per_request"]:
            raise UploadError(400, f"At most {UPLOAD_LIMITS['max_files_per_request']} files per request")
        with self._lock:
            existing_stems = {os.path.splitext(d)[0]: d for d in self.docs}
            uploads = [d for d in self.docs.values() if d["source"] == "upload"]
            if len(uploads) + len(files) > UPLOAD_LIMITS["max_uploaded_docs"]:
                raise UploadError(400, f"Upload limit reached ({UPLOAD_LIMITS['max_uploaded_docs']} docs); "
                                       "delete one first")
            prepared, seen = [], set()
            for original, data in files:
                name = sanitize_filename(original)
                if len(data) > UPLOAD_LIMITS["max_file_bytes"]:
                    raise UploadError(413, f"{name} is larger than 2 MB")
                stem = os.path.splitext(name)[0]
                clash = existing_stems.get(stem)
                if clash or stem in seen:
                    owner = self.docs.get(clash, {}).get("source") if clash else "upload"
                    what = "a base document" if owner == "base" else "an existing upload"
                    raise UploadError(409, f"{name} conflicts with {what} ({clash or name}); rename it")
                seen.add(stem)
                t0 = time.perf_counter()
                try:
                    text = read_text(name, data)
                except Exception:  # corrupt or encrypted PDF
                    raise UploadError(422, f"Could not read {name}")
                if not text.strip():
                    raise UploadError(422, f"No text found in {name}")
                read_ms = (time.perf_counter() - t0) * 1000
                t1 = time.perf_counter()
                chunk([{"doc_id": name, "text": text, "source": "upload"}])
                prepared.append({"doc_id": name, "data": data, "chars": len(text), "read_ms": read_ms,
                                 "chunk_ms": (time.perf_counter() - t1) * 1000})

            os.makedirs(self.uploads_dir, exist_ok=True)
            root = os.path.realpath(self.uploads_dir)
            for p in prepared:
                path = os.path.realpath(os.path.join(root, p["doc_id"]))
                if os.path.dirname(path) != root:  # defence in depth; sanitize_filename already strips paths
                    raise UploadError(400, "Invalid filename")
                with open(path, "wb") as f:
                    f.write(p.pop("data"))
            index_ms = self.rebuild()
        return prepared, index_ms

    def delete_upload(self, doc_id: str) -> float:
        with self._lock:
            doc = self.docs.get(doc_id)
            if doc is None:
                raise UploadError(404, "Unknown document")
            if doc["source"] != "upload" or not self.uploads_dir:
                raise UploadError(403, "Base documents can't be deleted")
            path = os.path.join(self.uploads_dir, doc_id)
            if os.path.exists(path):
                os.remove(path)
            return self.rebuild()


def default_uploads_dir() -> str:
    return config.root_path(os.environ.get("RAG_UPLOADS_DIR") or config.UPLOADS_DIR)
