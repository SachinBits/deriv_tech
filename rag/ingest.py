"""Load docs and split them into heading-aware chunks of at most CHUNK_CHARS."""

import io
import os
import re
from dataclasses import asdict, dataclass

from . import config
from .obs import log

DOC_EXTENSIONS = (".md", ".txt", ".pdf")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    heading: str
    text: str
    source: str = "base"

    def to_dict(self) -> dict:
        return asdict(self)


def pdf_text(data: bytes) -> str:
    """Extract text page by page with pypdf (imported lazily); pages without text are skipped."""
    from pypdf import PdfReader

    pages = [(page.extract_text() or "").strip() for page in PdfReader(io.BytesIO(data)).pages]
    return "\n\n".join(p for p in pages if p)


def read_text(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        return pdf_text(data)
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def load_docs(dirs=config.DOCS_DIR) -> list[dict]:
    """Load docs from one directory, or from [(dir, source), ...] with source "base" | "upload".

    Each doc is {doc_id, text, source, added_at}. Files are sorted by name within each directory,
    and a directory that doesn't exist is skipped.
    """
    if isinstance(dirs, str):
        dirs = [(dirs, "base")]
    docs = []
    for directory, source in dirs:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and name.lower().endswith(DOC_EXTENSIONS):
                with open(path, "rb") as f:
                    text = read_text(name, f.read())
                docs.append({"doc_id": name, "text": text, "source": source,
                             "added_at": os.path.getmtime(path)})
    return docs


def split_sentences(text: str) -> list[str]:
    """Split on line breaks, then on sentence punctuation. List markers are removed."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if line:
            out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Text before the first heading gets an empty heading."""
    sections, heading, body = [], "", []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append((heading, "\n".join(body)))
            heading, body = m.group(2), []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body)))
    return sections


def _pieces(paragraph: str, limit: int) -> list[str]:
    """Break a paragraph into units no longer than limit: sentences, then words."""
    if len(paragraph) <= limit:
        return [paragraph]
    units = []
    for sentence in split_sentences(paragraph):
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        current = ""
        for word in sentence.split():
            if current and len(current) + 1 + len(word) > limit:
                units.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            units.append(current)
    return units


def _pack(body: str, limit: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        for n, unit in enumerate(_pieces(paragraph, limit)):
            sep = "\n\n" if n == 0 else " "
            if current and len(current) + len(sep) + len(unit) > limit:
                chunks.append(current)
                current = unit
            else:
                current = f"{current}{sep}{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def chunk(docs: list[dict], chunk_chars: int = config.CHUNK_CHARS) -> list[Chunk]:
    chunks = []
    for doc in docs:
        stem = os.path.splitext(doc["doc_id"])[0]
        i = 0
        for heading, body in _sections(doc["text"]):
            for text in _pack(body, chunk_chars):
                if text.strip():
                    chunks.append(Chunk(doc["doc_id"], f"{stem}_{i}", heading, text.strip(),
                                        doc.get("source", "base")))
                    i += 1
    log("ingest", docs_loaded=len(docs), chunks_created=len(chunks))
    return chunks
