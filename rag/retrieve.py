"""TF-IDF retrieval over chunks."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config
from .ingest import Chunk
from .obs import log


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)


class Index:
    def __init__(self):
        self.vectorizer = make_vectorizer()
        self.chunks: list[Chunk] = []
        self.matrix = None

    def build(self, chunks: list[Chunk]) -> "Index":
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = list(chunks)
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)
        return self

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.chunks))

    @property
    def doc_sources(self) -> dict[str, str]:
        return {c.doc_id: c.source for c in self.chunks}

    def search(self, question: str, k: int = config.TOP_K,
               doc_ids: list[str] | None = None) -> list[dict]:
        """Top-k chunks by cosine similarity, optionally restricted to `doc_ids`."""
        query = self.vectorizer.transform([question])
        scores = linear_kernel(query, self.matrix).ravel()
        order = scores.argsort()[::-1]
        if doc_ids is not None:
            allowed = set(doc_ids)
            order = [i for i in order if self.chunks[i].doc_id in allowed]
        order = order[: max(1, k)]
        hits = [
            {
                "doc_id": self.chunks[i].doc_id,
                "chunk_id": self.chunks[i].chunk_id,
                "score": round(float(scores[i]), 4),
                "text": self.chunks[i].text,
            }
            for i in order
        ]
        log("retrieve", top_score=hits[0]["score"] if hits else 0.0,
            chunk_ids=[h["chunk_id"] for h in hits],
            **({"doc_ids": list(doc_ids)} if doc_ids is not None else {}))
        return hits


def build_index(docs_dir: str = config.DOCS_DIR) -> Index:
    from .ingest import chunk, load_docs

    return Index().build(chunk(load_docs(docs_dir)))
