"""Optional Supabase sink: persists query logs, eval runs and chunks via PostgREST.

Enabled only when SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_SECRET_KEY) are set.
Tables come from migration 001_init_support_qa; this module never creates or alters them.
Writes are best-effort: a failure is logged to logs/pipeline.jsonl and never affects answering.
"""

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

from .obs import log

TIMEOUT_S = 5
BATCH_SIZE = 500


class SupabaseSink:
    def __init__(self, url: str | None = None, key: str | None = None):
        self.url = (url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.key = (key or os.environ.get("SUPABASE_SERVICE_KEY")
                    or os.environ.get("SUPABASE_SECRET_KEY") or "")

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self) -> dict:
        # New Supabase API keys (sb_secret_...) are not JWTs: send them as `apikey` only,
        # never as `Authorization: Bearer`.
        return {"apikey": self.key, "Content-Type": "application/json", "Prefer": "return=minimal"}

    def _send(self, method: str, table: str, query: str = "", body: list[dict] | None = None,
              prefer: str | None = None, rows: int = 0) -> bool:
        headers = self._headers()
        if prefer:
            headers["Prefer"] = f"{headers['Prefer']},{prefer}"
        data = json.dumps(body, default=str).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}/rest/v1/{table}{query}", method=method,
                                     data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                log("sink", method=method, table=table, rows=rows, ok=True, status=resp.status)
                return True
        except urllib.error.HTTPError as e:
            error = e.read().decode(errors="replace")[:300]
            log("sink", method=method, table=table, rows=rows, ok=False, status=e.code, error=error)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log("sink", method=method, table=table, rows=rows, ok=False, error=f"{type(e).__name__}: {e}")
        return False

    def _post(self, table: str, rows: list[dict], query: str = "", prefer: str | None = None) -> bool:
        if not self.enabled or not rows:
            return False
        results = [self._send("POST", table, query, rows[i:i + BATCH_SIZE], prefer,
                              rows=len(rows[i:i + BATCH_SIZE]))
                   for i in range(0, len(rows), BATCH_SIZE)]
        return all(results)

    def delete_chunks(self, doc_id: str) -> bool:
        if not self.enabled:
            return False
        return self._send("DELETE", "chunks", f"?doc_id=eq.{urllib.parse.quote(doc_id, safe='')}")

    @staticmethod
    def _query_row(record: dict, client: str) -> dict:
        return {
            "client": client,
            "question": record["question"],
            "answer": record["answer"],
            "supported": record["supported"],
            "refusal_reason": record["refusal_reason"],
            "citations": record["citations"],
            "retrieved_sources": record["retrieved_sources"],
            "top_score": record["top_score"],
            "validation_passed": record["validation_passed"],
            "generator": record["generator"],
            "options": record["options_applied"],
            "latency_ms": round(record["latency_ms"]),
        }

    def log_query(self, record: dict, client: str) -> bool:
        return self.log_queries([record], client)

    def log_queries(self, records: list[dict], client: str) -> bool:
        return self._post("query_logs", [self._query_row(r, client) for r in records])

    def log_eval_run(self, run_id: str, generator: str, options: dict, summary: dict) -> bool:
        return self._post("eval_runs", [{"run_id": run_id, "generator": generator,
                                         "options": options, "summary": summary}])

    def upsert_chunks(self, chunks) -> bool:
        """Upsert on chunk_id. `embedding` and `fts` are left to the database."""
        index_in_doc = defaultdict(int)
        rows = []
        for c in chunks:
            rows.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "heading": c.heading,
                "chunk_index": index_in_doc[c.doc_id],
                "content": c.text,
                "content_hash": hashlib.sha256(c.text.encode()).hexdigest(),
            })
            index_in_doc[c.doc_id] += 1
        return self._post("chunks", rows, query="?on_conflict=chunk_id",
                          prefer="resolution=merge-duplicates")


def get_sink() -> SupabaseSink:
    return SupabaseSink()
