"""SQLite-backed cache for LLM judge results.

Caches judge verdicts keyed by SHA-256(query + answer + sorted context hashes),
scoped to a specific judge prompt version so cache entries are automatically
invalidated when the prompt changes.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


def compute_cache_key(query: str, answer: str, context_hashes: list[str]) -> str:
    """Compute a deterministic cache key from case inputs and outputs.

    Context hashes are sorted before hashing so that order doesn't matter.
    """
    sorted_hashes = sorted(context_hashes)
    raw = f"{query}\n{answer}\n{'|'.join(sorted_hashes)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JudgeCache:
    """Version-aware SQLite cache for judge results."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT NOT NULL,
                    version TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (cache_key, version)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, cache_key: str, version: str) -> Optional[list[dict]]:
        """Retrieve cached results, or None on miss or version mismatch."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT results FROM judge_cache WHERE cache_key = ? AND version = ?",
                (cache_key, version),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, version: str, results: list[dict]) -> None:
        """Store judge results, overwriting any existing entry for this key+version."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO judge_cache (cache_key, version, results, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, version, json.dumps(results), datetime.now().isoformat()),
            )

    def stats(self) -> dict:
        """Return basic cache statistics."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()
        return {"total": row[0]}
