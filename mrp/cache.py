"""SQLite cache for fetched movie features + OMDb API usage tracking."""
import sqlite3
import json
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from mrp.config import CACHE_DB


class Cache:
    """Persistent key-value store keyed by IMDb ID (tconst)."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or CACHE_DB)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS movie_cache (
                    imdb_id     TEXT PRIMARY KEY,
                    data        TEXT NOT NULL,
                    embedding   BLOB,
                    status      TEXT DEFAULT 'ok',
                    fetched_at  TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON movie_cache(status)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)


    # ── Movie data ─────────────────────────────────────────────────────────

    def get(self, imdb_id):
        """Return cached feature dict (with embedding as np.ndarray) or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data, embedding, status, fetched_at FROM movie_cache WHERE imdb_id = ?",
                (imdb_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["data"])
        data["status"] = row["status"]
        data["fetched_at"] = row["fetched_at"]
        if row["embedding"]:
            data["embedding"] = np.frombuffer(row["embedding"], dtype=np.float32)
        else:
            data["embedding"] = None
        return data

    def set(self, imdb_id, features):
        """Store feature dict.  Does NOT mutate the caller's dict."""
        data_copy = dict(features)
        embedding = data_copy.pop("embedding", None)
        emb_blob = (
            np.asarray(embedding, dtype=np.float32).tobytes()
            if embedding is not None
            else None
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO movie_cache
                   (imdb_id, data, embedding, status, fetched_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    imdb_id,
                    json.dumps(data_copy, default=str),
                    emb_blob,
                    data_copy.get("status", "ok"),
                    datetime.now().isoformat(),
                ),
            )

    def has(self, imdb_id):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM movie_cache WHERE imdb_id = ?", (imdb_id,)
            ).fetchone()
            return row is not None

    def get_all_cached_ids(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT imdb_id FROM movie_cache").fetchall()
            return {r["imdb_id"] for r in rows}

    def get_all_with_status(self):
        """Return dict {imdb_id: status} for every cached entry."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT imdb_id, status FROM movie_cache"
            ).fetchall()
            return {r["imdb_id"]: r["status"] for r in rows}

    def delete(self, imdb_id):
        with self._conn() as conn:
            conn.execute("DELETE FROM movie_cache WHERE imdb_id = ?", (imdb_id,))

    # ── Settings (small key-value store, e.g. active OMDb key index) ───────

    def get_setting(self, key):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_setting(self, key, value):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value))
            )

    def get_plot_count(self):
        """Count how many cached movies actually have plot text."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM movie_cache 
                   WHERE json_extract(data, '$.plot') IS NOT NULL 
                   AND json_extract(data, '$.plot') != ''"""
            ).fetchone()
            return row[0] if row else 0


# ── Module-level singleton ─────────────────────────────────────────────────
_instance = None


def get_cache() -> Cache:
    global _instance
    if _instance is None:
        _instance = Cache()
    return _instance