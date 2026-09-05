"""A sqlite embedding cache, keyed by sha256(model + text).

The single most valuable piece of scaffolding in this project. Re-running four
chunking strategies against two models on CPU means embedding the same text many
times; a forward pass costs milliseconds, a sqlite lookup costs microseconds.

Vectors are stored as raw float32 (`array('f').tobytes()`), which is 4 bytes per
dimension against ~14 for a JSON list of floats, and needs no parsing on read.
The dimension is implied by the blob length, so nothing has to agree on it.
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path

Vector = list[float]

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds.
_SELECT_BATCH = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key    TEXT PRIMARY KEY,
    model  TEXT NOT NULL,
    dim    INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_model ON embeddings(model);
"""


def cache_key(model: str, text: str) -> str:
    """Model name is part of the key: the same text under two models is two vectors."""
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(b"\x00")  # separator, so ("ab", "c") and ("a", "bc") differ
    digest.update(text.encode())
    return digest.hexdigest()


class EmbeddingCache:
    """Not thread-safe, and deliberately so — the pipeline is single-threaded."""

    def __init__(self, path: Path | str | None = None):
        self.path = path
        self.hits = 0
        self.misses = 0
        target = ":memory:" if path is None else str(path)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(target)
        self._db.executescript(_SCHEMA)
        # WAL survives a crash mid-run without a fsync per insert. A cache is
        # rebuildable, so durability is worth trading for throughput.
        if path is not None:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def get_many(self, model: str, texts: Sequence[str]) -> list[Vector | None]:
        """One entry per input text, `None` where the text is not cached."""
        keys = [cache_key(model, t) for t in texts]
        found: dict[str, Vector] = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), _SELECT_BATCH):
            window = unique[start : start + _SELECT_BATCH]
            placeholders = ",".join("?" * len(window))
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",  # noqa: S608
                window,
            )
            for key, blob in rows:
                vector = array("f")
                vector.frombytes(blob)
                found[key] = vector.tolist()
        out = [found.get(k) for k in keys]
        hits = sum(1 for v in out if v is not None)
        self.hits += hits
        self.misses += len(out) - hits
        return out

    def put_many(
        self, model: str, texts: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(texts) != len(vectors):
            raise ValueError(f"{len(texts)} texts but {len(vectors)} vectors")
        rows = [
            (cache_key(model, text), model, len(vector), array("f", vector).tobytes())
            for text, vector in zip(texts, vectors, strict=True)
        ]
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) VALUES (?, ?, ?, ?)",
                rows,
            )

    def count(self, model: str | None = None) -> int:
        if model is None:
            return self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return self._db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
        ).fetchone()[0]

    def clear(self, model: str | None = None) -> int:
        with self._db:
            if model is None:
                cursor = self._db.execute("DELETE FROM embeddings")
            else:
                cursor = self._db.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        return cursor.rowcount
