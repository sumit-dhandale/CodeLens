"""Qdrant collection management, idempotent indexing, and ANN search.

One collection per `{repo}__{model}__{strategy}`, so a chunking experiment can
never silently overwrite another one's points, and a query can never be answered
by vectors from a different model.

Indexing is a *sync*, not an append:

- Point IDs are `uuid5` over `path:class:symbol:start_line:strategy`, so
  re-indexing an edited function updates the same point instead of adding a
  second copy.
- Each payload carries a content hash, so an unchanged chunk is not re-embedded.
- Points whose key is absent from the current parse are deleted. That last half
  is the one everyone skips, and it is why a search after a refactor returns
  functions that no longer exist.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..chunking.code_chunker import Chunk
from ..config import Config
from ..parser.repo_fetcher import parse_repo_url

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

Vector = list[float]
EmbedFn = Callable[[list[str]], list[Vector]]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Fields worth filtering on. Milestone 4 uses these; creating them at collection
# time costs nothing and avoids a reindex later.
KEYWORD_FIELDS = ("language", "symbol_type", "directory", "file_path", "class_name")

_SCROLL_BATCH = 1000


def _slugify(value: str) -> str:
    return _UNSAFE.sub("-", value).strip("-")


def model_slug(model: str) -> str:
    """`sentence-transformers/all-MiniLM-L6-v2` -> `all-MiniLM-L6-v2`."""
    return _slugify(model.rsplit("/", 1)[-1])


def repo_slug(repo: str) -> str:
    try:
        owner, name = parse_repo_url(repo)
    except ValueError:
        return _slugify(repo.rstrip("/").rsplit("/", 1)[-1] or "local")
    return f"{_slugify(owner)}__{_slugify(name)}"


def collection_name(cfg: Config, strategy: str | None = None) -> str:
    parts = [repo_slug(cfg.repo), model_slug(cfg.model), strategy or cfg.strategy]
    if cfg.collection_suffix:
        parts.append(_slugify(cfg.collection_suffix))
    return "__".join(parts)


def payload_of(chunk: Chunk) -> dict[str, Any]:
    return {
        "point_key": chunk.point_key,
        "file_path": chunk.file_path,
        "directory": chunk.directory,
        "language": chunk.language,
        "strategy": chunk.strategy,
        "symbol_type": chunk.symbol_type,
        "symbol_name": chunk.symbol_name,
        "class_name": chunk.class_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "file_sha": chunk.file_sha,
        # Hash of what was embedded, not of the file: an edit elsewhere in the
        # file must not invalidate this chunk's vector.
        "content_hash": content_hash(chunk),
    }


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.embed_text.encode()).hexdigest()


@dataclass(frozen=True)
class SyncReport:
    collection: str
    dim: int
    chunks: int
    upserted: int
    unchanged: int
    deleted: int
    points: int

    def __str__(self) -> str:
        return (
            f"{self.collection}\n"
            f"  dim:       {self.dim}\n"
            f"  chunks:    {self.chunks}\n"
            f"  upserted:  {self.upserted}\n"
            f"  unchanged: {self.unchanged}\n"
            f"  deleted:   {self.deleted}\n"
            f"  points:    {self.points}"
        )


@dataclass(frozen=True)
class Hit:
    score: float
    file_path: str
    language: str
    symbol_type: str
    symbol_name: str
    class_name: str
    start_line: int
    end_line: int
    text: str

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> Hit:
        """Lets keyword and vector retrievers return the same type, so fusion
        can treat them interchangeably and `citation` is a usable join key."""
        return cls(
            score=score,
            file_path=chunk.file_path,
            language=chunk.language,
            symbol_type=chunk.symbol_type,
            symbol_name=chunk.symbol_name,
            class_name=chunk.class_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
        )

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.symbol_name}" if self.class_name else self.symbol_name

    def snippet(self, lines: int = 3) -> str:
        return "\n".join(self.text.splitlines()[:lines])


class QdrantStore:
    def __init__(
        self, cfg: Config, collection: str | None = None, client: QdrantClient | None = None
    ):
        self.cfg = cfg
        self.collection = collection or collection_name(cfg)
        self._client = client

    @property
    def is_local(self) -> bool:
        """Local mode is the pure-python engine: no server, and no payload indexes."""
        return not self.cfg.qdrant_url.startswith("http")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            from qdrant_client import QdrantClient

            # `:memory:` and a directory path both run the local engine, which is
            # what makes the tests serverless.
            if self.is_local:
                self._client = QdrantClient(location=self.cfg.qdrant_url or ":memory:")
            else:
                self._client = QdrantClient(url=self.cfg.qdrant_url)
        return self._client

    # --- collection lifecycle ---------------------------------------------

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def vector_dim(self) -> int | None:
        if not self.exists():
            return None
        params = self.client.get_collection(self.collection).config.params
        return int(params.vectors.size)  # type: ignore[union-attr]

    def ensure_collection(self, dim: int | None = None) -> bool:
        """True when the collection was created by this call.

        `dim=None` means "adopt the existing collection's dimension", which lets a
        fully-cached reindex skip loading the model entirely.
        """
        from qdrant_client import models

        existing_dim = self.vector_dim()
        if existing_dim is not None:
            if dim is not None and existing_dim != dim:
                raise RuntimeError(
                    f"collection {self.collection} holds {existing_dim}-dim vectors but "
                    f"the model produces {dim}; drop it or change --collection-suffix"
                )
            return False
        if dim is None:
            raise RuntimeError(f"collection {self.collection} does not exist; a dim is required")

        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if not self.is_local:
            for name in KEYWORD_FIELDS:
                self.client.create_payload_index(
                    self.collection, field_name=name, field_schema=models.PayloadSchemaType.KEYWORD
                )
        log.info("created collection %s (dim=%d)", self.collection, dim)
        return True

    def drop(self) -> bool:
        return self.exists() and self.client.delete_collection(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # --- indexing ----------------------------------------------------------

    def existing_hashes(self) -> dict[str, str]:
        """point id -> content_hash for everything currently in the collection."""
        out: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for point in points:
                out[str(point.id)] = (point.payload or {}).get("content_hash", "")
            if offset is None:
                return out

    def sync(self, chunks: Sequence[Chunk], embed: EmbedFn, dim: int | None = None) -> SyncReport:
        """Upsert changed chunks, skip unchanged ones, delete orphans.

        The collection is created *after* embedding, so its dimension comes from
        the vectors themselves and no caller has to load a model to ask.
        """
        from qdrant_client import models

        existing = self.existing_hashes() if self.exists() else {}

        current = {str(c.point_id): c for c in chunks}
        if len(current) != len(chunks):
            log.warning("%d chunks collapsed into %d point ids", len(chunks), len(current))

        changed = [c for pid, c in current.items() if existing.get(pid) != content_hash(c)]
        if changed:
            vectors = embed([c.embed_text for c in changed])
            self.ensure_collection(len(vectors[0]))
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=str(c.point_id), vector=v, payload=payload_of(c))
                    for c, v in zip(changed, vectors, strict=True)
                ],
                wait=True,
            )
        else:
            self.ensure_collection(dim)

        orphans = [pid for pid in existing if pid not in current]
        if orphans:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=orphans),
                wait=True,
            )

        return SyncReport(
            collection=self.collection,
            dim=self.vector_dim() or 0,
            chunks=len(chunks),
            upserted=len(changed),
            unchanged=len(current) - len(changed),
            deleted=len(orphans),
            points=self.count(),
        )

    # --- search ------------------------------------------------------------

    def search(
        self,
        vector: Vector,
        top_k: int = 5,
        query_filter: Any = None,
        exact: bool = False,
    ) -> list[Hit]:
        from qdrant_client import models

        response = self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            # exact=True brute-forces every vector: slow, but it is the ground
            # truth that Milestone 5 measures HNSW recall against.
            search_params=models.SearchParams(exact=exact),
            with_payload=True,
        )
        return [_hit(point.score, point.payload or {}) for point in response.points]


def _hit(score: float, payload: dict[str, Any]) -> Hit:
    return Hit(
        score=score,
        file_path=payload.get("file_path", ""),
        language=payload.get("language", ""),
        symbol_type=payload.get("symbol_type", ""),
        symbol_name=payload.get("symbol_name", ""),
        class_name=payload.get("class_name", ""),
        start_line=int(payload.get("start_line", 0)),
        end_line=int(payload.get("end_line", 0)),
        text=payload.get("text", ""),
    )


def make_filter(
    language: str | None = None,
    symbol_type: str | None = None,
    directory: str | None = None,
) -> Any:
    """`must` conditions over the indexed keyword fields, or None if unconstrained."""
    from qdrant_client import models

    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in (
            ("language", language),
            ("symbol_type", symbol_type),
            ("directory", directory),
        )
        if value
    ]
    return models.Filter(must=conditions) if conditions else None
