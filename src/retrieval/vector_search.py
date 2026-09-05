"""Query-side of the vector half: embed the query, ANN search, return hits.

The query goes through the *same* model as the chunks and gets no enrichment —
it is already English. Only the indexed side needs its identifiers split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..embeddings.embedder import Embedder
from ..vector_store.qdrant_store import Hit, QdrantStore


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[Hit]
    embed_ms: float
    search_ms: float
    exact: bool

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms


class VectorSearcher:
    def __init__(
        self, cfg: Config, store: QdrantStore | None = None, embedder: Embedder | None = None
    ):
        self.cfg = cfg
        self.store = store if store is not None else QdrantStore(cfg)
        self.embedder = embedder if embedder is not None else Embedder(cfg)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        query_filter: Any = None,
        exact: bool = False,
    ) -> SearchResult:
        if not query.strip():
            raise ValueError("empty query")
        if not self.store.exists():
            raise RuntimeError(
                f"collection {self.store.collection!r} does not exist; run `index` first"
            )

        started = time.perf_counter()
        # Queries are cached too: an eval sweep re-runs the same 32 strings
        # against every strategy.
        vector = self.embedder.embed(query)
        embedded = time.perf_counter()
        hits = self.store.search(
            vector, top_k=top_k or self.cfg.top_k, query_filter=query_filter, exact=exact
        )
        done = time.perf_counter()
        return SearchResult(
            query=query,
            hits=hits,
            embed_ms=(embedded - started) * 1000,
            search_ms=(done - embedded) * 1000,
            exact=exact,
        )
