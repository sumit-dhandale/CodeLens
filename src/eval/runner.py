"""Run every retriever over the gold set and print one comparison table.

Also dumps per-query results to `eval/results/`. With 32 queries the mean hides
everything interesting: a retriever that wins on average can be the only one
that fails the three queries you actually care about, and you cannot see that in
a table of five numbers.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..chunking import code_chunker
from ..chunking.code_chunker import Chunk
from ..config import ROOT, Config
from ..embeddings.embedder import Embedder
from ..parser.repository_loader import SourceFile
from ..retrieval import context, fusion, keyword_search
from ..retrieval.reranker import CrossEncoderReranker
from ..vector_store.qdrant_store import Hit, QdrantStore
from .gold import chunk_matches
from .metrics import QueryScore, Summary, summarize

RETRIEVERS = (
    "random",
    "grep",
    "bm25",
    "vector",
    "hybrid_weighted",
    "hybrid_rrf",
    "vector_rerank",
    "hybrid_rrf_rerank",
    "vector_mmr",
    "vector_cap",
)

RESULTS_DIR = ROOT / "eval" / "results"

Retriever = Callable[[str, int], list[Hit]]


@dataclass
class Harness:
    """Owns the corpus and the one store/embedder pair every retriever shares."""

    cfg: Config
    chunks: list[Chunk]
    store: QdrantStore | None = None
    embedder: Embedder | None = None
    reranker: CrossEncoderReranker | None = None

    @classmethod
    def build(cls, cfg: Config, files: list[SourceFile]) -> Harness:
        return cls(cfg=cfg, chunks=code_chunker.chunk(files, cfg, cfg.strategy))

    def __post_init__(self) -> None:
        self._bm25: keyword_search.BM25Index | None = None

    @property
    def bm25(self) -> keyword_search.BM25Index:
        if self._bm25 is None:
            self._bm25 = keyword_search.BM25Index(self.chunks)
        return self._bm25

    def _vector(self, query: str, top_k: int, with_vectors: bool = False) -> list[Hit]:
        if self.store is None or self.embedder is None:
            raise RuntimeError("vector retrievers need a store and an embedder")
        return self.store.search(self.embedder.embed(query), top_k=top_k, with_vectors=with_vectors)

    def _rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if self.reranker is None:
            raise RuntimeError("rerank retrievers need a reranker")
        return self.reranker.rerank(query, hits, top_k)

    def _random(self, query: str, top_k: int) -> list[Hit]:
        # Seeded on the query so the floor is reproducible across runs; a
        # different seed per query so it is not the same chunks every time.
        rng = random.Random(query)  # noqa: S311 - a baseline, not a secret
        picks = rng.sample(self.chunks, min(top_k, len(self.chunks)))
        return [Hit.from_chunk(c, 0.0) for c in picks]

    def retriever(self, name: str, depth: int | None = None) -> Retriever:
        """`depth` is how many candidates each arm of a hybrid retrieves before fusing."""
        depth = depth or self.cfg.rerank_top_n
        if name == "random":
            return self._random
        if name == "grep":
            return lambda q, k: keyword_search.grep_search(self.chunks, q, k)
        if name == "bm25":
            return self.bm25.search
        if name == "vector":
            return self._vector
        if name == "hybrid_weighted":
            return lambda q, k: fusion.weighted_fusion(
                self._vector(q, depth), self.bm25.search(q, depth), self.cfg.vector_weight, k
            )
        if name == "hybrid_rrf":
            return lambda q, k: fusion.reciprocal_rank_fusion(
                [self._vector(q, depth), self.bm25.search(q, depth)], top_k=k
            )
        # The whole point of the two-stage shape: retrieve `depth` candidates
        # cheaply, then spend the cross-encoder only on those.
        if name == "vector_rerank":
            return lambda q, k: self._rerank(q, self._vector(q, depth), k)
        if name == "hybrid_rrf_rerank":
            return lambda q, k: self._rerank(
                q,
                fusion.reciprocal_rank_fusion(
                    [self._vector(q, depth), self.bm25.search(q, depth)], top_k=depth
                ),
                k,
            )
        if name == "vector_mmr":
            return lambda q, k: context.mmr(
                self._vector(q, depth, with_vectors=True), self.cfg.mmr_lambda, k
            )
        if name == "vector_cap":
            return lambda q, k: context.per_file_cap(
                self._vector(q, depth), self.cfg.per_file_cap or 2, k
            )
        raise ValueError(f"unknown retriever {name!r}, expected one of {RETRIEVERS}")


def grade_hits(hits: Sequence[Hit], gold: list[dict]) -> list[int]:
    """Gold grade of each hit in rank order, 0 for unlabelled.

    `max` because a hit can satisfy two labels at once — a file-level label
    (`symbol: null`) and a symbol label both match the same chunk, and the
    stronger grade is the honest one.
    """
    return [
        max((g["grade"] for g in gold if chunk_matches(g, hit)), default=0)  # type: ignore[arg-type]
        for hit in hits
    ]


def evaluate(
    harness: Harness,
    gold: dict,
    retriever_names: Sequence[str],
    top_k: int = 5,
    threshold: float = 0.30,
    dump_dir: Path | None = RESULTS_DIR,
) -> list[Summary]:
    queries = gold["queries"]
    summaries = []

    for name in retriever_names:
        search = harness.retriever(name)
        scores: list[QueryScore] = []
        unanswerable: list[float] = []
        per_query = []
        elapsed_ms = []

        for query in queries:
            started = time.perf_counter()
            hits = search(query["query"], top_k)
            elapsed_ms.append((time.perf_counter() - started) * 1000)

            grades = grade_hits(hits, query["gold"])
            top_score = hits[0].score if hits else 0.0

            score = None
            if query["kind"] == "unanswerable":
                unanswerable.append(top_score)
            else:
                score = QueryScore.score(
                    query["id"],
                    query["kind"],
                    grades,
                    [g["grade"] for g in query["gold"]],
                    top_score,
                    top_k,
                )
                scores.append(score)

            per_query.append(
                {
                    "id": query["id"],
                    "query": query["query"],
                    "kind": query["kind"],
                    "grades": grades,
                    "top_score": top_score,
                    "hits": [
                        {"citation": h.citation, "symbol": h.qualified_name, "score": h.score}
                        for h in hits
                    ],
                    "score": asdict(score) if score else None,
                }
            )

        summary = summarize(
            name,
            scores,
            unanswerable,
            threshold,
            sum(elapsed_ms) / len(elapsed_ms) if elapsed_ms else 0.0,
        )
        summaries.append(summary)

        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            path = dump_dir / f"{harness.cfg.strategy}__{name}.json"
            path.write_text(
                json.dumps(
                    {"retriever": name, "summary": asdict(summary), "queries": per_query}, indent=1
                )
            )

    return summaries
