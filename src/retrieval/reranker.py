"""Cross-encoder reranking: retrieve 20 cheaply, reorder 20 expensively, return 5.

A **bi-encoder** (the embedder) encodes the query and each chunk *separately*, so
chunk vectors can be computed once at index time and reused forever. The price is
that the two texts never meet: the model scores them by the distance between two
summaries it wrote without knowing about each other.

A **cross-encoder** concatenates query and chunk into one input and runs the
transformer over the pair, so every query token can attend to every chunk token.
Much more accurate, and it cannot precompute anything — scoring N chunks means N
forward passes at query time. Hence the two-stage shape: ANN over 85 vectors in
3ms to get 20 candidates, then 20 forward passes to order them.

Expect this to help less on code than the benchmarks promise. `ms-marco-MiniLM`
was trained on English web passages, and Python is not English web passages.
Measuring that honestly is the point; see docs/12.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from functools import cached_property

from ..config import Config
from ..vector_store.qdrant_store import Hit

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """`score_pairs` is injectable so tests can run without torch."""

    def __init__(self, cfg: Config, score_pairs=None):
        self.model_name = cfg.rerank_model
        self.batch_size = cfg.batch_size
        self._custom_score = score_pairs
        self.scored = 0

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        model = CrossEncoder(self.model_name)
        log.info("loaded %s in %.1fs", self.model_name, time.perf_counter() - started)
        return model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        self.scored += len(texts)
        pairs = [(query, t) for t in texts]
        if self._custom_score is not None:
            return [float(s) for s in self._custom_score(pairs)]
        return [float(s) for s in self._model.predict(pairs, batch_size=self.batch_size)]

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 5) -> list[Hit]:
        """Reorder by cross-encoder score, keeping the top_k.

        The returned scores are the cross-encoder's logits, not cosines: they are
        unbounded and can be negative. Nothing downstream may compare them to a
        retrieval score, which is why the eval table reports rerank rows
        separately rather than alongside raw cosine rows.
        """
        if not hits:
            return []
        scores = self.score(query, [h.text for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].citation))
        return [replace(hits[i], score=scores[i]) for i in order[:top_k]]
