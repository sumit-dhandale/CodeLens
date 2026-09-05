"""Two ways to combine ranked lists, and they disagree about what a score is.

Cosine lives in roughly 0.1-0.5 on this corpus; BM25 is unbounded and depends on
corpus statistics. `w * cosine + (1-w) * bm25` on raw values is therefore
meaningless — whichever list happens to have larger numbers wins every time.

- `weighted_fusion` normalizes each list to 0-1 *per query* first, then blends.
  Keeps score magnitude as information, at the cost of a weight to tune and a
  normalization that a single outlier can distort.
- `reciprocal_rank_fusion` throws the scores away and uses only positions.
  Nothing to tune, immune to scale, and it usually wins — which is the most
  useful surprise in this milestone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ..vector_store.qdrant_store import Hit

# The constant from Cormack, Clarke & Buettcher (2009). Its job is to flatten
# the difference between rank 1 and rank 2 (1/61 vs 1/62) so a single list
# cannot dominate on the strength of one confident guess.
RRF_K = 60


def minmax(values: Sequence[float]) -> list[float]:
    """Scale to 0-1 per query. All-equal input maps to 1.0, not to 0/0."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _by_citation(hits: Sequence[Hit]) -> dict[str, Hit]:
    return {h.citation: h for h in hits}


def _normalized(hits: Sequence[Hit]) -> dict[str, float]:
    return dict(zip([h.citation for h in hits], minmax([h.score for h in hits]), strict=True))


def weighted_fusion(
    vector_hits: Sequence[Hit],
    keyword_hits: Sequence[Hit],
    weight: float = 0.7,
    top_k: int = 5,
) -> list[Hit]:
    """`weight` is the vector list's share. A chunk missing from a list scores 0 there."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    vector_scores = _normalized(vector_hits)
    keyword_scores = _normalized(keyword_hits)
    lookup = {**_by_citation(keyword_hits), **_by_citation(vector_hits)}

    fused = {
        citation: weight * vector_scores.get(citation, 0.0)
        + (1 - weight) * keyword_scores.get(citation, 0.0)
        for citation in lookup
    }
    return _rank(fused, lookup, top_k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], k: int = RRF_K, top_k: int = 5
) -> list[Hit]:
    """Sum of 1/(k + rank) across lists. Scores are ignored entirely."""
    fused: dict[str, float] = {}
    lookup: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            fused[hit.citation] = fused.get(hit.citation, 0.0) + 1.0 / (k + rank)
            lookup.setdefault(hit.citation, hit)
    return _rank(fused, lookup, top_k)


def _rank(fused: dict[str, float], lookup: dict[str, Hit], top_k: int) -> list[Hit]:
    # Tie-break on citation so the output is deterministic; two chunks with equal
    # fused scores would otherwise order by dict insertion, which depends on
    # which retriever ran first.
    order = sorted(fused, key=lambda c: (-fused[c], c))
    return [dataclasses.replace(lookup[c], score=fused[c]) for c in order[:top_k]]
