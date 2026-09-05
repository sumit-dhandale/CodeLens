"""Ranking metrics over graded relevance.

Every function takes `grades`: the gold grade of each *retrieved* item, in rank
order, with 0 for "not labelled relevant". A retriever returning
`[irrelevant, primary, secondary]` gives `[0, 2, 1]`. Keeping that shape means
the metrics never need to know what a chunk is.

Naming is deliberate. With a single gold item, "Recall@5" and "Hit@5" are the
same number, and calling it recall makes a 20% score look like a retrieval
failure when it is arithmetic: 1 relevant item out of 1 found is recall 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def hit_at_k(grades: list[int], k: int) -> float:
    """1.0 if anything relevant appears in the top k. The blunt, honest metric."""
    return 1.0 if any(g > 0 for g in grades[:k]) else 0.0


def recall_at_k(grades: list[int], k: int, total_relevant: int) -> float:
    """Fraction of all known relevant items retrieved in the top k."""
    if total_relevant <= 0:
        return 0.0
    return sum(1 for g in grades[:k] if g > 0) / total_relevant


def precision_at_k(grades: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for g in grades[:k] if g > 0) / k


def reciprocal_rank(grades: list[int]) -> float:
    """1/rank of the first relevant hit. Rewards being right at rank 1."""
    for index, grade in enumerate(grades, 1):
        if grade > 0:
            return 1.0 / index
    return 0.0


def dcg(grades: list[int]) -> float:
    # Exponential gain, (2^g - 1), so a grade-2 primary is worth 3x a grade-1
    # secondary rather than 2x. That gap is the reason for grading at all.
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, 1))


def ndcg_at_k(grades: list[int], k: int, ideal_grades: list[int]) -> float:
    """DCG of the actual ranking over DCG of the best possible ranking.

    `ideal_grades` is every gold grade for the query, so the denominator reflects
    what a perfect retriever could have achieved — including gold items that this
    retriever missed entirely.
    """
    best = dcg(sorted(ideal_grades, reverse=True)[:k])
    return dcg(grades[:k]) / best if best else 0.0


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    kind: str
    hit_at_1: float
    hit_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    top_score: float
    ranks: list[int] = field(default_factory=list)  # 1-based ranks of relevant hits

    @classmethod
    def score(
        cls,
        query_id: str,
        kind: str,
        grades: list[int],
        ideal_grades: list[int],
        top_score: float = 0.0,
        k: int = 5,
    ) -> QueryScore:
        return cls(
            query_id=query_id,
            kind=kind,
            hit_at_1=hit_at_k(grades, 1),
            hit_at_5=hit_at_k(grades, k),
            recall_at_5=recall_at_k(grades, k, len(ideal_grades)),
            mrr=reciprocal_rank(grades),
            ndcg_at_5=ndcg_at_k(grades, k, ideal_grades),
            top_score=top_score,
            ranks=[i for i, g in enumerate(grades, 1) if g > 0],
        )


@dataclass(frozen=True)
class Summary:
    retriever: str
    queries: int
    hit_at_1: float
    hit_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    unanswerable: int
    false_confidence: float  # unanswerable queries answered above the threshold
    mean_ms: float

    HEADER = (
        f"{'retriever':16s} {'n':>3s} {'hit@1':>6s} {'hit@5':>6s} "
        f"{'rec@5':>6s} {'mrr':>6s} {'ndcg@5':>7s} {'falsecf':>8s} {'ms':>7s}"
    )

    def row(self) -> str:
        return (
            f"{self.retriever:16s} {self.queries:3d} {self.hit_at_1:6.2f} {self.hit_at_5:6.2f} "
            f"{self.recall_at_5:6.2f} {self.mrr:6.2f} {self.ndcg_at_5:7.2f} "
            f"{self.false_confidence:8.2f} {self.mean_ms:7.1f}"
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    retriever: str,
    scores: list[QueryScore],
    unanswerable_top_scores: list[float],
    threshold: float,
    mean_ms: float = 0.0,
) -> Summary:
    """Answerable queries drive the ranking metrics; unanswerable ones drive abstention.

    Averaging them together would let a retriever look better by returning
    nothing, and worse by returning something for a query with no right answer.
    """
    return Summary(
        retriever=retriever,
        queries=len(scores),
        hit_at_1=_mean([s.hit_at_1 for s in scores]),
        hit_at_5=_mean([s.hit_at_5 for s in scores]),
        recall_at_5=_mean([s.recall_at_5 for s in scores]),
        mrr=_mean([s.mrr for s in scores]),
        ndcg_at_5=_mean([s.ndcg_at_5 for s in scores]),
        unanswerable=len(unanswerable_top_scores),
        false_confidence=_mean([1.0 if s >= threshold else 0.0 for s in unanswerable_top_scores]),
        mean_ms=mean_ms,
    )
