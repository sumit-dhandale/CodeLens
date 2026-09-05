"""Metric arithmetic, checked against hand-computed values."""

import math

from src.eval.metrics import (
    QueryScore,
    dcg,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_hit_at_k_is_binary_and_respects_the_cutoff():
    grades = [0, 0, 2, 0, 1]
    assert hit_at_k(grades, 1) == 0.0
    assert hit_at_k(grades, 3) == 1.0
    assert hit_at_k([0, 0], 5) == 0.0


def test_recall_needs_the_total_not_just_the_retrieved():
    # Two of three known relevant items found in the top 5.
    assert recall_at_k([2, 0, 1, 0, 0], 5, total_relevant=3) == 2 / 3
    # One gold item, found: recall 1.0. Calling this 0.2 would be the classic
    # mistake of dividing by k.
    assert recall_at_k([2, 0, 0, 0, 0], 5, total_relevant=1) == 1.0
    assert recall_at_k([2], 5, total_relevant=0) == 0.0


def test_precision_divides_by_k():
    assert precision_at_k([2, 0, 1, 0, 0], 5) == 0.4
    assert precision_at_k([], 0) == 0.0


def test_reciprocal_rank_finds_the_first_relevant_hit():
    assert reciprocal_rank([0, 2, 1]) == 0.5
    assert reciprocal_rank([2]) == 1.0
    assert reciprocal_rank([0, 0]) == 0.0


def test_dcg_uses_exponential_gain_and_log_discount():
    # One grade-2 hit at rank 1: (2^2 - 1) / log2(2) = 3.
    assert dcg([2]) == 3.0
    # Same hit at rank 2: 3 / log2(3).
    assert dcg([0, 2]) == 3.0 / math.log2(3)
    # A primary is worth more than two secondaries at the same rank.
    assert dcg([2]) > dcg([1, 1])


def test_ndcg_penalizes_missed_gold_items():
    ideal = [2, 1]
    assert ndcg_at_k([2, 1], 5, ideal) == 1.0  # perfect
    # Found the primary only: the denominator still includes the missed secondary.
    partial = ndcg_at_k([2], 5, ideal)
    assert 0 < partial < 1.0
    # Right items, wrong order.
    assert ndcg_at_k([1, 2], 5, ideal) < 1.0
    assert ndcg_at_k([0, 0], 5, []) == 0.0


def test_query_score_records_ranks_of_relevant_hits():
    score = QueryScore.score("q1", "code", [0, 2, 0, 1], [2, 1], top_score=0.42)
    assert score.ranks == [2, 4]
    assert score.hit_at_1 == 0.0
    assert score.hit_at_5 == 1.0
    assert score.recall_at_5 == 1.0
    assert score.mrr == 0.5


def test_summarize_keeps_abstention_separate_from_ranking():
    scores = [
        QueryScore.score("q1", "code", [2], [2]),
        QueryScore.score("q2", "code", [0], [2]),
    ]
    summary = summarize("vector", scores, [0.5, 0.1], threshold=0.3, mean_ms=1.5)
    assert summary.queries == 2  # unanswerable queries are not averaged in
    assert summary.hit_at_1 == 0.5
    assert summary.unanswerable == 2
    assert summary.false_confidence == 0.5  # one of two exceeded the threshold
    assert "vector" in summary.row()
