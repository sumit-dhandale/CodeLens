"""Tokenizer, BM25, grep baseline, and both fusion methods."""

import re

import pytest

from src import cli
from src.chunking import code_chunker
from src.config import Config
from src.eval import runner
from src.parser.repository_loader import SourceFile
from src.retrieval import fusion, keyword_search
from src.vector_store.qdrant_store import Hit

PY = '''"""Client module."""


class Client:
    def sendRequest(self, url):
        """Send one request."""
        return url

    def _should_retry(self, resp):
        """Decide whether to try again."""
        return resp.status >= 500


def parse_timeout(value):
    """Read a deadline from config."""
    return int(value)
'''


@pytest.fixture
def chunks():
    src = SourceFile(file_path="pkg/client.py", language="python", content=PY, sha="0" * 64)
    return code_chunker.chunk_file(src, Config(), "function")


def hit(citation: str, score: float) -> Hit:
    path, span = citation.split(":")
    start, end = span.split("-")
    return Hit(
        score=score,
        file_path=path,
        language="python",
        symbol_type="function",
        symbol_name=path,
        class_name="",
        start_line=int(start),
        end_line=int(end),
        text="",
    )


def test_tokenizer_splits_both_cases_and_keeps_the_whole():
    assert keyword_search.tokenize("_shouldRetry") == ["should", "retry", "shouldretry"]
    assert keyword_search.tokenize("parse_timeout") == ["parse", "timeout", "parsetimeout"]
    # A single word yields no redundant joined form.
    assert keyword_search.tokenize("url") == ["url"]
    # Punctuation is dropped; digits survive.
    assert keyword_search.tokenize("resp.status >= 500") == ["resp", "status", "500"]


def test_bm25_matches_an_english_query_against_a_camelcase_name(chunks):
    index = keyword_search.BM25Index(chunks)
    hits = index.search("should retry", top_k=3)
    assert hits[0].symbol_name == "_should_retry"
    # Naming the symbol exactly must also work, which is the whole point of
    # emitting the joined form alongside the parts.
    assert index.search("sendRequest", top_k=1)[0].symbol_name == "sendRequest"


def test_bm25_returns_nothing_when_no_term_matches(chunks):
    assert keyword_search.BM25Index(chunks).search("kubernetes ingress certificate") == []
    assert keyword_search.BM25Index([]).search("anything") == []


def test_grep_scores_by_fraction_of_terms_present(chunks):
    hits = keyword_search.grep_search(chunks, "parse timeout", top_k=5)
    assert hits[0].symbol_name == "parse_timeout"
    assert hits[0].score == 1.0
    assert keyword_search.grep_search(chunks, "no such term here", top_k=5) == []


def test_minmax_handles_the_degenerate_cases():
    assert fusion.minmax([]) == []
    assert fusion.minmax([0.4, 0.4]) == [1.0, 1.0]
    assert fusion.minmax([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


def test_weighted_fusion_normalizes_before_blending():
    # BM25's raw scores are 20x the cosines. Without normalization the keyword
    # list would win outright at any sane weight.
    vector_hits = [hit("a.py:1-2", 0.50), hit("b.py:1-2", 0.40)]
    keyword_hits = [hit("b.py:1-2", 12.0), hit("c.py:1-2", 6.0)]

    fused = fusion.weighted_fusion(vector_hits, keyword_hits, weight=0.7, top_k=3)
    assert [h.citation for h in fused] == ["a.py:1-2", "b.py:1-2", "c.py:1-2"]
    # a.py: top of the vector list (1.0) but absent from keyword (0.0).
    assert fused[0].score == pytest.approx(0.7)

    with pytest.raises(ValueError, match="weight"):
        fusion.weighted_fusion(vector_hits, keyword_hits, weight=1.5)


def test_rrf_ignores_scores_and_rewards_agreement():
    # b.py is second in both lists; a.py is first in one and absent from the other.
    vector_hits = [hit("a.py:1-2", 0.99), hit("b.py:1-2", 0.10)]
    keyword_hits = [hit("c.py:1-2", 99.0), hit("b.py:1-2", 1.0)]

    fused = fusion.reciprocal_rank_fusion([vector_hits, keyword_hits], top_k=3)
    assert fused[0].citation == "b.py:1-2"
    assert fused[0].score == pytest.approx(2 / (fusion.RRF_K + 2))
    # Agreement at rank 2 beats a lone rank-1 hit, which is exactly the
    # behaviour that cost RRF the hit@1 column in docs/11.
    assert fused[0].score > fused[1].score


def test_fusion_is_deterministic_under_ties():
    tied = [hit("b.py:1-2", 0.5), hit("a.py:1-2", 0.5)]
    assert [h.citation for h in fusion.reciprocal_rank_fusion([tied], top_k=2)] == [
        "b.py:1-2",
        "a.py:1-2",
    ]
    same_rank = fusion.reciprocal_rank_fusion([[hit("z.py:1-2", 1.0)], [hit("a.py:1-2", 1.0)]])
    assert [h.citation for h in same_rank] == ["a.py:1-2", "z.py:1-2"]


def test_grade_hits_takes_the_strongest_matching_label(chunks):
    index = keyword_search.BM25Index(chunks)
    hits = index.search("should retry", top_k=2)
    gold = [
        {"file": "pkg/client.py", "symbol": "Client._should_retry", "grade": 2},
        {"file": "pkg/client.py", "symbol": None, "grade": 1},  # any chunk of the file
    ]
    grades = runner.grade_hits(hits, gold)
    assert grades[0] == 2  # both labels match; the stronger one wins
    assert all(g >= 1 for g in grades)


def test_random_retriever_is_seeded_per_query(chunks):
    harness = runner.Harness(cfg=Config(), chunks=chunks)
    first = harness.retriever("random")("same query", 2)
    second = harness.retriever("random")("same query", 2)
    other = harness.retriever("random")("different query", 2)
    assert [h.citation for h in first] == [h.citation for h in second]
    assert [h.citation for h in first] != [h.citation for h in other]


def test_unknown_retriever_is_rejected(chunks):
    with pytest.raises(ValueError, match="unknown retriever"):
        runner.Harness(cfg=Config(), chunks=chunks).retriever("magic")


def test_vector_retriever_without_a_store_fails_loudly(chunks):
    with pytest.raises(RuntimeError, match="store and an embedder"):
        runner.Harness(cfg=Config(), chunks=chunks).retriever("vector")("q", 5)


def test_a_misquoted_flag_is_rejected_before_any_search_happens():
    # argparse treats a dashed token containing a space as a positional, so
    # `search "q --language python"` would otherwise embed the flag as text and
    # return plausible nonsense.
    assert cli.main(["search", "how do we store vectors --language python"]) == 1


def test_hyphenated_words_are_not_mistaken_for_flags():
    assert re.search(cli.FLAG_IN_QUERY, "what is a well-known port") is None
    assert re.search(cli.FLAG_IN_QUERY, "why is -1 returned") is None
    assert re.search(cli.FLAG_IN_QUERY, "find --top-k handling").group(1) == "--top-k"
