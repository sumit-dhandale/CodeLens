"""Reranking, context expansion, and the two diversity passes."""

import pytest

from src.config import Config
from src.parser.repository_loader import SourceFile
from src.retrieval import context
from src.retrieval.reranker import CrossEncoderReranker
from src.vector_store.qdrant_store import Hit

PY = '''"""Client module."""

import httpx
from typing import Any


class Client:
    """Talks to the API."""

    def __init__(self, url):
        self.url = url

    def send_request(self, url):
        """Send one request."""
        return url

    def close(self):
        return None


def helper():
    return 1
'''


@pytest.fixture
def files():
    return [SourceFile(file_path="pkg/client.py", language="python", content=PY, sha="0" * 64)]


def hit(symbol="send_request", class_name="Client", score=0.5, vector=(), path="pkg/client.py"):
    return Hit(
        score=score,
        file_path=path,
        language="python",
        symbol_type="method" if class_name else "function",
        symbol_name=symbol,
        class_name=class_name,
        start_line=13,
        end_line=15,
        text="    def send_request(self, url):\n        return url",
        vector=vector,
    )


def test_expansion_adds_imports_class_signature_and_siblings(files):
    expanded = context.ContextExpander(files).expand(hit())

    assert expanded.imports == ("import httpx", "from typing import Any")
    assert expanded.class_signature.startswith("class Client:")
    assert '"""Talks to the API."""' in expanded.class_signature
    assert set(expanded.siblings) == {"__init__", "close"}
    assert "send_request" not in expanded.siblings  # the hit itself is not its own sibling

    text = expanded.text
    assert "import httpx" in text
    assert "other methods: " in text
    assert text.endswith(hit().text)  # the verbatim source stays last and intact
    # The parent's *body* must not be inlined: that is what we chunked away.
    assert "self.url = url" not in text


def test_expansion_of_a_top_level_function_has_no_class_parts(files):
    expanded = context.ContextExpander(files).expand(hit(symbol="helper", class_name=""))
    assert expanded.class_signature == ""
    assert expanded.siblings == ()
    assert expanded.imports  # imports still apply


def test_expansion_of_an_unknown_file_degrades_quietly(files):
    expanded = context.ContextExpander(files).expand(hit(path="gone.py"))
    assert expanded.text == hit().text


def test_mmr_drops_a_near_duplicate_for_a_more_distinct_chunk():
    # b is nearly identical to a; c is orthogonal and slightly less relevant.
    a = hit(symbol="a", score=0.90, vector=(1.0, 0.0))
    b = hit(symbol="b", score=0.89, vector=(0.99, 0.141))
    c = hit(symbol="c", score=0.60, vector=(0.0, 1.0))

    relevance_only = context.mmr([a, b, c], lambda_=1.0, top_k=2)
    assert [h.symbol_name for h in relevance_only] == ["a", "b"]

    balanced = context.mmr([a, b, c], lambda_=0.5, top_k=2)
    assert [h.symbol_name for h in balanced] == ["a", "c"]


def test_mmr_without_vectors_returns_the_input_order():
    hits = [hit(symbol="a", score=0.9), hit(symbol="b", score=0.8)]
    assert context.mmr(hits, 0.5, top_k=2) == hits
    with pytest.raises(ValueError, match="lambda"):
        context.mmr(hits, 1.5)


def test_per_file_cap_breaks_up_a_single_file_sweep():
    hits = [hit(symbol=f"m{i}", path="a.py") for i in range(4)] + [hit(symbol="z", path="b.py")]
    capped = context.per_file_cap(hits, cap=2, top_k=5)
    assert [h.file_path for h in capped] == ["a.py", "a.py", "b.py"]
    # cap=0 means unlimited, not "drop everything".
    assert len(context.per_file_cap(hits, cap=0, top_k=5)) == 5


def test_reranker_reorders_by_pair_score_and_keeps_top_k():
    # A stand-in cross-encoder: score by how many query words the text contains.
    def score_pairs(pairs):
        return [sum(w in text for w in query.split()) for query, text in pairs]

    reranker = CrossEncoderReranker(Config(), score_pairs=score_pairs)
    hits = [
        hit(symbol="a", score=0.9),
        Hit(0.8, "b.py", "python", "function", "b", "", 1, 2, "send url request"),
        Hit(0.7, "c.py", "python", "function", "c", "", 1, 2, "unrelated"),
    ]
    reranked = reranker.rerank("send url", hits, top_k=2)

    assert [h.file_path for h in reranked] == ["b.py", "pkg/client.py"]
    assert reranked[0].score == 2.0  # the cross-encoder's score replaces the cosine
    assert reranker.scored == 3  # every candidate costs a forward pass
    assert reranker.rerank("q", [], top_k=5) == []
