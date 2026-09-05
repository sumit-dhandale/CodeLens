"""Indexing idempotency, orphan pruning, and search — against Qdrant's local mode.

`:memory:` runs qdrant-client's pure-python engine, so these tests need no
server and no torch.
"""

import pytest

from src.chunking import code_chunker
from src.config import Config
from src.embeddings.cache import EmbeddingCache
from src.embeddings.embedder import Embedder
from src.parser.repository_loader import SourceFile
from src.retrieval.vector_search import VectorSearcher
from src.vector_store import qdrant_store
from src.vector_store.qdrant_store import QdrantStore, collection_name, model_slug, repo_slug
from tests.test_embeddings import DIM, fake_encode

pytest.importorskip("qdrant_client")

BEFORE = '''"""Client module."""


class Client:
    """A client."""

    def send_request(self, url):
        """Send one request."""
        return url

    def _shouldRetry(self, resp):
        return resp.status >= 500


def helper():
    return 1
'''

# _shouldRetry deleted. Note this also shifts `helper` up by four lines.
AFTER = '''"""Client module."""


class Client:
    """A client."""

    def send_request(self, url):
        """Send one request."""
        return url


def helper():
    return 1
'''

# Only `helper`'s body changed, and no line moved.
EDITED_IN_PLACE = BEFORE.replace("    return 1\n", "    return 2\n")


def src(content: str) -> SourceFile:
    return SourceFile(file_path="pkg/client.py", language="python", content=content, sha="0" * 64)


def make(content: str, cfg: Config):
    chunks = code_chunker.chunk_file(src(content), cfg, "function")
    return chunks


@pytest.fixture
def cfg():
    return Config(qdrant_url=":memory:", cache_db=None, model="fake-model", top_k=5)


@pytest.fixture
def embedder(cfg):
    return Embedder(cfg, cache=EmbeddingCache(None), encode=fake_encode)


@pytest.fixture
def store(cfg):
    return QdrantStore(cfg, collection="test")


def test_collection_name_pins_repo_model_and_strategy():
    cfg = Config(repo="https://github.com/sumit-dhandale/OpsSense", strategy="function")
    assert collection_name(cfg) == "sumit-dhandale__OpsSense__all-MiniLM-L6-v2__function"
    assert collection_name(cfg, "file").endswith("__file")
    suffixed = collection_name(Config(collection_suffix="exp 3"))
    assert suffixed.endswith("__exp-3")
    assert model_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert repo_slug("/srv/checkouts/local-repo/") == "local-repo"


def test_reindexing_is_idempotent(cfg, store, embedder):
    chunks = make(BEFORE, cfg)
    first = store.sync(chunks, embedder.embed_batch, embedder.dim)
    assert first.upserted == len(chunks)
    assert first.points == len(chunks)

    encoded_after_first = embedder.encoded_count
    second = store.sync(chunks, embedder.embed_batch, embedder.dim)
    assert second.upserted == 0
    assert second.unchanged == len(chunks)
    assert second.points == first.points
    # Unchanged content hash means nothing is re-embedded at all.
    assert embedder.encoded_count == encoded_after_first


def test_in_place_edit_updates_one_point_and_deletes_nothing(cfg, store, embedder):
    store.sync(make(BEFORE, cfg), embedder.embed_batch, embedder.dim)
    report = store.sync(make(EDITED_IN_PLACE, cfg), embedder.embed_batch, embedder.dim)

    assert report.upserted == 1  # helper's body
    assert report.unchanged == report.chunks - 1
    assert report.deleted == 0
    assert report.points == report.chunks


def test_deleting_a_symbol_leaves_no_ghost_points(cfg, store, embedder):
    before = make(BEFORE, cfg)
    store.sync(before, embedder.embed_batch, embedder.dim)
    report = store.sync(make(AFTER, cfg), embedder.embed_batch, embedder.dim)

    assert report.chunks == len(before) - 1
    assert report.points == report.chunks  # no orphan survives
    # start_line is part of the point key, so removing _shouldRetry also churns
    # `helper`, which shifted up four lines. Correctness over churn: see docs/07.
    assert report.deleted == 2
    keys = {p.payload["point_key"] for p in store.client.scroll("test", limit=100)[0]}
    assert not any("shouldRetry" in k for k in keys)


def test_dimension_mismatch_is_refused(cfg, store, embedder):
    store.sync(make(BEFORE, cfg), embedder.embed_batch, embedder.dim)
    with pytest.raises(RuntimeError, match="dim"):
        store.ensure_collection(embedder.dim + 1)


def test_sync_infers_dim_from_the_vectors(cfg, store, embedder):
    assert store.vector_dim() is None
    report = store.sync(make(BEFORE, cfg), embedder.embed_batch)  # no dim argument
    assert report.dim == embedder.dim == DIM


def test_sync_of_an_empty_corpus_needs_a_dim(cfg, store, embedder):
    with pytest.raises(RuntimeError, match="does not exist"):
        store.sync([], embedder.embed_batch)


def test_search_returns_payload_and_respects_filters(cfg, store, embedder):
    chunks = make(BEFORE, cfg)
    store.sync(chunks, embedder.embed_batch, embedder.dim)

    target = next(c for c in chunks if c.symbol_name == "send_request")
    hits = store.search(embedder.embed(target.embed_text), top_k=3)
    assert hits[0].symbol_name == "send_request"
    assert hits[0].citation == target.citation
    assert hits[0].qualified_name == "Client.send_request"
    assert hits[0].score > 0.99  # a chunk is its own nearest neighbour
    assert hits[0].snippet(1) == target.text.splitlines()[0]

    only_md = store.search(
        embedder.embed("anything"),
        top_k=5,
        query_filter=qdrant_store.make_filter(language="markdown"),
    )
    assert only_md == []
    assert qdrant_store.make_filter() is None


def test_exact_search_agrees_with_ann_on_a_tiny_corpus(cfg, store, embedder):
    store.sync(make(BEFORE, cfg), embedder.embed_batch, embedder.dim)
    vector = embedder.embed("retry a failed request")
    ann = store.search(vector, top_k=3)
    exact = store.search(vector, top_k=3, exact=True)
    assert [h.citation for h in ann] == [h.citation for h in exact]


def test_searcher_rejects_missing_collection_and_empty_query(cfg, embedder):
    searcher = VectorSearcher(
        cfg, store=QdrantStore(cfg, collection="never-created"), embedder=embedder
    )
    with pytest.raises(ValueError, match="empty query"):
        searcher.search("  ")
    with pytest.raises(RuntimeError, match="does not exist"):
        searcher.search("anything")


def test_searcher_reports_timings(cfg, store, embedder):
    store.sync(make(BEFORE, cfg), embedder.embed_batch, embedder.dim)
    result = VectorSearcher(cfg, store=store, embedder=embedder).search("send a request", top_k=2)
    assert len(result.hits) == 2
    assert result.total_ms >= 0
    assert result.exact is False
