"""Cache and embedder, with a deterministic stand-in for the real model."""

import hashlib
import math

import pytest

from src.config import Config
from src.embeddings.cache import EmbeddingCache, cache_key
from src.embeddings.embedder import Embedder, unit

DIM = 8


def fake_encode(texts):
    """Deterministic, unnormalized, and *not* unit length, so `unit()` is exercised."""
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        out.append([float(digest[i]) for i in range(DIM)])
    return out


def test_cache_key_separates_model_and_text():
    assert cache_key("m", "text") != cache_key("mt", "ext")
    assert cache_key("m", "t") == cache_key("m", "t")


def test_cache_roundtrip_and_counters():
    with EmbeddingCache(None) as cache:
        assert cache.get_many("m", ["a", "b"]) == [None, None]
        assert (cache.hits, cache.misses) == (0, 2)

        cache.put_many("m", ["a"], [[0.5, -0.25]])
        assert cache.get_many("m", ["a", "b"]) == [[0.5, -0.25], None]
        assert (cache.hits, cache.misses) == (1, 3)

        # Same text, different model: a different entry.
        assert cache.get_many("other", ["a"]) == [None]
        assert cache.count() == 1
        assert cache.count("m") == 1
        assert cache.count("other") == 0


def test_cache_persists_across_connections(tmp_path):
    db = tmp_path / "nested" / "embeddings.sqlite3"
    with EmbeddingCache(db) as cache:
        cache.put_many("m", ["a"], [[1.0, 2.0, 3.0]])
    with EmbeddingCache(db) as reopened:
        assert reopened.get_many("m", ["a"]) == [[1.0, 2.0, 3.0]]
        assert reopened.clear("m") == 1
        assert reopened.count() == 0


def test_cache_handles_more_texts_than_sqlite_variable_limit():
    texts = [f"t{i}" for i in range(1200)]
    with EmbeddingCache(None) as cache:
        cache.put_many("m", texts, [[float(i)] for i in range(len(texts))])
        got = cache.get_many("m", texts)
    assert got == [[float(i)] for i in range(len(texts))]


def test_embedder_returns_unit_vectors():
    embedder = Embedder(Config(cache_db=None), cache=EmbeddingCache(None), encode=fake_encode)
    vector = embedder.embed("def retry(): ...")
    assert embedder.dim == DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_embedder_only_encodes_cache_misses():
    embedder = Embedder(Config(cache_db=None), cache=EmbeddingCache(None), encode=fake_encode)
    baseline = embedder.encoded_count  # dim probe already ran an encode

    first = embedder.embed_batch(["a", "b", "a"])
    # Three texts, two unique, so two forward passes.
    assert embedder.encoded_count - baseline == 2
    assert first[0] == first[2]

    again = embedder.embed_batch(["a", "b"])
    assert embedder.encoded_count - baseline == 2  # fully cached
    # Not bit-identical: the cache stores float32, while `fake_encode` computes
    # in float64. Real models emit float32 already, so there is no loss there.
    for cached, fresh in zip(again, first[:2], strict=True):
        assert cached == pytest.approx(fresh, rel=1e-6)


def test_embedder_can_bypass_the_cache():
    cache = EmbeddingCache(None)
    embedder = Embedder(Config(cache_db=None), cache=cache, encode=fake_encode)
    embedder.embed_batch(["a"], use_cache=False)
    assert cache.count() == 0


def test_unit_leaves_zero_vector_alone():
    assert unit([0.0, 0.0]) == [0.0, 0.0]
