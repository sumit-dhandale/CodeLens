"""Sentence-transformer embedding with a cache in front of it.

Two invariants that the rest of the pipeline relies on:

- **Unit vectors.** `normalize_embeddings=True`, so cosine similarity equals the
  dot product and Qdrant's cosine distance needs no extra work.
- **One model at index and query time.** Two models produce two unrelated
  coordinate systems; the collection name embeds the model to make a mismatch
  impossible rather than merely unlikely.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from functools import cached_property

from ..config import Config
from .cache import EmbeddingCache, Vector

log = logging.getLogger(__name__)

EncodeFn = Callable[[list[str]], list[Vector]]


def unit(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


class Embedder:
    """`encode` is injectable so tests (and the eval harness) can run without torch."""

    def __init__(
        self, cfg: Config, cache: EmbeddingCache | None = None, encode: EncodeFn | None = None
    ):
        self.cfg = cfg
        self.model_name = cfg.model
        self.batch_size = cfg.batch_size
        self.cache = cache if cache is not None else EmbeddingCache(cfg.cache_db)
        self._custom_encode = encode
        self._encoded = 0

    @cached_property
    def _model(self):
        # Imported here because torch costs seconds to load, and `chunk`/`load`
        # never need it.
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        model = SentenceTransformer(self.model_name)
        log.info("loaded %s in %.1fs", self.model_name, time.perf_counter() - started)
        return model

    @cached_property
    def dim(self) -> int:
        if self._custom_encode is not None:
            return len(self._custom_encode(["dimension probe"])[0])
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def encoded_count(self) -> int:
        """Texts that actually went through a forward pass, i.e. cache misses."""
        return self._encoded

    def _encode(self, texts: list[str]) -> list[Vector]:
        self._encoded += len(texts)
        if self._custom_encode is not None:
            return [unit(v) for v in self._custom_encode(texts)]
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed(self, text: str, use_cache: bool = True) -> Vector:
        return self.embed_batch([text], use_cache=use_cache)[0]

    def embed_batch(self, texts: Sequence[str], use_cache: bool = True) -> list[Vector]:
        if not texts:
            return []
        if not use_cache:
            return self._encode(list(texts))

        out = self.cache.get_many(self.model_name, texts)
        # Dedup the misses: `file`-strategy chunks of two identical files, or a
        # repeated query in an eval sweep, would otherwise each pay a forward pass.
        missing = list(dict.fromkeys(t for t, v in zip(texts, out, strict=True) if v is None))
        if missing:
            fresh = self._encode(missing)
            self.cache.put_many(self.model_name, missing, fresh)
            by_text = dict(zip(missing, fresh, strict=True))
            out = [by_text[t] if v is None else v for t, v in zip(texts, out, strict=True)]
        return out  # type: ignore[return-value]

    def close(self) -> None:
        self.cache.close()
