"""BM25 over code, where the tokenizer does the work.

Off-the-shelf tokenizers split on whitespace and punctuation, which turns
`_shouldRetry` into one token that matches nothing a human would type. Splitting
it into `should` and `retry` — *while keeping the original* — is the entire
difference between BM25 being useless on code and BM25 beating embeddings on
exact-name queries.

BM25 itself is `rank_bm25`: a term-frequency formula with document-length
normalization, worth understanding but not worth reimplementing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..chunking.code_chunker import Chunk, split_identifier
from ..vector_store.qdrant_store import Hit

# Identifiers, numbers, and nothing else. Punctuation carries no retrieval signal
# in either a query or a body, and indexing it would dilute every IDF.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """`self._shouldRetry(resp)` -> ['self', 'should', 'retry', 'shouldretry', 'resp'].

    Both forms are emitted: the split parts so English queries match, and the
    joined original so a query naming the symbol exactly still scores highest —
    it matches every part *and* the whole.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        parts = [p.lower() for p in split_identifier(raw)]
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append("".join(parts))
    return tokens


@dataclass
class BM25Index:
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        from rank_bm25 import BM25Okapi

        # Indexing `embed_text`, the same string the vector side embeds. Feeding
        # both retrievers identical input is what makes the comparison in
        # docs/11 a comparison of scoring methods rather than of preprocessing.
        corpus = [tokenize(c.embed_text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # A zero score means not one query term appears; returning those would
        # inflate hit@k with chunks BM25 has no opinion about.
        return [
            Hit.from_chunk(self.chunks[i], float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] > 0
        ]


def grep_search(chunks: list[Chunk], query: str, top_k: int = 5) -> list[Hit]:
    """The baseline that matters: plain case-insensitive substring counting.

    If embeddings cannot beat this on the gold set, the embeddings are not
    earning their 22 seconds and 200 KB.
    """
    terms = [t for t in tokenize(query) if len(t) > 2]
    if not terms:
        return []
    scored = []
    for chunk in chunks:
        haystack = chunk.embed_text.lower()
        matches = sum(1 for t in terms if t in haystack)
        if matches:
            scored.append((matches / len(terms), chunk))
    scored.sort(key=lambda pair: (-pair[0], pair[1].citation))
    return [Hit.from_chunk(c, score) for score, c in scored[:top_k]]
