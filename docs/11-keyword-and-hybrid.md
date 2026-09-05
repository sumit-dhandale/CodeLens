# 11 — BM25, fusion, and the result that contradicted the plan

## The table

`function` strategy, 85 chunks, k=5, 28 answerable queries, 4 unanswerable:

| retriever | hit@1 | hit@5 | rec@5 | mrr | ndcg@5 | falsecf | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.04 | 0.07 | 0.03 | 0.04 | 0.01 | 0.00 | 0.0 |
| grep | 0.07 | 0.32 | 0.24 | 0.14 | 0.14 | 0.75 | 0.3 |
| bm25 | 0.21 | 0.54 | 0.34 | 0.32 | 0.28 | 1.00 | 0.1 |
| **vector** | **0.46** | 0.75 | **0.65** | 0.55 | **0.56** | 0.25 | 1.9 |
| hybrid_weighted (w=0.7) | **0.46** | 0.75 | 0.63 | **0.58** | 0.54 | 1.00 | 2.2 |
| hybrid_rrf | 0.32 | 0.75 | 0.57 | 0.48 | 0.45 | 0.00 | 2.3 |

Read the baselines first, because they are the only rows that could have
invalidated the project:

- **random** at Hit@5 0.07 is the floor. Any retriever near it is broken.
- **grep** at 0.32 is what plain substring matching buys. Embeddings had to beat
  this or the 22 seconds of indexing and the 200 KB of vectors were theatre.
- **vector** at 0.75 is 2.3x grep and 1.4x BM25. That is the result that
  justifies everything in Milestones 1–3.

## The tokenizer is the whole story of BM25 on code

```python
tokenize("_shouldRetry")  # ['should', 'retry', 'shouldretry']
```

Split on snake_case and camelCase, lowercase, **and keep the joined form**. Both
halves matter:

- Without splitting, "should retry" matches nothing — `_shouldRetry` is one
  opaque token, and BM25 has no notion of similarity, only of term identity.
- Without the joined form, someone who types the symbol name exactly gets the
  same score as someone who typed two common words, losing the one query type
  BM25 should dominate.

Punctuation is dropped entirely. It carries no signal in either a query or a
body, and indexing it dilutes every IDF in the corpus. BM25 itself is
`rank_bm25` — a term-frequency formula with length normalization, worth
understanding, not worth reimplementing. The tokenizer was the lesson.

BM25 indexes `embed_text`, the same string the vector side embeds. Feeding both
retrievers identical input is what makes this a comparison of *scoring methods*
rather than of preprocessing.

## Where each one wins

From the per-query dumps, the ten queries where they disagree:

**BM25 wins** when the query happens to use the code's vocabulary:

```
q15  "measure what fraction of the relevant documents we retrieved"  ->  recall_at_k
q05  "read the incident markdown files off disk"                     ->  load_documents
```

`fraction`, `relevant`, `retrieved` are literally in the docstring. Cosine
similarity has no reason to prefer that chunk over five other metrics functions;
term overlap does.

**Vector wins** on the other eight, all of them paraphrases:

```
q11  "connect to the qdrant server"                     ->  get_client
q09  "how are point ids generated so re-indexing ..."   ->  _point_id
q28  "slowness caused by the network path rather ..."   ->  an incident about pool exhaustion
```

`get_client` shares no token with "connect to the qdrant server" beyond
"qdrant". This is the lexical gap, and closing it is the entire argument for
embeddings.

## Fusion: two implementations, and neither one won

Raw scores cannot be added: cosine sits in 0.1-0.5 here, BM25 is unbounded.
`w * cosine + (1-w) * bm25` on raw values is a coin flip decided by scale.

**Weighted fusion** min-max normalizes each list per query, then blends. The
weight sweep:

| w (vector share) | hit@1 | hit@5 | mrr | ndcg@5 |
| --- | --- | --- | --- | --- |
| 0.3 | 0.25 | 0.64 | 0.40 | 0.37 |
| 0.5 | 0.43 | **0.79** | 0.55 | 0.52 |
| 0.7 | 0.46 | 0.75 | **0.58** | 0.54 |
| 0.9 | 0.43 | 0.71 | 0.56 | 0.55 |
| 1.0 | 0.46 | 0.75 | 0.55 | **0.56** |

w=1.0 reproduces the `vector` row exactly, which is a free correctness check on
the implementation. w=0.5 gives the best Hit@5 in the whole project (0.79, one
query better than vector alone) while being *worse* than vector on nDCG. w=0.3
is much worse, because BM25 is the weaker retriever here and weighting it
heavily just adds noise.

**RRF** ignores scores and sums `1/(60 + rank)` across lists. It is the
scale-free, tuning-free option, and the plan predicted it would win.

**It did not.** Hit@1 0.32 against vector's 0.46, nDCG@5 0.45 against 0.56. The
reason is visible in the per-query dumps: RRF weights both lists equally by
construction, and on this corpus BM25 is *much* worse than vector (0.28 nDCG vs
0.56). A rank-1 vector hit gets 1/61; a rank-1 BM25 hit gets the same 1/61.
Fusing a strong retriever with a weak one at equal weight moves the strong one's
correct answers down. RRF wins when the arms are of comparable quality — which
is the condition nobody states when recommending it.

## What I would actually ship

`vector` alone, or `hybrid_weighted` at w=0.5-0.7 if Hit@5 is what matters more
than ordering. And I would keep the raw cosine around for abstention, because
both fusion methods destroy score calibration (docs/10).

With the caveat that on 28 queries these gaps are one or two queries wide.
Milestone 5 re-runs this against `encode/httpx` before any of it is called a
conclusion.

## Verify

```bash
python -m src.cli eval
python -m src.cli eval --retrievers hybrid_weighted --vector-weight 0.5
pytest tests/test_retrieval.py
```

## Alternatives considered

- **Qdrant's built-in full-text index / sparse vectors.** Qdrant can do BM25-ish
  matching server-side, and native sparse vectors would fuse inside the database.
  Both are the right production answer and both hide the tokenizer, which was the
  thing worth learning here.
- **Learned sparse retrieval** (SPLADE). Better than BM25 on paper, and it needs
  a model at index time — a Milestone 5 experiment, not a baseline.
- **Tuning `k1` and `b`.** `rank_bm25`'s defaults are 1.5 and 0.75. Tuning them
  on 28 queries would fit the noise, not the corpus.
- **Fusing three lists** (grep as well). grep is a subset of BM25's signal with
  worse ranking; adding it makes RRF's equal-weight problem worse.
