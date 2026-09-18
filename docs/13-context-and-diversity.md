# 13 — Context expansion and stopping the list repeating itself

Two problems that only appear once retrieval works.

## 1. A precise chunk is a decontextualized chunk

Function-level chunking wins on every metric (Experiment 2) precisely because a
chunk is small. The cost lands later: the thing you retrieved no longer says
what surrounds it. `--expand` puts back the minimum needed to read it:

```
$ python -m src.cli search "what is the embedding dimension" --symbol-type method --expand --top-k 1
1. 0.5872  OpsSense/src/embeddings/embedder.py:20-22  method Embedder.dim
      from sentence_transformers import SentenceTransformer
      from src.config import EMBEDDING_MODEL, VECTOR_SIZE

      class Embedder:
          # other methods: __init__, embed, embed_batch

          @property
          def dim(self) -> int:
              return VECTOR_SIZE
```

Three additions, each chosen for what it tells a reader — or an LLM — that the
chunk cannot:

- **Module imports.** `VECTOR_SIZE` is imported from `src.config`, so the answer
  to "what is the dimension" is one hop away and now visible.
- **The parent class signature and its docstring.** `dim` is a property of
  `Embedder`, not a free function.
- **Sibling method names, not bodies.** `__init__, embed, embed_batch` tells you
  what else the class does for about ten tokens. Inlining their bodies would
  defeat the purpose: we chunked the class apart because it does not fit.

`_signature` deliberately takes the class line and its docstring *only*. The
constraint is explicit in the code, because "expand the context" is exactly the
kind of feature that grows until every hit drags in its whole file.

### Why this is not in the metrics table

Expansion changes the text attached to a hit, not which hits come back, so
Hit@5 and nDCG cannot move by construction. It is a RAG-input feature and
Milestone 6's answer quality is where it gets measured. Reporting an unchanged
metric as evidence would be theatre.

## 2. Five hits, one class, one answer

The predicted failure of function-level chunking, reproduced exactly:

```
$ python -m src.cli search "how are embeddings created" --language python
1. 0.4396  src/embeddings/embedder.py:7-9    Embedder.__init__
2. 0.4180  src/embeddings/embedder.py:14-18  Embedder.embed_batch
3. 0.4060  src/embeddings/embedder.py:20-22  Embedder.dim
4. 0.4021  src/embeddings/embedder.py:11-12  Embedder.embed
5. 0.3065  src/embeddings/__init__.py:1-1    __init__.py
```

Four of five results are methods of one class, scoring within 0.04 of each
other. The top-5 spends four slots restating one answer.

### MMR

Maximal Marginal Relevance picks greedily by *marginal* value:

```
lambda * relevance - (1 - lambda) * max_similarity_to_already_selected
```

At `lambda=0.7` on the same query:

```
1. 0.4396  src/embeddings/embedder.py:7-9    Embedder.__init__
2. 0.4060  src/embeddings/embedder.py:20-22  Embedder.dim
3. 0.4180  src/embeddings/embedder.py:14-18  Embedder.embed_batch
4. 0.3065  src/embeddings/__init__.py:1-1    __init__.py
5. 0.2719  tests/test_embedder.py:11-22      test_semantic_vs_keyword
```

`Embedder.embed` is gone — it is nearly a duplicate of `embed_batch` — and a
test that demonstrates usage takes the slot. Note ranks 2 and 3 are now out of
score order: that is MMR working, not a sorting bug.

This needs candidate-to-candidate similarity, which a ranked list does not
carry, so `Hit` gained an optional `vector` and the store an optional
`with_vectors=True`. Without vectors `mmr` returns the input order rather than
pretending to diversify.

On the gold set, at k=5:

| retriever | hit@1 | hit@5 | rec@5 | mrr | ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| vector | 0.46 | 0.75 | **0.65** | 0.55 | **0.56** |
| vector_mmr (lambda=0.7) | 0.46 | **0.79** | 0.62 | **0.58** | 0.53 |

Hit@5 and MRR up, Recall@5 and nDCG down — and that is the honest shape of a
diversity trade, not a mixed result to explain away. Dropping a near-duplicate
sometimes drops a *labelled* near-duplicate (recall falls), while the freed slot
sometimes surfaces the one answer a single-file sweep was hiding (hit rises). If
the consumer is a human scanning five results, or an LLM that needs coverage
rather than confirmation, that is the right trade. If you are counting labelled
items, it is not.

### The per-file cap

`per_file_cap` is the zero-cost approximation: keep at most N hits per file, no
vectors, no tuning. On this corpus at cap=2 it changes **nothing** — identical
metrics to plain vector — because the top 5 rarely contains three chunks from
one file even when it contains four. The demo above is a 4-of-5 case that cap=2
would have fixed, and the gold set's queries mostly are not. Keeping a measured
null result is more useful than quietly deleting the feature: it is 12 lines,
and on a monorepo where one file legitimately dominates, it is the cheaper
half of this page.

## Verify

```bash
python -m src.cli search "how are embeddings created" --language python --mmr
python -m src.cli search "what is the embedding dimension" --symbol-type method --expand
python -m src.cli eval --retrievers vector,vector_mmr,vector_cap
pytest tests/test_context.py
```

## Alternatives considered

- **Expanding to the whole parent class.** Simpler, and it reintroduces the
  token blowup that made file-level chunking score 0.21.
- **Clustering the candidates, then taking one per cluster.** More principled
  than greedy MMR, needs a cluster count nobody can pick for 20 items.
- **Deduplicating by content hash.** Catches only exact duplicates. The problem
  here is near-duplicates, which is why similarity, not equality, is the test.
- **Expanding at index time** (embedding the class context into every method).
  That is chunk enrichment, already done in `embed_text` — and doing it again at
  display time would be double-counting.
