# 17 — Where it fails, and why retrieval is probabilistic

Every table in this project reports a mean. A mean of 0.56 nDCG says nothing
about *which* queries produced it. This page is the per-query data: four
categories of failure, each with the query that produced it, and then the
argument that these are not bugs.

All numbers: OpsSense, function chunking, k=5, `nDCG@5` per query.
Reproduce with `python -m src.cli eval --retrievers bm25,vector,hybrid_rrf,grep`
and read `eval/results/function__*.json`.

Across 28 scoreable queries: vector wins 11 outright, BM25 wins 2, they tie or
trade on 10, and **5 queries defeat both**.

## 1. Where BM25 wins

| id | query | bm25 | vector |
| --- | --- | --- | --- |
| q15 | measure what fraction of the relevant documents we retrieved | **0.83** | 0.00 |
| q05 | read the incident markdown files off disk | **0.36** | 0.00 |

**q15** is the clean case. The gold answer is a function called `recall_at_k`,
and BM25 puts it at rank 1 with a score of 13.1 — more than twice the next
result. "Fraction of relevant documents retrieved" *is* the definition of
recall, so the query and the identifier share the discipline's jargon, and the
identifier tokenizer from docs/11 splits `recall_at_k` into `recall`, `at`,
`k`, `recallatk`. Exact lexical match, exactly what BM25 is for.

Vector search scored 0.00 on it and returned `run_chunk_eval` (0.432),
`chunk_document`, `index_documents` — the *evaluation machinery*, not the
metric. Embeddings map "measure what fraction we retrieved" into the
neighbourhood of evaluation scripts, which is topically right and functionally
useless.

**q05** exposes a bias that only shows up on a corpus with prose in it. The
gold answer is `load_documents` in `loader.py`. Vector search returned, in
order: `OpsSense/README.md:1-6` (0.583), `README.md:7-23` (0.444), `README.md`
Step 2 — Document loading (0.442), the root `README.md` (0.437). Four documents
*about* loading, ranked above the code that loads.

A natural-language query embeds closest to natural-language text. The prose
that describes a function will usually beat the function, because the prose was
written in the same register as the question. This is the single most
under-appreciated failure mode of embedding search over a mixed repo, and it
is why `--language python` exists.

## 2. Where vector wins

| id | query | bm25 | vector | gold |
| --- | --- | --- | --- | --- |
| q07 | clean up inconsistent service names before storing them | 0.00 | **1.00** | `normalize_service` |
| q09 | how are point ids generated so re-indexing does not duplicate rows | 0.00 | **1.00** | `_point_id` |
| q14 | run a nearest neighbour query and return the top hits | 0.00 | **1.00** | `search` |
| q19 | command line entry point for searching from the terminal | 0.00 | **1.00** | `main` in `search.py` |
| q26 | the service kept running out of RAM | 0.31 | **1.00** | two OOM incidents |

Every one of these is a synonym problem, and BM25 scores zero on four of five.

- q07: the query says "clean up inconsistent", the code says `normalize`.
  Zero shared terms. BM25's top hit was `qdrant_up`, a test fixture.
- q14: "nearest neighbour query" versus a function called `search`. BM25's top
  hit was `run_chunk_eval` at 11.5, which contains the word "query" more often
  than the function that runs one does.
- q19: "command line entry point" versus `def main()`. There is no lexical
  bridge from a description of a concept to the conventional name for it.
- q26: "running out of RAM" versus incidents that say "memory pressure" and
  "OOM kill". This is the lexical-gap category the markdown incidents were
  indexed for, and it is where embeddings earn the whole pipeline.

Note q09's absolute score: `_point_id` came back at **0.46**. That is a correct
rank-1 answer with a cosine of 0.46, which matters for the next section.

## 3. Semantically adjacent but wrong

The most dangerous category, because the results look plausible.

**q16 — "collapse several chunks of the same incident into one result"**. Gold
is `unique_incident_ids`. Vector search returned:

```
0.512  scripts/chunk_documents.py:14-26   main
0.497  src/ingestion/chunker.py:25-40     chunk_document
0.493  scripts/eval_chunking.py:51-58     main
0.476  src/ingestion/chunker.py:6-22      chunk_text
0.476  scripts/eval_chunking.py:24-48     run_chunk_eval
```

Five results, all about chunks, none about collapsing. The word "chunks" is
the loudest token in the query and it dominated the embedding; "collapse
several… into one", the operation actually being asked for, contributed almost
nothing. And the top score is **0.512** — higher than q09's correct answer at
0.46. A confidently wrong result outscored a correct one.

**q04 — "where is the chunk size and overlap configured"**. Gold is
`config.py`. Top hit: `test_chunk_size_and_overlap` at 0.515 — the *test* named
after the query, not the configuration it tests. A config file is a list of
constants with no prose and no behaviour; there is very little for an embedding
model to represent. Configuration is close to the worst case for semantic
search, and the fix is a metadata filter or a grep, not a better model.

**q12 — "how many points are currently stored"**. Gold is `collection_info`.
Vector returned `_point_id` (0.301) and `recall_at_k` (0.307) — the model
matched on "points" and "stored" separately, and `_point_id` is *about* points
in a way that has nothing to do with counting them.

## 4. Where everything fails

Five of 28 queries score 0.00 on both retrievers: **q04, q12, q16, q22, q25**.

**q22 — "how do I run the project end to end"**, top vector score **0.148**,
top hit `qdrant_up`, a pytest fixture. Compare against the score bands from
docs/14: 0.40–0.55 means a real answer, 0.25–0.40 means topically adjacent,
0.10–0.20 means nothing relevant exists. 0.148 is the index saying *I have
nothing*, as loudly as cosine similarity is able to.

**q25 — "one record was hammered far more than the rest"** should have matched
two hot-key incidents. Vector search returned evaluation and chunking code at
0.23–0.28; BM25 at least returned incident documents, but the wrong ones. The
gap from "hammered" to "hot key" and "partition skew" is one that a
general-purpose sentence embedding does not bridge.

Note that q04, q12, q16 and q25 all fail *while returning five confident-looking
results*. Nothing in the output says "I failed". Only the absolute score hints
at it, and section 5 explains why even that hint is unreliable.

## 5. Why retrieval is probabilistic

Not "buggy". Structurally probabilistic, for five compounding reasons.

**Embedding is lossy compression with a fixed budget.** An arbitrary function
becomes 384 floats. Information is necessarily discarded, and *which*
information is discarded was decided by the model's training objective, long
before your query existed. q16 shows the shape of that loss: the nouns survived
the compression and the verb did not.

**The similarity function answers a different question than you asked.**
Cosine similarity between a query and a chunk estimates "would these plausibly
occur in similar contexts". You wanted "does this chunk answer this question".
Those correlate well enough to build a product on and badly enough to produce
section 3. The reranker in docs/12 attacks exactly this gap, and on code it
made things *worse* — cross-encoders are trained on web passages, which is a
different distribution again.

**The score has no calibrated zero.** From docs/14's distribution: irrelevant
results have a 75th-percentile score of 0.430, and relevant results have a
median of 0.416. **The distributions overlap.** A correct answer at 0.46 (q09)
scores lower than a wrong one at 0.512 (q16). There is no threshold that
separates them, which is why `falsecf` is a reported column and not a solved
problem, and why "return nothing if the best score is below T" cannot be
implemented honestly here.

**Approximation is layered on top.** HNSW returns approximate neighbours
(docs/16). At this corpus size recall against exact search is 1.000, so it
contributes nothing to the failures above — but at 100k vectors with default
`ef_search` it would, and nothing in the response distinguishes "the index
didn't find it" from "it isn't there".

**Chunk boundaries decide what is retrievable at all.** A chunking strategy is
a claim about what the unit of an answer is. q04 fails partly because
`config.py`'s answer is a two-line constant assignment that function-level
chunking never produced as a chunk. No ranking improvement can retrieve
something that was never indexed as a unit.

On top of all that, the gold set is itself a set of judgments. Grade 1 —
"acceptable" — is one person's opinion, and every metric in this repo inherits
that.

### What follows from it

- **Report per-query deltas, not just means.** A 0.02 nDCG improvement across
  28 queries is noise; one query going 0.00 → 1.00 is a finding. Both look the
  same in a mean.
- **Hybrid is insurance, not an improvement.** `hybrid_rrf` scores 0.45 mean
  nDCG against vector's 0.56 — worse on average, because RRF weights a weak arm
  equally with a strong one (docs/11). But it is the only retriever that scored
  0.00 false-confidence, and it rescued q23 (0.99 versus vector's 0.57). Fusion
  buys tail coverage with mean quality.
- **An abstaining generator is the real safety net.** Since no score threshold
  works, the place to catch "nothing relevant was found" is the LLM, which sees
  the chunks rather than their scores. docs/15 measures this: with the project
  prompt, 3 of 3 unanswerable questions were refused; with a naive prompt, 0 of
  3.

## Verify

```bash
python -m src.cli eval --retrievers bm25,vector,hybrid_rrf,grep
python -m src.cli search "collapse several chunks of the same incident into one result"
python -m src.cli search "how do I run the project end to end"     # top score ~0.15
python -m src.cli search "measure what fraction of the relevant documents we retrieved"
```

## Alternatives considered

- **Fixing the failures by hand-tuning the gold set.** The five hard queries
  could be relabelled until the numbers improve. That is how benchmarks stop
  meaning anything.
- **A score threshold for abstention.** Measured and rejected above: the
  distributions overlap, so any threshold trades false silence for false
  confidence at a rate nobody would accept.
- **A code-trained embedding model.** Experiment 1 tried it. It won on
  OpsSense and lost on httpx — a conclusion that did not replicate, which is
  the entire reason the second corpus exists.
- **Query expansion with an LLM before retrieval.** Would plausibly fix q07 and
  q25 by generating "normalize", "hot key", "partition skew" as alternates. It
  also puts a generation call in front of every search, and it was not in the
  plan. This is the most promising unexplored direction in the project.
