# 12 — Cross-encoder reranking, and the experiment that said no

## Bi-encoder vs cross-encoder

The embedder is a **bi-encoder**: it encodes the query and each chunk
*separately* and compares the two vectors. Because the chunk never sees the
query, its vector can be computed once at index time and reused forever. That
is the only reason searching 1465 chunks takes 2ms.

A **cross-encoder** concatenates query and chunk into a single input and runs
the transformer over the pair, so every query token can attend to every chunk
token. It is strictly more expressive — it can notice that "the *client* sends"
and "the *server* sends" are opposites, which two independently-written summaries
cannot. And it can precompute nothing: scoring N chunks is N forward passes at
query time.

| | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| Input | query and chunk, separately | the pair, jointly |
| Precomputable | yes, at index time | no |
| Cost for 1465 chunks | 1 forward pass + ANN | 1465 forward passes |
| Output | a reusable vector | one score, for this pair only |

Hence the two-stage shape: a cheap retriever narrows 1465 chunks to 20, and the
expensive model only ever sees 20. That is `vector_rerank` — `--rerank` on the
CLI, 20 candidates in about 60-250ms.

## The result

`cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20, cut to 5:

| corpus | variant | hit@1 | hit@5 | ndcg@5 | ms |
| --- | --- | --- | --- | --- | --- |
| OpsSense | vector top-5 | **0.46** | **0.75** | **0.56** | 1.9 |
| OpsSense | top-20 -> rerank 5 | 0.25 | 0.57 | 0.33 | 244 |
| httpx | vector top-5 | 0.31 | **0.69** | **0.48** | 1.9 |
| httpx | top-20 -> rerank 5 | 0.31 | 0.56 | 0.40 | 184 |

Reranking made retrieval **substantially worse on both corpora**, at 100x the
latency. On OpsSense it cost 23 points of nDCG; on httpx, 8. This is the result
the plan predicted directionally and understated in magnitude: the expectation
was "helps less than you'd hope", and the measurement is "actively harmful".

## Why — and the evidence, not the story

The easy explanation is domain mismatch: `ms-marco` was trained on English web
search, and Python is not English web search. That is a hypothesis, and the
gold set can test it, because its queries are tagged by kind. Splitting the
OpsSense run:

| query kind | n | vector hit@1 | reranked hit@1 | vector mrr | reranked mrr |
| --- | --- | --- | --- | --- | --- |
| code | 21 | **0.48** | 0.14 | **0.56** | 0.27 |
| lexical-gap (prose incidents) | 6 | 0.50 | **0.67** | 0.64 | **0.71** |

The cross-encoder **improves** ranking on the prose queries and collapses on the
code queries. That is the domain-mismatch hypothesis confirmed rather than
asserted: the model is good, at English. Given markdown it earns its latency;
given Python it reorders confidently and wrongly.

## The rescue attempt that half-worked

A second hypothesis: the bi-encoder sees *enriched* text (path, split
identifiers, docstring, then body — docs/04) while the cross-encoder was being
handed raw source. Maybe the model was fine and its input was starved.

| OpsSense variant | hit@1 | hit@5 | ndcg@5 |
| --- | --- | --- | --- |
| top-20 -> rerank 5 (raw source) | 0.25 | 0.57 | 0.33 |
| top-20 -> rerank 5 (enriched text) | 0.32 | 0.64 | 0.41 |
| no rerank | **0.46** | **0.75** | **0.56** |

Enrichment recovers roughly a third of the damage — so input starvation was
real, and it was not the main cause. Both hypotheses were partly right, and
neither makes reranking worth shipping here.

## What I would actually do

Not rerank, on this corpus, with this model. The honest options if reranking
matters:

- A code-trained cross-encoder. `ms-marco` is the default recommendation
  everywhere and the wrong default for code.
- Rerank only prose. The per-kind table says it works; routing by
  `language == "markdown"` is a payload filter away.
- Spend the latency budget on a better first stage instead. bge-small as the
  bi-encoder (Experiment 1) buys +7 points of hit@5 for 0ms of query time, which
  is strictly better than -18 points for 240ms.

That last line is the real lesson of this milestone. A two-stage architecture is
not automatically better than a one-stage one; it is a bet that stage two is
better than stage one at ordering *this* kind of text, and that bet is
measurable before you ship it.

## Verify

```bash
python -m src.cli search "where do we retry a failed request" --rerank
python -m src.cli eval --retrievers vector,vector_rerank
python -m scripts.experiments.exp7_rerank
pytest tests/test_context.py
```

## Alternatives considered

- **ColBERT / late interaction.** Sits between the two: token-level vectors,
  precomputable, much better than a bi-encoder at fine distinctions. The storage
  cost is one vector per token, which is a different project.
- **LLM reranking** (ask a model to order the 20). Better than a cross-encoder
  and far slower; Milestone 6 has the LLM plumbing, so it is a cheap follow-up.
- **Training a cross-encoder on this repo's queries.** 32 gold queries is not a
  training set.
