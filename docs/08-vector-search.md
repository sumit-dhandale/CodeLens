# 08 — Vector search: HNSW, exact search, and reading the scores

## The query path

Embed the query with the same model, ANN search the collection, return payload.
The query gets **no enrichment** — no path prefix, no identifier splitting. It is
already English; the asymmetry is the point. Enrichment exists to drag code
*towards* English, not the reverse.

Query vectors go through the same cache, because an eval sweep runs the same 32
queries against every strategy and every model. Cached, the whole search is 3ms;
cold, it is 6.9 seconds, essentially all of it importing torch.

## What HNSW is doing

Exhaustive search compares the query to all N vectors: exact, and O(N). HNSW
builds a multi-layer graph where each node links to its approximate neighbours,
sparse at the top and dense at the bottom. A search enters at the top, greedily
walks towards the query, drops a layer, repeats. Roughly O(log N) comparisons.

It is *approximate*: the greedy walk can end in a local minimum and miss a true
nearest neighbour. Three knobs, swept in Milestone 6:

| Knob | Controls | Trade-off |
| --- | --- | --- |
| `m` | edges per node | recall and memory up, build time up |
| `ef_construct` | candidate list while building | index quality up, build time up |
| `ef_search` | candidate list while querying | recall up, latency up — tunable per query |

`--exact` sets Qdrant's `exact=True`, which brute-forces every vector and is
therefore ground truth. On 85 vectors ANN and exact agree completely (11ms exact
vs 3ms ANN — at this size the graph is pure overhead). The flag exists so
Milestone 6 can measure real recall at 100k vectors, where the two diverge and
the gap is the whole point of tuning.

## Reading the scores

Real output at `da938e09`, `function` strategy, top 5:

```
$ python -m src.cli search "how are chunks split for embedding" --language python --top-k 3
1. 0.4860  OpsSense/src/ingestion/chunker.py:43-49  function chunk_documents
2. 0.4752  OpsSense/tests/test_chunker.py:5-9       function test_chunk_size_and_overlap
3. 0.4517  OpsSense/src/ingestion/chunker.py:6-22   function chunk_text
```

Correct, and note the ceiling: **0.49, not 0.95**. A chunk compared against
itself scores 1.0; anything else is a genuine semantic gap between an English
question and Python source. Calibrate expectations to the corpus, not to the
demos in vector-database marketing.

Three score bands from this run, which is the most useful thing in this doc:

| Band | Example | Meaning |
| --- | --- | --- |
| 0.40 – 0.55 | "how are chunks split" → `chunk_documents` | a real answer |
| 0.25 – 0.40 | "retry a failed request" → incident write-ups | topically adjacent |
| 0.10 – 0.20 | "retry a failed request" `--language python` | nothing relevant exists |

The last row is the honest one. OpsSense has no retry logic, so the top hit is
`qdrant_up()` at 0.144 — a function about connections, returned because *something*
had to be. The engine cannot say "no". Every result is the nearest of what
exists, and nearest is not the same as relevant. That is why Milestone 4 measures
against a gold set instead of eyeballing output, why the gold set has
unanswerable queries, and why Experiment 4 sweeps a similarity threshold.

## Why markdown wins the vague queries

Unfiltered, "where do we retry a failed request" returns ten incident write-ups
at 0.24–0.38 — the entire top 10, with not one line of Python. Prose about timeouts and
retries embeds much closer to an English question than a function whose signature
is `def send(self, url)`. Both behaviours are correct and both are useful:
`--language python` when you want code, no filter when you want context. The
filter is a one-word answer to a problem that looks like it needs a better model.

## Verify

```bash
python -m src.cli search "how are chunks split for embedding" --language python
python -m src.cli search "how are chunks split for embedding" --exact
python -m src.cli search "where do we retry a failed request" --top-k 10
```

## Alternatives considered

- **numpy brute force, no vector DB.** For 85 chunks it is faster and simpler.
  It gives up filtered search, persistence, payloads, quantization, and the
  recall-vs-latency curve that half this project exists to measure.
- **IVF / product quantization** (FAISS). Better memory profile at millions of
  vectors, worse recall at these sizes, and Qdrant's scalar quantization covers
  the same trade-off with one config flag (Milestone 6).
- **Asymmetric query encoding** (an instruction prefix, as `bge` and `e5` want).
  Real technique, and Experiment 1 is the place to measure it — it needs the
  model comparison harness to say anything meaningful.
