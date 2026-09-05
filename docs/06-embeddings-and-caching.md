# 06 — Embeddings, and the cache that makes experiments affordable

## What an embedding actually is

`all-MiniLM-L6-v2` is a 6-layer transformer that reads up to 256 wordpieces and
emits one 384-dimensional vector. The vector is not a summary you can read; it is
a position in a space that was *trained* so that texts humans call similar land
near each other. Nothing about the number 384 is meaningful on its own — it is a
capacity choice. Bigger models (768, 1024) separate finer distinctions and cost
proportionally more to compute and store.

Two properties are worth internalising because they explain most surprises:

- **The space is model-specific.** A vector from MiniLM and a vector from
  `bge-small` are both 384 numbers and are completely incomparable. This is why
  the collection name embeds the model (docs/07) instead of trusting the operator
  to remember.
- **Similarity is relative, not absolute.** A cosine of 0.45 does not mean "45%
  relevant". It means "closer than the 0.28 below it". Score distributions shift
  per model, per query phrasing, and per corpus, which is why Experiment 4 sweeps
  the threshold rather than hardcoding one.

## Unit vectors, so cosine *is* the dot product

`Embedder` passes `normalize_embeddings=True`, so every stored vector has length
1. Then:

```
cos(a, b) = (a · b) / (|a| |b|) = a · b
```

Cosine, dot product, and Euclidean distance all rank identically on unit vectors
(`|a - b|² = 2 - 2(a·b)`), so the choice of metric stops mattering — as long as
nothing un-normalized ever gets in. `unit()` enforces this for injected encoders,
and a zero vector is left alone rather than dividing by zero.

| Metric | Ranking on unit vectors | Use when |
| --- | --- | --- |
| Cosine | identical | vectors may not be normalized; the safe default |
| Dot | identical | vectors are normalized *and* you want the cheapest kernel |
| Euclidean | identical (inverted) | magnitude carries meaning, e.g. raw counts |

## The cache is the most valuable 100 lines in the project

Milestone 5 runs four chunking strategies against several models. Without a
cache, every re-run of a pipeline stage re-pays for every forward pass. With one,
only genuinely new text costs anything.

The key is `sha256(model + "\0" + text)`. Three details:

- **The model is in the key.** Same text, different model, different entry.
- **The separator matters.** Without the `\0`, `("ab", "c")` and `("a", "bc")`
  would hash identically — a real collision, not a theoretical one, once model
  names and code text are concatenated freely.
- **The text is the key, not the chunk id.** So a function that moves between
  files, or is reached under a different chunking strategy, still hits.

Vectors are stored as raw float32 (`array('f').tobytes()`): 1536 bytes for 384
dimensions, against roughly 7 KB for a JSON list of floats, and no parsing on
read. The dimension is implied by the blob length, so no schema migration is
needed to switch models. Measured: 88 vectors in a 200 KB sqlite file.

One consequence worth knowing: a cached vector is float32, so it will not be
*bit-identical* to a float64 computation of the same text. Real models emit
float32 already, so nothing is lost in practice — but the test asserts
approximate equality, not exact.

`get_many` batches lookups 500 keys at a time, because SQLite's
`SQLITE_MAX_VARIABLE_NUMBER` is 999 on older builds and a single `IN (...)` over
10 000 chunks would fail at runtime rather than at review time. `WAL` plus
`synchronous=NORMAL` trades durability for throughput, which is the right trade
for data that is rebuildable by definition.

## Deduplication before the forward pass

`embed_batch` looks up everything, then dedups the misses with `dict.fromkeys`
before encoding. Duplicate text is common: two identical `__init__.py` files
under the `file` strategy, or the same 32 eval queries re-run against every
strategy in one process.

## Laziness that shows up on the clock

`sentence_transformers` imports torch, which costs about 7 seconds. So the model
lives behind a `cached_property` and is only touched by an actual forward pass —
`load`, `parse`, and `chunk` never pay it, and neither does a fully cached index
run or a repeated query. Measured on OpsSense's 85 function-level chunks:

| Run | Forward passes | Wall time |
| --- | --- | --- |
| First index (cold cache, cold model) | 85 | 22.3s |
| Re-index, nothing changed | 0 | 0.5s |
| `--drop` then full rebuild, warm cache | 0 | **0.2s** |
| Index the `class` strategy afterwards | 1 of 82 | 0.2s |
| Query, cold | 1 | 6.9s (7s of it is torch) |
| Query, cached | 0 | 3ms |

The `class` row is the cache paying off in a way that was not designed for:
`embed_text` does not include the strategy name, so a top-level function chunked
under `class` produces byte-identical text to the same function chunked under
`function`. 81 of 82 vectors came back for free.

The rebuild row is worth pausing on. Dropping the collection and re-indexing 85
chunks costs 0.2 seconds and does not import torch at all, because `sync` takes
the vector dimension from the vectors it just fetched from the cache rather than
asking the model for it. An eval sweep re-indexes constantly; this is the
difference between a 3-second sweep and a 3-minute one.

## Verify

```bash
python -m src.cli index --strategy function      # watch the cache counters
python -m src.cli index --strategy function      # embedded: 0
pytest tests/test_embeddings.py
```

## Alternatives considered

- **No cache, just re-embed.** Fine for one pass over 85 chunks. The project is
  a matrix of models × strategies; this is where the time goes.
- **Cache in a pickle or npz file.** `pickle` on a file that a later run appends
  to is a deserialization footgun for no gain, and neither format supports
  partial reads — you would load every vector to look up one.
- **Cache keyed on chunk id.** Cheaper key construction, but a renamed function
  or a re-chunk invalidates entries whose text never changed. The text *is* the
  identity.
- **A vector-length header in the blob.** Unnecessary: `len(blob) // 4` is the
  dimension, and `array.frombytes` validates the length for us.
