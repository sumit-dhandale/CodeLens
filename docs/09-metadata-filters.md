# 09 — Metadata filters: why Qdrant does this and a numpy array cannot

`--language`, `--symbol-type`, and `--directory` become Qdrant `must` conditions
over indexed payload fields:

```bash
python -m src.cli search "how do we store vectors" --symbol-type method --top-k 5
1. 0.3647  OpsSense/src/embeddings/embedder.py:20-22  method Embedder.dim
2. 0.3191  OpsSense/src/embeddings/embedder.py:14-18  method Embedder.embed_batch
3. 0.2159  OpsSense/src/embeddings/embedder.py:11-12  method Embedder.embed
4. 0.1476  OpsSense/src/embeddings/embedder.py:7-9    method Embedder.__init__
```

Four results for `--top-k 5`, because the corpus contains exactly four `method`
chunks. The engine is not hiding a fifth: it returned everything that satisfies
the filter, in similarity order.

## Pre-filter, post-filter, and filtered HNSW

Three ways to combine a filter with a vector search, and the difference is not
cosmetic:

| Approach | How | Failure mode |
| --- | --- | --- |
| Post-filter | search top-k, then drop non-matching | Ask for 5, get 1. The filtered-out results were the top 5; nothing refills them. |
| Pre-filter | select matching IDs, brute-force those | Correct, and O(matches). Fine for 4 chunks, hopeless for 4 million. |
| Filtered HNSW | walk the graph, only accepting matching nodes | What Qdrant does. Correct *and* sublinear — usually. |

Post-filtering is what you write by hand, and it is why a naive implementation
returns three results for `top_k=5` and looks broken. The demo above returning a
full ranked set from a 4-chunk slice of an 85-chunk corpus is the visible
consequence of Qdrant filtering *during* traversal.

## Why a very restrictive filter can hurt recall

HNSW's speed comes from its graph being well connected: the greedy walk assumes
that from any node it can reach the query's neighbourhood. A filter deletes
nodes from that graph. Delete enough and the graph fragments — the walk arrives
somewhere with no accepted neighbours and stops, missing matching vectors that
sit in a disconnected component.

Qdrant handles this with a cardinality estimate: below a threshold of matching
points it abandons the graph and brute-forces the filtered subset instead, which
is exact. So the pathological case is not "very restrictive" (that becomes exact
search) nor "barely restrictive" (the graph stays connected) but the awkward
middle, where the graph is sparse enough to fragment yet large enough that
Qdrant still trusts it. Experiment 5 measures exactly this by comparing filtered
ANN against filtered `--exact`.

## Payload indexes

Created at collection time on `language`, `symbol_type`, `directory`,
`file_path`, and `class_name`:

```bash
$ curl -s localhost:6333/collections/...__function | jq '.result.payload_schema | keys'
["class_name", "directory", "file_path", "language", "symbol_type"]
```

Without an index, a filter is a linear scan of payloads — Qdrant must read every
point to know whether it matches, which defeats the cardinality estimate it uses
to choose a strategy. All five are `keyword` type: exact-match string fields, not
tokenized text. `directory` therefore matches `OpsSense/src/ingestion` exactly
and not its subdirectories, which is worth knowing before you file a bug.

Local mode ignores payload indexes and logs a warning per call, so
`ensure_collection` skips creating them there. The filters themselves still work
in local mode — just by scanning — which is why the tests can cover filtering
without a server.

## What filters are actually for

The honest use is not precision tuning, it is scoping a corpus that mixes kinds
of text. Unfiltered, "where do we retry a failed request" returns ten markdown
incident write-ups and zero Python (docs/08): prose about retries out-embeds
code about retries, every time. `--language python` is a one-word fix for what
looks like a model problem.

The second use is cost. A filtered search over 4 candidates does not touch the
other 81 vectors, and that scales: filtering to one directory of a monorepo turns
a million-vector search into a ten-thousand-vector one.

## The quoting trap

argparse treats a dashed token *containing a space* as a positional argument. So

```bash
python -m src.cli search "how do we store vectors --language python"
```

silently embedded the flag as part of the query and returned confident nonsense —
which is exactly how I found it, via a `zsh` loop that (unlike `bash`) does not
word-split unquoted variables. `cmd_search` now rejects a query containing a long
flag. A search engine that answers a malformed question with plausible results is
worse than one that errors.

## Verify

```bash
python -m src.cli search "how do we store vectors" --symbol-type method
python -m src.cli search "where do we retry a failed request" --language python
python -m src.cli search "q --language python"      # rejected
```

## Alternatives considered

- **Post-filtering in Python.** Two lines, no payload indexes, and it returns
  fewer results than asked for exactly when the filter matters most.
- **A collection per language.** Perfect filter isolation and no index needed,
  but cross-language search becomes a manual merge of ranked lists, and the
  collection count multiplies by the strategy and model axes.
- **Tokenized `text` payload index** for keyword matching inside Qdrant. Qdrant
  supports a full-text index, which would put BM25-ish matching in the database.
  Rejected because implementing the tokenizer myself is the point (docs/11).
