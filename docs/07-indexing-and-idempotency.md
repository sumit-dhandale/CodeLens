# 07 — Indexing: collections, stable IDs, and the delete nobody writes

## One collection per experiment

```
sumit-dhandale__OpsSense__all-MiniLM-L6-v2__function
{repo}__________________  {model}__________  {strategy}
```

The name is built from the repo, the model, and the chunking strategy, plus an
optional `--collection-suffix`. This is not tidiness. Two failure modes it makes
structurally impossible:

- **Cross-model queries.** MiniLM and `bge-small` both emit 384 dimensions.
  Query one model's vectors with the other's and Qdrant returns confident
  nonsense — no error, no warning, just bad results. Separate collections mean
  the mistake cannot be made.
- **Experiment cross-contamination.** Running the `file` strategy after the
  `function` strategy would otherwise leave a collection holding both, and every
  number in Experiment 2 would be wrong in a way that averages out to plausible.

`ensure_collection` additionally refuses to write 768-dim vectors into a 384-dim
collection, which is the one mismatch a shared name cannot prevent (same model
name, different revision).

## Point IDs: `uuid5`, not autoincrement

```python
point_id = uuid5(NS, f"{file_path}:{class_name}:{symbol_name}:{start_line}:{strategy}")
```

Qdrant accepts unsigned integers or UUIDs as IDs. `uuid5` is a hash of a name, so
the same chunk always derives the same ID on any machine, in any order, without
consulting the database. That single property is what makes `upsert` an update
instead of an insert, and therefore what makes re-indexing idempotent. Verified:
indexing OpsSense twice leaves `points_count` at 85.

The alternative — a counter, or a random `uuid4` — means the second run cannot
recognise the first run's points, so you get 170 points, every result duplicated,
and a corpus that grows every time CI runs.

## Three levels of change detection

| What changed | What happens | Why |
| --- | --- | --- |
| Nothing | 0 upserts, 0 forward passes | payload `content_hash` matches |
| A function body, no lines moved | 1 upsert, 1 forward pass | same ID, new vector, updated in place |
| A function deleted | orphan points deleted | its ID is absent from the current parse |

The `content_hash` is `sha256(embed_text)`, deliberately *not* the file's hash.
Editing a docstring at the top of a file must not invalidate the vectors of forty
functions below it. Cheap hash, no forward pass, no Qdrant write.

## The honest cost of putting `start_line` in the ID

Two `def foo()` in one file are different chunks and need different IDs, so the
line number is in the key. The price: deleting a method shifts every symbol below
it, changing their IDs, so they are deleted and re-inserted even though their
text is identical. `tests/test_deleting_a_symbol_leaves_no_ghost_points`
documents exactly this — removing one method churns two points, not one.

That churn is nearly free (the *vectors* still come from the cache, since the
text did not change; only the Qdrant write repeats) and the alternative is worse:
a key without the line number silently merges two same-named functions into one
point, so one of them becomes unretrievable. Correctness over churn.

If the churn ever mattered, the fix is a content-addressed key
(`path:symbol:sha256(text)[:12]`), which is stable under line shifts but changes
on every edit — trading insert churn for update churn. Neither is free.

## The delete half

Indexing is a *sync*: every point currently in the collection whose ID is not in
the current parse is deleted. Almost every tutorial stops at upsert, and the
result is a search engine that confidently returns a function you deleted three
refactors ago, complete with a line range that now points at something else.

Sequencing matters. Scroll existing IDs → embed only changed chunks → create the
collection → upsert → delete orphans → count. The collection is created *after*
embedding so its dimension comes from the vectors themselves; that is why a warm
`--drop` rebuild never imports torch (docs/06).

## Payload

Alongside the vector: `file_path`, `directory`, `language`, `strategy`,
`symbol_type`, `symbol_name`, `class_name`, `start_line`, `end_line`, the
verbatim `text`, `file_sha`, and `content_hash`. Storing the source text in the
payload means a hit is renderable and citable from one round trip, without
reopening the repo — which matters because the checkout is pinned to a SHA that
the working tree may have moved past.

Keyword payload indexes are created on the filterable fields at collection time,
where Milestone 4 needs them. They are skipped in local mode, which does not
support them and logs a warning per call if you ask.

## Local mode

`--qdrant-url :memory:` runs qdrant-client's pure-python engine in-process. That
is what lets `tests/test_vector_store.py` test real indexing, real filtering, and
real ANN search with no server and no torch. Local mode is not a mock: it is the
same client API over a different backend, so the tests exercise the code that
production runs.

## Verify

```bash
docker compose up -d
python -m src.cli index --strategy function       # upserted: 85
python -m src.cli index --strategy function       # upserted: 0, points: 85
curl -s localhost:6333/collections | python -m json.tool
pytest tests/test_vector_store.py
```

## Alternatives considered

- **Delete the collection and rebuild every time.** Honest, simple, and what
  most projects do. It is also a full re-embed of everything on every run unless
  the cache saves you, and it makes the index unavailable mid-rebuild.
- **A single collection with a `strategy` payload filter.** One collection, more
  filtering. Filtered HNSW recall degrades as the filter gets more restrictive
  (docs/09), so a strategy comparison would be measuring the filter as much as
  the strategy.
- **Integer point IDs from a counter.** Requires a persistent counter and a
  path-to-id mapping table, which is a second database to keep in sync with the
  first.
