# 01 — Vector databases, and why Qdrant instead of a numpy array

## The four words you need

- **Vector** — a fixed-length list of floats produced by an embedding model. Every
  chunk of code becomes one. MiniLM-L6 emits 384 dimensions. Similarity between
  two chunks is a cheap arithmetic operation on their vectors.
- **Point** — one stored record: an ID, its vector, and a payload. A point in this
  project is one chunk of code.
- **Payload** — arbitrary JSON attached to a point: `file_path`, `language`,
  `symbol_type`, `start_line`. It is what turns a similarity hit into a citation,
  and what lets us filter ("only Python, only functions") before or during search.
- **Collection** — a named set of points sharing one vector size and distance
  metric. We use one collection per `(repo, model, chunking strategy)` so that
  experiments never contaminate each other's numbers.

## Why not just a numpy array?

For 90 chunks, honestly, numpy would work. `matrix @ query` over 90×384 floats is
microseconds, and exhaustive search is exactly correct — no recall loss at all.
The reasons to run a real vector database anyway:

1. **Exhaustive search is linear.** At 90 chunks it's free. At 10⁶ chunks it is
   ~1.5 GB of floats scanned per query. Qdrant's HNSW index is sublinear: it walks
   a navigable small-world graph and touches a few hundred candidates instead of a
   million. Milestone 6 measures exactly this trade-off, including the recall we
   give up for that speed.
2. **Filters belong next to the index.** "Nearest Python functions, excluding
   tests" is not something a bare matrix multiply expresses. Post-filtering a
   top-10 can return zero rows; Qdrant applies filters *during* graph traversal.
3. **Persistence and updates.** Re-indexing after an edit must update points in
   place and delete points for symbols that no longer exist. That is a storage
   problem, not a math problem.
4. **Payloads travel with the vector.** Keeping a parallel `chunks.json` in sync
   with a `vectors.npy` is a bug waiting to happen.

The honest framing: for this corpus Qdrant is over-engineering, and we use it
because measuring the ANN recall/latency curve is one of the project's goals.

## Running it

```bash
docker compose up -d      # REST on 6333, gRPC on 6334
curl localhost:6333/healthz
```

Storage is bind-mounted to `./.qdrant_storage`, so `docker compose down` does not
lose the index. The image tag is pinned — HNSW defaults and quantization options
have changed between minor versions, which would quietly invalidate benchmarks.
