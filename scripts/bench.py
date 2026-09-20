"""ANN performance and recall benchmarks on synthetic vectors.

Why synthetic: the question here is about Qdrant and HNSW, not about MiniLM.
Embedding 100k real chunks would take hours of CPU and would measure the
embedding model, not the index. Random unit vectors of the same dimension
(384) exercise exactly the code path we care about, and they are worse than
real data in a useful way: real embeddings cluster, and clustered data is
*easier* for HNSW, so every recall number here is a pessimistic bound.

Three benchmarks:

- `scale`   1k / 10k / 100k points: build time, p50/p95 query latency, size.
- `sweep`   m / ef_construct / ef_search vs recall against exact search.
- `quant`   scalar quantization: memory and recall cost.

Run: `python -m scripts.bench all`
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import ROOT, Config

DIM = 384
SEED = 20260906
PREFIX = "bench__"
STORAGE = ROOT / ".qdrant_storage" / "collections"

QUERIES = 50  # query vectors per measurement
RECALL_K = 10
UPSERT_BATCH = 1000

# Qdrant only builds HNSW once a segment crosses indexing_threshold_kb (20MB by
# default) and brute-forces below it. At 1k points that default would silently
# make "ANN latency" mean "exact latency", so every benchmark collection forces
# the index on.
FORCE_INDEX = models.OptimizersConfigDiff(indexing_threshold=1)


def rng_vectors(n: int, seed: int = SEED) -> np.ndarray:
    """Unit-norm random vectors: cosine distance then behaves like dot product."""
    v = np.random.default_rng(seed).standard_normal((n, DIM), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@dataclass
class Bench:
    client: QdrantClient
    name: str

    def build(
        self,
        vectors: np.ndarray,
        m: int = 16,
        ef_construct: int = 100,
        quantize: bool = False,
    ) -> float:
        """Create, fill, and wait for the index. Returns wall-clock seconds."""
        self.drop()
        started = time.perf_counter()
        self.client.create_collection(
            self.name,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
            optimizers_config=FORCE_INDEX,
            quantization_config=(
                models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                )
                if quantize
                else None
            ),
        )
        ids = range(len(vectors))
        for start in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[start : start + UPSERT_BATCH]
            self.client.upsert(
                self.name,
                points=models.Batch(
                    ids=list(ids)[start : start + UPSERT_BATCH],
                    vectors=batch.tolist(),
                ),
                wait=True,
            )
        self._await_green()
        return time.perf_counter() - started

    def _await_green(self, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self.client.get_collection(self.name)
            if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count:
                return
            time.sleep(0.5)
        raise RuntimeError(f"{self.name} never finished indexing")

    def query(
        self, vector, top_k: int = RECALL_K, ef: int | None = None, exact: bool = False
    ) -> list[int]:
        params = models.SearchParams(exact=exact, hnsw_ef=ef)
        if exact:
            # Rescoring off: with quantization on, the honest "what did the
            # index return" number is the quantized one.
            params = models.SearchParams(exact=True)
        response = self.client.query_points(
            self.name, query=vector.tolist(), limit=top_k, search_params=params,
            with_payload=False,
        )
        return [int(p.id) for p in response.points]

    def measure(self, queries: np.ndarray, ef: int | None = None) -> tuple[float, float]:
        """p50 / p95 query latency in milliseconds."""
        timings = []
        for q in queries:
            started = time.perf_counter()
            self.query(q, ef=ef)
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        return statistics.median(timings), timings[int(len(timings) * 0.95) - 1]

    def recall(self, queries: np.ndarray, truth: list[list[int]], ef: int | None = None) -> float:
        hits = 0
        for q, gold in zip(queries, truth, strict=True):
            hits += len(set(self.query(q, ef=ef)) & set(gold))
        return hits / (len(queries) * RECALL_K)

    def truth(self, queries: np.ndarray) -> list[list[int]]:
        """Brute-force top-k. This is the only thing recall can be measured against."""
        return [self.query(q, exact=True) for q in queries]

    def size_mb(self) -> float:
        """On-disk bytes for this collection, via the bind-mounted storage dir."""
        path = STORAGE / self.name
        if not path.exists():
            return float("nan")
        out = subprocess.run(  # noqa: S603 - fixed binary, argument is a local path
            [shutil.which("du") or "/usr/bin/du", "-sk", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return int(out.stdout.split()[0]) / 1024 if out.stdout else float("nan")

    def drop(self) -> None:
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)


def rss_mb(url: str) -> float:
    """Qdrant's own `memory_resident_bytes` from /metrics.

    Process-wide, so it only means something when the benchmark collection is
    the only large one on the server - which it is, since every other
    collection here holds a few thousand points.
    """
    import urllib.request

    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:  # noqa: S310
        for line in response.read().decode().splitlines():
            if line.startswith("memory_resident_bytes "):
                return float(line.split()[1]) / 1024**2
    return float("nan")


def table(headers, rows) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True))  # noqa: E731
    md = ["| " + " | ".join(str(h) for h in headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    print(line(headers))
    for row in rows:
        print(line(row))
    print()
    return "\n".join(md)


# --- benchmarks --------------------------------------------------------------


def bench_scale(client: QdrantClient, url: str, sizes=(1_000, 10_000, 100_000)) -> str:
    queries = rng_vectors(QUERIES, seed=SEED + 1)
    rows = []
    for n in sizes:
        bench = Bench(client, f"{PREFIX}scale_{n}")
        build_s = bench.build(rng_vectors(n))
        p50, p95 = bench.measure(queries)
        exact_p50, _ = _exact_latency(bench, queries)
        rows.append([
            f"{n:,}",
            f"{build_s:.1f}",
            f"{n / build_s:,.0f}",
            f"{_round_trip(bench):.2f}",
            f"{p50:.2f}",
            f"{p95:.2f}",
            f"{exact_p50:.2f}",
            f"{exact_p50 / p50:.1f}x",
            f"{bench.size_mb():.1f}",
            f"{rss_mb(url):.0f}",
        ])
        bench.drop()
    return table(
        ("points", "build s", "pts/s", "http ms", "p50 ms", "p95 ms", "exact p50 ms",
         "speedup", "disk MB", "server RSS MB"),
        rows,
    )


def _round_trip(bench: Bench) -> float:
    """Median latency of a do-nothing HTTP call: the floor every query pays.

    Without this column the scale table looks like HNSW is barely faster than
    brute force, when most of both numbers is the REST round trip.
    """
    timings = []
    for _ in range(QUERIES):
        started = time.perf_counter()
        bench.client.get_collection(bench.name)
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def _exact_latency(bench: Bench, queries: np.ndarray) -> tuple[float, float]:
    timings = []
    for q in queries:
        started = time.perf_counter()
        bench.query(q, exact=True)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return statistics.median(timings), timings[int(len(timings) * 0.95) - 1]


def bench_sweep(client: QdrantClient, n: int = 10_000) -> tuple[str, str, str]:
    """The recall-vs-speed curve, one knob at a time.

    Run at 10k rather than 100k because `m` and `ef_construct` each require a
    full rebuild, and the shape of the curve - not its absolute latency - is
    the thing being measured.
    """
    vectors, queries = rng_vectors(n), rng_vectors(QUERIES, seed=SEED + 1)
    bench = Bench(client, f"{PREFIX}sweep")

    build_s = bench.build(vectors, m=16, ef_construct=100)
    truth = bench.truth(queries)
    print(f"ground truth from exact search, {n} points, built in {build_s:.1f}s\n")

    ef_rows = []
    for ef in (4, 8, 16, 32, 64, 128, 256, 512):
        p50, p95 = bench.measure(queries, ef=ef)
        ef_rows.append([ef, f"{bench.recall(queries, truth, ef=ef):.3f}", f"{p50:.2f}", f"{p95:.2f}"])
    ef_table = table(("ef_search", f"recall@{RECALL_K}", "p50 ms", "p95 ms"), ef_rows)

    m_rows = []
    for m in (4, 8, 16, 32, 64):
        build_s = bench.build(vectors, m=m, ef_construct=100)
        p50, _ = bench.measure(queries, ef=64)
        m_rows.append([
            m,
            f"{build_s:.1f}",
            f"{bench.recall(queries, truth, ef=64):.3f}",
            f"{p50:.2f}",
            f"{bench.size_mb():.1f}",
        ])
    m_table = table(("m", "build s", f"recall@{RECALL_K} (ef=64)", "p50 ms", "disk MB"), m_rows)

    efc_rows = []
    for efc in (16, 32, 64, 128, 256, 512):
        build_s = bench.build(vectors, m=16, ef_construct=efc)
        p50, _ = bench.measure(queries, ef=64)
        efc_rows.append([
            efc, f"{build_s:.1f}", f"{bench.recall(queries, truth, ef=64):.3f}", f"{p50:.2f}"
        ])
    efc_table = table(("ef_construct", "build s", f"recall@{RECALL_K} (ef=64)", "p50 ms"), efc_rows)

    bench.drop()
    return ef_table, m_table, efc_table


def bench_real(client: QdrantClient, source: str) -> str:
    """The same ef_search sweep, on real embeddings instead of random ones.

    Random unit vectors in 384 dimensions are nearly equidistant from each
    other, which is the worst case HNSW can be handed: there is no cluster
    structure for the graph to exploit. Real embeddings cluster hard. Running
    both sweeps is the only way to know which of those two curves the synthetic
    numbers above are describing.
    """
    points, _ = client.scroll(source, limit=20_000, with_vectors=True, with_payload=False)
    vectors = np.array([p.vector for p in points], dtype=np.float32)
    if len(vectors) < QUERIES * 2:
        raise RuntimeError(f"{source} has only {len(vectors)} points")
    # Queries are held-out corpus vectors: a nearest-neighbour problem with the
    # same distribution as the index, which is what a real query embedding is.
    queries, vectors = vectors[:QUERIES], vectors[QUERIES:]

    bench = Bench(client, f"{PREFIX}real")
    bench.build(vectors, m=16, ef_construct=100)
    truth = bench.truth(queries)
    rows = []
    for ef in (4, 8, 16, 32, 64, 128, 256, 512):
        p50, _ = bench.measure(queries, ef=ef)
        rows.append([ef, f"{bench.recall(queries, truth, ef=ef):.3f}", f"{p50:.2f}"])
    bench.drop()
    print(f"source: {source}, {len(vectors)} indexed, {QUERIES} held-out queries\n")
    return table(("ef_search", f"recall@{RECALL_K} (real)", "p50 ms"), rows)


def bench_quant(client: QdrantClient, url: str, n: int = 100_000) -> str:
    vectors, queries = rng_vectors(n), rng_vectors(QUERIES, seed=SEED + 1)
    bench = Bench(client, f"{PREFIX}quant")

    bench.build(vectors, m=16, ef_construct=100)
    truth = bench.truth(queries)

    rows = []
    for label, quantize in (("float32", False), ("int8 scalar", True)):
        build_s = bench.build(vectors, m=16, ef_construct=100, quantize=quantize)
        p50, p95 = bench.measure(queries, ef=64)
        rows.append([
            label,
            f"{build_s:.1f}",
            f"{bench.recall(queries, truth, ef=64):.3f}",
            f"{p50:.2f}",
            f"{p95:.2f}",
            f"{bench.size_mb():.1f}",
            f"{rss_mb(url):.0f}",
        ])
    bench.drop()
    return table(
        ("vectors", "build s", f"recall@{RECALL_K} (ef=64)", "p50 ms", "p95 ms", "disk MB",
         "server RSS MB"),
        rows,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.bench")
    parser.add_argument(
        "what", choices=("scale", "sweep", "real", "quant", "all"), default="all", nargs="?"
    )
    parser.add_argument("--qdrant-url", default=Config.qdrant_url)
    parser.add_argument(
        "--real-collection",
        default="encode__httpx__all-MiniLM-L6-v2__function",
        help="existing collection to draw real embeddings from",
    )
    args = parser.parse_args(argv)

    client = QdrantClient(url=args.qdrant_url)
    sections = []
    if args.what in ("scale", "all"):
        print("\n=== scale ===")
        sections.append(("Scale", bench_scale(client, args.qdrant_url)))
    if args.what in ("sweep", "all"):
        print("\n=== ef_search / m / ef_construct ===")
        ef_t, m_t, efc_t = bench_sweep(client)
        sections += [("ef_search", ef_t), ("m", m_t), ("ef_construct", efc_t)]
    if args.what in ("real", "all"):
        print("\n=== real embeddings ===")
        sections.append(("Real embeddings", bench_real(client, args.real_collection)))
    if args.what in ("quant", "all"):
        print("\n=== quantization ===")
        sections.append(("Quantization", bench_quant(client, args.qdrant_url)))

    out = ROOT / "docs" / "experiments" / "bench-ann.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(f"## {title}\n\n{md}" for title, md in sections)
    out.write_text(
        f"# ANN benchmarks (synthetic, dim {DIM})\n\n"
        f"_Generated by `scripts/bench.py` on {time.strftime('%Y-%m-%d')}._\n\n{body}\n"
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
