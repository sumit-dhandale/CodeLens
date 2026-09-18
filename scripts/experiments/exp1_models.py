"""Experiment 1 — does a bigger or code-trained embedding model retrieve better?

Each model gets its own collection, so nothing is compared across coordinate
systems. Index time is reported per model because it is half the decision: a
model that is 3 points better and 4x slower is not obviously better.
"""

from __future__ import annotations

import sys
import time

from . import _common as c

MODELS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    ("st-codesearch-distilroberta", "flax-sentence-embeddings/st-codesearch-distilroberta-base"),
]


def main(repo: str | None = None, gold: str | None = None) -> None:
    rows = []
    cfg = None
    for label, model in MODELS:
        cfg = c.base_cfg(repo, gold, model=model, strategy="function")
        files = c.load_files(cfg)

        started = time.perf_counter()
        _, embedder, points = c.ensure_index(cfg, files)
        index_s = time.perf_counter() - started

        harness = c.harness_for(cfg, files)
        summary = c.evaluate(harness, ["vector"])[0]
        row = c.summary_row(label, summary)
        rows.append([*row, embedder.dim, points, f"{index_s:.1f}"])
        embedder.close()

    body = "Same chunks, same gold set, one collection per model.\n\n" + c.md_table(
        [*c.SUMMARY_HEADERS, "dim", "points", "index s"], rows
    )
    c.write("exp1-embedding-models.md", "Experiment 1 — embedding models", body, cfg)


if __name__ == "__main__":
    main(*sys.argv[1:3])
