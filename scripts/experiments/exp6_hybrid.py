"""Experiment 6 — keyword vs vector vs hybrid, with the fusion weight swept.

The baselines matter as much as the contenders: `random` is the floor and `grep`
is what costs nothing. An embedding pipeline that cannot beat grep has not
earned its index.
"""

from __future__ import annotations

import sys

from . import _common as c

RETRIEVERS = ("random", "grep", "bm25", "vector", "hybrid_rrf")
WEIGHTS = (0.3, 0.5, 0.7, 0.9, 1.0)


def main(repo: str | None = None, gold: str | None = None) -> None:
    cfg = c.base_cfg(repo, gold, strategy="function")
    files = c.load_files(cfg)
    harness = c.harness_for(cfg, files)

    rows = [
        c.summary_row(name, summary)
        for name, summary in zip(RETRIEVERS, c.evaluate(harness, RETRIEVERS), strict=True)
    ]

    weight_rows = []
    for weight in WEIGHTS:
        harness.cfg = harness.cfg.override(vector_weight=weight)
        summary = c.evaluate(harness, ["hybrid_weighted"])[0]
        weight_rows.append(c.summary_row(f"w={weight}", summary))

    body = (
        c.md_table(c.SUMMARY_HEADERS, rows)
        + "\n\n## Weighted fusion, sweeping the vector share\n\n"
        + "w=1.0 is pure vector, which reproduces the `vector` row above and so "
        "doubles as a correctness check on the fusion code.\n\n"
        + c.md_table(c.SUMMARY_HEADERS, weight_rows)
    )
    c.write("exp6-hybrid.md", "Experiment 6 — keyword, vector, hybrid", body, cfg)
    harness.embedder.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
