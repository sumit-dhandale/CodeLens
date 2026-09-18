"""Experiment 2 — chunking strategy, reported next to the truncation rate.

The truncation column is the explanation, not a footnote: a chunk longer than
the model's 256 wordpieces is silently cut, so its vector represents the first
third of the text. Any strategy that produces long chunks is being scored on
partial information.
"""

from __future__ import annotations

import sys

from src.chunking import code_chunker
from src.config import CHUNK_STRATEGIES

from . import _common as c


def main(repo: str | None = None, gold: str | None = None) -> None:
    rows = []
    # The strategy is the variable here, so the header should not name one.
    label_cfg = c.base_cfg(repo, gold, strategy="all four")
    for strategy in CHUNK_STRATEGIES:
        cfg = c.base_cfg(repo, gold, strategy=strategy)
        files = c.load_files(cfg)
        chunks = code_chunker.chunk(files, cfg, strategy)
        stats = code_chunker.stats(chunks, strategy, model=cfg.model)

        harness = c.harness_for(cfg, files)
        summary = c.evaluate(harness, ["vector"])[0]
        rows.append(
            [
                *c.summary_row(strategy, summary),
                stats.chunks,
                f"{stats.mean_tokens:.0f}",
                f"{stats.truncation_rate:.0%}",
            ]
        )
        harness.embedder.close()

    body = (
        "One collection per strategy. Token counts are exact wordpieces against "
        "the indexing model's tokenizer.\n\n"
        + c.md_table([*c.SUMMARY_HEADERS, "chunks", "mean tok", "trunc"], rows)
    )
    c.write("exp2-chunking.md", "Experiment 2 — chunking strategy", body, label_cfg)


if __name__ == "__main__":
    main(*sys.argv[1:3])
