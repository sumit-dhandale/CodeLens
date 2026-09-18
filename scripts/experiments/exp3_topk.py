"""Experiment 3 — how deep should k go?

Recall only ever rises with k, so the number that decides k is on the right of
the table: the context tokens you must pay an LLM to read. This is the
retrieval-side half of the RAG cost equation.
"""

from __future__ import annotations

import sys

from src.chunking.code_chunker import count_tokens_batch

from . import _common as c

KS = (1, 3, 5, 10, 20)


def main(repo: str | None = None, gold: str | None = None) -> None:
    cfg = c.base_cfg(repo, gold, strategy="function")
    files = c.load_files(cfg)
    harness = c.harness_for(cfg, files)
    gold = c.gold_for(cfg)
    search = harness.retriever("vector")

    rows = []
    for k in KS:
        summary = c.evaluate(harness, ["vector"], top_k=k)[0]
        # What the answering model would actually have to read at this k.
        tokens = [
            sum(count_tokens_batch([h.text for h in search(q["query"], k)], cfg.model))
            for q in gold["queries"]
        ]
        rows.append([*c.summary_row(f"k={k}", summary), f"{sum(tokens) / len(tokens):.0f}"])

    body = (
        "hit@1 and the other @5 metrics are computed at each k, so `hit@5` here "
        "means hit@k.\n\n" + c.md_table([*c.SUMMARY_HEADERS, "ctx tokens"], rows)
    )
    c.write("exp3-topk.md", "Experiment 3 — top-k", body, cfg)
    harness.embedder.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
