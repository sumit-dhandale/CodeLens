"""Experiment 5 — filters help exactly as much as they are correct.

Three variants of the same search: no filter, a blanket `language=python`
filter, and an oracle filter set to the language of each query's own gold
answer. The oracle is not shippable — it needs the answer to pick the filter —
but it bounds how much a *perfect* query router could win, and the gap between
it and the blanket filter is what a wrong guess costs.
"""

from __future__ import annotations

import sys

from src.eval.metrics import QueryScore, summarize
from src.eval.runner import grade_hits
from src.vector_store.qdrant_store import make_filter

from . import _common as c


def _language_of(query: dict, by_path: dict[str, str]) -> str | None:
    for label in query["gold"]:
        if language := by_path.get(label["file"]):
            return language
    return None


def main(repo: str | None = None, gold: str | None = None) -> None:
    cfg = c.base_cfg(repo, gold, strategy="function")
    files = c.load_files(cfg)
    by_path = {f.file_path: f.language for f in files}
    harness = c.harness_for(cfg, files)
    gold = c.gold_for(cfg)
    store, embedder = harness.store, harness.embedder

    variants = {
        "no filter": lambda q: None,
        "language=python": lambda q: make_filter(language="python"),
        "oracle language": lambda q: make_filter(language=_language_of(q, by_path)),
    }

    rows = []
    for label, choose in variants.items():
        scores, unanswerable = [], []
        for query in gold["queries"]:
            hits = store.search(embedder.embed(query["query"]), top_k=5, query_filter=choose(query))
            grades = grade_hits(hits, query["gold"])
            if query["kind"] == "unanswerable":
                unanswerable.append(hits[0].score if hits else 0.0)
            else:
                scores.append(
                    QueryScore.score(
                        query["id"], query["kind"], grades, [g["grade"] for g in query["gold"]]
                    )
                )
        rows.append(c.summary_row(label, summarize(label, scores, unanswerable, 0.30)))

    body = (
        "The oracle filter picks the language of the query's own gold answer, so "
        "it is an upper bound on query routing rather than something you can ship.\n\n"
        + c.md_table(c.SUMMARY_HEADERS, rows)
    )
    c.write("exp5-filters.md", "Experiment 5 — metadata filters", body, cfg)
    embedder.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
