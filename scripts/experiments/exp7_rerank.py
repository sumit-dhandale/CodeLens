"""Experiment 7 — is two-stage retrieval worth it on code?

Top-5 straight from the index, against top-20 reordered by a cross-encoder. Also
tests the obvious rescue: the cross-encoder normally sees raw source, while the
bi-encoder sees the *enriched* text (path, split identifiers, docstring). Giving
the cross-encoder the same enriched text isolates whether the problem is the
model or its input.

The per-kind split is the diagnostic: `ms-marco-MiniLM` was trained on English
web passages, so if the domain mismatch is the explanation, it should do least
badly on the prose queries.
"""

from __future__ import annotations

import dataclasses
import sys
import time

from src.eval.metrics import QueryScore, summarize
from src.eval.runner import grade_hits
from src.retrieval import fusion

from . import _common as c


def main(repo: str | None = None, gold: str | None = None) -> None:
    cfg = c.base_cfg(repo, gold, strategy="function")
    files = c.load_files(cfg)
    harness = c.harness_for(cfg, files, with_reranker=True)
    gold = c.gold_for(cfg)
    depth = cfg.rerank_top_n
    enriched = {ch.citation: ch.embed_text for ch in harness.chunks}

    def vector(query: str, k: int):
        return harness._vector(query, k)

    def rerank_raw(query: str, k: int):
        return harness.reranker.rerank(query, vector(query, depth), k)

    def rerank_enriched(query: str, k: int):
        hits = vector(query, depth)
        swapped = [dataclasses.replace(h, text=enriched.get(h.citation, h.text)) for h in hits]
        reordered = harness.reranker.rerank(query, swapped, k)
        by_citation = {h.citation: h for h in hits}
        return [by_citation[h.citation] for h in reordered]

    def rrf_rerank(query: str, k: int):
        fused = fusion.reciprocal_rank_fusion(
            [vector(query, depth), harness.bm25.search(query, depth)], top_k=depth
        )
        return harness.reranker.rerank(query, fused, k)

    variants = {
        "vector top-5": vector,
        f"vector top-{depth} -> rerank 5": rerank_raw,
        f"vector top-{depth} -> rerank 5 (enriched)": rerank_enriched,
        f"hybrid_rrf top-{depth} -> rerank 5": rrf_rerank,
    }

    rows, kind_rows = [], []
    for label, fn in variants.items():
        scores, unanswerable, elapsed = [], [], []
        for query in gold["queries"]:
            started = time.perf_counter()
            hits = fn(query["query"], 5)
            elapsed.append((time.perf_counter() - started) * 1000)
            grades = grade_hits(hits, query["gold"])
            if query["kind"] == "unanswerable":
                unanswerable.append(hits[0].score if hits else 0.0)
            else:
                scores.append(
                    QueryScore.score(
                        query["id"], query["kind"], grades, [g["grade"] for g in query["gold"]]
                    )
                )
        rows.append(
            c.summary_row(
                label, summarize(label, scores, unanswerable, 0.30, sum(elapsed) / len(elapsed))
            )
        )
        for kind in ("code", "lexical-gap", "docs"):
            subset = [s for s in scores if s.kind == kind]
            if subset:
                kind_rows.append(
                    c.summary_row(f"{label} [{kind}]", summarize(label, subset, [], 0.30))
                )

    body = (
        c.md_table(c.SUMMARY_HEADERS, rows)
        + "\n\n## Split by query kind\n\n"
        + "`ms-marco-MiniLM` was trained on English web passages. If domain "
        "mismatch explains the damage, the prose rows should suffer least.\n\n"
        + c.md_table(c.SUMMARY_HEADERS, kind_rows)
    )
    c.write("exp7-rerank.md", "Experiment 7 — cross-encoder reranking", body, cfg)
    harness.embedder.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
