"""Experiment 4 — is there a similarity threshold that means "I don't know"?

A vector index always returns its top-k, so the only way to abstain is to refuse
scores below some cutoff. This sweeps the cutoff and prints the score
distributions it is trying to separate. If those distributions overlap — and
they do — no cutoff can be correct, and that is the finding.
"""

from __future__ import annotations

import sys

from src.eval.runner import grade_hits

from . import _common as c

THRESHOLDS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def main(repo: str | None = None, gold: str | None = None) -> None:
    cfg = c.base_cfg(repo, gold, strategy="function")
    harness = c.harness_for(cfg, c.load_files(cfg))
    gold = c.gold_for(cfg)
    search = harness.retriever("vector")

    per_query = []
    relevant, irrelevant = [], []
    for query in gold["queries"]:
        hits = search(query["query"], 5)
        grades = grade_hits(hits, query["gold"])
        per_query.append((query["kind"], [h.score for h in hits], grades))
        for hit, grade in zip(hits, grades, strict=True):
            (relevant if grade > 0 else irrelevant).append(hit.score)

    rows = []
    for threshold in THRESHOLDS:
        answerable = [(s, g) for kind, s, g in per_query if kind != "unanswerable"]
        kept_hit = [
            1.0
            if any(grade > 0 and score >= threshold for score, grade in zip(s, g, strict=True))
            else 0.0
            for s, g in answerable
        ]
        unanswerable = [s for kind, s, _ in per_query if kind == "unanswerable"]
        abstained = [
            1.0 if not any(score >= threshold for score in s) else 0.0 for s in unanswerable
        ]
        rows.append(
            [
                f"{threshold:.2f}",
                f"{sum(kept_hit) / len(kept_hit):.2f}",
                f"{sum(abstained) / len(abstained):.2f}",
                sum(1 for s, _ in answerable for score in s if score >= threshold),
            ]
        )

    distribution = c.md_table(
        ("score", "n", "min", "p25", "median", "p75", "max"),
        [
            [
                label,
                len(values),
                f"{min(values):.3f}",
                f"{_percentile(values, 0.25):.3f}",
                f"{_percentile(values, 0.50):.3f}",
                f"{_percentile(values, 0.75):.3f}",
                f"{max(values):.3f}",
            ]
            for label, values in (("relevant", relevant), ("irrelevant", irrelevant))
        ],
    )

    body = (
        "`hit@5 kept` is the fraction of answerable queries whose correct answer "
        "survives the cutoff; `abstained` is the fraction of unanswerable queries "
        "that return nothing at all. A usable threshold would push both to 1.0.\n\n"
        + c.md_table(("threshold", "hit@5 kept", "abstained", "hits kept"), rows)
        + "\n\n## Score distributions\n\n"
        + distribution
    )
    c.write("exp4-threshold.md", "Experiment 4 — similarity threshold", body, cfg)
    harness.embedder.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
