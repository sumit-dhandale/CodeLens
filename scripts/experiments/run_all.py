"""Run every experiment against OpsSense, and the corpus-sensitive ones against httpx.

Experiments 1, 2, 6 and 7 are the ones whose conclusions could plausibly be an
artifact of an 85-chunk corpus, so they are repeated on httpx (1465 chunks).
"""

from __future__ import annotations

import sys

from . import (
    exp1_models,
    exp2_chunking,
    exp3_topk,
    exp4_threshold,
    exp5_filters,
    exp6_hybrid,
    exp7_rerank,
)

OPSSENSE = [
    exp1_models,
    exp2_chunking,
    exp3_topk,
    exp4_threshold,
    exp5_filters,
    exp6_hybrid,
    exp7_rerank,
]
HTTPX = [exp1_models, exp2_chunking, exp6_hybrid, exp7_rerank]
HTTPX_REPO = "https://github.com/encode/httpx"


def main(only: str = "") -> None:
    if only in ("", "opssense"):
        for module in OPSSENSE:
            print(f"\n=== {module.__name__} (opssense) ===")
            module.main()
    if only in ("", "httpx"):
        for module in HTTPX:
            print(f"\n=== {module.__name__} (httpx) ===")
            module.main(HTTPX_REPO, "httpx")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
