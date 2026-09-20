"""Run a set of questions through `ask` and tabulate citation validity.

The `ask` command answers one question. This asks several and reports the only
two things that matter across a batch: did the model abstain when it should
have, and did it cite anything that was never retrieved.

Run: `python -m scripts.rag_probe [--provider ollama|extractive]`
"""

from __future__ import annotations

import argparse
import json

from scripts.experiments._common import checkout_path
from src.config import ROOT, Config
from src.embeddings.embedder import Embedder
from src.parser import repository_loader
from src.rag.generator import NAIVE_PROMPT, SYSTEM_PROMPT, Generator
from src.retrieval.context import ContextExpander
from src.retrieval.vector_search import VectorSearcher

# Two answerable, two lexical-gap, two unanswerable. The unanswerable ones are
# the whole point: a RAG system's failure mode is confidently answering them.
QUESTIONS = [
    ("q09", "answerable", "how are point ids generated so re-indexing does not duplicate rows"),
    ("q14", "answerable", "run a nearest neighbour query and return the top hits"),
    ("q23", "lexical-gap", "we ran out of database connections under load"),
    ("q28", "lexical-gap", "slowness caused by the network path rather than by the datastore itself"),
    ("q29", "unanswerable", "how does the cross-encoder reranker score candidates"),
    ("q30", "unanswerable", "where do we validate the JWT on an incoming request"),
    ("q31", "unanswerable", "kubernetes deployment manifest for the api"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.rag_probe")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expand", action="store_true", default=True)
    parser.add_argument("--prompt", choices=("project", "naive"), default="project")
    args = parser.parse_args(argv)

    cfg = Config(llm_provider=args.provider, top_k=args.top_k)
    searcher = VectorSearcher(cfg, embedder=Embedder(cfg))
    expander = ContextExpander(repository_loader.load(checkout_path(cfg), cfg))
    generator = Generator(
        cfg,
        expander=expander if args.expand else None,
        system=SYSTEM_PROMPT if args.prompt == "project" else NAIVE_PROMPT,
    )

    rows, records = [], []
    for qid, kind, question in QUESTIONS:
        result = searcher.search(question, top_k=cfg.top_k)
        answer = generator.answer(question, result.hits, retrieval_ms=result.total_ms)
        check = answer.citations
        rows.append(
            f"| {qid} | {kind} | {'yes' if answer.abstained else 'no'} | "
            f"{len(check.grounded)} | {len(check.drifted)} | "
            f"{', '.join(check.invented) or '-'} | {answer.generation_ms:.0f} |"
        )
        records.append(
            {
                "id": qid,
                "kind": kind,
                "question": question,
                "answer": answer.text,
                "top_hit": answer.hits[0].citation if answer.hits else None,
                "top_score": round(answer.hits[0].score, 4) if answer.hits else None,
                "abstained": answer.abstained,
                "grounded": list(check.grounded),
                "drifted": list(check.drifted),
                "invented": list(check.invented),
            }
        )
        print(f"--- {qid} ({kind}) {question}\n{answer.text}\n")

    print("| id | kind | abstained | grounded | drift | invented | gen ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(row)

    out = ROOT / "eval" / "results" / f"rag_{args.provider}_{args.prompt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    searcher.embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
