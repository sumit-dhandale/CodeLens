"""argparse entry point: python -m src.cli <command> [flags]."""

from __future__ import annotations

import argparse
import collections
import logging
import re
import sys
import time
from pathlib import Path

from .chunking import code_chunker
from .config import LLM_PROVIDERS, Config
from .embeddings.embedder import Embedder
from .eval import gold as gold_mod
from .eval import metrics, runner
from .parser import code_parser, repo_fetcher, repository_loader
from .rag import generator as rag_generator
from .retrieval import context as context_mod
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.vector_search import VectorSearcher
from .vector_store import qdrant_store

# A long flag inside the query text. Only `--word`, so "well-known" and "-1"
# stay searchable.
FLAG_IN_QUERY = r"(?:^|\s)(--[a-z][a-z-]+)"


def _resolve_checkout(cfg: Config) -> repo_fetcher.Checkout:
    """Reuse the newest local checkout of cfg.repo, cloning only if there is none."""
    owner, name = repo_fetcher.parse_repo_url(cfg.repo)
    existing = sorted(
        cfg.repos_dir.glob(f"{owner}__{name}@*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if existing:
        path = existing[0]
        return repo_fetcher.Checkout(
            owner=owner, name=name, sha=path.name.split("@", 1)[1], path=path
        )
    return repo_fetcher.fetch(cfg)


def cmd_fetch(cfg: Config, _args: argparse.Namespace) -> int:
    checkout = repo_fetcher.fetch(cfg)
    print(f"{checkout.slug}\n  sha:  {checkout.sha}\n  path: {checkout.path}")
    return 0


def cmd_load(cfg: Config, args: argparse.Namespace) -> int:
    root = Path(args.path).resolve() if args.path else _resolve_checkout(cfg).path
    files = repository_loader.load(root, cfg)
    by_language = collections.Counter(f.language for f in files)
    for language, count in sorted(by_language.items()):
        print(f"{count:5d}  {language}")
    print(f"{len(files):5d}  total ({root})")
    if args.list:
        for f in files:
            print(f"  {f.language:10s} {f.file_path}")
    return 0


def cmd_parse(cfg: Config, args: argparse.Namespace) -> int:
    root = Path(args.path).resolve() if args.path else _resolve_checkout(cfg).path
    symbols = [s for f in repository_loader.load(root, cfg) for s in code_parser.parse(f)]
    by_type = collections.Counter(s.symbol_type for s in symbols)
    for symbol_type, count in sorted(by_type.items()):
        print(f"{count:5d}  {symbol_type}")
    print(f"{len(symbols):5d}  total")
    if args.list:
        for s in sorted(symbols, key=lambda s: (s.file_path, s.start_line)):
            print(
                f"  {s.symbol_type:8s} {s.file_path}:{s.start_line}-{s.end_line}  {s.qualified_name}"
            )
    return 0


def cmd_chunk(cfg: Config, args: argparse.Namespace) -> int:
    root = Path(args.path).resolve() if args.path else _resolve_checkout(cfg).path
    files = repository_loader.load(root, cfg)
    token_model = args.token_model if args.token_model is not None else cfg.model

    if args.stats:
        by_strategy = code_chunker.chunk_all_strategies(files, cfg)
        for index, strategy in enumerate(code_chunker.STRATEGIES):
            chunks = by_strategy[strategy]
            st = code_chunker.stats(chunks, strategy, model=token_model)
            if index == 0:
                print(
                    f"{'strategy':10s} {'chunks':>7s} {'mean tok':>9s} {'p95':>6s} {'max':>6s} {'trunc':>7s}"
                )
            print(
                f"{st.strategy:10s} {st.chunks:7d} {st.mean_tokens:9.1f} "
                f"{st.p95_tokens:6d} {st.max_tokens:6d} {st.truncation_rate:6.0%}"
            )
            if args.list:
                for c in chunks:
                    print(f"  {c.citation:50s} {c.symbol_type:8s} {c.symbol_name}")
    else:
        chunks = code_chunker.chunk(files, cfg, cfg.strategy)
        st = code_chunker.stats(chunks, cfg.strategy, model=token_model)
        print(
            f"{'strategy':10s} {'chunks':>7s} {'mean tok':>9s} {'p95':>6s} {'max':>6s} {'trunc':>7s}"
        )
        print(
            f"{st.strategy:10s} {st.chunks:7d} {st.mean_tokens:9.1f} "
            f"{st.p95_tokens:6d} {st.max_tokens:6d} {st.truncation_rate:6.0%}"
        )
        if args.list:
            for c in chunks:
                print(f"  {c.citation:50s} {c.symbol_type:8s} {c.symbol_name}")
    return 0


def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    root = Path(args.path).resolve() if args.path else _resolve_checkout(cfg).path
    files = repository_loader.load(root, cfg)
    chunks = code_chunker.chunk(files, cfg, cfg.strategy)

    embedder = Embedder(cfg)
    store = qdrant_store.QdrantStore(cfg)
    if args.drop and store.drop():
        print(f"dropped {store.collection}")

    started = time.perf_counter()
    # No dim is passed: sync takes it from the vectors, so a fully-cached
    # reindex never imports torch at all.
    report = store.sync(chunks, embedder.embed_batch)
    print(report)
    print(
        f"  embedded:  {embedder.encoded_count} (cache {embedder.cache.hits} hit / "
        f"{embedder.cache.misses} miss)\n"
        f"  elapsed:   {time.perf_counter() - started:.1f}s"
    )
    embedder.close()
    return 0


def _query_text(args: argparse.Namespace) -> str:
    # argparse treats a token containing a space as positional even when it
    # starts with a dash, so a quoting mistake turns "--language python" into
    # part of the query and silently returns plausible nonsense.
    query = " ".join(args.query)
    if flag := re.search(FLAG_IN_QUERY, query):
        raise ValueError(
            f"query contains what looks like a flag: {flag.group(1)!r} - check quoting"
        )
    return query


def _retrieve(cfg: Config, args: argparse.Namespace, query: str):
    """The shared search pipeline behind both `search` and `ask`."""
    searcher = VectorSearcher(cfg)
    query_filter = qdrant_store.make_filter(
        language=args.language, symbol_type=args.symbol_type, directory=args.directory
    )
    # Reranking and MMR both need a candidate pool deeper than what we show.
    two_stage = args.rerank or args.mmr
    result = searcher.search(
        query,
        top_k=cfg.rerank_top_n if two_stage else cfg.top_k,
        query_filter=query_filter,
        exact=args.exact,
        with_vectors=args.mmr,
    )

    hits, extra = result.hits, ""
    if args.mmr:
        hits = context_mod.mmr(hits, cfg.mmr_lambda, cfg.top_k)
        extra += f", mmr lambda={cfg.mmr_lambda}"
    if args.rerank:
        started = time.perf_counter()
        hits = CrossEncoderReranker(cfg).rerank(query, hits, cfg.top_k)
        extra += f", rerank {(time.perf_counter() - started) * 1000:.0f}ms of {len(result.hits)}"
    return searcher, hits[: cfg.top_k], result, extra


def _expander(cfg: Config) -> context_mod.ContextExpander:
    return context_mod.ContextExpander(repository_loader.load(_resolve_checkout(cfg).path, cfg))


def cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    query = _query_text(args)
    searcher, hits, result, extra = _retrieve(cfg, args, query)

    expander = _expander(cfg) if args.expand else None
    if not hits:
        print("no results")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:2d}. {hit.score:.4f}  {hit.citation}  {hit.symbol_type} {hit.qualified_name}")
        body = expander.expand(hit).text if expander else hit.snippet(args.snippet)
        for line in body.splitlines():
            print(f"      {line}")
    mode = "exact" if result.exact else "hnsw"
    print(
        f"\n{len(hits)} hits in {result.total_ms:.0f}ms "
        f"(embed {result.embed_ms:.0f}ms, {mode} search {result.search_ms:.0f}ms{extra})"
    )
    searcher.embedder.close()
    return 0


def cmd_ask(cfg: Config, args: argparse.Namespace) -> int:
    query = _query_text(args)
    searcher, hits, result, extra = _retrieve(cfg, args, query)

    generator = rag_generator.Generator(cfg, expander=_expander(cfg) if args.expand else None)
    if args.show_prompt:
        prompt = generator.prompt(query, hits)
        print(f"--- system ---\n{prompt.system}\n\n--- user ---\n{prompt.user}\n")
        searcher.embedder.close()
        return 0

    answer = generator.answer(query, hits, retrieval_ms=result.total_ms)
    print(answer.text)

    print("\nsources:")
    for rank, hit in enumerate(answer.hits, 1):
        print(f"  [{rank}] {hit.score:.4f}  {hit.citation}  {hit.symbol_type} {hit.qualified_name}")

    check = answer.citations
    flag = "ok" if check.ok else "SUSPECT"
    print(f"\ncitations: {flag} - {check.summary()}")
    if answer.abstained:
        print("model declared the context insufficient")
    print(
        f"{answer.provider}/{answer.model}, {answer.prompt_chars} prompt chars, "
        f"retrieval {answer.retrieval_ms:.0f}ms{extra}, generation {answer.generation_ms:.0f}ms"
    )
    searcher.embedder.close()
    return 0 if check.ok else 2


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    root = Path(args.path).resolve() if args.path else _resolve_checkout(cfg).path
    files = repository_loader.load(root, cfg)
    harness = runner.Harness.build(cfg, files)

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    if any(n.startswith(("vector", "hybrid")) for n in names):
        harness.store = qdrant_store.QdrantStore(cfg)
        harness.embedder = Embedder(cfg)
        if not harness.store.exists():
            raise RuntimeError(
                f"collection {harness.store.collection!r} missing; run `index` first"
            )
    if any(n.endswith("rerank") for n in names):
        harness.reranker = CrossEncoderReranker(cfg)

    gold = gold_mod.load_gold(Path(args.gold) if args.gold else cfg.gold_set)
    summaries = runner.evaluate(
        harness,
        gold,
        names,
        top_k=cfg.top_k,
        threshold=args.threshold,
        dump_dir=None if args.no_dump else runner.RESULTS_DIR,
    )

    print(f"{harness.cfg.strategy} strategy, {len(harness.chunks)} chunks, k={cfg.top_k}\n")
    print(metrics.Summary.HEADER)
    for summary in summaries:
        print(summary.row())
    if not args.no_dump:
        print(f"\nper-query results in {runner.RESULTS_DIR}")
    if harness.embedder is not None:
        harness.embedder.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared as a parent so every flag works both before and after the subcommand.
    # SUPPRESS keeps the subparser from overwriting a flag given before the subcommand.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--repo", help="https://github.com/<owner>/<name>")
    common.add_argument("--ref", help="branch or tag to clone (default HEAD)")
    common.add_argument("--max-file-bytes", type=int, dest="max_file_bytes")
    common.add_argument("--strategy", choices=code_chunker.STRATEGIES)
    common.add_argument("--chunk-lines", type=int, dest="chunk_lines")
    common.add_argument("--chunk-overlap", type=int, dest="chunk_overlap")
    common.add_argument("--model", help="sentence-transformers model name")
    common.add_argument("--batch-size", type=int, dest="batch_size")
    common.add_argument("--qdrant-url", dest="qdrant_url", help="http url, or :memory: for local")
    common.add_argument("--collection-suffix", dest="collection_suffix")
    common.add_argument("--top-k", type=int, dest="top_k")
    common.add_argument(
        "--vector-weight",
        type=float,
        dest="vector_weight",
        help="vector share in hybrid_weighted fusion (0-1)",
    )

    parser = argparse.ArgumentParser(prog="python -m src.cli", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "fetch", parents=[common], help="shallow clone the repo and pin its SHA"
    ).set_defaults(func=cmd_fetch)

    load = sub.add_parser("load", parents=[common], help="discover and read indexable files")
    load.add_argument("--path", help="load a local directory instead of a checkout")
    load.add_argument("--list", action="store_true", help="print every file")
    load.set_defaults(func=cmd_load)

    parse_cmd = sub.add_parser("parse", parents=[common], help="ast/heading parse into symbols")
    parse_cmd.add_argument("--path", help="parse a local directory instead of a checkout")
    parse_cmd.add_argument("--list", action="store_true", help="print every symbol")
    parse_cmd.set_defaults(func=cmd_parse)

    chunk_cmd = sub.add_parser("chunk", parents=[common], help="chunk and report token stats")
    chunk_cmd.add_argument("--path", help="chunk a local directory instead of a checkout")
    chunk_cmd.add_argument("--list", action="store_true", help="print every chunk")
    chunk_cmd.add_argument(
        "--stats", action="store_true", help="compare all strategies instead of just --strategy"
    )
    chunk_cmd.add_argument(
        "--token-model",
        dest="token_model",
        default=None,
        help="HF model for exact token counts (default: embedding model from config)",
    )
    chunk_cmd.set_defaults(func=cmd_chunk)

    index_cmd = sub.add_parser(
        "index", parents=[common], help="embed chunks and sync them into Qdrant"
    )
    index_cmd.add_argument("--path", help="index a local directory instead of a checkout")
    index_cmd.add_argument(
        "--drop", action="store_true", help="delete the collection first instead of syncing"
    )
    index_cmd.set_defaults(func=cmd_index)

    # `search` and `ask` run the identical retrieval pipeline; only the last
    # step differs, so the flags that shape retrieval live in one parent.
    retrieval = argparse.ArgumentParser(add_help=False)
    retrieval.add_argument("query", nargs="+")
    retrieval.add_argument("--language", help="filter: python, markdown, ...")
    retrieval.add_argument("--symbol-type", dest="symbol_type", help="filter: function, class, ...")
    retrieval.add_argument("--directory", help="filter: exact directory of the chunk's file")
    retrieval.add_argument(
        "--exact", action="store_true", help="brute-force search, the ANN recall baseline"
    )
    retrieval.add_argument(
        "--rerank", action="store_true", help=f"cross-encode the top {Config.rerank_top_n}"
    )
    retrieval.add_argument("--mmr", action="store_true", help="diversify with MMR")
    retrieval.add_argument(
        "--expand", action="store_true", help="add imports and parent class to each hit"
    )

    search_cmd = sub.add_parser(
        "search", parents=[common, retrieval], help="vector search the index"
    )
    search_cmd.add_argument("--snippet", type=int, default=3, help="lines of source per hit")
    search_cmd.set_defaults(func=cmd_search)

    ask_cmd = sub.add_parser(
        "ask", parents=[common, retrieval], help="retrieve, then answer with cited context"
    )
    ask_cmd.add_argument("--llm-provider", dest="llm_provider", choices=LLM_PROVIDERS)
    ask_cmd.add_argument("--llm-model", dest="llm_model", help="e.g. llama3.2, gpt-4o-mini")
    ask_cmd.add_argument("--llm-base-url", dest="llm_base_url")
    ask_cmd.add_argument(
        "--show-prompt",
        dest="show_prompt",
        action="store_true",
        help="print the assembled prompt and exit without calling a model",
    )
    ask_cmd.set_defaults(func=cmd_ask)

    eval_cmd = sub.add_parser(
        "eval", parents=[common], help="score every retriever against the gold set"
    )
    eval_cmd.add_argument("--path", help="evaluate a local directory instead of a checkout")
    eval_cmd.add_argument("--gold", help="path to a gold set json")
    # Named --retrievers, not --strategies: --strategy already means the
    # chunking strategy everywhere else in this CLI.
    eval_cmd.add_argument(
        "--retrievers",
        default=",".join(runner.RETRIEVERS),
        help=f"comma-separated subset of {','.join(runner.RETRIEVERS)}",
    )
    eval_cmd.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="score above which an answer counts as confident, for false-confidence",
    )
    eval_cmd.add_argument("--no-dump", action="store_true", help="skip writing eval/results/*.json")
    eval_cmd.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = Config().override(**vars(args))
    try:
        cfg.validate()
        return args.func(cfg, args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
