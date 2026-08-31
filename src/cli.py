"""argparse entry point: python -m src.cli <command> [flags]."""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

from .chunking import code_chunker
from .config import Config
from .parser import code_parser, repo_fetcher, repository_loader


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
