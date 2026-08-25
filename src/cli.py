"""argparse entry point: python -m src.cli <command> [flags]."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .config import Config
from .parser import repo_fetcher, repository_loader


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


def build_parser() -> argparse.ArgumentParser:
    # Shared as a parent so every flag works both before and after the subcommand.
    # SUPPRESS keeps the subparser from overwriting a flag given before the subcommand.
    common = argparse.ArgumentParser(
        add_help=False, argument_default=argparse.SUPPRESS
    )
    common.add_argument("--repo", help="https://github.com/<owner>/<name>")
    common.add_argument("--ref", help="branch or tag to clone (default HEAD)")
    common.add_argument("--max-file-bytes", type=int, dest="max_file_bytes")

    parser = argparse.ArgumentParser(prog="python -m src.cli", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "fetch", parents=[common], help="shallow clone the repo and pin its SHA"
    ).set_defaults(func=cmd_fetch)

    load = sub.add_parser(
        "load", parents=[common], help="discover and read indexable files"
    )
    load.add_argument("--path", help="load a local directory instead of a checkout")
    load.add_argument("--list", action="store_true", help="print every file")
    load.set_defaults(func=cmd_load)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config().override(**vars(args))
    try:
        return args.func(cfg, args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
