"""Guards the gold set against rot: every label must name a chunk that exists."""

import pytest

from src.chunking import code_chunker
from src.config import ROOT, Config
from src.eval.gold import build_name_index, load_gold
from src.parser import repository_loader

GOLD = ROOT / "eval" / "opssense.json"


@pytest.fixture(scope="module")
def corpus():
    cfg = Config()
    checkouts = sorted(cfg.repos_dir.glob("*OpsSense@*"))
    if not checkouts:
        pytest.skip("no OpsSense checkout; run `python -m src.cli fetch` first")
    checkout = checkouts[0]
    files = repository_loader.load(checkout, cfg)
    names: set[tuple[str, str]] = set()
    for strategy in ("file", "class", "function"):
        names |= build_name_index(code_chunker.chunk(files, cfg, strategy))
    return checkout.name.split("@", 1)[1], {f.file_path for f in files}, names


def test_gold_sha_matches_checkout(corpus):
    checkout_sha, _, _ = corpus
    gold = load_gold(GOLD)
    pinned = gold["sha"]
    assert checkout_sha.startswith(pinned), (
        f"gold set pinned to {pinned}, checkout is {checkout_sha} — re-label or re-fetch"
    )


def test_every_gold_label_resolves(corpus):
    _, paths, names = corpus
    gold = load_gold(GOLD)
    missing = [
        (q["id"], g)
        for q in gold["queries"]
        for g in q["gold"]
        if g["file"] not in paths or (g["symbol"] and (g["file"], g["symbol"]) not in names)
    ]
    assert not missing


def test_gold_set_shape():
    gold = load_gold(GOLD)
    queries = gold["queries"]
    assert len({q["id"] for q in queries}) == len(queries)
    assert sum(1 for q in queries if q["kind"] == "unanswerable") >= 4
    assert sum(1 for q in queries if q["kind"] == "lexical-gap") >= 5
    assert all(q["gold"] or q["kind"] == "unanswerable" for q in queries)
    assert all(g["grade"] in (1, 2) for q in queries for g in q["gold"])
