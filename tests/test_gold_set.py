"""Guards the gold sets against rot: every label must name a chunk that exists."""

import pytest

from src.chunking import code_chunker
from src.config import ROOT, Config
from src.eval.gold import build_name_index, load_gold
from src.parser import repository_loader

GOLD_SETS = {
    "opssense": ROOT / "eval" / "opssense.json",
    "httpx": ROOT / "eval" / "httpx.json",
}


@pytest.fixture(scope="module")
def corpora():
    """Chunk each labelled repo once, under every strategy a label could target."""
    out = {}
    for name, path in GOLD_SETS.items():
        gold = load_gold(path)
        cfg = Config(repo=gold["repo"])
        owner, repo_name = gold["repo"].rstrip("/").split("/")[-2:]
        checkouts = sorted(cfg.repos_dir.glob(f"{owner}__{repo_name}@*"))
        if not checkouts:
            continue
        files = repository_loader.load(checkouts[0], cfg)
        names: set[tuple[str, str]] = set()
        for strategy in ("file", "class", "function"):
            names |= build_name_index(code_chunker.chunk(files, cfg, strategy))
        out[name] = (
            checkouts[0].name.split("@", 1)[1],
            gold,
            {f.file_path for f in files},
            names,
        )
    return out


def _corpus(corpora, name):
    if name not in corpora:
        pytest.skip(f"no {name} checkout; run `python -m src.cli fetch --repo ...` first")
    return corpora[name]


@pytest.mark.parametrize("name", list(GOLD_SETS))
def test_gold_sha_matches_checkout(corpora, name):
    checkout_sha, gold, _, _ = _corpus(corpora, name)
    assert checkout_sha.startswith(gold["sha"]), (
        f"{name} gold set pinned to {gold['sha']}, checkout is {checkout_sha} - re-label or re-fetch"
    )


@pytest.mark.parametrize("name", list(GOLD_SETS))
def test_every_gold_label_resolves(corpora, name):
    _, gold, paths, names = _corpus(corpora, name)
    missing = [
        (q["id"], g)
        for q in gold["queries"]
        for g in q["gold"]
        if g["file"] not in paths or (g["symbol"] and (g["file"], g["symbol"]) not in names)
    ]
    assert not missing


@pytest.mark.parametrize("name", list(GOLD_SETS))
def test_gold_set_shape(corpora, name):
    _, gold, _, _ = _corpus(corpora, name)
    queries = gold["queries"]
    assert len({q["id"] for q in queries}) == len(queries)
    assert sum(1 for q in queries if q["kind"] == "unanswerable") >= 4
    assert all(q["gold"] or q["kind"] == "unanswerable" for q in queries)
    assert all(g["grade"] in (1, 2) for q in queries for g in q["gold"])


def test_opssense_covers_the_query_kinds_it_was_designed_for(corpora):
    _, gold, _, _ = _corpus(corpora, "opssense")
    kinds = [q["kind"] for q in gold["queries"]]
    assert sum(1 for k in kinds if k == "lexical-gap") >= 5
