import pytest

from src.chunking import code_chunker
from src.chunking.code_chunker import (
    STRATEGIES,
    chunk_file,
    chunk_file_all_strategies,
    count_tokens,
    count_tokens_batch,
    split_identifier,
    stats,
)
from src.config import Config
from src.parser.repository_loader import SourceFile

PY = '''"""Module doc."""

import os
from typing import Any


def top_level(x):
    """Adds one."""
    def helper():
        return 1
    return x + helper()


class Client:
    """A client."""

    @property
    def base_url(self) -> str:
        return "/"

    async def _shouldRetry(self, resp: Any) -> bool:
        return resp.status >= 500


class Empty:
    pass
'''

MD = """# Title

intro

## Section A

```bash
# not a heading
```

body a

## Section B

body b
"""


def src(content: str, path: str = "pkg/client.py", language: str = "python") -> SourceFile:
    return SourceFile(file_path=path, language=language, content=content, sha="0" * 64)


def test_split_identifier():
    assert split_identifier("_shouldRetry_HTTPRequest2") == [
        "should",
        "Retry",
        "HTTP",
        "Request",
        "2",
    ]


def test_function_strategy_prefers_methods_but_keeps_empty_classes():
    names = {c.symbol_name for c in chunk_file(src(PY), Config(), "function")}
    assert {"top_level", "base_url", "_shouldRetry", "Empty"} == names
    assert "Client" not in names


def test_class_strategy_keeps_classes_whole():
    names = {c.symbol_name for c in chunk_file(src(PY), Config(), "class")}
    assert names == {"top_level", "Client", "Empty"}


def test_embed_text_enriches_and_text_stays_verbatim():
    chunk = next(
        c for c in chunk_file(src(PY), Config(), "function") if c.symbol_name == "_shouldRetry"
    )
    assert chunk.text.lstrip().startswith("async def _shouldRetry")
    assert "should retry" in chunk.embed_text
    assert "pkg / client.py" in chunk.embed_text
    assert chunk.embed_text.endswith(chunk.text)
    assert chunk.citation == f"pkg/client.py:{chunk.start_line}-{chunk.end_line}"


def test_markdown_embed_text_keeps_section_path_verbatim():
    chunk = next(
        c
        for c in chunk_file(src(MD, "README.md", "markdown"), Config(), "function")
        if c.symbol_name == "Section A"
    )
    assert "Title" in chunk.embed_text
    assert "Section A" in chunk.embed_text


def test_symbolless_file_still_yields_one_chunk():
    chunks = chunk_file(src("A = 1\nB = 2\n", "pkg/config.py"), Config(), "function")
    assert [c.symbol_type for c in chunks] == ["module"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_covers_the_file(strategy):
    chunks = chunk_file(src(PY), Config(chunk_lines=10, chunk_overlap=2), strategy)
    assert chunks
    assert all(1 <= c.start_line <= c.end_line <= len(PY.splitlines()) for c in chunks)
    assert all(c.strategy == strategy for c in chunks)


def test_all_strategies_match_single_pass():
    source = src(PY)
    cfg = Config(chunk_lines=10, chunk_overlap=2)
    combined = chunk_file_all_strategies(source, cfg)
    for strategy in STRATEGIES:
        assert combined[strategy] == chunk_file(source, cfg, strategy)


def test_fixed_windows_overlap():
    chunks = chunk_file(src(PY), Config(chunk_lines=10, chunk_overlap=4), "fixed")
    assert [c.start_line for c in chunks][:3] == [1, 7, 13]


@pytest.mark.parametrize("lines,overlap", [(10, 10), (10, 12), (0, 0), (10, -1)])
def test_fixed_rejects_non_advancing_windows(lines, overlap):
    with pytest.raises(ValueError):
        chunk_file(src(PY), Config(chunk_lines=lines, chunk_overlap=overlap), "fixed")


def test_config_validate_rejects_bad_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        Config(strategy="semantic").validate()


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        chunk_file(src(PY), Config(strategy="semantic"), "semantic")


def test_stats_counts_truncation():
    st = stats(chunk_file(src(PY), Config(), "file"), "file", max_seq_length=5)
    assert st.chunks == 1 and st.truncation_rate == 1.0
    assert stats([], "file").chunks == 0


def test_token_counting():
    assert count_tokens("") == 0
    assert count_tokens_batch([]) == []
    assert count_tokens_batch(["a", "hello world"]) == [
        count_tokens("a"),
        count_tokens("hello world"),
    ]
    assert count_tokens("a" * 40) > count_tokens("a" * 4)


def test_unknown_token_model_warns_once_then_estimates(caplog):
    with caplog.at_level("WARNING"):
        counts = count_tokens_batch(["hello"] * 3, "definitely/not-a-model")
    assert counts == [count_tokens("hello")] * 3
    assert len([r for r in caplog.records if "no tokenizer" in r.message]) == 1


def test_stats_defaults_to_config_model(monkeypatch):
    seen: list[str] = []

    def fake_batch(texts, model=""):
        seen.append(model)
        return [1] * len(texts)

    monkeypatch.setattr(code_chunker, "count_tokens_batch", fake_batch)
    chunks = chunk_file(src(PY), Config(model="my-model"), "file")
    stats(chunks, "file", model=Config().model)
    assert seen == ["sentence-transformers/all-MiniLM-L6-v2"]
