from src.chunking import code_chunker
from src.chunking.ids import point_id, point_key
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

DUP = """def foo():
    return 1

def foo():
    return 2
"""


def src(content: str, path: str = "pkg/client.py", language: str = "python") -> SourceFile:
    return SourceFile(file_path=path, language=language, content=content, sha="0" * 64)


def test_point_key_includes_start_line():
    chunks = code_chunker.chunk_file(src(DUP), Config(), "function")
    foo = [c for c in chunks if c.symbol_name == "foo"]
    assert len(foo) == 2
    assert foo[0].point_key != foo[1].point_key
    assert foo[0].point_id != foo[1].point_id


def test_point_id_is_stable_uuid():
    chunk = code_chunker.chunk_file(src(PY), Config(), "function")[0]
    assert (
        point_id(
            chunk.file_path,
            chunk.class_name,
            chunk.symbol_name,
            chunk.start_line,
            chunk.strategy,
        )
        == chunk.point_id
    )
    assert str(chunk.point_id) == str(
        point_id(
            chunk.file_path,
            chunk.class_name,
            chunk.symbol_name,
            chunk.start_line,
            chunk.strategy,
        )
    )
    assert chunk.point_key == point_key(
        chunk.file_path,
        chunk.class_name,
        chunk.symbol_name,
        chunk.start_line,
        chunk.strategy,
    )
