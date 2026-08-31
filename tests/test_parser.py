
from src.parser.code_parser import parse
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


def by_name(symbols):
    return {s.symbol_name: s for s in symbols}


def test_python_symbols():
    symbols = by_name(parse(src(PY)))
    assert symbols["client.py"].symbol_type == "module"
    assert symbols["client.py"].imports == ("import os", "from typing import Any")
    assert symbols["top_level"].symbol_type == "function"
    assert symbols["top_level"].docstring == "Adds one."
    assert symbols["_shouldRetry"].symbol_type == "method"
    assert symbols["_shouldRetry"].class_name == "Client"
    assert "helper" not in symbols, "nested closures must stay inside their parent chunk"


def test_decorator_included_in_span():
    sym = by_name(parse(src(PY)))["base_url"]
    body = "\n".join(PY.splitlines()[sym.start_line - 1 : sym.end_line])
    assert body.lstrip().startswith("@property")


def test_syntax_error_falls_back_to_whole_file():
    symbols = parse(src("def broken(:\n"))
    assert [s.symbol_type for s in symbols] == ["module"]


def test_pathological_nesting_does_not_abort_the_run():
    symbols = parse(src("x = " + "[" * 20_000 + "]" * 20_000, "pkg/deep.py"))
    assert [s.symbol_type for s in symbols] == ["module"]


def test_parse_is_cached_and_immutable():
    source = src(PY)
    assert parse(source) is parse(source)
    assert isinstance(parse(source), tuple)


def test_markdown_sections_skip_fenced_headings():
    sections = [s for s in parse(src(MD, "README.md", "markdown")) if s.symbol_type == "section"]
    assert [s.symbol_name for s in sections] == ["Title", "Section A", "Section B"]
    assert sections[1].class_name == "Title"
    body = "\n".join(MD.splitlines()[sections[1].start_line - 1 : sections[1].end_line])
    assert "not a heading" in body and "body b" not in body
