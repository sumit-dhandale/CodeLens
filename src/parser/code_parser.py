"""Turn a SourceFile into Symbol records.

Python goes through stdlib `ast`, which reports an exact `end_lineno` for every
node — no brace counting, no regex, no tree-sitter dependency. Markdown is split
on ATX headings. Every other language falls back to a single module symbol, which
still makes it retrievable via the file and fixed chunking strategies.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache

from .repository_loader import SourceFile

# Nodes we treat as callables. Async variants carry the same attributes.
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

_HEADING = re.compile(r"\A(#{1,6})\s+(.+?)\s*#*\s*\Z")
_FENCE = re.compile(r"\A\s*(```|~~~)")


@dataclass(frozen=True)
class Symbol:
    file_path: str
    language: str
    symbol_type: str  # module | class | function | method | section
    symbol_name: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    class_name: str = ""  # owning class for methods, heading path for sections
    docstring: str = ""
    imports: tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return f"{self.class_name}.{self.symbol_name}" if self.class_name else self.symbol_name


def _line_count(content: str) -> int:
    return len(content.splitlines()) or 1


def _module_symbol(src: SourceFile, docstring: str = "", imports: tuple[str, ...] = ()) -> Symbol:
    return Symbol(
        file_path=src.file_path,
        language=src.language,
        symbol_type="module",
        symbol_name=src.file_path.rsplit("/", 1)[-1],
        start_line=1,
        end_line=_line_count(src.content),
        docstring=docstring,
        imports=imports,
    )


def _import_lines(tree: ast.Module) -> tuple[str, ...]:
    """Module-level imports, rendered back to source-ish text.

    Used both as retrieval context (an httpx import is a strong hint about what a
    file does) and later by context expansion, which shows a retrieved method
    alongside the imports it depends on.
    """
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.append("import " + ", ".join(a.asname or a.name for a in node.names))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            out.append(f"from {module} import " + ", ".join(a.name for a in node.names))
    return tuple(out)


def _start_line(node: ast.AST) -> int:
    """Include decorators: ast reports `lineno` at `def`, so `@retry` would be cut off."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def parse_python(src: SourceFile) -> tuple[Symbol, ...]:
    try:
        tree = ast.parse(src.content, filename=src.file_path)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # Third-party source is untrusted input: a py2 file raises SyntaxError, a
        # deeply nested literal raises RecursionError, and neither should abort a
        # run over thousands of files. An unparseable file is still indexed whole.
        return (_module_symbol(src),)

    imports = _import_lines(tree)
    symbols = [_module_symbol(src, ast.get_docstring(tree) or "", imports)]

    def emit(node, symbol_type: str, class_name: str) -> None:
        symbols.append(
            Symbol(
                file_path=src.file_path,
                language=src.language,
                symbol_type=symbol_type,
                symbol_name=node.name,
                start_line=_start_line(node),
                end_line=node.end_lineno or node.lineno,
                class_name=class_name,
                docstring=ast.get_docstring(node) or "",
                imports=imports,
            )
        )

    # Deliberately two levels deep, not ast.walk: a nested closure would otherwise
    # become a chunk that overlaps its parent, double-counting the same lines.
    for node in tree.body:
        if isinstance(node, _FUNC_NODES):
            emit(node, "function", "")
        elif isinstance(node, ast.ClassDef):
            emit(node, "class", "")
            for child in node.body:
                if isinstance(child, _FUNC_NODES):
                    emit(child, "method", node.name)
    return tuple(symbols)


def parse_markdown(src: SourceFile) -> tuple[Symbol, ...]:
    """Split on ATX headings; a section runs until the next heading of any level.

    Headings inside fenced code blocks are ignored — `# comment` in a bash block
    is not a section boundary.
    """
    lines = src.content.splitlines()
    symbols = [_module_symbol(src)]

    starts: list[tuple[int, int, str]] = []  # (line_no, level, title)
    fence: str | None = None
    for i, line in enumerate(lines, start=1):
        match = _FENCE.match(line)
        if match:
            token = match.group(1)
            fence = None if fence == token else (fence or token)
            continue
        if fence:
            continue
        heading = _HEADING.match(line)
        if heading:
            starts.append((i, len(heading.group(1)), heading.group(2)))

    path: list[str] = []  # ancestor headings, so a section knows where it sits
    for index, (line_no, level, title) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        del path[level - 1 :]
        symbols.append(
            Symbol(
                file_path=src.file_path,
                language=src.language,
                symbol_type="section",
                symbol_name=title,
                start_line=line_no,
                end_line=max(end, line_no),
                class_name=" > ".join(path),
            )
        )
        path.append(title)
    return tuple(symbols)


# The chunker parses the same file once per strategy, so a comparison run over
# four strategies re-runs ast.parse four times. SourceFile is frozen and the
# caller already holds every file in memory, so caching costs no extra bytes.
# Call parse.cache_clear() after mutating a SourceFile in place in a long-lived
# process; tests build a fresh frozen SourceFile per case so they never need this.
@lru_cache(maxsize=1024)
def parse(src: SourceFile) -> tuple[Symbol, ...]:
    """Symbols for one file. The first entry is always the whole-file module symbol.

    Returns a tuple because the result is shared between callers via the cache.
    """
    if src.language == "python":
        return parse_python(src)
    if src.language == "markdown":
        return parse_markdown(src)
    return (_module_symbol(src),)
