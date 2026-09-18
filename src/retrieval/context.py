"""Post-retrieval result shaping: expand each hit, and stop the list repeating itself.

Two different problems, both of which appear only *after* retrieval works:

1. A function-level chunk is precise but decontextualized. `def send(self, url)`
   retrieved alone does not say which class it belongs to, what `self` has, or
   that the file imports `httpx`. Context expansion puts that back.
2. Function-level chunking makes the top 5 collapse into five methods of one
   class — all correct, all redundant, and between them they answer one question
   instead of five. MMR and a per-file cap trade a little relevance for coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..parser import code_parser
from ..parser.repository_loader import SourceFile
from ..vector_store.qdrant_store import Hit

MAX_SIBLINGS = 12


@dataclass(frozen=True)
class Expanded:
    hit: Hit
    imports: tuple[str, ...]
    class_signature: str
    siblings: tuple[str, ...]

    @property
    def text(self) -> str:
        """The hit's source, prefixed by just enough of its surroundings to read it."""
        blocks = []
        if self.imports:
            blocks.append("\n".join(self.imports))
        if self.class_signature:
            header = self.class_signature
            if self.siblings:
                header += "\n    # other methods: " + ", ".join(self.siblings)
            blocks.append(header)
        blocks.append(self.hit.text)
        return "\n\n".join(blocks)


class ContextExpander:
    """Needs the corpus: a Hit knows its line span, not what surrounds it."""

    def __init__(self, files: Sequence[SourceFile], max_imports: int = 10):
        self._files = {f.file_path: f for f in files}
        self.max_imports = max_imports

    def expand(self, hit: Hit) -> Expanded:
        src = self._files.get(hit.file_path)
        if src is None:
            return Expanded(hit, (), "", ())

        symbols = code_parser.parse(src)
        lines = src.content.splitlines()

        imports = symbols[0].imports[: self.max_imports] if symbols else ()

        signature, siblings = "", ()
        if hit.class_name:
            parent = next(
                (
                    s
                    for s in symbols
                    if s.symbol_type == "class" and s.symbol_name == hit.class_name
                ),
                None,
            )
            if parent is not None:
                signature = _signature(lines, parent)
                siblings = tuple(
                    s.symbol_name
                    for s in symbols
                    if s.symbol_type == "method"
                    and s.class_name == hit.class_name
                    and s.symbol_name != hit.symbol_name
                )[:MAX_SIBLINGS]
        return Expanded(hit, imports, signature, siblings)


def _signature(lines: list[str], symbol) -> str:
    """The `class Foo(Base):` line plus its docstring, never the whole body.

    Including the body would defeat the purpose: the reason we retrieve one
    method rather than the class is that the class does not fit in the context
    window.
    """
    head = lines[symbol.start_line - 1] if symbol.start_line <= len(lines) else ""
    if not symbol.docstring:
        return head
    summary = symbol.docstring.strip().splitlines()[0]
    return f'{head}\n    """{summary}"""'


# --- diversity ---------------------------------------------------------------


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def mmr(hits: Sequence[Hit], lambda_: float = 0.7, top_k: int = 5) -> list[Hit]:
    """Maximal Marginal Relevance: greedily pick the best *marginal* addition.

    Each step maximizes `lambda*relevance - (1-lambda)*max_similarity_to_already_picked`,
    so a chunk nearly identical to one already selected has to be much more
    relevant to earn its slot. Requires vectors on the hits (`with_vectors=True`);
    without them there is nothing to measure redundancy with, so the input order
    is returned untouched rather than silently pretending to diversify.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lambda_}")
    if not hits or not all(h.vector for h in hits):
        return list(hits[:top_k])

    remaining = list(hits)
    selected = [remaining.pop(0)]  # the top hit is always kept
    while remaining and len(selected) < top_k:
        best_index, best_value = 0, float("-inf")
        for index, candidate in enumerate(remaining):
            redundancy = max(dot(candidate.vector, s.vector) for s in selected)
            value = lambda_ * candidate.score - (1 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        selected.append(remaining.pop(best_index))
    return selected


def per_file_cap(hits: Sequence[Hit], cap: int, top_k: int = 5) -> list[Hit]:
    """Keep at most `cap` hits per file, preserving order. The zero-cost MMR.

    Needs no vectors and no tuning, and on a corpus where redundancy means "five
    methods of one class" it does most of MMR's job.
    """
    if cap <= 0:
        return list(hits[:top_k])
    seen: dict[str, int] = {}
    out = []
    for hit in hits:
        if seen.get(hit.file_path, 0) >= cap:
            continue
        seen[hit.file_path] = seen.get(hit.file_path, 0) + 1
        out.append(hit)
        if len(out) == top_k:
            break
    return out
