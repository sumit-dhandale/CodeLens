"""Four chunking strategies behind one signature.

Every chunk carries two texts:

- `text` is verbatim source, used for display and citation.
- `embed_text` is enriched with the relative path, the owning class, the symbol
  name split on snake_case and camelCase, and the docstring, before the body.

The enrichment is the whole trick. An English query like "retry a failed request"
has no lexical overlap with `_shouldRetry`, but it does overlap with the enriched
form "should retry", so the embedding lands much closer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import CHUNK_STRATEGIES, Config
from ..parser import code_parser
from ..parser.repository_loader import SourceFile
from .ids import point_id as make_point_id
from .ids import point_key as make_point_key

log = logging.getLogger(__name__)

STRATEGIES = CHUNK_STRATEGIES

# snake_case, camelCase, and ACRONYMBoundaries, plus bare digits.
_IDENT_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")
_WORDISH = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


@dataclass(frozen=True)
class Chunk:
    file_path: str
    language: str
    strategy: str
    symbol_type: str
    symbol_name: str
    class_name: str
    start_line: int
    end_line: int
    text: str
    embed_text: str
    file_sha: str

    @property
    def directory(self) -> str:
        return self.file_path.rsplit("/", 1)[0] if "/" in self.file_path else ""

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def point_key(self) -> str:
        return make_point_key(
            self.file_path, self.class_name, self.symbol_name, self.start_line, self.strategy
        )

    @property
    def point_id(self):
        return make_point_id(
            self.file_path, self.class_name, self.symbol_name, self.start_line, self.strategy
        )


def split_identifier(name: str) -> list[str]:
    """`_shouldRetry_HTTPRequest2` -> ['should', 'Retry', 'HTTP', 'Request', '2']."""
    return _IDENT_PART.findall(name)


def _embed_text(sym: code_parser.Symbol, body: str) -> str:
    words = split_identifier(sym.symbol_name)
    header = [
        sym.file_path.replace("/", " / ").replace("_", " "),
        f"{sym.language} {sym.symbol_type} {sym.qualified_name}",
    ]
    if sym.class_name:
        header.append(sym.class_name)
    if words:
        header.append(" ".join(words).lower())
    if sym.docstring:
        header.append(sym.docstring.strip())
    return "\n".join([*header, "", body])


def _chunk(sym: code_parser.Symbol, src: SourceFile, lines: list[str], strategy: str) -> Chunk:
    text = "\n".join(lines[sym.start_line - 1 : sym.end_line])
    return Chunk(
        file_path=sym.file_path,
        language=sym.language,
        strategy=strategy,
        symbol_type=sym.symbol_type,
        symbol_name=sym.symbol_name,
        class_name=sym.class_name,
        start_line=sym.start_line,
        end_line=sym.end_line,
        text=text,
        embed_text=_embed_text(sym, text),
        file_sha=src.sha,
    )


def _prepare(src: SourceFile) -> tuple[list[str], tuple[code_parser.Symbol, ...]]:
    return src.content.splitlines() or [""], code_parser.parse(src)


def _wanted_symbols(
    strategy: str, module: code_parser.Symbol, rest: tuple[code_parser.Symbol, ...]
) -> list[code_parser.Symbol]:
    if strategy == "file":
        return [module]
    if strategy == "class":
        wanted = [s for s in rest if s.symbol_type in ("class", "function", "section")]
    else:
        wanted = [s for s in rest if s.symbol_type in ("function", "method", "section")]
        covered = {s.class_name for s in wanted if s.symbol_type == "method"}
        wanted += [s for s in rest if s.symbol_type == "class" and s.symbol_name not in covered]
    if not wanted:
        return [module]
    wanted.sort(key=lambda s: s.start_line)
    return wanted


def _chunk_symbols(
    src: SourceFile,
    lines: list[str],
    symbols: tuple[code_parser.Symbol, ...],
    strategy: str,
) -> list[Chunk]:
    module, rest = symbols[0], symbols[1:]
    return [_chunk(s, src, lines, strategy) for s in _wanted_symbols(strategy, module, rest)]


def _fixed(src: SourceFile, lines: list[str], cfg: Config) -> list[Chunk]:
    step = cfg.chunk_lines - cfg.chunk_overlap
    out = []
    for start in range(0, len(lines), step):
        window = lines[start : start + cfg.chunk_lines]
        if any(line.strip() for line in window):
            sym = code_parser.Symbol(
                file_path=src.file_path,
                language=src.language,
                symbol_type="window",
                symbol_name=f"L{start + 1}",
                start_line=start + 1,
                end_line=start + len(window),
            )
            out.append(_chunk(sym, src, lines, "fixed"))
        if start + cfg.chunk_lines >= len(lines):
            break
    return out


def chunk_file(src: SourceFile, cfg: Config, strategy: str | None = None) -> list[Chunk]:
    cfg.validate()
    strategy = strategy or cfg.strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    lines, symbols = _prepare(src)
    if strategy == "fixed":
        return _fixed(src, lines, cfg)
    return _chunk_symbols(src, lines, symbols, strategy)


def chunk_file_all_strategies(src: SourceFile, cfg: Config) -> dict[str, list[Chunk]]:
    """Parse and split once, then emit all four strategy views."""
    cfg.validate()
    lines, symbols = _prepare(src)
    return {
        "file": _chunk_symbols(src, lines, symbols, "file"),
        "class": _chunk_symbols(src, lines, symbols, "class"),
        "function": _chunk_symbols(src, lines, symbols, "function"),
        "fixed": _fixed(src, lines, cfg),
    }


def chunk(files: list[SourceFile], cfg: Config, strategy: str | None = None) -> list[Chunk]:
    return [c for f in files for c in chunk_file(f, cfg, strategy)]


def chunk_all_strategies(files: list[SourceFile], cfg: Config) -> dict[str, list[Chunk]]:
    out = {s: [] for s in STRATEGIES}
    for f in files:
        for strategy, chunks in chunk_file_all_strategies(f, cfg).items():
            out[strategy].extend(chunks)
    return out


# --- token accounting -------------------------------------------------------
# MiniLM truncates at 256 wordpieces. A file-level chunk blows through that
# silently: the vector then represents the first third of the file and nothing
# else, which is most of the explanation for the chunking experiment's results.


@lru_cache(maxsize=4)
def _hf_tokenizer(model: str):
    """None when transformers is missing or the model name does not resolve.

    Cached on the failure too, so a bad name warns once instead of once per chunk.
    """
    try:
        from transformers import AutoTokenizer  # heavy, and optional

        return AutoTokenizer.from_pretrained(model)
    except Exception as exc:
        log.warning("no tokenizer for %r (%s); falling back to estimates", model, exc)
        return None


def _estimate(text: str) -> int:
    # ponytail: heuristic worth roughly +/-10% of the real wordpiece count. Good
    # enough to rank strategies; pass a model to get exact numbers. Integer
    # arithmetic rather than math.ceil because this runs per word per chunk.
    return sum((len(w) + 3) >> 2 for w in _WORDISH.findall(text))


def count_tokens(text: str, model: str = "") -> int:
    """Wordpiece count, exact when transformers is installed and estimated otherwise."""
    return count_tokens_batch([text], model)[0]


def count_tokens_batch(texts: list[str], model: str = "") -> list[int]:
    """Batched because a tokenizer call per chunk is dominated by Python overhead."""
    if not texts:
        return []
    if model and (tokenizer := _hf_tokenizer(model)) is not None:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]
    return [_estimate(t) for t in texts]


@dataclass(frozen=True)
class Stats:
    strategy: str
    chunks: int
    mean_tokens: float
    p95_tokens: int
    max_tokens: int
    truncated: int
    max_seq_length: int

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.chunks if self.chunks else 0.0


def stats(chunks: list[Chunk], strategy: str, max_seq_length: int = 256, model: str = "") -> Stats:
    counts = sorted(count_tokens_batch([c.embed_text for c in chunks], model))
    if not counts:
        return Stats(strategy, 0, 0.0, 0, 0, 0, max_seq_length)
    return Stats(
        strategy=strategy,
        chunks=len(counts),
        mean_tokens=sum(counts) / len(counts),
        p95_tokens=counts[min(int(0.95 * len(counts)), len(counts) - 1)],
        max_tokens=counts[-1],
        truncated=sum(1 for n in counts if n > max_seq_length),
        max_seq_length=max_seq_length,
    )
