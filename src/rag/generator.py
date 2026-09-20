"""Retrieval-augmented answering, with the citations checked afterwards.

The generator does three things and deliberately no more:

1. Formats the retrieved hits as numbered blocks headed by `path:start-end`, so
   the model has a citation key it can copy rather than compose. A model asked
   to "cite the file" invents plausible paths; a model asked to copy `[3]` back
   usually copies it.
2. Sends the prompt verbatim - the one from the project brief, unedited, so a
   change in answer quality is attributable to retrieval rather than to prompt
   tinkering.
3. Validates every citation in the answer against the set that was actually
   retrieved. This is the part that turns "the answer looks right" into a
   measurement. An LLM that cites `src/core/retry.py:40-58` when no such file
   was retrieved has hallucinated, and we can say so mechanically, without
   reading the code.

Post-validation cannot make the model honest. It can only make dishonesty
visible, which is the difference between a demo and a system.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import LANGUAGES, Config
from ..retrieval.context import ContextExpander
from ..vector_store.qdrant_store import Hit
from . import llm as llm_mod

# From the project brief, unchanged. Edit this and the numbers in docs/15 stop
# describing this prompt.
SYSTEM_PROMPT = """You are a codebase analysis assistant.

Answer the user's question using ONLY the retrieved code context.

For every important claim:
- identify the file
- identify the class/function
- provide line numbers when available

Do not invent implementation details.

If the retrieved code is insufficient to answer the question,
explicitly say that the available context is insufficient.

Distinguish between:
1. Facts directly visible in the code.
2. Reasonable inferences.

Prefer concise technical explanations."""

# The ablation: what a RAG prompt looks like when nobody has thought about
# grounding yet. docs/15 measures the difference between these two strings.
NAIVE_PROMPT = """You are a helpful coding assistant. Answer the user's question
about this codebase. Cite the files and line numbers you used."""

# `path/to/file.py:12-40`, with the line range optional.
CITATION_RE = re.compile(r"(?P<path>[\w./-]+\.[A-Za-z][A-Za-z0-9]{0,7}):(?P<start>\d+)(?:\s*[-\u2013]\s*(?P<end>\d+))?")
# A bare path or filename with no line numbers. Restricted to extensions we
# actually index, so ordinary prose about "setup.py" in a docstring does not
# register as a citation.
PATH_RE = re.compile(
    r"(?:[\w.-]+/)*[\w-]+\.(?:" + "|".join(e.lstrip(".") for e in LANGUAGES) + r")\b"
)
# A reference back to a numbered block: `[3]`.
REF_RE = re.compile(r"\[(\d+)\]")

ABSTENTION_MARKERS = ("insufficient", "not enough context", "cannot answer", "does not contain")


@dataclass(frozen=True)
class CitationCheck:
    """Which references in the answer the retrieved set can back up."""

    grounded: tuple[str, ...]  # exact `path:start-end` that was retrieved
    drifted: tuple[str, ...]  # right file, line span the model made up
    invented: tuple[str, ...]  # file never retrieved
    bad_refs: tuple[int, ...]  # `[7]` when only 5 blocks were given

    @property
    def ok(self) -> bool:
        return not (self.invented or self.bad_refs)

    @property
    def total(self) -> int:
        return len(self.grounded) + len(self.drifted) + len(self.invented)

    def summary(self) -> str:
        parts = [f"{len(self.grounded)} grounded"]
        if self.drifted:
            parts.append(f"{len(self.drifted)} line-drift")
        if self.invented:
            parts.append(f"{len(self.invented)} INVENTED: {', '.join(self.invented)}")
        if self.bad_refs:
            parts.append(f"dangling block refs {list(self.bad_refs)}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    hits: tuple[Hit, ...]
    citations: CitationCheck
    provider: str
    model: str
    retrieval_ms: float
    generation_ms: float
    prompt_chars: int

    @property
    def abstained(self) -> bool:
        low = self.text.lower()
        return any(marker in low for marker in ABSTENTION_MARKERS)

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


def validate(text: str, hits: Sequence[Hit]) -> CitationCheck:
    """Split the answer's references into grounded, drifted and invented.

    `drifted` is kept separate from `invented` on purpose. "Right file, wrong
    lines" is the model paraphrasing a real citation; "file that was never
    retrieved" is the model making one up. Collapsing them would hide which
    failure mode the prompt actually has.
    """
    retrieved_spans = {h.citation for h in hits}
    retrieved_paths = {h.file_path for h in hits}

    grounded, drifted, invented = [], [], []
    seen: set[str] = set()
    for match in CITATION_RE.finditer(text):
        path, start = match.group("path"), match.group("start")
        end = match.group("end") or start
        span = f"{path}:{start}-{end}"
        if span in seen:
            continue
        seen.add(span)
        if span in retrieved_spans:
            grounded.append(span)
        elif _known_file(path, retrieved_paths):
            drifted.append(span)
        else:
            invented.append(span)

    # Paths mentioned with no line numbers: still a claim about a file. The
    # prompt says "line numbers when available", so a bare path to a retrieved
    # file is grounded, not a violation.
    for match in PATH_RE.finditer(text):
        path = match.group(0)
        if path in seen or any(s.startswith(f"{path}:") for s in seen):
            continue
        seen.add(path)
        (grounded if _known_file(path, retrieved_paths) else invented).append(path)

    bad_refs = sorted({int(n) for n in REF_RE.findall(text) if not 1 <= int(n) <= len(hits)})
    return CitationCheck(tuple(grounded), tuple(drifted), tuple(invented), tuple(bad_refs))


def _known_file(claim: str, retrieved: set[str]) -> bool:
    """Does this path refer to a file that was actually retrieved?

    Deliberately forgiving: case-insensitive, and a bare `qdrant_store.py`
    counts as referring to `OpsSense/src/vector_store/qdrant_store.py`. A model
    that drops the directory prefix or lowercases the repo name is being
    sloppy, not dishonest, and a hallucination detector that fires on
    transcription noise is one nobody will trust when it fires for real.
    """
    claim = claim.lower()
    for path in retrieved:
        path = path.lower()
        if path == claim or path.endswith(f"/{claim}") or claim.endswith(f"/{path}"):
            return True
        if path.rsplit("/", 1)[-1] == claim.rsplit("/", 1)[-1]:
            return True
    return False


class Generator:
    def __init__(
        self,
        cfg: Config,
        llm: llm_mod.LLM | None = None,
        expander: ContextExpander | None = None,
        system: str = SYSTEM_PROMPT,
    ):
        self.cfg = cfg
        self.llm = llm or llm_mod.build(cfg)
        self.expander = expander
        self.system = system

    def blocks(self, hits: Sequence[Hit]) -> tuple[llm_mod.Block, ...]:
        return tuple(
            llm_mod.Block(
                index=i,
                citation=hit.citation,
                qualified_name=hit.qualified_name,
                symbol_type=hit.symbol_type,
                # Expansion adds the imports and parent class signature, which
                # is what makes a bare method body interpretable.
                text=self.expander.expand(hit).text if self.expander else hit.text,
            )
            for i, hit in enumerate(hits, 1)
        )

    def prompt(self, question: str, hits: Sequence[Hit]) -> llm_mod.Prompt:
        return llm_mod.Prompt(self.system, question, self.blocks(hits))

    def answer(self, question: str, hits: Sequence[Hit], retrieval_ms: float = 0.0) -> Answer:
        prompt = self.prompt(question, hits)
        completion = self.llm.complete(prompt)
        return Answer(
            question=question,
            text=completion.text,
            hits=tuple(hits),
            citations=validate(completion.text, hits),
            provider=completion.provider,
            model=completion.model,
            retrieval_ms=retrieval_ms,
            generation_ms=completion.elapsed_ms,
            prompt_chars=prompt.chars,
        )
