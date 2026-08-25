"""Single source of tunables. CLI flags override these fields by name."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# extension -> language. Anything not listed here is not indexed.
LANGUAGES = {
    ".py": "python",
    ".rb": "ruby",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".md": "markdown",
}

# Directory names skipped anywhere in the tree.
IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".idea",
        ".vscode",
        "site-packages",
    }
)

# Lockfiles and generated artifacts: real source extensions, zero retrieval value.
IGNORE_FILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Gemfile.lock",
        "go.sum",
    }
)

# Substrings marking a file as generated/minified.
IGNORE_SUFFIXES = (".min.js", ".min.css", ".bundle.js", "_pb2.py", ".g.dart")


@dataclass
class Config:
    # repo fetch
    repo: str = "https://github.com/sumit-dhandale/OpsSense"
    ref: str = "HEAD"
    repos_dir: Path = ROOT / ".repos"

    # loading
    max_file_bytes: int = 512 * 1024

    # chunking
    strategy: str = "function"
    chunk_lines: int = 60
    chunk_overlap: int = 10

    # embeddings
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_db: Path = ROOT / ".cache" / "embeddings.sqlite3"

    # vector store
    qdrant_url: str = "http://localhost:6333"
    collection_suffix: str = ""

    # retrieval
    top_k: int = 5
    rerank_top_n: int = 20
    vector_weight: float = 0.7
    fusion: str = "rrf"

    # rag
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"

    languages: dict[str, str] = field(default_factory=lambda: dict(LANGUAGES))

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    def override(self, **kwargs) -> "Config":
        """Return a copy with non-None kwargs applied. Unknown keys are ignored
        so the CLI can pass its whole namespace through."""
        known = self.field_names()
        updates = {k: v for k, v in kwargs.items() if k in known and v is not None}
        return dataclasses.replace(self, **updates)
