"""Walk a checkout and emit the files worth indexing."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import IGNORE_DIRS, IGNORE_FILES, IGNORE_SUFFIXES, Config


@dataclass(frozen=True)
class SourceFile:
    file_path: str  # relative to the repo root, posix separators
    language: str
    content: str
    sha: str  # sha256 of the content

    @property
    def directory(self) -> str:
        return str(Path(self.file_path).parent).replace("\\", "/")


def _is_ignored_name(name: str) -> bool:
    return name in IGNORE_FILES or name.endswith(IGNORE_SUFFIXES)


def discover(root: Path, cfg: Config) -> list[Path]:
    """Absolute paths of candidate files under root, ignore rules applied."""
    root = root.resolve()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            if filename.startswith(".") or _is_ignored_name(filename):
                continue
            if Path(filename).suffix.lower() not in cfg.languages:
                continue
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size > cfg.max_file_bytes:
                continue
            found.append(path)
    return found


def load(root: Path, cfg: Config) -> list[SourceFile]:
    """Read every discovered file. Undecodable files are treated as binary and skipped."""
    root = root.resolve()
    files: list[SourceFile] = []
    for path in discover(root, cfg):
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        files.append(
            SourceFile(
                file_path=path.relative_to(root).as_posix(),
                language=cfg.languages[path.suffix.lower()],
                content=content,
                sha=hashlib.sha256(raw).hexdigest(),
            )
        )
    return files
