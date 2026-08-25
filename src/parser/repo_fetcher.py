"""Shallow-clone a GitHub repo and pin it to a commit SHA.

The SHA is part of the checkout directory name: eval labels reference file paths
and symbol names at one commit, so an unpinned clone silently rots the gold set.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..config import Config

_ALLOWED_HOSTS = {"github.com", "www.github.com"}
_SEGMENT = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_REF = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")


@dataclass(frozen=True)
class Checkout:
    owner: str
    name: str
    sha: str
    path: Path

    @property
    def slug(self) -> str:
        return f"{self.owner}__{self.name}@{self.sha[:12]}"


def parse_repo_url(url: str) -> tuple[str, str]:
    """Return (owner, name) for an https GitHub URL, or raise ValueError.

    Restricted to https://github.com to keep an untrusted --repo value from
    reaching git as a local path, an ssh target, or an ext:: transport.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"only https://github.com URLs are supported, got: {url!r}")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2:
        raise ValueError(f"expected https://github.com/<owner>/<repo>, got: {url!r}")
    owner, name = parts[0], parts[1].removesuffix(".git")
    if not _SEGMENT.match(owner) or not _SEGMENT.match(name):
        raise ValueError(f"unsafe owner/repo in URL: {url!r}")
    return owner, name


def _git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout.strip()


def fetch(cfg: Config) -> Checkout:
    """Clone cfg.repo at cfg.ref into cfg.repos_dir, reusing an existing checkout."""
    owner, name = parse_repo_url(cfg.repo)
    ref = cfg.ref.strip()
    if not _REF.match(ref):
        raise ValueError(f"unsafe ref: {ref!r}")

    cfg.repos_dir.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{owner}/{name}.git"

    tmp = Path(tempfile.mkdtemp(dir=cfg.repos_dir, prefix=".tmp-"))
    try:
        args = ["clone", "--depth", "1", "--quiet"]
        if ref != "HEAD":
            args += ["--branch", ref]
        _git(*args, clone_url, str(tmp))
        sha = _git("rev-parse", "HEAD", cwd=tmp)
        dest = cfg.repos_dir / f"{owner}__{name}@{sha[:12]}"
        if dest.exists():
            shutil.rmtree(tmp)
        else:
            tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return Checkout(owner=owner, name=name, sha=sha, path=dest)
