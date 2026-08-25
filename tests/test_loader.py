import pytest

from src.config import Config
from src.parser.repo_fetcher import parse_repo_url
from src.parser.repository_loader import load


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text("def handler():\n    return 1\n")
    (tmp_path / "README.md").write_text("# Title\n")
    (tmp_path / "notes.txt").write_text("not indexed\n")
    (tmp_path / "yarn.lock").write_text("lock\n")
    (tmp_path / "vendor.min.js").write_text("var a=1\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.py").write_text("cached\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("dep\n")
    return tmp_path


def test_load_applies_ignore_rules(repo):
    files = load(repo, Config())
    assert [f.file_path for f in files] == ["README.md", "pkg/app.py"]
    assert [f.language for f in files] == ["markdown", "python"]
    assert files[1].directory == "pkg"
    assert len(files[1].sha) == 64


def test_size_cap(repo):
    (repo / "big.py").write_text("x = 1\n" * 5000)
    assert "big.py" not in {f.file_path for f in load(repo, Config(max_file_bytes=100))}


def test_parse_repo_url():
    assert parse_repo_url("https://github.com/sumit-dhandale/OpsSense.git") == (
        "sumit-dhandale",
        "OpsSense",
    )


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:o/r.git",
        "https://evil.com/o/r",
        "https://github.com/only-owner",
        "https://github.com/../../etc/passwd",
        "ext::sh -c whoami",
    ],
)
def test_parse_repo_url_rejects(url):
    with pytest.raises(ValueError):
        parse_repo_url(url)
