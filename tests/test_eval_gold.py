from src.chunking.code_chunker import chunk_file
from src.config import Config
from src.eval.gold import chunk_matches
from src.parser.repository_loader import SourceFile

PY = "def handler():\n    return 1\n"


def test_chunk_matches_label():
    chunk = chunk_file(
        SourceFile(file_path="pkg/app.py", language="python", content=PY, sha="0" * 64),
        Config(),
        "function",
    )[0]
    assert chunk_matches({"file": "pkg/app.py", "symbol": "handler"}, chunk)
    assert chunk_matches({"file": "pkg/app.py", "symbol": None}, chunk)
    assert not chunk_matches({"file": "other.py", "symbol": "handler"}, chunk)
