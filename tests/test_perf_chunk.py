import time

import pytest

from src.chunking import code_chunker
from src.config import Config
from src.parser.repository_loader import SourceFile

FIXTURE = (
    '''"""Synthetic module."""

def handler(value: int) -> int:
    """Double the input."""
    return value * 2


class Worker:
    def run(self) -> None:
        pass
'''
    * 5
)


@pytest.mark.slow
def test_chunk_all_strategies_under_budget():
    files = [
        SourceFile(
            file_path=f"pkg/mod_{i}.py",
            language="python",
            content=FIXTURE,
            sha=f"{i:064d}",
        )
        for i in range(200)
    ]
    cfg = Config()
    start = time.perf_counter()
    for strategy in code_chunker.STRATEGIES:
        code_chunker.stats(code_chunker.chunk(files, cfg, strategy), strategy)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
