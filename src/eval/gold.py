"""Gold-set loading and chunk matching for eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chunking.code_chunker import Chunk


def load_gold(path: Path) -> dict:
    return json.loads(path.read_text())


def chunk_label_names(chunk: Chunk) -> set[str]:
    names = {chunk.symbol_name}
    if chunk.class_name:
        names.add(f"{chunk.class_name}.{chunk.symbol_name}")
    return names


def chunk_matches(label: dict, chunk: Chunk) -> bool:
    if label["file"] != chunk.file_path:
        return False
    symbol = label.get("symbol")
    if symbol is None:
        return True
    return symbol in chunk_label_names(chunk)


def build_name_index(chunks: list[Chunk]) -> set[tuple[str, str]]:
    names: set[tuple[str, str]] = set()
    for chunk in chunks:
        for name in chunk_label_names(chunk):
            names.add((chunk.file_path, name))
    return names
