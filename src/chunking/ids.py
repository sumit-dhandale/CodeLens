"""Stable point IDs for Qdrant.

A chunk's identity is its line span, not just its symbol name: two `def foo()`
blocks in one file would collide on path:class:symbol alone. start_line disambiguates
them while keeping the ID stable across content edits at the same span.
"""

from __future__ import annotations

import uuid
from uuid import NAMESPACE_URL

_POINT_NS = uuid.uuid5(NAMESPACE_URL, "codelens/chunk-point")


def point_key(
    file_path: str,
    class_name: str,
    symbol_name: str,
    start_line: int,
    strategy: str,
) -> str:
    return f"{file_path}:{class_name}:{symbol_name}:{start_line}:{strategy}"


def point_id(
    file_path: str,
    class_name: str,
    symbol_name: str,
    start_line: int,
    strategy: str,
) -> uuid.UUID:
    return uuid.uuid5(
        _POINT_NS, point_key(file_path, class_name, symbol_name, start_line, strategy)
    )
