"""Path resolution relative to config / workspace roots."""

from __future__ import annotations

from pathlib import Path


def resolve_under(base_dir: Path, relative_or_absolute: str) -> Path:
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()
