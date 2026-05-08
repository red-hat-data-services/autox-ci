"""Parse KFP run timestamps and durations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def duration_seconds(start: datetime | None, end: datetime | None) -> str:
    if not start or not end:
        return ""
    return str(max(0.0, (end - start).total_seconds()))
