"""Inspect KFP run objects for lifecycle state."""

from __future__ import annotations

from enum import Enum
from typing import Any

TERMINAL_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "COMPLETED",
        "ERROR",
        "SKIPPED",
    }
)


def unwrap_run_from_get_run(detail: Any) -> Any | None:
    """KFP v1 ``RunDetail`` exposes ``.run``; v2 ``client.get_run`` often returns ``V2beta1Run`` directly."""
    if detail is None:
        return None
    inner = getattr(detail, "run", None)
    return inner if inner is not None else detail


def read_run_state(run: Any) -> str:
    for attr in ("state", "status", "phase"):
        v = getattr(run, attr, None)
        if v is None:
            continue
        if isinstance(v, Enum):
            return v.name
        text = str(v)
        if "." in text and (text.startswith("RunState.") or "RunState" in type(v).__name__):
            return text.split(".")[-1]
        return text
    return ""


def is_terminal_state(state: str) -> bool:
    return state.upper() in TERMINAL_STATES


def is_success_state(state: str) -> bool:
    u = state.upper()
    return u in ("SUCCEEDED", "COMPLETED")
