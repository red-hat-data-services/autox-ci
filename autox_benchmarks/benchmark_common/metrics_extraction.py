"""Best-effort extraction of run metadata for downstream CSV / analysis."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from benchmark_common.run_state import unwrap_run_from_get_run


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def run_to_metrics_dict(run: Any) -> dict[str, Any]:
    """Compact, JSON-serializable summary (no pipeline_spec — it breaks CSV and is not useful here)."""
    payload: dict[str, Any] = {"pipeline_spec": "omitted"}
    for attr in ("metrics", "runtime_context"):
        v = getattr(run, attr, None)
        if v is not None:
            payload[attr] = _to_jsonable(v)

    rd = getattr(run, "run_details", None)
    if rd is not None:
        tasks_out: list[dict[str, Any]] = []
        for t in getattr(rd, "task_details", None) or []:
            entry: dict[str, Any] = {
                "task_id": _to_jsonable(getattr(t, "task_id", None)),
                "task_name": _to_jsonable(getattr(t, "task_name", None)),
                "display_name": _to_jsonable(getattr(t, "display_name", None)),
                "state": _to_jsonable(getattr(t, "state", None)),
            }
            for attr in ("inputs", "outputs"):
                o = getattr(t, attr, None)
                if o is not None:
                    entry[attr] = _to_jsonable(o)
            tasks_out.append(entry)
        payload["task_details"] = tasks_out

    return payload


def extract_metrics_blob(run_detail: Any) -> str:
    """Single JSON object per run; valid JSON for downstream summarize_benchmark_results.py."""
    run = unwrap_run_from_get_run(run_detail)
    if run is None:
        run = run_detail
    try:
        return json.dumps(run_to_metrics_dict(run), default=str)
    except (TypeError, ValueError):
        return json.dumps({"error": "serialization_failed"}, default=str)
