"""CSV row builders for AutoRAG benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from benchmark_common.metrics_extraction import extract_metrics_blob
from benchmark_common.run_state import read_run_state, unwrap_run_from_get_run
from benchmark_common.run_timing import duration_seconds, parse_timestamp


def base_row_for_dataset(
    dataset: dict[str, Any],
    dataset_index: int,
    run_name: str,
    suite: str = "",
    rhoai_version: str = "",
) -> dict[str, Any]:
    ds_id = str(dataset.get("id", dataset.get("name", f"dataset_{dataset_index}")))
    name = str(dataset.get("name", ds_id))
    return {
        "suite": suite,
        "rhoai_version": rhoai_version,
        "dataset_id": ds_id,
        "dataset_name": name,
        "input_data_key": dataset.get("input_data_key", ""),
        "test_data_key": dataset.get("test_data_key", ""),
        "optimization_metric": dataset.get("optimization_metric", ""),
        "run_name": run_name,
    }


def dry_run_row(base: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        "run_id": "",
        "state": "DRY_RUN",
        "started_at": "",
        "finished_at": "",
        "duration_seconds": "",
        "error": "",
        "metrics_blob": json.dumps(arguments),
    }


def timeout_row(base: dict[str, Any], run_id: str, timeout_seconds: float) -> dict[str, Any]:
    return {
        **base,
        "run_id": run_id,
        "state": "TIMEOUT",
        "started_at": "",
        "finished_at": "",
        "duration_seconds": str(timeout_seconds),
        "error": "wait timeout",
        "metrics_blob": "",
    }


def completed_row(base: dict[str, Any], run_id: str, run_detail: Any) -> dict[str, Any]:
    run = unwrap_run_from_get_run(run_detail)
    if run is None:
        run = run_detail
    state = read_run_state(run)
    created = parse_timestamp(getattr(run, "created_at", None))
    finished = parse_timestamp(getattr(run, "finished_at", None))
    err = ""
    if hasattr(run, "error") and run.error:
        err = str(run.error)
    elif getattr(run, "error_message", None):
        err = str(run.error_message)
    return {
        **base,
        "run_id": run_id,
        "state": state,
        "started_at": created.isoformat() if created else "",
        "finished_at": finished.isoformat() if finished else "",
        "duration_seconds": duration_seconds(created, finished),
        "error": err,
        "metrics_blob": extract_metrics_blob(run_detail),
    }


def submit_error_row(base: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        **base,
        "run_id": "",
        "state": "SUBMIT_ERROR",
        "started_at": "",
        "finished_at": "",
        "duration_seconds": "",
        "error": message,
        "metrics_blob": "",
    }


def run_name_for_dataset(prefix: str, dataset_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_id)
    return f"{prefix}-{safe}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
