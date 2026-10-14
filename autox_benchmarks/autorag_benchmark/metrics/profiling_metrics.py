"""Extract indexing volume, throughput, and component timing from S3 artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from autorag_benchmark.indexing_report import flatten_report
from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME


def extract_profiling_metrics(indexing_run_id: str, config: dict[str, Any], bucket: str, *, indexing_wall_time: float | None = None) -> dict[str, Any]:
    """Collect metrics whose source is not limited to ``indexing_report``.

    The artifact schemas have changed across pipeline versions, so descriptor and
    status parsing is deliberately defensive and omits a field when unavailable.
    """
    from autorag_benchmark.pattern_scores import create_s3_client

    client = create_s3_client(config)
    root = f"{INDEXING_PIPELINE_NAME}/{indexing_run_id}/"
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    result: dict[str, Any] = {}
    report_key = next((key for key in keys if key.endswith("indexing_report.json") or key.endswith("/indexing_report")), None)
    if report_key:
        result.update(flatten_report(_read_json(client, bucket, report_key)))
    descriptor_key = next((key for key in keys if key.endswith("documents_descriptor.json")), None)
    if descriptor_key:
        result.update(_descriptor_metrics(_read_json(client, bucket, descriptor_key)))
    statuses = [_read_json(client, bucket, key) for key in keys if key.endswith("component_status.json")]
    result.update(_component_timings(statuses))
    total_documents = result.get("indexing_total_documents")
    total_chunks = result.get("indexing_total_chunks")
    if isinstance(total_documents, (int, float)) and total_documents:
        completed = result.get("indexing_completed")
        if isinstance(completed, (int, float)):
            result["indexing_ingestion_success_rate"] = round(completed / total_documents, 6)
    if indexing_wall_time and indexing_wall_time > 0:
        if isinstance(total_documents, (int, float)):
            result["indexing_documents_per_second"] = round(total_documents / indexing_wall_time, 6)
        if isinstance(total_chunks, (int, float)):
            result["indexing_chunks_per_second"] = round(total_chunks / indexing_wall_time, 6)
        if isinstance(result.get("indexing_input_data_total_bytes"), (int, float)):
            result["indexing_bytes_per_second"] = round(result["indexing_input_data_total_bytes"] / indexing_wall_time, 3)
    return result


def _read_json(client: Any, bucket: str, key: str) -> Any:
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def _descriptor_metrics(data: Any) -> dict[str, Any]:
    docs = data.get("documents", data) if isinstance(data, dict) else data
    if not isinstance(docs, list):
        return {}
    sizes: list[int | float] = []
    for document in docs:
        if not isinstance(document, dict):
            continue
        for key in ("size_bytes", "size", "file_size", "content_length"):
            value = document.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sizes.append(value)
                break
    total = sum(sizes)
    result: dict[str, Any] = {"indexing_discovered_document_count": len(docs)}
    if sizes:
        result.update({
            "indexing_input_data_total_bytes": total,
            "indexing_input_data_total_mb": round(total / 1024**2, 3),
            "indexing_avg_document_size_bytes": round(total / len(sizes), 3),
        })
    return result


def _component_timings(statuses: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for status in statuses:
        if not isinstance(status, dict):
            continue
        name = str(status.get("component_name") or status.get("component") or status.get("name") or "").lower()
        duration = _duration_from_status(status)
        if duration is None:
            continue
        if "discovery" in name:
            result["indexing_discovery_duration_seconds"] = duration
        elif "extraction" in name:
            result["indexing_text_extraction_duration_seconds"] = duration
        elif "index" in name:
            result["indexing_chunking_embedding_duration_seconds"] = duration
    return result


def _duration_from_status(status: dict[str, Any]) -> float | None:
    for key in ("duration_seconds", "duration", "elapsed_seconds"):
        value = status.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    start = status.get("started_at") or status.get("start_time") or status.get("start")
    end = status.get("finished_at") or status.get("end_time") or status.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        return round((datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds(), 3)
    except ValueError:
        return None
