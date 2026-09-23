"""Collect indexing_report.json from S3 after a documents-indexing-pipeline run."""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Any

from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME

logger = logging.getLogger(__name__)


def extract_indexing_report(
    run_id: str,
    config: dict[str, Any],
    bucket: str,
    *,
    pipeline_name: str = INDEXING_PIPELINE_NAME,
) -> dict[str, Any]:
    """Download and parse the indexing report from S3.

    Returns a flat dict suitable for merging into a CSV row, with keys prefixed
    by ``indexing_``.  Returns ``{"indexing_error": "..."}`` on failure.
    """
    from autorag_benchmark.pattern_scores import create_s3_client

    try:
        s3_client = create_s3_client(config)
    except Exception as e:
        logger.warning("Could not create S3 client: %s", e)
        return {"indexing_error": f"S3 client creation failed: {e}"}

    prefix = f"{pipeline_name}/{run_id}/"
    report_key = find_indexing_report_key(s3_client, bucket, prefix)
    if not report_key:
        return {"indexing_error": f"indexing_report.json not found under s3://{bucket}/{prefix}"}

    try:
        response = s3_client.get_object(Bucket=bucket, Key=report_key)
        report = json.loads(response["Body"].read())
    except Exception as e:
        logger.error("Error reading indexing report %s: %s", report_key, e)
        return {"indexing_error": str(e)}

    return flatten_report(report)


def find_indexing_report_key(s3_client: Any, bucket: str, prefix: str) -> str | None:
    """Walk the S3 prefix tree to find the indexing_report.json artifact."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("indexing_report.json") or key.endswith("indexing_report"):
                return key
    return None


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested indexing report into prefixed columns."""
    flat: dict[str, Any] = {}

    flat["indexing_total_documents"] = report.get("total_documents")
    flat["indexing_completed"] = report.get("completed")
    flat["indexing_failed"] = report.get("failed")
    flat["indexing_total_chunks"] = report.get("total_chunks")

    settings = report.get("settings", {})
    vsb = settings.get("vector_store_binding", {})
    flat["indexing_collection_name"] = vsb.get("collection_name", "")
    flat["indexing_provider_type"] = vsb.get("provider_type", "")

    chunking = settings.get("chunking", {})
    flat["indexing_chunking_method"] = chunking.get("method", "")
    flat["indexing_chunk_size"] = chunking.get("chunk_size")
    flat["indexing_chunk_overlap"] = chunking.get("chunk_overlap")

    embedding = settings.get("embedding", {})
    flat["indexing_embedding_model"] = embedding.get("model_id", "")

    params = embedding.get("embedding_params", {})
    dimension = params.get("embedding_dimension", embedding.get("embedding_dimension"))
    flat["indexing_embedding_dimension"] = dimension
    total_chunks = _number(report.get("total_chunks"))
    if total_chunks is not None and isinstance(dimension, (int, float)):
        flat["indexing_estimated_vector_storage_mb"] = round(total_chunks * dimension * 4 / 1024**2, 3)
        flat["indexing_estimated_embedding_api_calls"] = total_chunks

    documents = report.get("documents") or []
    chunk_counts = [_chunk_count(document) for document in documents if _chunk_count(document) is not None]
    if chunk_counts:
        flat.update(_chunk_distribution(chunk_counts))

    return flat


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _chunk_count(document: Any) -> float | int | None:
    if not isinstance(document, dict):
        return None
    for key in ("chunk_count", "chunks_count", "total_chunks", "chunks"):
        value = document.get(key)
        if isinstance(value, list):
            return len(value)
        if _number(value) is not None:
            return _number(value)
    return None


def _percentile(values: list[float | int], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _chunk_distribution(values: list[float | int]) -> dict[str, Any]:
    return {
        "indexing_chunks_per_document_mean": round(statistics.fmean(values), 3),
        "indexing_chunks_per_document_std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "indexing_chunks_per_document_min": min(values),
        "indexing_chunks_per_document_p25": round(_percentile(values, .25), 3),
        "indexing_chunks_per_document_median": round(statistics.median(values), 3),
        "indexing_chunks_per_document_p75": round(_percentile(values, .75), 3),
        "indexing_chunks_per_document_max": max(values),
        "indexing_documents_with_few_chunks": sum(count <= 2 for count in values),
        "indexing_documents_with_many_chunks": sum(count > 200 for count in values),
    }
