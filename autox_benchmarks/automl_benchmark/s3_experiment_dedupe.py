"""S3 experiment index for skipping identical benchmark runs."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from benchmark_common.run_state import is_success_state
from benchmark_common.s3_client import make_s3_client, s3_cfg_usable

logger = logging.getLogger(__name__)

INDEX_SCHEMA = "1"


def _s3_join(*parts: str) -> str:
    return "/".join(p.strip().strip("/") for p in parts if p and str(p).strip())


def experiment_index_key(benchmark_s3_prefix: str, fingerprint: str) -> str:
    return _s3_join(benchmark_s3_prefix, "experiment_index", "v1", f"{fingerprint}.json")


def _is_not_found(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    err = exc.response.get("Error") or {}
    return err.get("Code", "") in ("404", "NoSuchKey", "NotFound")


def try_load_cached_result_row(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_s3_prefix: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """
    If an experiment index exists for ``fingerprint``, download the referenced
    ``results.csv`` and return the first data row as a dict. Returns ``None`` on miss or error.
    """
    if not s3_cfg_usable(s3_cfg):
        return None
    index_key = experiment_index_key(benchmark_s3_prefix, fingerprint)
    client = make_s3_client(s3_cfg)
    try:
        idx_body = client.get_object(Bucket=bucket, Key=index_key)["Body"].read()
        index_doc = json.loads(idx_body.decode("utf-8"))
    except Exception as e:
        if not _is_not_found(e):
            logger.debug("Experiment index read failed for %s: %s", index_key, e)
        return None

    if not isinstance(index_doc, dict):
        return None
    results_key = index_doc.get("results_csv_key")
    if not isinstance(results_key, str) or not results_key.strip():
        return None

    try:
        csv_body = client.get_object(Bucket=bucket, Key=results_key)["Body"].read().decode("utf-8")
    except Exception as e:
        logger.warning("Could not download cached results %s: %s", results_key, e)
        return None

    try:
        rdr = csv.DictReader(io.StringIO(csv_body))
        rows = list(rdr)
    except Exception as e:
        logger.warning("Could not parse cached results CSV: %s", e)
        return None
    if not rows:
        return None
    row = {k: (v if v is not None else "") for k, v in rows[0].items()}
    if not str(row.get("run_id") or "").strip():
        logger.warning("Cached results row missing run_id; ignoring cache hit")
        return None
    if not is_success_state(str(row.get("state") or "")):
        logger.debug("Cached results row state is not success; ignoring cache hit")
        return None

    row["dedupe_cache_hit"] = "true"
    row["experiment_fingerprint"] = fingerprint
    row["dedupe_index_s3_uri"] = f"s3://{bucket}/{index_key}"
    row["dedupe_results_s3_uri"] = f"s3://{bucket}/{results_key}"
    logger.info(
        "Skipping duplicate experiment for dataset_id=%s (fingerprint=%s…)",
        row.get("dataset_id", ""),
        fingerprint[:12],
    )
    return row


def write_experiment_index(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_s3_prefix: str,
    fingerprint: str,
    batch_id: str,
    prior_run_id: str,
    results_csv_key: str,
    metadata_json_key: str,
    aggregated_merged_csv_key: str,
    dataset_results_subpath: str,
) -> None:
    """Upsert index so future runs can skip identical experiments."""
    if not s3_cfg_usable(s3_cfg):
        return
    from datetime import datetime, timezone

    key = experiment_index_key(benchmark_s3_prefix, fingerprint)
    doc = {
        "schema_version": INDEX_SCHEMA,
        "experiment_fingerprint": fingerprint,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "prior_batch_id": batch_id,
        "prior_run_id": prior_run_id,
        "results_csv_key": results_csv_key,
        "metadata_json_key": metadata_json_key,
        "aggregated_merged_csv_key": aggregated_merged_csv_key,
        "dataset_results_subpath": dataset_results_subpath,
    }
    body = json.dumps(doc, indent=2).encode("utf-8")
    client = make_s3_client(s3_cfg)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )
    logger.info("Wrote experiment index s3://%s/%s", bucket, key)
