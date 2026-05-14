"""Upload benchmark results and metadata to S3 under benchmarks/{batch_id}/."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autorag_benchmark.settings import BenchmarkSettings

logger = logging.getLogger(__name__)


def build_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _join_key(*parts: str) -> str:
    return "/".join(p.strip().strip("/") for p in parts if p and str(p).strip())


def _row_to_csv_bytes(row: dict[str, Any]) -> bytes:
    keys = sorted(row.keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerow({k: "" if row.get(k) is None else row.get(k) for k in keys})
    return buf.getvalue().encode("utf-8")


def _s3_cfg_usable(s3_cfg: dict[str, Any]) -> bool:
    """Check if S3 config has required keys."""
    return bool(
        s3_cfg.get("endpoint_url")
        and s3_cfg.get("aws_access_key_id")
        and s3_cfg.get("aws_secret_access_key")
    )


def _make_s3_client(s3_cfg: dict[str, Any]) -> Any:
    """Create boto3 S3 client from config dict."""
    import boto3

    return boto3.client("s3", **s3_cfg)


def _put_s3_bytes(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
) -> None:
    client = _make_s3_client(s3_cfg)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def _dataset_results_subpath(dataset: dict[str, Any]) -> str:
    """Generate subpath for dataset results (dataset_id)."""
    return str(dataset.get("id", "unknown")).strip()


def _build_run_metadata(
    *,
    row: dict[str, Any],
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    s3_cfg: dict[str, Any],
    pipeline_yaml_path: Path,
    s3_benchmark_key_prefix: str,
    arguments: dict[str, Any] | None,
    dataset_filter: str,
    fail_fast: bool,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Build metadata dict for a single dataset run."""
    return {
        "dataset_id": dataset.get("id"),
        "dataset_name": dataset.get("name"),
        "run_id": row.get("run_id"),
        "run_name": row.get("run_name"),
        "state": row.get("state"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "duration_seconds": row.get("duration_seconds"),
        "error": row.get("error"),
        "optimization_metric": row.get("optimization_metric"),
        "pipeline_yaml": str(pipeline_yaml_path),
        "pipeline_arguments": arguments or {},
        "s3_benchmark_prefix": s3_benchmark_key_prefix,
        "dataset_filter": dataset_filter,
        "fail_fast": fail_fast,
        "repo_root": str(repo_root) if repo_root else None,
        "config": {
            "pipeline": cfg.get("pipeline"),
            "run": cfg.get("run"),
            "storage": {k: v for k, v in (cfg.get("storage") or {}).items() if "secret" not in k.lower()},
        },
    }


def _build_batch_metadata(
    *,
    batch_id: str,
    started_at: str,
    finished_at: str,
    manifest_relative: str,
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    dataset_ids: list[str],
    row_count: int,
    output_csv_relative: str,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Build metadata dict for entire batch."""
    return {
        "batch_id": batch_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset_manifest": manifest_relative,
        "dataset_ids": dataset_ids,
        "run_count": row_count,
        "output_csv": output_csv_relative,
        "repo_root": str(repo_root) if repo_root else None,
        "settings": {
            "optimization_metric": settings.optimization_metric,
            "optimization_max_rag_patterns": settings.optimization_max_rag_patterns,
            "poll_interval_seconds": settings.poll_interval_seconds,
            "timeout_seconds": settings.timeout_seconds,
            "enable_caching": settings.enable_caching,
            "experiment_name": settings.experiment_name,
            "run_name_prefix": settings.run_name_prefix,
        },
        "config": {
            "pipeline": cfg.get("pipeline"),
            "run": cfg.get("run"),
            "storage": {k: v for k, v in (cfg.get("storage") or {}).items() if "secret" not in k.lower()},
        },
    }


def upload_single_dataset_results(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    batch_id: str,
    dataset: dict[str, Any],
    row: dict[str, Any],
    pipeline_yaml_path: Path,
    arguments: dict[str, Any] | None,
    dataset_filter: str,
    fail_fast: bool,
    repo_root: Path | None,
) -> None:
    """Upload results and metadata for a single dataset run to S3."""
    if not settings.upload_benchmark_results or not _s3_cfg_usable(s3_cfg):
        return

    sub = _dataset_results_subpath(dataset)
    prefix = _join_key(settings.benchmark_s3_prefix, batch_id, "datasets", sub)

    meta = _build_run_metadata(
        row=row,
        dataset=dataset,
        settings=settings,
        cfg=cfg,
        s3_cfg=s3_cfg,
        pipeline_yaml_path=pipeline_yaml_path,
        s3_benchmark_key_prefix=prefix + "/",
        arguments=arguments,
        dataset_filter=dataset_filter,
        fail_fast=fail_fast,
        repo_root=repo_root,
    )

    meta_body = json.dumps(meta, indent=2, default=str).encode("utf-8")
    results_body = _row_to_csv_bytes(row)

    try:
        _put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=_join_key(prefix, "metadata.json"),
            body=meta_body,
            content_type="application/json; charset=utf-8",
        )
        _put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=_join_key(prefix, "results.csv"),
            body=results_body,
            content_type="text/csv; charset=utf-8",
        )
        logger.info("Uploaded benchmark artifacts to s3://%s/%s/", bucket, prefix)
    except Exception as e:
        logger.warning("S3 upload failed for dataset %s: %s", row.get("dataset_id"), e)


def upload_batch_aggregated(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    batch_id: str,
    started_at: str,
    output_csv: Path,
    rows: list[dict[str, Any]],
    dataset_filter: str,
    repo_root: Path | None,
) -> None:
    """Upload aggregated batch results and metadata to S3."""
    if not settings.upload_benchmark_results or not _s3_cfg_usable(s3_cfg):
        return

    agg_prefix = _join_key(settings.benchmark_s3_prefix, batch_id, "aggregated")
    finished_at = datetime.now(timezone.utc).isoformat()
    manifest_rel = str(cfg.get("dataset_manifest_path") or "")

    try:
        rel_out = str(output_csv.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel_out = str(output_csv)

    dataset_ids = [str(r.get("dataset_id") or "") for r in rows if r.get("dataset_id")]

    batch_meta = _build_batch_metadata(
        batch_id=batch_id,
        started_at=started_at,
        finished_at=finished_at,
        manifest_relative=manifest_rel,
        settings=settings,
        cfg=cfg,
        dataset_ids=dataset_ids,
        row_count=len(rows),
        output_csv_relative=rel_out,
        repo_root=repo_root,
    )
    batch_meta["cli_dataset_filter"] = dataset_filter

    try:
        _put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=_join_key(agg_prefix, "batch_metadata.json"),
            body=json.dumps(batch_meta, indent=2, default=str).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

        if output_csv.is_file():
            _put_s3_bytes(
                s3_cfg=s3_cfg,
                bucket=bucket,
                key=_join_key(agg_prefix, "benchmark_runs.csv"),
                body=output_csv.read_bytes(),
                content_type="text/csv; charset=utf-8",
            )

        logger.info("Uploaded batch aggregated artifacts to s3://%s/%s/", bucket, agg_prefix)
    except Exception as e:
        logger.warning("S3 batch upload failed: %s", e)
