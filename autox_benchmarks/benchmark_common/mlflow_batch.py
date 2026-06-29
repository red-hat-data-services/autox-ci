"""Load batch results from S3 (or local files) and push to MLflow after orchestration."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_common.mlflow_ingest import KIND_PRESETS, ingest_dataframe
from benchmark_common.mlflow_settings import MlflowSettings, mlflow_settings_from_env
from benchmark_common.s3_upload import join_s3_key, try_get_s3_object_bytes

logger = logging.getLogger(__name__)


def aggregated_csv_key(benchmark_s3_prefix: str, batch_id: str, benchmark_kind: str) -> str:
    csv_name = KIND_PRESETS[benchmark_kind]["aggregate_csv"]
    return join_s3_key(benchmark_s3_prefix, batch_id, "aggregated", csv_name)


def batch_metadata_key(benchmark_s3_prefix: str, batch_id: str) -> str:
    return join_s3_key(benchmark_s3_prefix, batch_id, "aggregated", "batch_metadata.json")


def load_batch_metadata_from_s3(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_s3_prefix: str,
    batch_id: str,
) -> dict[str, Any] | None:
    raw = try_get_s3_object_bytes(
        s3_cfg=s3_cfg,
        bucket=bucket,
        key=batch_metadata_key(benchmark_s3_prefix, batch_id),
    )
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not parse batch_metadata.json for batch %s", batch_id)
        return None


def load_batch_dataframe_from_s3(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_s3_prefix: str,
    batch_id: str,
    benchmark_kind: str,
) -> pd.DataFrame:
    key = aggregated_csv_key(benchmark_s3_prefix, batch_id, benchmark_kind)
    raw = try_get_s3_object_bytes(s3_cfg=s3_cfg, bucket=bucket, key=key)
    if not raw:
        raise FileNotFoundError(f"Missing batch CSV s3://{bucket}/{key}")
    return pd.read_csv(io.BytesIO(raw))


def load_automl_batch_dataframe_local(output_csv: Path) -> pd.DataFrame:
    from automl_benchmark.leaderboard_merge import merge_benchmark_csv_with_leaderboards

    return merge_benchmark_csv_with_leaderboards(
        output_csv,
        include_metrics_blob=False,
        include_rows_without_leaderboard=True,
    )


def log_batch_to_mlflow(
    *,
    cfg: dict[str, Any],
    bucket: str,
    benchmark_s3_prefix: str,
    batch_id: str,
    output_csv: Path,
    benchmark_kind: str,
    dry_run: bool = False,
    settings: MlflowSettings | None = None,
) -> dict[str, Any] | None:
    """
    After S3 upload, ingest this batch into MLflow (same hierarchy as MLFlow.ipynb).

    Uses local merged CSV for AutoML when available; otherwise reads from S3.
    AutoRAG uses local output_csv or S3 benchmark_runs.csv.
    """
    if dry_run:
        return None

    try:
        mlflow_settings = settings or mlflow_settings_from_env()
    except ValueError as exc:
        logger.error("%s", exc)
        return None
    if mlflow_settings is None:
        return None

    kind = benchmark_kind if benchmark_kind in KIND_PRESETS else mlflow_settings.benchmark_kind
    s3_cfg = cfg.get("s3") if isinstance(cfg.get("s3"), dict) else None
    source_uri = f"s3://{bucket}/{aggregated_csv_key(benchmark_s3_prefix, batch_id, kind)}"

    try:
        if kind == "automl" and output_csv.is_file():
            df = load_automl_batch_dataframe_local(output_csv)
            source_uri = f"local:{output_csv.resolve()} (same as {source_uri})"
        elif output_csv.is_file() and kind == "autorag":
            df = pd.read_csv(output_csv)
            source_uri = f"local:{output_csv.resolve()} (same as {source_uri})"
        elif s3_cfg:
            df = load_batch_dataframe_from_s3(
                s3_cfg=s3_cfg,
                bucket=bucket,
                benchmark_s3_prefix=benchmark_s3_prefix,
                batch_id=batch_id,
                benchmark_kind=kind,
            )
        else:
            logger.warning("MLflow upload skipped: no local CSV and no S3 config")
            return None
    except Exception as exc:
        logger.warning("MLflow ingest failed loading batch %s: %s", batch_id, exc)
        return None

    batch_metadata = None
    if s3_cfg:
        batch_metadata = load_batch_metadata_from_s3(
            s3_cfg=s3_cfg,
            bucket=bucket,
            benchmark_s3_prefix=benchmark_s3_prefix,
            batch_id=batch_id,
        )

    try:
        return ingest_dataframe(
            df,
            settings=mlflow_settings,
            batch_id=batch_id,
            source_uri=source_uri,
            batch_metadata=batch_metadata,
        )
    except Exception as exc:
        logger.warning("MLflow ingest failed for batch %s: %s", batch_id, exc, exc_info=True)
        return None
