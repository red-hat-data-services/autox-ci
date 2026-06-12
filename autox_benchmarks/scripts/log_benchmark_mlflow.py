#!/usr/bin/env python3
"""Log one benchmark batch from S3 to MLflow (CLI equivalent of MLFlow.ipynb)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_common.credentials import load_benchmark_dotenv, load_credentials_overlay
from benchmark_common.mlflow_batch import (
    aggregated_csv_key,
    load_batch_dataframe_from_s3,
    load_batch_metadata_from_s3,
)
from benchmark_common.mlflow_ingest import ingest_dataframe
from benchmark_common.mlflow_settings import mlflow_settings_from_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one benchmark batch CSV from S3 into MLflow.")
    parser.add_argument("batch_id", help="Batch id folder under benchmark_s3_prefix (e.g. 20260529T120000Z)")
    parser.add_argument(
        "--kind",
        choices=("automl", "autorag"),
        default=None,
        help="Benchmark kind (default: BENCHMARK_MLFLOW_KIND or automl)",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Path to .env")
    parser.add_argument("--bucket", default=None, help="Override train/test data bucket")
    parser.add_argument("--prefix", default=None, help="Override benchmark_s3_prefix")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    load_benchmark_dotenv(args.env_file)

    settings = mlflow_settings_from_env()
    if settings is None:
        print("Set BENCHMARK_UPLOAD_MLFLOW=true in .env to enable MLflow ingest.", file=sys.stderr)
        return 1

    kind = args.kind or settings.benchmark_kind
    overlay, _ = load_credentials_overlay(env_file=args.env_file)
    storage = overlay.get("storage") or {}
    s3_cfg = overlay.get("s3") or {}

    if kind == "autorag":
        bucket = args.bucket or str(storage.get("test_data_bucket_name") or "").strip()
    else:
        bucket = args.bucket or str(storage.get("train_data_bucket_name") or "").strip()
    prefix = (args.prefix or str(storage.get("benchmark_s3_prefix") or "benchmarks")).strip().strip("/")

    if not bucket:
        print("Bucket not configured in .env", file=sys.stderr)
        return 1

    key = aggregated_csv_key(prefix, args.batch_id, kind)
    source_uri = f"s3://{bucket}/{key}"

    df = load_batch_dataframe_from_s3(
        s3_cfg=s3_cfg,
        bucket=bucket,
        benchmark_s3_prefix=prefix,
        batch_id=args.batch_id,
        benchmark_kind=kind,
    )
    meta = load_batch_metadata_from_s3(
        s3_cfg=s3_cfg,
        bucket=bucket,
        benchmark_s3_prefix=prefix,
        batch_id=args.batch_id,
    )
    summary = ingest_dataframe(
        df,
        settings=settings,
        batch_id=args.batch_id,
        source_uri=source_uri,
        batch_metadata=meta,
    )
    print(f"Logged {summary.get('entities_logged', 0)} entity runs to {settings.experiment_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
