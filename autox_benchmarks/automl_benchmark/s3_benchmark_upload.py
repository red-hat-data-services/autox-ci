"""Upload benchmark results and metadata to S3 under benchmarks/{batch_id}/."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automl_benchmark.benchmark_run_metadata import (
    build_batch_metadata,
    build_run_metadata,
    dataset_results_subpath,
    sha256_file,
)
from automl_benchmark.experiment_fingerprint import compute_experiment_fingerprint
from benchmark_common.run_state import is_success_state
from benchmark_common.s3_client import s3_cfg_usable
from benchmark_common.s3_upload import (
    build_batch_id,
    join_s3_key,
    put_s3_bytes,
    row_to_csv_bytes,
    try_get_s3_object_bytes,
)
from automl_benchmark.s3_experiment_dedupe import write_experiment_index
from automl_benchmark.settings import BenchmarkSettings

logger = logging.getLogger(__name__)

__all__ = [
    "build_batch_id",
    "join_s3_key",
    "put_s3_bytes",
    "row_to_csv_bytes",
    "try_get_s3_object_bytes",
    "upload_batch_aggregated",
    "upload_single_dataset_results",
]


def _aggregated_pipeline_ir_uploads(settings: BenchmarkSettings) -> list[tuple[str, Path]]:
    """Stable S3 names under ``aggregated/`` for tabular and time-series compiled IR (dedupe same path)."""
    tab = settings.pipeline_yaml.resolve()
    ts = settings.timeseries_pipeline_yaml.resolve()
    out: list[tuple[str, Path]] = []
    if tab == ts:
        if tab.is_file():
            out.append(("autogluon-tabular-training-pipeline.yaml", tab))
        return out
    if tab.is_file():
        out.append(("autogluon-tabular-training-pipeline.yaml", tab))
    if ts.is_file():
        out.append(("autogluon-timeseries-training-pipeline.yaml", ts))
    return out


def _merge_joined_results_dedupe(existing: Any, new: Any) -> Any:
    """Append batch merged leaderboard rows; drop exact duplicate rows (same as prior S3 content)."""
    import pandas as pd

    if existing is None or getattr(existing, "empty", True):
        return new.drop_duplicates(keep="first")
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    return combined.drop_duplicates(keep="first")


def upload_single_dataset_results(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    batch_id: str,
    dataset: dict[str, Any],
    row: dict[str, Any],
    pipeline_ir_path: Path,
    output_csv_parent: Path,
    arguments: dict[str, Any] | None,
    dataset_filter: str,
    fail_fast: bool,
    artifact_s3_root: str,
    repo_root: Path | None,
    experiment_fingerprint: str | None = None,
) -> None:
    if not settings.upload_benchmark_results or not s3_cfg_usable(s3_cfg):
        return
    sub = dataset_results_subpath(dataset)
    prefix = join_s3_key(settings.benchmark_s3_prefix, batch_id, "datasets", sub)
    fp = experiment_fingerprint or compute_experiment_fingerprint(
        pipeline_ir_path=pipeline_ir_path.resolve(),
        pipeline_arguments=dict(arguments or {}),
        dataset=dataset,
        settings=settings,
        cfg=cfg,
        s3_cfg=s3_cfg,
        dataset_filter=dataset_filter,
    )
    meta = build_run_metadata(
        row=row,
        dataset=dataset,
        settings=settings,
        cfg=cfg,
        s3_cfg=s3_cfg,
        pipeline_ir_path=pipeline_ir_path,
        s3_benchmark_key_prefix=prefix + "/",
        arguments=arguments,
        dataset_filter=dataset_filter,
        fail_fast=fail_fast,
        artifact_s3_root=artifact_s3_root,
        repo_root=repo_root,
        experiment_fingerprint=fp,
    )
    meta_body = json.dumps(meta, indent=2, default=str).encode("utf-8")
    results_body = row_to_csv_bytes(row)
    try:
        put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=join_s3_key(prefix, "metadata.json"),
            body=meta_body,
            content_type="application/json; charset=utf-8",
        )
        put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=join_s3_key(prefix, "results.csv"),
            body=results_body,
            content_type="text/csv; charset=utf-8",
        )
        rel_html = str(row.get("leaderboard_html_path") or "").strip()
        if rel_html:
            local_html = output_csv_parent / rel_html
            if local_html.is_file():
                put_s3_bytes(
                    s3_cfg=s3_cfg,
                    bucket=bucket,
                    key=join_s3_key(prefix, "leaderboard.html"),
                    body=local_html.read_bytes(),
                    content_type="text/html; charset=utf-8",
                )
        logger.info("Uploaded benchmark artifacts to s3://%s/%s/", bucket, prefix)
        if is_success_state(str(row.get("state", ""))):
            agg_merged_key = join_s3_key(
                settings.benchmark_s3_prefix, batch_id, "aggregated", "merged_leaderboards.csv"
            )
            write_experiment_index(
                s3_cfg=s3_cfg,
                bucket=bucket,
                benchmark_s3_prefix=settings.benchmark_s3_prefix,
                fingerprint=fp,
                batch_id=batch_id,
                prior_run_id=str(row.get("run_id") or ""),
                results_csv_key=join_s3_key(prefix, "results.csv"),
                metadata_json_key=join_s3_key(prefix, "metadata.json"),
                aggregated_merged_csv_key=agg_merged_key,
                dataset_results_subpath=sub,
            )
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
    if not settings.upload_benchmark_results or not s3_cfg_usable(s3_cfg):
        return
    agg_prefix = join_s3_key(settings.benchmark_s3_prefix, batch_id, "aggregated")
    joined_results_key = join_s3_key(settings.benchmark_s3_prefix, "joined_results.csv")
    finished_at = datetime.now(timezone.utc).isoformat()
    manifest_rel = str(cfg.get("dataset_manifest_path") or "")
    try:
        rel_out = str(output_csv.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel_out = str(output_csv)

    dataset_ids = [str(r.get("dataset_id") or "") for r in rows if r.get("dataset_id")]

    pipeline_entries: list[dict[str, Any]] = []
    try:
        for s3_name, ir_path in _aggregated_pipeline_ir_uploads(settings):
            body = ir_path.read_bytes()
            put_s3_bytes(
                s3_cfg=s3_cfg,
                bucket=bucket,
                key=join_s3_key(agg_prefix, s3_name),
                body=body,
                content_type="application/yaml; charset=utf-8",
            )
            pipeline_entries.append(
                {
                    "relative_s3_key": f"aggregated/{s3_name}",
                    "compiled_ir_sha256": sha256_file(ir_path),
                    "local_source_path": str(ir_path),
                }
            )
            logger.info("Uploaded pipeline IR to s3://%s/%s", bucket, join_s3_key(agg_prefix, s3_name))

        if output_csv.is_file():
            put_s3_bytes(
                s3_cfg=s3_cfg,
                bucket=bucket,
                key=join_s3_key(agg_prefix, "benchmark_runs.csv"),
                body=output_csv.read_bytes(),
                content_type="text/csv; charset=utf-8",
            )

        merged_bytes = b""
        merged_df: Any = None
        try:
            from automl_benchmark.leaderboard_merge import merge_benchmark_csv_with_leaderboards

            merged_df = merge_benchmark_csv_with_leaderboards(
                output_csv,
                include_metrics_blob=False,
                include_rows_without_leaderboard=True,
            )
            buf = io.StringIO()
            merged_df.to_csv(buf, index=False)
            merged_bytes = buf.getvalue().encode("utf-8")
        except Exception as e:
            logger.warning("Could not build merged leaderboard CSV for S3: %s", e)

        if merged_bytes:
            put_s3_bytes(
                s3_cfg=s3_cfg,
                bucket=bucket,
                key=join_s3_key(agg_prefix, "merged_leaderboards.csv"),
                body=merged_bytes,
                content_type="text/csv; charset=utf-8",
            )

        if merged_df is not None and not merged_df.empty:
            try:
                import pandas as pd

                old_raw = try_get_s3_object_bytes(s3_cfg=s3_cfg, bucket=bucket, key=joined_results_key)
                old_df = None
                if old_raw:
                    try:
                        old_df = pd.read_csv(io.BytesIO(old_raw))
                    except Exception as e:
                        logger.warning("Could not parse existing joined_results.csv; rewriting: %s", e)
                out_df = _merge_joined_results_dedupe(old_df, merged_df)
                jbuf = io.StringIO()
                out_df.to_csv(jbuf, index=False)
                put_s3_bytes(
                    s3_cfg=s3_cfg,
                    bucket=bucket,
                    key=joined_results_key,
                    body=jbuf.getvalue().encode("utf-8"),
                    content_type="text/csv; charset=utf-8",
                )
                logger.info("Updated joined results s3://%s/%s (%d rows)", bucket, joined_results_key, len(out_df))
            except Exception as e:
                logger.warning("Could not update joined_results.csv: %s", e)

        batch_meta = build_batch_metadata(
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
            aggregated_pipeline_ir=pipeline_entries or None,
            joined_results_s3_key=joined_results_key,
        )
        batch_meta["cli_dataset_filter"] = dataset_filter

        put_s3_bytes(
            s3_cfg=s3_cfg,
            bucket=bucket,
            key=join_s3_key(agg_prefix, "batch_metadata.json"),
            body=json.dumps(batch_meta, indent=2, default=str).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

        logger.info("Uploaded batch aggregated artifacts to s3://%s/%s/", bucket, agg_prefix)
    except Exception as e:
        logger.warning("S3 batch upload failed: %s", e)
