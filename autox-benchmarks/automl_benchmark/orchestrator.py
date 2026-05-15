"""Coordinates manifest loading, pipeline submissions, waits, and CSV export."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automl_benchmark.config_loader import load_merged_benchmark_config
from benchmark_common.kfp_client import create_kfp_client
from benchmark_common.manifest import load_dataset_entries
from automl_benchmark.pipeline_params import (
    build_pipeline_arguments,
    is_timeseries_dataset,
    pipeline_file_for_dataset,
)
from benchmark_common.pipeline_run import extract_run_id, submit_pipeline_package, wait_for_terminal_run
from automl_benchmark.s3_leaderboard_artifact import (
    discover_leaderboard_html_s3_uri,
    download_leaderboard_html_to_dir,
)
from automl_benchmark.result_rows import (
    base_row_for_dataset,
    completed_row,
    dry_run_row,
    run_name_for_dataset,
    submit_error_row,
    timeout_row,
)
from benchmark_common.results_csv import write_results_csv
from benchmark_common.run_state import is_success_state
from automl_benchmark.settings import benchmark_settings_from_config, BenchmarkSettings
from automl_benchmark.experiment_fingerprint import compute_experiment_fingerprint
from automl_benchmark.s3_benchmark_upload import (
    build_batch_id,
    upload_batch_aggregated,
    upload_single_dataset_results,
)
from automl_benchmark.s3_client import s3_cfg_usable
from automl_benchmark.s3_experiment_dedupe import try_load_cached_result_row
from benchmark_common.pipeline_package_resolve import resolve_automl_pipeline_package_paths

logger = logging.getLogger(__name__)


def _infer_repo_root(config_path: Path) -> Path | None:
    p = config_path.resolve().parent
    if (p / ".git").is_dir():
        return p
    pp = p.parent
    if (pp / ".git").is_dir():
        return pp
    return None


def _maybe_upload_benchmark_row(
    *,
    cfg: dict[str, Any],
    settings: BenchmarkSettings,
    batch_id: str,
    ds: dict[str, Any],
    row: dict[str, Any],
    pipeline_file: Path,
    arguments: dict[str, Any],
    output_csv: Path,
    dataset_filter: str,
    fail_fast: bool,
    repo_root: Path | None,
    experiment_fingerprint: str | None = None,
) -> None:
    if not settings.upload_benchmark_results:
        return
    s3_cfg = cfg.get("s3")
    if not isinstance(s3_cfg, dict) or not s3_cfg_usable(s3_cfg):
        return
    if not str(row.get("run_id") or "").strip():
        return
    root = (
        settings.artifact_s3_root_timeseries
        if is_timeseries_dataset(ds)
        else settings.artifact_s3_root_tabular
    )
    upload_single_dataset_results(
        s3_cfg=s3_cfg,
        bucket=settings.train_data_bucket_name,
        settings=settings,
        cfg=cfg,
        batch_id=batch_id,
        dataset=ds,
        row=row,
        pipeline_ir_path=pipeline_file.resolve(),
        output_csv_parent=output_csv.resolve().parent,
        arguments=arguments,
        dataset_filter=dataset_filter,
        fail_fast=fail_fast,
        artifact_s3_root=root,
        repo_root=repo_root,
        experiment_fingerprint=experiment_fingerprint,
    )


def _dataset_matches_filter(ds: dict[str, Any], dataset_filter: str) -> bool:
    if dataset_filter == "all":
        return True
    ts = is_timeseries_dataset(ds)
    if dataset_filter == "tabular":
        return not ts
    if dataset_filter == "timeseries":
        return ts
    return True


def _validate_dataset_entry(ds: dict[str, Any], ds_id: str) -> str | None:
    if not ds.get("train_data_file_key"):
        return f"Dataset {ds_id} missing train_data_file_key"
    if is_timeseries_dataset(ds):
        if not ds.get("id_column") or not ds.get("timestamp_column"):
            return (
                f"Dataset {ds_id} (task_type=timeseries) requires id_column and timestamp_column"
            )
        if not (ds.get("target") or ds.get("label_column")):
            return f"Dataset {ds_id} (task_type=timeseries) requires target or label_column"
        return None
    if not ds.get("label_column") or not ds.get("task_type"):
        return f"Dataset {ds_id} missing label_column or task_type"
    return None


class BenchmarkOrchestrator:
    """High-level benchmark run: one pipeline run per dataset entry, then aggregate CSV."""

    def __init__(self, config_path: Path, credentials_ini_path: Path | None = None) -> None:
        self.config_path = config_path.resolve()
        self.credentials_ini_path = credentials_ini_path

    def load_config_and_datasets(
        self,
        *,
        dataset_filter: str = "all",
        tabular_package_path_cli: str | None = None,
        timeseries_package_path_cli: str | None = None,
    ) -> tuple[dict[str, Any], BenchmarkSettings, list[dict[str, Any]], Path]:
        cfg, config_dir = load_merged_benchmark_config(self.config_path, self.credentials_ini_path)
        datasets = load_dataset_entries(cfg, config_dir)

        needs_tabular = False
        needs_ts = False
        for ds in datasets:
            if not _dataset_matches_filter(ds, dataset_filter):
                continue
            if is_timeseries_dataset(ds):
                needs_ts = True
            else:
                needs_tabular = True

        resolve_automl_pipeline_package_paths(
            cfg,
            config_dir,
            cli_tabular=tabular_package_path_cli,
            cli_timeseries=timeseries_package_path_cli,
            needs_tabular=needs_tabular,
            needs_timeseries=needs_ts,
        )
        settings = benchmark_settings_from_config(cfg, config_dir)
        return cfg, settings, datasets, config_dir

    def execute(
        self,
        *,
        output_csv: Path,
        dry_run: bool = False,
        fail_fast: bool = False,
        dataset_filter: str = "all",
        skip_identical_runs: bool = True,
        tabular_package_path_cli: str | None = None,
        timeseries_package_path_cli: str | None = None,
    ) -> int:
        try:
            cfg, settings, datasets, _ = self.load_config_and_datasets(
                dataset_filter=dataset_filter,
                tabular_package_path_cli=tabular_package_path_cli,
                timeseries_package_path_cli=timeseries_package_path_cli,
            )
        except Exception as e:
            logger.error("%s", e)
            return 1

        batch_id = build_batch_id()
        started_at = datetime.now(timezone.utc).isoformat()
        repo_root = _infer_repo_root(self.config_path)

        needs_tabular = False
        needs_ts = False
        for ds in datasets:
            if not _dataset_matches_filter(ds, dataset_filter):
                continue
            if is_timeseries_dataset(ds):
                needs_ts = True
            else:
                needs_tabular = True
        if needs_tabular and not settings.pipeline_yaml.is_file():
            logger.error("Tabular pipeline package not found: %s", settings.pipeline_yaml)
            return 1
        if needs_ts and not settings.timeseries_pipeline_yaml.is_file():
            logger.error("Time series pipeline package not found: %s", settings.timeseries_pipeline_yaml)
            return 1

        rows: list[dict[str, Any]] = []
        client = None
        if not dry_run:
            try:
                client = create_kfp_client(cfg)
            except Exception as e:
                logger.error("KFP client failed: %s", e)
                return 1

        for i, ds in enumerate(datasets):
            ds_id = str(ds.get("id", ds.get("name", f"dataset_{i}")))
            if not _dataset_matches_filter(ds, dataset_filter):
                logger.info("Skipping dataset %s (dataset_filter=%s)", ds_id, dataset_filter)
                continue

            err = _validate_dataset_entry(ds, ds_id)
            if err:
                logger.error("%s", err)
                if fail_fast:
                    return 1
                continue

            try:
                arguments = build_pipeline_arguments(ds, settings)
            except ValueError as e:
                logger.error("Dataset %s: %s", ds_id, e)
                if fail_fast:
                    return 1
                continue

            pipeline_file = pipeline_file_for_dataset(ds, settings)
            run_name = run_name_for_dataset(settings.run_name_prefix, ds_id)
            base = base_row_for_dataset(ds, i, run_name, settings.top_n)

            if dry_run:
                rows.append(dry_run_row(base, arguments))
                logger.info("DRY_RUN %s pipeline=%s -> %s", ds_id, pipeline_file.name, arguments)
                continue

            assert client is not None

            experiment_fp: str | None = None
            if skip_identical_runs:
                s3_cfg_dedupe = cfg.get("s3")
                if isinstance(s3_cfg_dedupe, dict) and s3_cfg_usable(s3_cfg_dedupe):
                    experiment_fp = compute_experiment_fingerprint(
                        pipeline_ir_path=pipeline_file.resolve(),
                        pipeline_arguments=dict(arguments),
                        dataset=ds,
                        settings=settings,
                        cfg=cfg,
                        s3_cfg=s3_cfg_dedupe,
                        dataset_filter=dataset_filter,
                    )
                    cached = try_load_cached_result_row(
                        s3_cfg=s3_cfg_dedupe,
                        bucket=settings.train_data_bucket_name,
                        benchmark_s3_prefix=settings.benchmark_s3_prefix,
                        fingerprint=experiment_fp,
                    )
                    if cached is not None:
                        rows.append(cached)
                        continue

            try:
                run_result = submit_pipeline_package(
                    client,
                    pipeline_file=str(pipeline_file),
                    arguments=arguments,
                    run_name=run_name,
                    experiment_name=settings.experiment_name,
                    enable_caching=settings.enable_caching,
                )
                rid = extract_run_id(run_result)
                logger.info("Started run_id=%s dataset=%s", rid, ds_id)

                detail, timed_out = wait_for_terminal_run(
                    client,
                    rid,
                    timeout_seconds=settings.timeout_seconds,
                    poll_interval_seconds=settings.poll_interval_seconds,
                )
                if timed_out:
                    tout = timeout_row(base, rid, settings.timeout_seconds)
                    rows.append(tout)
                    _maybe_upload_benchmark_row(
                        cfg=cfg,
                        settings=settings,
                        batch_id=batch_id,
                        ds=ds,
                        row=tout,
                        pipeline_file=pipeline_file,
                        arguments=arguments,
                        output_csv=output_csv,
                        dataset_filter=dataset_filter,
                        fail_fast=fail_fast,
                        repo_root=repo_root,
                        experiment_fingerprint=experiment_fp,
                    )
                    logger.error("Timeout waiting for run %s", rid)
                    if fail_fast:
                        break
                    continue

                if detail is None:
                    detail = client.get_run(rid)
                row = completed_row(base, rid, detail)
                state = str(row.get("state", ""))
                if is_success_state(state) and rid.strip():
                    s3_cfg = cfg.get("s3")
                    if isinstance(s3_cfg, dict) and s3_cfg:
                        root = (
                            settings.artifact_s3_root_timeseries
                            if is_timeseries_dataset(ds)
                            else settings.artifact_s3_root_tabular
                        )
                        row["leaderboard_html_s3_uri"] = discover_leaderboard_html_s3_uri(
                            bucket=settings.train_data_bucket_name,
                            s3_cfg=s3_cfg,
                            run_id=rid,
                            is_timeseries=is_timeseries_dataset(ds),
                            artifact_root_prefix=root,
                        )
                        uri = str(row.get("leaderboard_html_s3_uri") or "").strip()
                        if uri:
                            local_rel = download_leaderboard_html_to_dir(
                                s3_cfg,
                                uri,
                                output_csv.resolve().parent,
                                run_id=rid,
                            )
                            if local_rel:
                                row["leaderboard_html_path"] = local_rel
                rows.append(row)
                _maybe_upload_benchmark_row(
                    cfg=cfg,
                    settings=settings,
                    batch_id=batch_id,
                    ds=ds,
                    row=row,
                    pipeline_file=pipeline_file,
                    arguments=arguments,
                    output_csv=output_csv,
                    dataset_filter=dataset_filter,
                    fail_fast=fail_fast,
                    repo_root=repo_root,
                    experiment_fingerprint=experiment_fp,
                )

                state = rows[-1].get("state", "")
                if not is_success_state(str(state)) and fail_fast:
                    logger.error("Run %s ended with state=%s", rid, state)
                    break

            except Exception as e:
                logger.exception("Run failed for dataset %s", ds_id)
                rows.append(submit_error_row(base, str(e)))
                if fail_fast:
                    break

        write_results_csv(rows, output_csv)
        logger.info("Wrote %d row(s) to %s", len(rows), output_csv)

        if not dry_run:
            s3_cfg = cfg.get("s3")
            if (
                settings.upload_benchmark_results
                and isinstance(s3_cfg, dict)
                and s3_cfg_usable(s3_cfg)
            ):
                upload_batch_aggregated(
                    s3_cfg=s3_cfg,
                    bucket=settings.train_data_bucket_name,
                    settings=settings,
                    cfg=cfg,
                    batch_id=batch_id,
                    started_at=started_at,
                    output_csv=output_csv.resolve(),
                    rows=rows,
                    dataset_filter=dataset_filter,
                    repo_root=repo_root,
                )
        return 0
