"""Coordinates manifest loading, pipeline submissions, waits, and CSV export (AutoRAG)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autorag_benchmark.config_loader import load_merged_benchmark_config
from autorag_benchmark.e2e_evaluation import run_full_corpus_evaluation
from autorag_benchmark.indexing_from_hpo import (
    build_indexing_arguments,
    extract_best_pattern_settings,
    resolve_indexing_pipeline_target,
)
from autorag_benchmark.indexing_report_generator import generate_indexing_report
from autorag_benchmark.metrics import calculate_scale_drift, extract_profiling_metrics, extract_quality_metrics
from autorag_benchmark.pattern_scores import extract_pattern_scores_tabular
from autorag_benchmark.pipeline_params import build_pipeline_arguments
from autorag_benchmark.result_rows import (
    base_row_for_dataset,
    completed_row,
    dry_run_row,
    indexing_row,
    run_name_for_dataset,
    submit_error_row,
    timeout_row,
)
from autorag_benchmark.s3_benchmark_upload import (
    build_batch_id,
    upload_batch_aggregated,
    upload_single_dataset_results,
)
from autorag_benchmark.settings import BenchmarkSettings, benchmark_settings_from_config
from benchmark_common.kfp_client import create_kfp_client
from benchmark_common.managed_pipelines import PipelineRunTarget
from benchmark_common.manifest import load_dataset_entries
from benchmark_common.pipeline_run import extract_run_id, filter_pipeline_arguments, redact_arguments, submit_pipeline_run, wait_for_terminal_run
from benchmark_common.pipeline_target_resolve import resolve_autorag_pipeline_target
from benchmark_common.results_csv import write_results_csv
from benchmark_common.run_state import is_success_state, read_run_state, unwrap_run_from_get_run
from benchmark_common.run_timing import duration_seconds, parse_timestamp

logger = logging.getLogger(__name__)


def _dataset_matches_filter(ds: dict[str, Any], dataset_filter: str) -> bool:
    return dataset_filter == "all"


def _validate_dataset_entry(ds: dict[str, Any], ds_id: str) -> str | None:
    if not ds.get("test_data_key"):
        return f"Dataset {ds_id} missing test_data_key (path to test data JSON file)"
    return None


class BenchmarkOrchestrator:
    """High-level RAG benchmark run: one pipeline run per dataset entry, then aggregate CSV."""

    def __init__(self, config_path: Path, env_file: Path | None = None) -> None:
        self.config_path = config_path.resolve()
        self.env_file = env_file

    def load_config_and_datasets(
        self,
        *,
        package_path_cli: str | None = None,
        client: Any = None,
        create_client: bool = False,
    ) -> tuple[dict[str, Any], BenchmarkSettings, list[dict[str, Any]], PipelineRunTarget, Any]:
        cfg, config_dir = load_merged_benchmark_config(self.config_path, self.env_file)
        if client is None and create_client:
            client = create_kfp_client(cfg)
        target = resolve_autorag_pipeline_target(
            cfg, config_dir, client, cli_package=package_path_cli,
        )
        settings = benchmark_settings_from_config(cfg, config_dir)
        datasets = load_dataset_entries(cfg, config_dir)
        return cfg, settings, datasets, target, client

    def execute(
        self,
        *,
        output_csv: Path,
        dry_run: bool = False,
        fail_fast: bool = False,
        dataset_filter: str = "all",
        package_path_cli: str | None = None,
    ) -> int:
        try:
            cfg, settings, datasets, target, client = self.load_config_and_datasets(
                package_path_cli=package_path_cli,
                create_client=not dry_run,
            )
        except Exception as e:
            logger.error("%s", e)
            return 1

        is_managed = settings.pipeline_mode == "managed"

        # S3 upload preparation
        batch_id = build_batch_id()
        started_at = datetime.now(timezone.utc).isoformat()
        s3_cfg = cfg.get("s3")
        if not isinstance(s3_cfg, dict):
            s3_cfg = {}
        bucket = settings.test_data_bucket_name
        repo_root = None
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
        except Exception:
            pass

        if settings.upload_benchmark_results:
            logger.info("Benchmark batch_id=%s (results will upload to s3://%s/%s/)", batch_id, bucket, settings.benchmark_s3_prefix)

        rows: list[dict[str, Any]] = []
        pipeline_file = Path(target.package_path) if target.package_path else None

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
            except (ValueError, KeyError) as e:
                logger.error("Dataset %s: %s", ds_id, e)
                if fail_fast:
                    return 1
                continue

            if not is_managed and pipeline_file is not None:
                arguments = filter_pipeline_arguments(arguments, pipeline_file)

            run_name = run_name_for_dataset(settings.run_name_prefix, ds_id)
            base = base_row_for_dataset(ds, i, run_name, suite=settings.suite, rhoai_version=settings.rhoai_version)

            if dry_run:
                rows.append(dry_run_row(base, arguments))
                label = target.kfp_pipeline_name if is_managed else (pipeline_file.name if pipeline_file else "unknown")
                logger.info("DRY_RUN %s pipeline=%s -> %s", ds_id, label, redact_arguments(arguments))
                continue

            assert client is not None
            try:
                run_result = submit_pipeline_run(
                    client,
                    target,
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
                    rows.append(timeout_row(base, rid, settings.timeout_seconds))
                    logger.error("Timeout waiting for run %s", rid)
                    if fail_fast:
                        break
                    continue

                if detail is None:
                    detail = client.get_run(rid)

                base_row = completed_row(base, rid, detail)

                # Extract pattern scores for successful runs and create one row per pattern
                state = base_row.get("state", "")
                dataset_rows: list[dict[str, Any]] = []

                if is_success_state(str(state)):
                    logger.info("Extracting pattern scores from S3 for run %s (bucket: %s)", rid, bucket)

                    # Check if S3 credentials are configured
                    s3_cfg = cfg.get("s3", {})
                    if not s3_cfg.get("aws_access_key_id") or not s3_cfg.get("aws_secret_access_key"):
                        logger.warning("S3 credentials not configured in config. Skipping pattern extraction.")
                        logger.warning("To enable pattern extraction, set AWS_* keys in .env")
                        dataset_rows.append(base_row)
                    else:
                        try:
                            optimization_metric = ds.get("optimization_metric", "faithfulness")
                            pattern_rows = extract_pattern_scores_tabular(
                                run_id=rid,
                                config=cfg,
                                bucket=bucket,
                                optimization_metric=optimization_metric,
                            )

                            if pattern_rows:
                                # Create one row per pattern
                                for pattern_row in pattern_rows:
                                    row = {**base_row, **pattern_row}
                                    dataset_rows.append(row)

                                logger.info(
                                    "Extracted %d pattern rows for dataset %s (best score: %.4f)",
                                    len(pattern_rows),
                                    ds_id,
                                    max((p.get("final_score") or 0) for p in pattern_rows),
                                )
                                logger.info("Pattern configuration columns: %s", list(pattern_rows[0].keys()))
                            else:
                                # No patterns extracted, keep base row only
                                logger.warning("No patterns extracted for run %s, using base row only", rid)
                                logger.warning("Check if pattern.json files exist in S3 at: s3://%s/documents-rag-optimization-pipeline/%s/rag-templates-optimization/*/rag_patterns/*/pattern.json", bucket, rid)
                                dataset_rows.append(base_row)

                        except Exception as e:
                            logger.error("Failed to extract pattern scores for run %s: %s", rid, e, exc_info=True)
                            logger.error("Verify S3 credentials and pattern.json file locations")
                            # Fall back to base row on error
                            dataset_rows.append(base_row)
                else:
                    # Failed run, keep single base row
                    dataset_rows.append(base_row)

                rows.extend(dataset_rows)

                # Upload dataset results to S3 (use first row for metadata, will contain all patterns)
                upload_row = dataset_rows[0] if dataset_rows else base_row
                upload_single_dataset_results(
                    s3_cfg=s3_cfg,
                    bucket=bucket,
                    settings=settings,
                    cfg=cfg,
                    batch_id=batch_id,
                    dataset=ds,
                    row=upload_row,
                    pipeline_yaml_path=pipeline_file,
                    kfp_pipeline_name=target.kfp_pipeline_name,
                    arguments=arguments,
                    dataset_filter=dataset_filter,
                    fail_fast=fail_fast,
                    repo_root=repo_root,
                )

                # ── Step 2: HPO -> Indexing -> E2E evaluation ──
                if (
                    is_success_state(str(state))
                    and settings.run_indexing
                    and not dry_run
                ):
                    try:
                        indexing_rows = self._run_indexing_from_hpo(
                            cfg=cfg,
                            settings=settings,
                            client=client,
                            dataset=ds,
                            hpo_run_id=rid,
                            base_row=base,
                            bucket=bucket,
                        )
                        rows.extend(indexing_rows)
                    except Exception as idx_exc:
                        logger.exception(
                            "Indexing step failed for dataset %s (HPO run %s)", ds_id, rid,
                        )
                        rows.append(indexing_row(
                            base, rid, "", "ERROR", "",
                            input_data_key_hpo=ds.get("input_data_key", ""),
                            indexing_report={"indexing_error": str(idx_exc)},
                        ))

                if not is_success_state(str(state)) and fail_fast:
                    logger.error("Run %s ended with state=%s", rid, state)
                    break

            except Exception as exc:
                logger.exception("Run failed for dataset %s", ds_id)
                error_row = submit_error_row(base, str(exc))
                rows.append(error_row)

                # Upload error result to S3
                upload_single_dataset_results(
                    s3_cfg=s3_cfg,
                    bucket=bucket,
                    settings=settings,
                    cfg=cfg,
                    batch_id=batch_id,
                    dataset=ds,
                    row=error_row,
                    pipeline_yaml_path=pipeline_file,
                    kfp_pipeline_name=target.kfp_pipeline_name,
                    arguments=arguments,
                    dataset_filter=dataset_filter,
                    fail_fast=fail_fast,
                    repo_root=repo_root,
                )

                if fail_fast:
                    break

        write_results_csv(rows, output_csv)
        logger.info("Wrote %d row(s) to %s", len(rows), output_csv)
        if settings.generate_report:
            # The HTML report is a nice-to-have; never let it block the S3 upload.
            report_path = output_csv.with_name(f"{output_csv.stem}_indexing_report.html")
            try:
                generate_indexing_report(rows, report_path)
                logger.info("Wrote indexing report to %s", report_path)
            except Exception as exc:
                logger.warning("Could not write indexing report: %s", exc)

        upload_batch_aggregated(
            s3_cfg=s3_cfg,
            bucket=bucket,
            settings=settings,
            cfg=cfg,
            batch_id=batch_id,
            started_at=started_at,
            output_csv=output_csv,
            rows=rows,
            dataset_filter=dataset_filter,
            repo_root=repo_root,
        )

        return 0

    def _run_indexing_from_hpo(
        self,
        *,
        cfg: dict[str, Any],
        settings: BenchmarkSettings,
        client: Any,
        dataset: dict[str, Any],
        hpo_run_id: str,
        base_row: dict[str, Any],
        bucket: str,
    ) -> list[dict[str, Any]]:
        """Chain: extract best HPO pattern -> submit indexing pipeline -> collect results."""
        ds_id = dataset.get("id", "unknown")

        pattern_settings = extract_best_pattern_settings(
            run_id=hpo_run_id,
            config=cfg,
            bucket=bucket,
            pattern_name_override=settings.indexing_pattern_name or None,
        )

        full_key = dataset.get("full_input_data_key") or dataset.get("input_data_key", "")
        if not dataset.get("full_input_data_key"):
            logger.warning(
                "Dataset %s has no full_input_data_key; indexing will reuse the HPO "
                "subsample key %r, which defeats full-corpus evaluation.",
                ds_id, full_key,
            )
        indexing_args = build_indexing_arguments(
            pattern_settings, settings, full_input_data_key=full_key,
        )

        indexing_target = resolve_indexing_pipeline_target(cfg, client)

        indexing_run_name = run_name_for_dataset(
            f"{settings.run_name_prefix}-indexing", ds_id,
        )
        logger.info(
            "Submitting indexing run %s for dataset %s (HPO run %s, pattern %s)",
            indexing_run_name,
            ds_id,
            hpo_run_id,
            pattern_settings.get("pattern_name"),
        )

        idx_result = submit_pipeline_run(
            client,
            indexing_target,
            arguments=indexing_args,
            run_name=indexing_run_name,
            experiment_name=settings.experiment_name,
            enable_caching=settings.enable_caching,
        )
        idx_rid = extract_run_id(idx_result)
        logger.info("Indexing run_id=%s for dataset=%s", idx_rid, ds_id)

        idx_detail, idx_timed_out = wait_for_terminal_run(
            client,
            idx_rid,
            timeout_seconds=settings.indexing_timeout_seconds or settings.timeout_seconds,
            poll_interval_seconds=settings.poll_interval_seconds,
        )

        pattern_name = pattern_settings.get("pattern_name", "")
        row_kwargs = {
            "pattern_name": pattern_name,
            "pattern_score": pattern_settings.get("final_score"),
            "input_data_key_hpo": dataset.get("input_data_key", ""),
            "input_data_key_indexing": full_key,
        }

        if idx_timed_out:
            return [indexing_row(
                base_row, hpo_run_id, idx_rid, "TIMEOUT",
                settings.indexing_timeout_seconds or settings.timeout_seconds,
                **row_kwargs,
            )]

        if idx_detail is None:
            idx_detail = client.get_run(idx_rid)
        idx_run = unwrap_run_from_get_run(idx_detail) or idx_detail
        idx_state = read_run_state(idx_run)
        created = parse_timestamp(getattr(idx_run, "created_at", None))
        finished = parse_timestamp(getattr(idx_run, "finished_at", None))
        dur = duration_seconds(created, finished)

        report_flat, e2e_metrics = self._collect_indexing_metrics(
            cfg=cfg,
            settings=settings,
            dataset=dataset,
            hpo_run_id=hpo_run_id,
            indexing_run_id=idx_rid,
            indexing_state=str(idx_state),
            indexing_duration=dur,
            pattern_name=pattern_name,
            bucket=bucket,
        )

        return [indexing_row(
            base_row, hpo_run_id, idx_rid, idx_state, dur,
            indexing_report=report_flat,
            e2e_metrics=e2e_metrics,
            **row_kwargs,
        )]

    def _collect_indexing_metrics(
        self,
        *,
        cfg: dict[str, Any],
        settings: BenchmarkSettings,
        dataset: dict[str, Any],
        hpo_run_id: str,
        indexing_run_id: str,
        indexing_state: str,
        indexing_duration: float | str,
        pattern_name: str,
        bucket: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Gather indexing profiling, HPO baseline, e2e quality, and scale-drift metrics.

        Returns ``(indexing_report_columns, e2e_metric_columns)``.
        """
        wall_time = indexing_duration if isinstance(indexing_duration, (int, float)) else None

        report_flat: dict[str, Any] = {}
        if is_success_state(indexing_state):
            # extract_profiling_metrics already flattens indexing_report.json, so we
            # do not call extract_indexing_report separately (avoids a duplicate read).
            try:
                report_flat = extract_profiling_metrics(
                    indexing_run_id, cfg, bucket, indexing_wall_time=wall_time,
                )
            except Exception as exc:
                logger.warning("Could not collect indexing metrics: %s", exc)
                report_flat = {"indexing_profiling_error": str(exc)}

        try:
            hpo_quality = extract_quality_metrics(
                hpo_run_id, cfg, bucket, pattern_name=pattern_name, prefix="hpo_",
            )
        except Exception as exc:
            logger.warning("Could not collect HPO quality baseline: %s", exc)
            hpo_quality = {"hpo_quality_error": str(exc)}

        e2e_flat: dict[str, Any] = {}
        if is_success_state(indexing_state) and settings.run_e2e_evaluation:
            e2e_flat = run_full_corpus_evaluation(
                indexing_run_id=indexing_run_id,
                hpo_run_id=hpo_run_id,
                config_path=self.config_path,
                env_file=self.env_file,
                config=cfg,
                bucket=bucket,
                test_data_key=str(dataset.get("test_data_key") or ""),
                pattern_name=pattern_name,
                timeout_seconds=settings.indexing_timeout_seconds or settings.timeout_seconds,
            )

        scale_drift = calculate_scale_drift(hpo_quality, e2e_flat) if e2e_flat else {}
        return report_flat, {**hpo_quality, **e2e_flat, **scale_drift}
