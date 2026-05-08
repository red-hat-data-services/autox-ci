"""Coordinates manifest loading, pipeline submissions, waits, and CSV export (AutoRAG)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autorag_benchmark.config_loader import load_merged_benchmark_config
from autorag_benchmark.pipeline_params import build_pipeline_arguments, pipeline_file_for_dataset
from autorag_benchmark.result_rows import (
    base_row_for_dataset,
    completed_row,
    dry_run_row,
    run_name_for_dataset,
    submit_error_row,
    timeout_row,
)
from autorag_benchmark.settings import BenchmarkSettings, benchmark_settings_from_config
from benchmark_common.kfp_client import create_kfp_client
from benchmark_common.manifest import load_dataset_entries
from benchmark_common.pipeline_run import extract_run_id, submit_pipeline_package, wait_for_terminal_run
from benchmark_common.results_csv import write_results_csv
from benchmark_common.run_state import is_success_state

logger = logging.getLogger(__name__)


def _dataset_matches_filter(ds: dict[str, Any], dataset_filter: str) -> bool:
    return dataset_filter == "all"


def _validate_dataset_entry(ds: dict[str, Any], ds_id: str) -> str | None:
    if not ds.get("test_data_key"):
        return f"Dataset {ds_id} missing test_data_key (path to test data JSON file)"
    return None


class BenchmarkOrchestrator:
    """High-level RAG benchmark run: one pipeline run per dataset entry, then aggregate CSV."""

    def __init__(self, config_path: Path, credentials_ini_path: Path | None = None) -> None:
        self.config_path = config_path.resolve()
        self.credentials_ini_path = credentials_ini_path

    def load_config_and_datasets(self) -> tuple[dict[str, Any], BenchmarkSettings, list[dict[str, Any]]]:
        cfg, config_dir = load_merged_benchmark_config(self.config_path, self.credentials_ini_path)
        settings = benchmark_settings_from_config(cfg, config_dir)
        datasets = load_dataset_entries(cfg, config_dir)
        return cfg, settings, datasets

    def execute(
        self,
        *,
        output_csv: Path,
        dry_run: bool = False,
        fail_fast: bool = False,
        dataset_filter: str = "all",
    ) -> int:
        try:
            _cfg, settings, datasets = self.load_config_and_datasets()
        except Exception as e:
            logger.error("%s", e)
            return 1

        if not settings.pipeline_yaml.is_file():
            logger.error("RAG pipeline package not found: %s", settings.pipeline_yaml)
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
            except (ValueError, KeyError) as e:
                logger.error("Dataset %s: %s", ds_id, e)
                if fail_fast:
                    return 1
                continue

            pipeline_file = pipeline_file_for_dataset(ds, settings)
            run_name = run_name_for_dataset(settings.run_name_prefix, ds_id)
            base = base_row_for_dataset(ds, i, run_name)

            if dry_run:
                rows.append(dry_run_row(base, arguments))
                logger.info("DRY_RUN %s pipeline=%s -> %s", ds_id, pipeline_file.name, arguments)
                continue

            assert client is not None
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
                    rows.append(timeout_row(base, rid, settings.timeout_seconds))
                    logger.error("Timeout waiting for run %s", rid)
                    if fail_fast:
                        break
                    continue

                if detail is None:
                    detail = client.get_run(rid)
                rows.append(completed_row(base, rid, detail))

                state = rows[-1].get("state", "")
                if not is_success_state(str(state)) and fail_fast:
                    logger.error("Run %s ended with state=%s", rid, state)
                    break

            except Exception as exc:
                logger.exception("Run failed for dataset %s", ds_id)
                rows.append(submit_error_row(base, str(exc)))
                if fail_fast:
                    break

        write_results_csv(rows, output_csv)
        logger.info("Wrote %d row(s) to %s", len(rows), output_csv)
        return 0
