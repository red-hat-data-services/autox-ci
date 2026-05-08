"""Command-line entry for the benchmark orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from automl_benchmark.orchestrator import BenchmarkOrchestrator


def default_config_path() -> Path:
    env_p = os.environ.get("BENCHMARK_CONFIG_PATH")
    if env_p:
        return Path(env_p)
    return Path("config/benchmark.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AutoML KFP benchmark suite and aggregate CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: $BENCHMARK_CONFIG_PATH or config/benchmark.yaml)",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Required INI: [kfp]/[storage]/[pipeline]/[s3] — cluster identity (default: "
            "$BENCHMARK_CREDENTIALS_PATH or config/credentials.ini)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_runs.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print runs only; do not call KFP")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first pipeline failure",
    )
    parser.add_argument(
        "--dataset-filter",
        choices=("all", "tabular", "timeseries"),
        default="all",
        metavar="MODE",
        help=(
            "Run only tabular (binary/multiclass/regression) or only time series "
            "(task_type=timeseries) manifest rows"
        ),
    )
    parser.add_argument(
        "--rerun-identical-experiments",
        action="store_true",
        help=(
            "Always submit KFP runs even when S3 has an experiment_index entry for the same "
            "pipeline, dataset, parameters, and environment (dedupe is on by default)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg_path = (args.config or default_config_path()).resolve()
    if not cfg_path.is_file():
        logging.getLogger(__name__).error(
            "Config not found: %s (copy from config/benchmark.example.yaml)",
            cfg_path,
        )
        return 1

    orch = BenchmarkOrchestrator(cfg_path, credentials_ini_path=args.credentials)
    return orch.execute(
        output_csv=args.output,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        dataset_filter=args.dataset_filter,
        skip_identical_runs=not args.rerun_identical_experiments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
