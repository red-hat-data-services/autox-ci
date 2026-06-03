"""Command-line entry for the AutoRAG benchmark orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from autorag_benchmark.orchestrator import BenchmarkOrchestrator


def default_config_path() -> Path:
    env_p = os.environ.get("BENCHMARK_CONFIG_PATH")
    if env_p:
        return Path(env_p)
    return Path("config/benchmark.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG optimization KFP benchmark suite and aggregate CSV.")
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
        choices=("all",),
        default="all",
        metavar="MODE",
        help="Dataset filter (only 'all' supported for RAG benchmarks)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--package-path",
        type=str,
        default=os.environ.get("BENCHMARK_PACKAGE_PATH")
        or os.environ.get("RAG_PACKAGE_PATH")
        or None,
        metavar="PATH",
        help=(
            "Compiled RAG pipeline YAML (skips Git compile). "
            "Default: $BENCHMARK_PACKAGE_PATH or $RAG_PACKAGE_PATH"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg_path = (args.config or default_config_path()).resolve()
    if not cfg_path.is_file():
        logging.getLogger(__name__).error(
            "Config not found: %s (copy from templates/benchmark.autorag.example.yaml)",
            cfg_path,
        )
        return 1

    orch = BenchmarkOrchestrator(cfg_path, credentials_ini_path=args.credentials)
    return orch.execute(
        output_csv=args.output,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        dataset_filter=args.dataset_filter,
        package_path_cli=args.package_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
