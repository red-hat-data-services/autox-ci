"""Command-line entry for the AutoRAG benchmark orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
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
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to .env with cluster/S3 settings (default: .env in project root)",
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
    parser.add_argument(
        "--managed-pipelines",
        action="store_true",
        help=(
            "Use DSPA-managed KFP pipelines instead of uploading a compiled YAML. "
            "Equivalent to BENCHMARK_USE_MANAGED_PIPELINES=true"
        ),
    )
    parser.add_argument(
        "--run-indexing",
        action="store_true",
        help=(
            "After HPO, extract the best pattern and submit a documents-indexing-pipeline "
            "run on the full corpus (using full_input_data_key from the manifest)."
        ),
    )
    parser.add_argument(
        "--pattern-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override automatic best-pattern selection with a specific pattern name.",
    )
    parser.add_argument(
        "--run-e2e-evaluation",
        action="store_true",
        help="After indexing, evaluate the selected HPO pattern against the full QA set and indexed collection.",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Write a self-contained HTML indexing and full-corpus quality report next to the CSV.",
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

    if args.managed_pipelines:
        os.environ["BENCHMARK_USE_MANAGED_PIPELINES"] = "true"

    if args.run_indexing:
        os.environ["BENCHMARK_RUN_INDEXING"] = "true"
    if args.pattern_name:
        os.environ["BENCHMARK_INDEXING_PATTERN_NAME"] = args.pattern_name
    if args.run_e2e_evaluation:
        os.environ["BENCHMARK_RUN_E2E_EVALUATION"] = "true"
    if args.generate_report:
        os.environ["BENCHMARK_GENERATE_REPORT"] = "true"

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

    orch = BenchmarkOrchestrator(cfg_path, env_file=args.env_file)
    return orch.execute(
        output_csv=args.output,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        dataset_filter=args.dataset_filter,
        package_path_cli=args.package_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
