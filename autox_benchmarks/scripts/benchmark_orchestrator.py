#!/usr/bin/env python3
"""
Submit AutoGluon KFP pipeline runs (tabular and/or time series) for each dataset in a
manifest, wait for completion, and write a single CSV summary. Successful runs with S3
credentials also write ``leaderboards/<run_id>.html`` next to the CSV when the leaderboard
object is found.

Implementation lives in the ``automl_benchmark`` package; this file is a thin CLI wrapper.

Configuration:
  - YAML ($BENCHMARK_CONFIG_PATH / config/benchmark.yaml): ``pipeline.compile`` (default: clone
    opendatahub-io/pipelines-components and compile tabular/time-series ``pipeline.py``), or optional
    static ``pipeline.package_path`` / ``timeseries_package_path`` when those files exist; run tuning;
    manifest. Static IR overrides (first match wins): CLI ``--tabular-package-path`` /
    ``--timeseries-package-path``, env ``$BENCHMARK_TABULAR_PACKAGE_PATH`` /
    ``$BENCHMARK_TIMESERIES_PACKAGE_PATH``, ``benchmark.yaml`` ``pipeline:``, or
    ``.env`` ``BENCHMARK_TABULAR_PACKAGE_PATH`` / ``BENCHMARK_TIMESERIES_PACKAGE_PATH``.
  - .env (required): KFP host/namespace/token, bucket, pipeline secret name, AWS keys for
    leaderboard discovery, uploads, and experiment dedupe (skip identical runs by default;
    use ``--rerun-identical-experiments`` to force new pipelines).
    Copy ``.env.example`` to ``.env`` (or ``--env-file PATH``).

Usage:
  pip install -e .
  cp .env.example .env
  cp templates/benchmark.example.yaml config/benchmark.yaml
  python scripts/benchmark_orchestrator.py --output results/benchmark_runs.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from automl_benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
