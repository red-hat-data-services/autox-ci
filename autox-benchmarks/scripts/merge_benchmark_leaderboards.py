#!/usr/bin/env python3
"""
Merge ``benchmark_runs.csv`` with leaderboard HTML tables saved under ``leaderboards/``.

For each row with a non-empty ``leaderboard_html_path`` (relative to the CSV directory),
parses the HTML table and emits one CSV row per leaderboard row, prefixed with benchmark
metadata (dataset_id, run_id, state, etc.).

Requires ``lxml`` (``pip install -r requirements-benchmark.txt``).

Examples:
  python scripts/merge_benchmark_leaderboards.py results/benchmark_runs.csv \\
      -o results/benchmark_leaderboard_merged.csv

  # Keep runs that have no HTML file (single row with leaderboard_parse_ok=false)
  python scripts/merge_benchmark_leaderboards.py results/benchmark_runs.csv -o out.csv \\
      --include-without-leaderboard

  # Attach the large metrics_blob column
  python scripts/merge_benchmark_leaderboards.py results/benchmark_runs.csv -o out.csv \\
      --include-metrics-blob
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge benchmark_runs.csv with parsed leaderboard HTML into one long CSV.",
    )
    parser.add_argument(
        "benchmark_csv",
        type=Path,
        help="Path to benchmark_runs.csv (leaderboard HTML paths are relative to this file's directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-without-leaderboard",
        action="store_true",
        help="Emit a row for runs missing HTML or empty tables (leaderboard_parse_ok / note)",
    )
    parser.add_argument(
        "--include-metrics-blob",
        action="store_true",
        help="Include the metrics_blob column from benchmark_runs (large)",
    )
    args = parser.parse_args()

    from automl_benchmark.leaderboard_merge import merge_benchmark_csv_with_leaderboards

    inp = args.benchmark_csv.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    try:
        df = merge_benchmark_csv_with_leaderboards(
            inp,
            include_metrics_blob=args.include_metrics_blob,
            include_rows_without_leaderboard=args.include_without_leaderboard,
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Merge failed: {e}", file=sys.stderr)
        return 1

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} row(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
