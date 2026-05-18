#!/usr/bin/env python3
"""
Build benchmark_summary.csv from benchmark_runs.csv:

  1. experiment_config_json — dataset, parameters, run metadata (no metrics_blob)
  2. best_model — when discoverable in metrics / KFP task outputs
  3. score_name, score — one row per metric (long format)

Truncated metrics_blob from older runs: pass --credentials to re-fetch each run from KFP.

Examples:
  python scripts/summarize_benchmark_results.py results/benchmark_runs.csv -o results/benchmark_summary.csv
  python scripts/summarize_benchmark_results.py results/benchmark_runs.csv -o out.csv \\
      --credentials config/credentials.ini
  python scripts/summarize_benchmark_results.py results/benchmark_runs.csv -o out.csv \\
      --credentials config/credentials.ini --refetch
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(v) if v is not None else "" for k, v in r.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark_runs.csv into metrics-focused CSV.")
    parser.add_argument("input_csv", type=Path, help="Path to benchmark_runs.csv")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help="credentials.ini for KFP re-fetch",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Always re-fetch run from KFP when credentials are provided",
    )
    args = parser.parse_args()

    from automl_benchmark.benchmark_summary import records_to_summary_rows
    from benchmark_common.ini_credentials import load_credentials_ini
    from benchmark_common.kfp_client import create_kfp_client

    inp = args.input_csv.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    with open(inp, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    client = None
    if args.credentials is not None:
        cred_path = args.credentials.resolve()
        if not cred_path.is_file():
            print(f"Credentials not found: {cred_path}", file=sys.stderr)
            return 1
        ini_cfg = load_credentials_ini(cred_path)
        client = create_kfp_client(ini_cfg)

    rows = records_to_summary_rows(records, client, force_refetch=args.refetch)
    _write_csv(args.output.resolve(), rows)
    print(f"Wrote {len(rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
