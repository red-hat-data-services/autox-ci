#!/usr/bin/env python3
"""
Build a dataset_manifest YAML for the benchmark orchestrator from local
downloaded_datasets/classification|regression CSVs (same layout as download_initial_datasets.py).

- label_column: last column in the header (matches how OpenML exports are usually saved).
- task_type: regression for regression/; for classification/, binary if ≤2 distinct
  label values in the file, else multiclass (scanned in chunks for large files).

Usage:
  pip install pandas pyyaml
  python scripts/generate_dataset_manifest.py --root downloaded_datasets \\
      --s3-key-prefix benchmark > config/dataset_manifest.generated.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _infer_task_classification(path: Path, label: str) -> str:
    import pandas as pd

    seen: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[label], chunksize=100_000, dtype=str):
        col = chunk[label].dropna()
        for u in col.unique():
            seen.add(str(u))
            if len(seen) > 2:
                return "multiclass"
    return "binary" if len(seen) == 2 else "multiclass"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("downloaded_datasets"),
        help="Root with classification/ and regression/ subdirs",
    )
    parser.add_argument(
        "--s3-key-prefix",
        type=str,
        default="benchmark",
        help="Prefix for train_data_file_key (no leading/trailing slash)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    prefix = args.s3_key_prefix.strip().strip("/")

    try:
        import yaml
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e} (pip install pyyaml pandas)", file=sys.stderr)
        return 1

    datasets: list[dict] = []
    for kind, task_default in (("classification", None), ("regression", "regression")):
        ddir = root / kind
        if not ddir.is_dir():
            continue
        for csv_path in sorted(ddir.glob("*.csv")):
            header = pd.read_csv(csv_path, nrows=0)
            cols = list(header.columns)
            if not cols:
                continue
            label = cols[-1]
            if task_default:
                task = task_default
            else:
                task = _infer_task_classification(csv_path, label)
            stem = csv_path.stem
            key = f"{prefix}/{kind}/{csv_path.name}" if prefix else f"{kind}/{csv_path.name}"
            datasets.append(
                {
                    "id": stem,
                    "name": stem,
                    "train_data_file_key": key,
                    "label_column": label,
                    "task_type": task,
                }
            )

    print(yaml.safe_dump({"datasets": datasets}, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
