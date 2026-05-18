#!/usr/bin/env python3
"""
Create a smaller CSV for quick pipeline / benchmark tests (e.g. 200 rows).

Examples:
  python scripts/make_benchmark_sample.py \\
    downloaded_datasets/classification/breast-w.csv \\
    -o samples/breast-w_n200.csv -n 200 --label Class --stratify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a fixed-size random sample of a tabular CSV.")
    parser.add_argument("input_csv", type=Path, help="Source CSV path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("-n", type=int, default=200, help="Number of rows to keep (default 200)")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Target column name (default: last column in header)",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratified sample by label (needs scikit-learn); keeps class proportions",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pip install pandas", file=sys.stderr)
        return 1

    src = args.input_csv.resolve()
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        return 1

    df = pd.read_csv(src)
    if len(df) == 0:
        print("Empty CSV", file=sys.stderr)
        return 1

    label = args.label
    if not label:
        label = str(df.columns[-1])
    if label not in df.columns:
        print(f"Label column {label!r} not in columns: {list(df.columns)}", file=sys.stderr)
        return 1

    n = min(args.n, len(df))
    if args.stratify:
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except ImportError:
            print("Stratified sampling requires: pip install scikit-learn", file=sys.stderr)
            return 1
        y = df[label]
        n_classes = y.nunique()
        if n < n_classes:
            print(f"-n must be >= number of classes ({n_classes}) for stratify", file=sys.stderr)
            return 1
        split = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=args.seed)
        idx, _ = next(split.split(df.drop(columns=[label]), y))
        sample = df.iloc[idx].reset_index(drop=True)
    else:
        sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Wrote {len(sample)} rows to {out} (label={label!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
