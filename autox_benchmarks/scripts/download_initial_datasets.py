#!/usr/bin/env python3
"""
Download datasets listed in initial_datasets.csv (classification, regression, and ts).

Uses OpenML via scikit-learn's fetch_openml, with an openml-python fallback when the
name does not resolve. Rows with type ``ts`` are written under ``timeseries/``.

Usage:
  pip install scikit-learn openml
  python scripts/download_initial_datasets.py
  python scripts/download_initial_datasets.py --csv path/to/initial_datasets.csv --out ./my_data
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return safe or "unnamed"


def _save_sklearn_bunch(bunch, out_path: Path) -> None:
    """Persist fetch_openml result to CSV."""
    if getattr(bunch, "frame", None) is not None:
        bunch.frame.to_csv(out_path, index=False)
        return
    X = bunch.data
    y = bunch.target
    if hasattr(X, "copy"):
        df = X.copy()
    else:
        import pandas as pd  # lazy: sklearn frames use pandas when as_frame=True

        df = pd.DataFrame(X, columns=getattr(bunch, "feature_names", None))
    target_name = getattr(bunch.target, "name", None) or "target"
    if target_name in df.columns:
        target_name = "target"
    df[target_name] = y
    df.to_csv(out_path, index=False)


def _download_sklearn(name: str):
    from sklearn.datasets import fetch_openml

    try:
        bunch = fetch_openml(name=name, as_frame=True, parser="auto")
        return bunch, None
    except Exception as e:
        return None, str(e)


def _download_openml_fallback(name: str, out_path: Path) -> tuple[bool, str]:
    try:
        import openml
        import pandas as pd
    except ImportError as e:
        return False, f"missing package: {e} (pip install openml pandas)"

    try:
        listed = openml.datasets.list_datasets(name=name, exact_match=True, output_format="dataframe")
        if listed is None or len(listed) == 0:
            listed = openml.datasets.list_datasets(name=name, exact_match=False, output_format="dataframe")
        if listed is None or len(listed) == 0:
            return False, f"no OpenML dataset matched name={name!r}"

        did = int(listed.iloc[0]["did"])
        ds = openml.datasets.get_dataset(did, download_data=True)
        target = ds.default_target_attribute
        X, y, _, _ = ds.get_data(target=target)
        if hasattr(X, "copy"):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        col = "target"
        if col in df.columns:
            col = "target_y"
        df[col] = y
        df.to_csv(out_path, index=False)
        return True, ""
    except Exception as e:
        return False, str(e)


def download_one(name: str, out_path: Path) -> tuple[bool, str]:
    """Download a single dataset by OpenML-style name. Returns (ok, error_message)."""
    bunch, err = _download_sklearn(name)
    if bunch is not None:
        try:
            _save_sklearn_bunch(bunch, out_path)
            return True, ""
        except Exception as e:
            err = f"sklearn save failed: {e}"
    else:
        err = err or "unknown"

    ok2, err2 = _download_openml_fallback(name, out_path)
    if ok2:
        return True, ""
    return False, f"sklearn: {err}; openml: {err2}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download classification, regression, and timeseries (ts) datasets from CSV via OpenML."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "initial_datasets.csv",
        help="Path to initial_datasets.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("downloaded_datasets"),
        help="Output directory (classification/, regression/, and timeseries/ are created inside)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.csv.is_file():
        logger.error("CSV not found: %s", args.csv)
        return 1

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "type" not in reader.fieldnames or "dataset" not in reader.fieldnames:
            logger.error("CSV must have 'type' and 'dataset' columns.")
            return 1
        rows = list(reader)

    class_dir = args.out / "classification"
    reg_dir = args.out / "regression"
    ts_dir = args.out / "timeseries"
    class_dir.mkdir(parents=True, exist_ok=True)
    reg_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    sub = [r for r in rows if r.get("type") in ("classification", "regression", "ts")]
    skipped = len(rows) - len(sub)
    if skipped:
        logger.info("Skipping %d row(s) with type not in {classification, regression, ts}.", skipped)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in sub:
        key = (r.get("dataset", "").strip(), r.get("type", ""))
        if not key[0]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    total = len(deduped)
    ok_n = 0
    fail: list[tuple[str, str, str]] = []

    for n, row in enumerate(deduped, start=1):
        name = str(row["dataset"]).strip()
        kind = row["type"]
        if not name:
            continue
        if kind == "classification":
            out_dir = class_dir
        elif kind == "regression":
            out_dir = reg_dir
        else:
            out_dir = ts_dir
        fname = _sanitize_filename(name) + ".csv"
        out_path = out_dir / fname

        logger.info("[%d/%d] %s / %s -> %s", n, total, kind, name, out_path)
        success, msg = download_one(name, out_path)
        if success:
            ok_n += 1
        else:
            fail.append((kind, name, msg))
            logger.warning("FAILED %s / %s: %s", kind, name, msg)

    logger.info("Done. Succeeded: %d / %d. Failed: %d", ok_n, total, len(fail))
    if fail:
        fail_path = args.out / "download_failures.txt"
        lines = [f"{k}\t{n}\t{e}" for k, n, e in fail]
        fail_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote failures to %s", fail_path)

    return 0 if not fail else 2


if __name__ == "__main__":
    sys.exit(main())
