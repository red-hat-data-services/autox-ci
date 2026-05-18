"""Join benchmark_runs.csv rows with tables parsed from saved leaderboard HTML files."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Columns copied from benchmark_runs (exclude huge / redundant for the merged leaderboard view).
BENCHMARK_META_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "dataset_name",
    "task_type",
    "label_column",
    "train_data_file_key",
    "run_name",
    "top_n",
    "run_id",
    "state",
    "started_at",
    "finished_at",
    "duration_seconds",
    "error",
    "leaderboard_html_s3_uri",
    "leaderboard_html_path",
)


def _read_html_tables(html: str) -> list[Any]:
    import pandas as pd

    # StringIO avoids pandas treating a long HTML string as a filesystem path (which
    # raises FileNotFoundError with the markup as the "path").
    return pd.read_html(io.StringIO(html), flavor="lxml")


def pick_leaderboard_table(html: str) -> Any:
    """
    Parse HTML and return the DataFrame best matching an AutoGluon-style leaderboard table.

    Chooses the table with the most data cells (rows * cols), requiring at least 2 rows
    and 2 columns when possible.
    """
    import pandas as pd

    try:
        tables = _read_html_tables(html)
    except ImportError as e:
        raise ImportError(
            "Parsing leaderboard HTML requires lxml. Install with: pip install lxml"
        ) from e
    except ValueError as e:
        if "No tables found" in str(e):
            return pd.DataFrame()
        raise

    if not tables:
        return pd.DataFrame()

    def score(df: Any) -> int:
        if df is None or df.empty:
            return 0
        return int(df.shape[0] * df.shape[1])

    best = max(tables, key=score)
    if score(best) == 0:
        return pd.DataFrame()
    return best.copy()


def _meta_row_from_record(rec: dict[str, str], *, include_metrics_blob: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in BENCHMARK_META_COLUMNS:
        if k in rec:
            out[k] = rec[k]
    if include_metrics_blob and "metrics_blob" in rec:
        out["metrics_blob"] = rec["metrics_blob"]
    return out


def _rename_colliding_columns(lb: Any, reserved: set[str]) -> Any:
    import pandas as pd

    if lb.empty:
        return lb
    rename = {c: f"lb_{c}" for c in lb.columns if c in reserved}
    if rename:
        lb = lb.rename(columns=rename)
    return lb


def merge_benchmark_csv_with_leaderboards(
    benchmark_csv: Path,
    *,
    include_metrics_blob: bool = False,
    include_rows_without_leaderboard: bool = False,
) -> Any:
    """
    Build one long-form DataFrame: benchmark metadata repeated per leaderboard row.

    ``benchmark_csv`` parent directory resolves ``leaderboard_html_path`` (relative paths).
    """
    import csv

    import pandas as pd

    benchmark_csv = benchmark_csv.resolve()
    base = benchmark_csv.parent
    if not benchmark_csv.is_file():
        raise FileNotFoundError(benchmark_csv)

    parts: list[Any] = []
    reserved = set(BENCHMARK_META_COLUMNS)
    if include_metrics_blob:
        reserved.add("metrics_blob")

    with open(benchmark_csv, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    for rec in records:
        meta = _meta_row_from_record(rec, include_metrics_blob=include_metrics_blob)
        rel = (rec.get("leaderboard_html_path") or "").strip()
        path = (base / rel) if rel else Path()

        if not rel or not path.is_file():
            if include_rows_without_leaderboard:
                row = {**meta, "leaderboard_parse_ok": False, "leaderboard_parse_note": "no_file"}
                parts.append(pd.DataFrame([row]))
            else:
                logger.warning(
                    "Skipping dataset_id=%s run_id=%s: missing leaderboard file (%r)",
                    rec.get("dataset_id", ""),
                    rec.get("run_id", ""),
                    str(path) if rel else "",
                )
            continue

        try:
            html = path.read_text(encoding="utf-8", errors="replace")
            lb = pick_leaderboard_table(html)
        except Exception as e:
            logger.warning("Failed to parse leaderboard HTML %s: %s", path, e)
            if include_rows_without_leaderboard:
                row = {
                    **meta,
                    "leaderboard_parse_ok": False,
                    "leaderboard_parse_note": str(e)[:500],
                }
                parts.append(pd.DataFrame([row]))
            continue

        if lb.empty:
            logger.warning("No tables parsed from %s", path)
            if include_rows_without_leaderboard:
                row = {**meta, "leaderboard_parse_ok": False, "leaderboard_parse_note": "no_table"}
                parts.append(pd.DataFrame([row]))
            continue

        lb = _rename_colliding_columns(lb, reserved)
        n = len(lb)
        meta_df = pd.DataFrame([meta] * n).reset_index(drop=True)
        lb = lb.reset_index(drop=True)
        merged = pd.concat([meta_df, lb], axis=1)
        merged["leaderboard_parse_ok"] = True
        merged["leaderboard_parse_note"] = ""
        parts.append(merged)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    return out
