"""Join benchmark_runs.csv rows with tables parsed from saved leaderboard HTML files."""

from __future__ import annotations

import io
import json
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
    "preset",
    "run_id",
    "state",
    "started_at",
    "finished_at",
    "duration_seconds",
    "error",
    "leaderboard_html_s3_uri",
    "leaderboard_html_path",
    "leaderboard_scores_path",
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


def _leaderboard_df_from_scores_json(path: Path) -> Any:
    import pandas as pd

    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("leaderboard") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _best_score_fields_from_leaderboard(lb: Any) -> dict[str, Any]:
    """Flatten the top-ranked leaderboard row into ``best_model`` / ``best_<metric>`` fields."""
    out: dict[str, Any] = {
        "best_model": "",
        "leaderboard_parse_ok": False,
        "leaderboard_parse_note": "",
    }
    if lb is None or getattr(lb, "empty", True):
        out["leaderboard_parse_note"] = "no_table"
        return out

    best = lb.iloc[0]
    model_col = next((c for c in lb.columns if str(c).lower() == "model"), None)
    if model_col is not None:
        out["best_model"] = str(best[model_col])

    skip = {"model", "rank", "notebook", "predictor"}
    for col in lb.columns:
        cl = str(col).lower()
        if cl in skip:
            continue
        key = f"best_{col}" if not str(col).startswith("best_") else str(col)
        val = best[col]
        if hasattr(val, "item"):
            try:
                val = val.item()
            except Exception:
                pass
        out[key] = val

    out["leaderboard_parse_ok"] = True
    return out


def enrich_result_row_with_scores(row: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    """
    Add ``best_model`` / ``best_<metric>`` onto a benchmark result row from local artifacts.

    Preference: ``leaderboard_scores_path`` (metrics.json aggregate) then HTML parse.
    Mutates and returns ``row``.
    """
    results_dir = results_dir.resolve()
    scores_rel = str(row.get("leaderboard_scores_path") or "").strip()
    html_rel = str(row.get("leaderboard_html_path") or "").strip()

    lb = None
    note = ""
    if scores_rel:
        scores_path = results_dir / scores_rel
        if scores_path.is_file():
            try:
                lb = _leaderboard_df_from_scores_json(scores_path)
                note = "scores_json"
            except Exception as e:
                logger.warning("Failed to parse scores JSON %s: %s", scores_path, e)
                row["leaderboard_parse_ok"] = False
                row["leaderboard_parse_note"] = f"scores_json:{e}"[:500]
                return row

    if (lb is None or getattr(lb, "empty", True)) and html_rel:
        html_path = results_dir / html_rel
        if html_path.is_file():
            try:
                html = html_path.read_text(encoding="utf-8", errors="replace")
                lb = pick_leaderboard_table(html)
                note = "html"
            except Exception as e:
                logger.warning("Failed to parse leaderboard HTML %s: %s", html_path, e)
                row["leaderboard_parse_ok"] = False
                row["leaderboard_parse_note"] = str(e)[:500]
                return row

    if lb is None or getattr(lb, "empty", True):
        row["best_model"] = ""
        row["leaderboard_parse_ok"] = False
        row["leaderboard_parse_note"] = "no_leaderboard_artifact"
        return row

    fields = _best_score_fields_from_leaderboard(lb)
    fields["leaderboard_parse_note"] = note or fields.get("leaderboard_parse_note") or ""
    row.update(fields)
    return row


def _append_merged_part(
    parts: list[Any],
    *,
    meta: dict[str, Any],
    lb: Any,
    reserved: set[str],
    note: str = "",
) -> None:
    import pandas as pd

    if lb is None or lb.empty:
        return
    lb = _rename_colliding_columns(lb, reserved)
    n = len(lb)
    meta_df = pd.DataFrame([meta] * n).reset_index(drop=True)
    lb = lb.reset_index(drop=True)
    merged = pd.concat([meta_df, lb], axis=1)
    merged["leaderboard_parse_ok"] = True
    merged["leaderboard_parse_note"] = note
    parts.append(merged)


def merge_benchmark_csv_with_leaderboards(
    benchmark_csv: Path,
    *,
    include_metrics_blob: bool = False,
    include_rows_without_leaderboard: bool = False,
) -> Any:
    """
    Build one long-form DataFrame: benchmark metadata repeated per leaderboard row.

    Preference order per run:
    1. ``leaderboard_scores_path`` (JSON from ``models_artifact/*/metrics/metrics.json``)
    2. ``leaderboard_html_path`` (parsed HTML table; legacy / fallback)
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
        scores_rel = (rec.get("leaderboard_scores_path") or "").strip()
        html_rel = (rec.get("leaderboard_html_path") or "").strip()
        scores_path = (base / scores_rel) if scores_rel else Path()
        html_path = (base / html_rel) if html_rel else Path()

        lb = None
        note = ""
        if scores_rel and scores_path.is_file():
            try:
                lb = _leaderboard_df_from_scores_json(scores_path)
                note = "scores_json"
            except Exception as e:
                logger.warning("Failed to parse scores JSON %s: %s", scores_path, e)
                if include_rows_without_leaderboard:
                    row = {
                        **meta,
                        "leaderboard_parse_ok": False,
                        "leaderboard_parse_note": f"scores_json:{e}"[:500],
                    }
                    parts.append(pd.DataFrame([row]))
                    continue

        if (lb is None or lb.empty) and html_rel and html_path.is_file():
            try:
                html = html_path.read_text(encoding="utf-8", errors="replace")
                lb = pick_leaderboard_table(html)
                note = "html"
            except Exception as e:
                logger.warning("Failed to parse leaderboard HTML %s: %s", html_path, e)
                if include_rows_without_leaderboard:
                    row = {
                        **meta,
                        "leaderboard_parse_ok": False,
                        "leaderboard_parse_note": str(e)[:500],
                    }
                    parts.append(pd.DataFrame([row]))
                continue

        if lb is None or lb.empty:
            if include_rows_without_leaderboard:
                row = {**meta, "leaderboard_parse_ok": False, "leaderboard_parse_note": "no_file"}
                parts.append(pd.DataFrame([row]))
            else:
                logger.warning(
                    "Skipping dataset_id=%s run_id=%s: missing scores JSON and HTML",
                    rec.get("dataset_id", ""),
                    rec.get("run_id", ""),
                )
            continue

        _append_merged_part(parts, meta=meta, lb=lb, reserved=reserved, note=note)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)
