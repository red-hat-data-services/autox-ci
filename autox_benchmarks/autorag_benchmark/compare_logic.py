"""Compare AutoRAG benchmark CSVs: baseline vs batch, join on dataset + pattern."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

# Pattern name column for matching
PATTERN_COLUMN = "pattern_name"

# Metadata columns that shouldn't be treated as scores
BENCHMARK_META_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "dataset_name",
    "input_data_key",
    "test_data_key",
    "optimization_metric",
    "run_name",
    "run_id",
    "state",
    "started_at",
    "finished_at",
    "duration_seconds",
    "error",
    "metrics_blob",
    "execution_time",
    "pattern_name",
    "batch_id",
    "compare_batch_id",
)

# Preferred score columns for RAG benchmarks
SCORE_COLUMN_PREFERENCES: tuple[str, ...] = (
    "final_score",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_similarity",
    "answer_correctness",
)

_META_AND_ID_COLS = frozenset(BENCHMARK_META_COLUMNS)

# RAG metrics where lower is better (most are higher is better)
_LOWER_IS_BETTER_RE = re.compile(r"(error|loss)", re.I)


def load_merged_csv(source: Path | str | bytes) -> Any:
    """Load an AutoRAG benchmark CSV from path or raw bytes."""
    import pandas as pd

    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(Path(source))


def detect_pattern_column(df: Any) -> str:
    """Return the column name used for pattern identity."""
    if PATTERN_COLUMN in df.columns:
        return PATTERN_COLUMN
    # Fallback options
    for cand in ["pattern", "pattern_id", "template"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"Could not detect pattern column. Expected '{PATTERN_COLUMN}' or similar."
    )


def numeric_score_columns(df: Any) -> list[str]:
    """RAG metric columns that are numeric."""
    import pandas as pd

    out: list[str] = []
    for c in df.columns:
        if c in _META_AND_ID_COLS or str(c).startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(str(c))
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                out.append(str(c))
    return out


def detect_score_column(df: Any, *, preferred: str | None = None) -> str:
    """Pick primary metric column for comparison."""
    if preferred and preferred in df.columns:
        return preferred
    for cand in SCORE_COLUMN_PREFERENCES:
        if cand in df.columns:
            return cand
    # Fall back to first numeric column
    numeric_cols = numeric_score_columns(df)
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError("Could not detect a score column for comparison.")


def list_available_score_columns(*dfs: Any) -> list[str]:
    """All numeric score columns present in any of the dataframes."""
    seen = set()
    for df in dfs:
        if df is not None and not df.empty:
            seen.update(numeric_score_columns(df))
    return sorted(seen)


def score_is_lower_better(score_col: str) -> bool:
    """Heuristic: is this a metric where lower values are better?"""
    return _LOWER_IS_BETTER_RE.search(score_col) is not None


def collapse_baseline_latest_per_dataset(baseline_df: Any) -> Any:
    """
    Keep only the latest run per (dataset_name, pattern_name) based on finished_at.

    Useful for rolling joined_results.csv that accumulates runs over time.
    """
    import pandas as pd

    if baseline_df.empty:
        return baseline_df

    if "finished_at" not in baseline_df.columns:
        return baseline_df

    pattern_col = detect_pattern_column(baseline_df)

    # Sort by finished_at descending, then drop duplicates on (dataset_name, pattern)
    baseline_df = baseline_df.copy()
    baseline_df["_finished_at_ts"] = pd.to_datetime(baseline_df["finished_at"], errors="coerce")
    baseline_df.sort_values("_finished_at_ts", ascending=False, inplace=True)

    collapsed = baseline_df.drop_duplicates(subset=["dataset_name", pattern_col], keep="first")
    collapsed.drop(columns=["_finished_at_ts"], inplace=True, errors="ignore")

    return collapsed


def compare_to_baseline(
    baseline_df: Any,
    compare_df: Any,
    *,
    score_column: str,
    compare_batch_id: str = "compare",
    collapse_baseline: bool = False,
) -> tuple[Any, list[str]]:
    """
    Join baseline and compare on (dataset_name, pattern_name), compute deltas.

    Returns:
        (matched_df, warnings)

    matched_df columns:
        - dataset_name, pattern_name
        - baseline_score, compare_score, delta, pct_change
        - baseline_* and compare_* for other relevant columns
    """
    import pandas as pd

    warnings: list[str] = []

    if baseline_df.empty:
        warnings.append("Baseline is empty")
        return pd.DataFrame(), warnings
    if compare_df.empty:
        warnings.append("Compare is empty")
        return pd.DataFrame(), warnings

    try:
        pattern_col = detect_pattern_column(baseline_df)
    except ValueError as e:
        warnings.append(str(e))
        return pd.DataFrame(), warnings

    if collapse_baseline:
        baseline_df = collapse_baseline_latest_per_dataset(baseline_df)

    if score_column not in baseline_df.columns and score_column not in compare_df.columns:
        warnings.append(f"Score column '{score_column}' not found in either dataframe")
        return pd.DataFrame(), warnings

    # Prepare dataframes for merge
    baseline_prep = baseline_df.copy()
    compare_prep = compare_df.copy()

    # Add batch_id to compare
    compare_prep["compare_batch_id"] = compare_batch_id

    # Merge on dataset_name + pattern_name
    merge_cols = ["dataset_name", pattern_col]

    matched = baseline_prep.merge(
        compare_prep,
        on=merge_cols,
        how="inner",
        suffixes=("_baseline", "_compare"),
    )

    if matched.empty:
        warnings.append(
            f"No matching (dataset_name, {pattern_col}) pairs between baseline and compare"
        )
        # Return empty DataFrame with expected columns
        empty_cols = [
            "dataset_name",
            pattern_col,
            "baseline_score",
            "compare_score",
            "delta",
            "pct_change",
            "compare_batch_id",
        ]
        return pd.DataFrame(columns=empty_cols), warnings

    # Compute baseline_score and compare_score
    baseline_score_col = f"{score_column}_baseline" if f"{score_column}_baseline" in matched.columns else score_column
    compare_score_col = f"{score_column}_compare" if f"{score_column}_compare" in matched.columns else score_column

    if baseline_score_col not in matched.columns or compare_score_col not in matched.columns:
        warnings.append(f"Could not find score columns after merge: {baseline_score_col}, {compare_score_col}")
        return matched, warnings

    matched["baseline_score"] = pd.to_numeric(matched[baseline_score_col], errors="coerce")
    matched["compare_score"] = pd.to_numeric(matched[compare_score_col], errors="coerce")

    # Compute delta and pct_change
    matched["delta"] = matched["compare_score"] - matched["baseline_score"]
    matched["pct_change"] = (
        (matched["delta"] / matched["baseline_score"].replace(0, float("nan"))) * 100
    )

    # Rename optimization_metric columns if present
    if "optimization_metric_baseline" in matched.columns:
        matched.rename(columns={"optimization_metric_baseline": "baseline_optimization_metric"}, inplace=True)
    if "optimization_metric_compare" in matched.columns:
        matched.rename(columns={"optimization_metric_compare": "compare_optimization_metric"}, inplace=True)

    return matched, warnings


def score_matrix_for_heatmap(
    df: Any,
    *,
    score_column: str,
    collapse_latest: bool = False,
    optimization_metrics: list[str] | None = None,
) -> Any:
    """
    Build a pivot table: datasets (rows) × patterns (columns), values = score.

    Args:
        df: Benchmark dataframe
        score_column: Metric column to use for values
        collapse_latest: If True, keep only latest run per (dataset, pattern)
        optimization_metrics: If provided, filter to datasets with these metrics

    Returns:
        Pandas DataFrame pivot table
    """
    import pandas as pd

    if df.empty:
        return pd.DataFrame()

    try:
        pattern_col = detect_pattern_column(df)
    except ValueError:
        return pd.DataFrame()

    df_work = df.copy()

    if collapse_latest:
        df_work = collapse_baseline_latest_per_dataset(df_work)

    if optimization_metrics and "optimization_metric" in df_work.columns:
        df_work = df_work[df_work["optimization_metric"].isin(optimization_metrics)]

    if score_column not in df_work.columns:
        return pd.DataFrame()

    # Pivot: datasets as rows, patterns as columns
    pivot = df_work.pivot_table(
        index="dataset_name",
        columns=pattern_col,
        values=score_column,
        aggfunc="mean",  # Average if duplicates
    )

    return pivot


def align_score_matrix(baseline_matrix: Any, compare_matrix: Any) -> Any:
    """
    Align compare matrix to have same rows/columns as baseline (insert NaN where missing).
    """
    if baseline_matrix.empty or compare_matrix.empty:
        return compare_matrix

    # Reindex to match baseline
    aligned = compare_matrix.reindex(
        index=baseline_matrix.index,
        columns=baseline_matrix.columns,
        fill_value=float("nan"),
    )
    return aligned


def coverage_stats(
    matched_df: Any,
    baseline_df: Any,
    compare_df: Any,
    *,
    batch_id: str = "compare",
) -> dict:
    """
    Compute coverage statistics: how many datasets/patterns overlap.
    """
    try:
        pattern_col = detect_pattern_column(baseline_df)
    except ValueError:
        pattern_col = PATTERN_COLUMN

    baseline_datasets = set(baseline_df["dataset_name"].dropna().unique()) if "dataset_name" in baseline_df.columns else set()
    compare_datasets = set(compare_df["dataset_name"].dropna().unique()) if "dataset_name" in compare_df.columns else set()

    # matched_df might be empty or might not have dataset_name column
    matched_datasets = set()
    if not matched_df.empty and "dataset_name" in matched_df.columns:
        matched_datasets = set(matched_df["dataset_name"].dropna().unique())

    baseline_patterns = set(baseline_df[pattern_col].dropna().unique()) if pattern_col in baseline_df.columns else set()
    compare_patterns = set(compare_df[pattern_col].dropna().unique()) if pattern_col in compare_df.columns else set()

    dataset_coverage_pct = (
        100.0 * len(matched_datasets) / len(baseline_datasets)
        if baseline_datasets
        else 0.0
    )

    return {
        "batch_id": batch_id,
        "matched_rows": len(matched_df),
        "baseline_datasets": len(baseline_datasets),
        "compare_datasets": len(compare_datasets),
        "matched_datasets": len(matched_datasets),
        "dataset_coverage_pct": round(dataset_coverage_pct, 1),
        "baseline_patterns": len(baseline_patterns),
        "compare_patterns": len(compare_patterns),
    }


def baseline_only_keys(
    baseline_df: Any,
    compare_df: Any,
    *,
    collapse_baseline: bool = False,
) -> Any:
    """Return (dataset_name, pattern_name) keys present in baseline but not compare."""
    import pandas as pd

    if baseline_df.empty:
        return pd.DataFrame()

    try:
        pattern_col = detect_pattern_column(baseline_df)
    except ValueError:
        return pd.DataFrame()

    baseline_work = baseline_df.copy()
    if collapse_baseline:
        baseline_work = collapse_baseline_latest_per_dataset(baseline_work)

    baseline_keys = set(
        zip(baseline_work["dataset_name"], baseline_work[pattern_col])
    )
    compare_keys = set()
    if not compare_df.empty and pattern_col in compare_df.columns:
        compare_keys = set(
            zip(compare_df["dataset_name"], compare_df[pattern_col])
        )

    only_baseline = baseline_keys - compare_keys
    if not only_baseline:
        return pd.DataFrame(columns=["dataset_name", pattern_col])

    return pd.DataFrame(list(only_baseline), columns=["dataset_name", pattern_col])


def compare_only_keys(
    baseline_df: Any,
    compare_df: Any,
    *,
    collapse_baseline: bool = False,
) -> Any:
    """Return (dataset_name, pattern_name) keys present in compare but not baseline."""
    import pandas as pd

    if compare_df.empty:
        return pd.DataFrame()

    try:
        pattern_col = detect_pattern_column(compare_df)
    except ValueError:
        return pd.DataFrame()

    baseline_work = baseline_df.copy()
    if collapse_baseline:
        baseline_work = collapse_baseline_latest_per_dataset(baseline_work)

    baseline_keys = set()
    if not baseline_work.empty and pattern_col in baseline_work.columns:
        baseline_keys = set(
            zip(baseline_work["dataset_name"], baseline_work[pattern_col])
        )

    compare_keys = set(
        zip(compare_df["dataset_name"], compare_df[pattern_col])
    )

    only_compare = compare_keys - baseline_keys
    if not only_compare:
        return pd.DataFrame(columns=["dataset_name", pattern_col])

    return pd.DataFrame(list(only_compare), columns=["dataset_name", pattern_col])
