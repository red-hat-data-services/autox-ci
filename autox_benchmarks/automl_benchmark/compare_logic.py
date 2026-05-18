"""Compare merged leaderboard CSVs: baseline collapse, join on dataset + model, deltas."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from automl_benchmark.leaderboard_merge import BENCHMARK_META_COLUMNS

MODEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "model",
    "model_name",
    "name",
    "lb_model",
    "lb_model_name",
    "lb_name",
)

SCORE_COLUMN_PREFERENCES: tuple[str, ...] = (
    "score_val",
    "score_test",
    "lb_score_val",
    "lb_score_test",
    "accuracy",
    "lb_accuracy",
    "r2",
    "lb_r2",
    "f1",
    "lb_f1",
    "auc",
    "lb_auc",
)

_META_AND_ID_COLS = frozenset(
    {
        *BENCHMARK_META_COLUMNS,
        "leaderboard_parse_ok",
        "leaderboard_parse_note",
        "metrics_blob",
        "model",
        "model_name",
        "name",
        "lb_model",
        "lb_model_name",
        "lb_name",
        "batch_id",
        "compare_batch_id",
    }
)

_LOWER_IS_BETTER_RE = re.compile(r"(rmse|mse|mae|error|loss|log_loss|nll)", re.I)


def load_merged_csv(source: Path | str | bytes) -> Any:
    """Load a merged leaderboard CSV from path or raw bytes."""
    import pandas as pd

    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(Path(source))


def detect_model_column(df: Any) -> str:
    """Return the column name used for model identity."""
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in MODEL_COLUMN_CANDIDATES:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(
        "Could not detect model column. Expected one of: "
        + ", ".join(MODEL_COLUMN_CANDIDATES)
    )


def numeric_score_columns(df: Any) -> list[str]:
    """Leaderboard metric columns that are numeric."""
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
    numeric = numeric_score_columns(df)
    if not numeric:
        raise ValueError("No numeric score column found in merged leaderboard CSV")
    return numeric[0]


def score_is_lower_better(score_column: str) -> bool:
    return bool(_LOWER_IS_BETTER_RE.search(score_column))


def _normalize_model_series(df: Any, model_col: str) -> Any:
    import pandas as pd

    s = df[model_col].astype(str).str.strip()
    out = df.copy()
    out["_model_key"] = s
    return out


def collapse_baseline_latest_per_dataset(df: Any) -> Any:
    """
    For each ``dataset_name``, keep all rows from the run with the latest ``finished_at``.

    Rows without ``finished_at`` are dropped when other rows for the same dataset have timestamps.
    """
    import pandas as pd

    if df.empty:
        return df.copy()
    if "dataset_name" not in df.columns:
        raise ValueError("CSV missing required column: dataset_name")
    if "finished_at" not in df.columns:
        return df.copy()

    work = df.copy()
    work["_finished_ts"] = pd.to_datetime(work["finished_at"], utc=True, errors="coerce")
    has_ts = work["_finished_ts"].notna()
    if not has_ts.any():
        work = work.drop(columns=["_finished_ts"])
        return work

    latest = work.loc[has_ts].groupby("dataset_name", dropna=False)["_finished_ts"].transform("max")
    mask = has_ts & (work["_finished_ts"] == latest)
    # Datasets with only NaT finished_at: keep all their rows
    only_nat_names = set(work["dataset_name"].astype(str)) - set(
        work.loc[has_ts, "dataset_name"].astype(str)
    )
    keep_no_ts = work["dataset_name"].astype(str).isin(only_nat_names) & ~has_ts
    out = work.loc[mask | keep_no_ts].drop(columns=["_finished_ts"])
    return out.reset_index(drop=True)


def _dedupe_model_rows(df: Any, *, model_col: str) -> tuple[Any, list[str]]:
    """Drop duplicate (dataset_name, model) rows, keeping first. Return warnings."""
    import pandas as pd

    work = _normalize_model_series(df, model_col)
    dup_mask = work.duplicated(subset=["dataset_name", "_model_key"], keep="first")
    warnings: list[str] = []
    if dup_mask.any():
        n = int(dup_mask.sum())
        warnings.append(f"Dropped {n} duplicate row(s) for same dataset_name + model")
    out = work.loc[~dup_mask].copy()
    return out, warnings


def compare_to_baseline(
    baseline_df: Any,
    compare_df: Any,
    *,
    score_column: str | None = None,
    model_column: str | None = None,
    compare_batch_id: str | None = None,
    collapse_baseline: bool = True,
) -> tuple[Any, list[str]]:
    """
  Inner-join baseline vs compare on (dataset_name, model).

  Returns a DataFrame with baseline_score, compare_score, delta, pct_change, and meta columns.
    """
    import pandas as pd

    warnings: list[str] = []
    base = baseline_df.copy()
    if collapse_baseline:
        base = collapse_baseline_latest_per_dataset(base)

    model_col_base = model_column or detect_model_column(base)
    model_col_cmp = model_column or detect_model_column(compare_df)
    if model_col_cmp != model_col_base:
        warnings.append(
            f"Compare uses model column {model_col_cmp!r}; baseline uses {model_col_base!r}"
        )

    score_col = score_column or detect_score_column(base)
    if score_col not in compare_df.columns:
        alt = detect_score_column(compare_df)
        if alt != score_col:
            warnings.append(f"Compare score column {alt!r} differs from baseline {score_col!r}")
            score_col = alt

    base, w1 = _dedupe_model_rows(base, model_col=model_col_base)
    cmp, w2 = _dedupe_model_rows(compare_df, model_col=model_col_cmp)
    warnings.extend(w1)
    warnings.extend(w2)

    base = _normalize_model_series(base, model_col_base)
    cmp = _normalize_model_series(cmp, model_col_cmp)

    merged = base.merge(
        cmp,
        on=["dataset_name", "_model_key"],
        how="inner",
        suffixes=("_baseline", "_compare"),
    )

    if merged.empty:
        empty_cols = [
            "dataset_name",
            "model",
            "score_column",
            "baseline_score",
            "compare_score",
            "delta",
            "pct_change",
            "compare_batch_id",
        ]
        return pd.DataFrame(columns=empty_cols), warnings

    b_score_col = f"{score_col}_baseline" if f"{score_col}_baseline" in merged.columns else score_col
    c_score_col = f"{score_col}_compare" if f"{score_col}_compare" in merged.columns else score_col
    if b_score_col not in merged.columns:
        b_score_col = score_col
    if c_score_col not in merged.columns:
        for c in merged.columns:
            if str(c).startswith(score_col) and str(c).endswith("_compare"):
                c_score_col = c
                break

    baseline_score = pd.to_numeric(merged[b_score_col], errors="coerce")
    compare_score = pd.to_numeric(merged[c_score_col], errors="coerce")
    delta = compare_score - baseline_score
    pct = (delta / baseline_score.replace(0, pd.NA)) * 100.0

    model_out = merged["_model_key"]
    b_model_col = f"{model_col_base}_baseline"
    if b_model_col in merged.columns:
        model_out = merged[b_model_col].fillna(merged["_model_key"])

    out = pd.DataFrame(
        {
            "dataset_name": merged["dataset_name"],
            "model": model_out,
            "score_column": score_col,
            "baseline_score": baseline_score,
            "compare_score": compare_score,
            "delta": delta,
            "pct_change": pct,
            "compare_batch_id": compare_batch_id or "",
        }
    )

    for meta in ("task_type", "run_id", "finished_at", "state"):
        bcol = f"{meta}_baseline" if f"{meta}_baseline" in merged.columns else meta
        ccol = f"{meta}_compare" if f"{meta}_compare" in merged.columns else None
        if bcol in merged.columns:
            out[f"baseline_{meta}"] = merged[bcol]
        if ccol and ccol in merged.columns:
            out[f"compare_{meta}"] = merged[ccol]

    return out, warnings


def baseline_only_keys(
    baseline_df: Any,
    compare_df: Any,
    *,
    model_column: str | None = None,
    collapse_baseline: bool = True,
) -> Any:
    """Rows in baseline (latest per dataset) with no matching (dataset_name, model) in compare."""
    import pandas as pd

    base = collapse_baseline_latest_per_dataset(baseline_df) if collapse_baseline else baseline_df.copy()
    model_col = model_column or detect_model_column(base)
    base = _normalize_model_series(base, model_col)
    cmp = _normalize_model_series(compare_df, model_column or detect_model_column(compare_df))
    keys = set(zip(base["dataset_name"].astype(str), base["_model_key"].astype(str)))
    cmp_keys = set(zip(cmp["dataset_name"].astype(str), cmp["_model_key"].astype(str)))
    only = keys - cmp_keys
    if not only:
        return pd.DataFrame(columns=["dataset_name", "model"])
    mask = [((str(d), str(m)) in only) for d, m in zip(base["dataset_name"], base["_model_key"])]
    out = base.loc[mask, ["dataset_name", "_model_key"]].rename(columns={"_model_key": "model"})
    return out.reset_index(drop=True)


def compare_only_keys(
    baseline_df: Any,
    compare_df: Any,
    *,
    model_column: str | None = None,
    collapse_baseline: bool = True,
) -> Any:
    """Rows in compare with no matching (dataset_name, model) in collapsed baseline."""
    import pandas as pd

    base = collapse_baseline_latest_per_dataset(baseline_df) if collapse_baseline else baseline_df.copy()
    model_col = model_column or detect_model_column(compare_df)
    base = _normalize_model_series(base, model_column or detect_model_column(base))
    cmp = _normalize_model_series(compare_df, model_col)
    keys = set(zip(base["dataset_name"].astype(str), base["_model_key"].astype(str)))
    cmp_keys = set(zip(cmp["dataset_name"].astype(str), cmp["_model_key"].astype(str)))
    only = cmp_keys - keys
    if not only:
        return pd.DataFrame(columns=["dataset_name", "model"])
    mask = [((str(d), str(m)) in only) for d, m in zip(cmp["dataset_name"], cmp["_model_key"])]
    out = cmp.loc[mask, ["dataset_name", "_model_key"]].rename(columns={"_model_key": "model"})
    return out.reset_index(drop=True)


def coverage_stats(matched: Any, baseline_df: Any, compare_df: Any, *, batch_id: str = "") -> dict[str, Any]:
    """Summary counts for a single compare batch."""
    import pandas as pd

    base_ds = set(baseline_df["dataset_name"].astype(str).unique()) if "dataset_name" in baseline_df.columns else set()
    cmp_ds = set(compare_df["dataset_name"].astype(str).unique()) if "dataset_name" in compare_df.columns else set()
    matched_ds = set(matched["dataset_name"].astype(str).unique()) if not matched.empty else set()
    n_base_ds = len(base_ds)
    pct = (len(matched_ds) / n_base_ds * 100.0) if n_base_ds else 0.0
    return {
        "batch_id": batch_id,
        "matched_pairs": len(matched),
        "matched_datasets": len(matched_ds),
        "baseline_datasets": n_base_ds,
        "compare_datasets": len(cmp_ds),
        "dataset_coverage_pct": pct,
    }


def score_matrix_for_heatmap(
    df: Any,
    *,
    score_column: str,
    model_column: str | None = None,
    collapse_latest: bool = False,
    task_types: list[str] | None = None,
) -> Any:
    """
    Pivot merged leaderboard rows to a dataset_name × model score matrix for heatmaps.

    Returns a DataFrame indexed by dataset_name with one column per model.
    """
    import pandas as pd

    work = collapse_baseline_latest_per_dataset(df) if collapse_latest else df.copy()
    if work.empty or "dataset_name" not in work.columns:
        return pd.DataFrame()
    if score_column not in work.columns:
        raise ValueError(f"Score column not in data: {score_column}")

    model_col = model_column or detect_model_column(work)
    if task_types and "task_type" in work.columns:
        work = work[work["task_type"].isin(task_types)]

    work = _normalize_model_series(work, model_col)
    work = work.assign(_score=pd.to_numeric(work[score_column], errors="coerce"))
    work = work.dropna(subset=["_score"])
    work = work.drop_duplicates(subset=["dataset_name", "_model_key"], keep="first")

    pivot = work.pivot(index="dataset_name", columns="_model_key", values="_score")
    pivot.index = pivot.index.astype(str)
    pivot.columns = pivot.columns.astype(str)
    return pivot.sort_index().sort_index(axis=1)


def align_score_matrix(reference: Any, other: Any) -> Any:
    """Reindex *other* to *reference* rows/columns (NaN where missing)."""
    if reference.empty:
        return other
    return other.reindex(index=reference.index, columns=reference.columns)


def list_available_score_columns(*dfs: Any) -> list[str]:
    """Union of numeric score columns across dataframes."""
    seen: list[str] = []
    for df in dfs:
        for c in numeric_score_columns(df):
            if c not in seen:
                seen.append(c)
    for pref in SCORE_COLUMN_PREFERENCES:
        if pref in seen:
            seen.remove(pref)
            seen.insert(0, pref)
    return seen
