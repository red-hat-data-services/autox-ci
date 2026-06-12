"""Log benchmark batch CSVs to MLflow (4-level nested hierarchy from MLFlow.ipynb)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from benchmark_common.mlflow_settings import MlflowSettings, apply_mlflow_env

logger = logging.getLogger(__name__)

META_COLS = frozenset(
    {
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
        "rank",
        "model",
        "pattern_name",
        "leaderboard_parse_ok",
        "leaderboard_parse_note",
        "metrics_blob",
        "Notebook",
        "Predictor",
        "optimization_metric",
        "final_score",
        "execution_time",
    }
)

SCORE_PREFERENCES = (
    "score_val",
    "score_test",
    "final_score",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "roc_auc",
    "r2",
    "rmse",
    "mae",
    "mape",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
)

DATASET_PARAM_COLS = [
    "optimization_metric",
    "run_id",
    "run_name",
    "state",
    "started_at",
    "finished_at",
    "top_n",
    "dataset_id",
    "label_column",
]
ENTITY_PARAM_COLS = [
    "run_id",
    "run_name",
    "state",
    "finished_at",
    "rank",
    "pattern_name",
]

KIND_PRESETS: dict[str, dict[str, Any]] = {
    "automl": {
        "entity_candidates": ("model", "model_name", "name", "lb_model", "lb_model_name", "lb_name"),
        "dataset_candidates": ("dataset_name", "dataset_id"),
        "aggregate_csv": "merged_leaderboards.csv",
        "task_type_fallback": "unknown",
    },
    "autorag": {
        "entity_candidates": ("pattern_name", "pattern_id", "name"),
        "dataset_candidates": ("dataset_name", "dataset_id"),
        "aggregate_csv": "benchmark_runs.csv",
        "task_type_fallback": "rag",
    },
}


def normalize_task_type(raw: str, *, normalize: bool, fallback: str) -> str:
    t = str(raw).strip().lower()
    if not t or t in ("nan", "none"):
        return fallback
    if not normalize:
        return t
    if t in ("binary", "multiclass", "classification"):
        return "classification"
    if t == "regression":
        return "regression"
    if t in ("timeseries", "time_series", "ts"):
        return "timeseries"
    return t


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in columns), None)


def numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in frame.columns:
        if col in META_COLS or str(col).startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out.append(col)
            continue
        coerced = pd.to_numeric(frame[col], errors="coerce")
        if coerced.notna().any():
            out.append(col)
    return out


def pick_primary_metric(metric_cols: list[str]) -> str | None:
    for name in SCORE_PREFERENCES:
        if name in metric_cols:
            return name
    return metric_cols[0] if metric_cols else None


def aggregate_metric_stats(group: pd.DataFrame, metric: str) -> dict[str, float]:
    vals = pd.to_numeric(group[metric], errors="coerce").dropna()
    if vals.empty:
        return {}
    return {
        f"{metric}.max": float(vals.max()),
        f"{metric}.mean": float(vals.mean()),
        f"{metric}.min": float(vals.min()),
    }


def log_row_params(row: pd.Series, columns: list[str]) -> None:
    """Log CSV columns as MLflow params; raw task_type -> source_task_type."""
    import mlflow

    for col in columns:
        if col not in row.index:
            continue
        val = row.get(col)
        if pd.isna(val):
            continue
        param_name = "source_task_type" if col == "task_type" else col
        mlflow.log_param(param_name, str(val)[:250])


def prepare_work_frame(
    df: pd.DataFrame,
    *,
    settings: MlflowSettings,
    batch_id: str,
    source_uri: str,
) -> tuple[pd.DataFrame, str, str, list[str], str | None]:
    preset = KIND_PRESETS[settings.benchmark_kind]
    work = df.copy()
    if settings.filter_parse_ok and "leaderboard_parse_ok" in work.columns:
        work = work[work["leaderboard_parse_ok"].astype(str).str.lower().isin(("true", "1", "yes"))]

    dataset_col = first_existing(work.columns, preset["dataset_candidates"])
    entity_col = first_existing(work.columns, preset["entity_candidates"])
    if not dataset_col:
        raise ValueError(f"No dataset column in CSV (tried {preset['dataset_candidates']})")
    if not entity_col:
        raise ValueError(f"No entity column in CSV (tried {preset['entity_candidates']})")

    fallback = preset["task_type_fallback"]
    if "task_type" in work.columns:
        work["_task_type_key"] = work["task_type"].map(
            lambda v: normalize_task_type(v, normalize=settings.task_type_normalize, fallback=fallback)
        )
    else:
        work["_task_type_key"] = fallback

    run_col = "run_id" if "run_id" in work.columns else dataset_col
    work["_dataset_key"] = work[run_col].astype(str) + "::" + work[dataset_col].astype(str)
    work["_benchmark_key"] = batch_id
    work["_source_uri"] = source_uri

    metric_cols = numeric_metric_columns(work)
    primary = pick_primary_metric(metric_cols)
    return work, dataset_col, entity_col, metric_cols, primary


def ingest_dataframe(
    df: pd.DataFrame,
    *,
    settings: MlflowSettings,
    batch_id: str,
    source_uri: str,
    batch_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Log one benchmark batch CSV to MLflow using nested runs:

    benchmark -> task_type -> dataset -> entity (model/pattern)
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise ImportError("MLflow upload requires: pip install -e \".[mlflow]\"") from exc

    apply_mlflow_env(settings)
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    client = MlflowClient()

    work, dataset_col, entity_col, metric_cols, primary_metric = prepare_work_frame(
        df,
        settings=settings,
        batch_id=batch_id,
        source_uri=source_uri,
    )
    if work.empty:
        logger.warning("No rows to log to MLflow after filtering")
        return {"entities_logged": 0, "entities_skipped": 0}

    summary: dict[str, Any] = {
        "experiment": settings.experiment_name,
        "workspace": settings.workspace,
        "source_uri": source_uri,
        "batch_id": batch_id,
        "benchmark_kind": settings.benchmark_kind,
        "primary_metric": primary_metric,
        "metric_columns": metric_cols,
        "benchmark_runs": [],
        "entities_logged": 0,
        "entities_skipped": 0,
        "task_types_logged": 0,
        "datasets_logged": 0,
    }

    ingested_at = datetime.now(timezone.utc).isoformat()

    for benchmark_key, benchmark_group in work.groupby("_benchmark_key", sort=False):
        benchmark_name = f"benchmark::{benchmark_key}"[:250]
        benchmark_metrics: dict[str, float] = {}
        if primary_metric:
            benchmark_metrics.update(aggregate_metric_stats(benchmark_group, primary_metric))

        with mlflow.start_run(run_name=benchmark_name) as benchmark_run:
            mlflow.set_tags(
                {
                    "run_level": "benchmark",
                    "benchmark_kind": settings.benchmark_kind,
                    "batch_id": str(benchmark_key),
                    "source_csv": source_uri,
                    "ingested_at": ingested_at,
                }
            )
            if batch_metadata:
                mlflow.log_param("batch_started_at", str(batch_metadata.get("started_at", ""))[:250])
                mlflow.log_param("batch_finished_at", str(batch_metadata.get("finished_at", ""))[:250])
            mlflow.log_param("dataset_count", int(benchmark_group[dataset_col].nunique()))
            mlflow.log_param("entity_count", int(len(benchmark_group)))
            if primary_metric:
                mlflow.log_metrics(benchmark_metrics)

            for task_key, task_group in benchmark_group.groupby("_task_type_key", sort=False):
                task_metrics = aggregate_metric_stats(task_group, primary_metric) if primary_metric else {}
                with mlflow.start_run(run_name=str(task_key)[:250], nested=True):
                    mlflow.set_tags(
                        {
                            "run_level": "task_type",
                            "task_type": str(task_key),
                            "batch_id": str(benchmark_key),
                            "benchmark_kind": settings.benchmark_kind,
                        }
                    )
                    mlflow.log_param("task_type", str(task_key))
                    mlflow.log_param("dataset_count", int(task_group[dataset_col].nunique()))
                    mlflow.log_param("entity_count", int(len(task_group)))
                    if task_metrics:
                        mlflow.log_metrics(task_metrics)
                    summary["task_types_logged"] += 1

                    for dataset_key, dataset_group in task_group.groupby("_dataset_key", sort=False):
                        dataset_label = str(dataset_group[dataset_col].iloc[0])[:200]
                        run_id = ""
                        if "run_id" in dataset_group.columns and pd.notna(dataset_group["run_id"].iloc[0]):
                            run_id = str(dataset_group["run_id"].iloc[0])
                        dataset_run_name = f"{dataset_label} | {run_id}".strip(" |")[:250]
                        dataset_metrics = (
                            aggregate_metric_stats(dataset_group, primary_metric) if primary_metric else {}
                        )

                        with mlflow.start_run(run_name=dataset_run_name, nested=True):
                            mlflow.set_tags(
                                {
                                    "run_level": "dataset",
                                    "dataset_name": dataset_label,
                                    "batch_id": str(benchmark_key),
                                    "task_type": str(task_key),
                                }
                            )
                            if run_id:
                                mlflow.log_param("run_id", run_id[:250])
                            mlflow.log_param("dataset_name", dataset_label[:250])
                            mlflow.log_param("entity_count", int(len(dataset_group)))
                            if "task_type" in dataset_group.columns:
                                mlflow.log_param("task_type", str(task_key))
                            log_row_params(dataset_group.iloc[0], DATASET_PARAM_COLS)
                            if dataset_metrics:
                                mlflow.log_metrics(dataset_metrics)
                            summary["datasets_logged"] += 1

                            for _, row in dataset_group.iterrows():
                                entity_name = str(row[entity_col])
                                row_metrics = {
                                    col: float(pd.to_numeric(row[col], errors="coerce"))
                                    for col in metric_cols
                                    if pd.notna(pd.to_numeric(row.get(col), errors="coerce"))
                                }
                                if not row_metrics:
                                    summary["entities_skipped"] += 1
                                    continue

                                with mlflow.start_run(run_name=entity_name[:250], nested=True):
                                    mlflow.set_tags(
                                        {
                                            "run_level": "entity",
                                            "entity_name": entity_name[:250],
                                            "dataset_name": dataset_label,
                                            "batch_id": str(benchmark_key),
                                            "task_type": str(task_key),
                                        }
                                    )
                                    mlflow.log_param("entity_name", entity_name[:250])
                                    mlflow.log_param("dataset_name", dataset_label[:250])
                                    if "task_type" in row.index and pd.notna(row.get("task_type")):
                                        mlflow.log_param("task_type", str(task_key))
                                    log_row_params(row, ENTITY_PARAM_COLS)
                                    mlflow.log_metrics(row_metrics)
                                    summary["entities_logged"] += 1

            summary["benchmark_runs"].append(
                {
                    "benchmark_run_id": benchmark_run.info.run_id,
                    "batch_id": str(benchmark_key),
                    "entities": int(len(benchmark_group)),
                }
            )

    experiment = client.get_experiment_by_name(settings.experiment_name)
    summary["experiment_id"] = experiment.experiment_id if experiment else None
    logger.info(
        "MLflow ingest complete: batch=%s entities=%d skipped=%d experiment=%s",
        batch_id,
        summary["entities_logged"],
        summary["entities_skipped"],
        settings.experiment_name,
    )
    return summary
