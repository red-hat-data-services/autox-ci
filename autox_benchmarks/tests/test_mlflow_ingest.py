"""Unit tests for MLflow ingest helpers (no tracking server)."""

from __future__ import annotations

import pandas as pd

from benchmark_common.mlflow_ingest import (
    aggregate_metric_stats,
    normalize_task_type,
    numeric_metric_columns,
    pick_primary_metric,
    prepare_work_frame,
)
from benchmark_common.mlflow_settings import MlflowSettings


def _settings(kind: str = "automl") -> MlflowSettings:
    return MlflowSettings(
        enabled=True,
        tracking_uri="http://example",
        token="t",
        workspace="ns",
        experiment_name="exp",
        benchmark_kind=kind,
        task_type_normalize=True,
        filter_parse_ok=False,
    )


def test_normalize_task_type_binary_to_classification() -> None:
    assert normalize_task_type("binary", normalize=True, fallback="unknown") == "classification"


def test_normalize_task_type_rag_fallback() -> None:
    assert normalize_task_type("", normalize=True, fallback="rag") == "rag"


def test_numeric_metric_columns_skips_meta() -> None:
    df = pd.DataFrame({"dataset_name": ["a"], "accuracy": [0.9], "run_id": ["r1"]})
    assert numeric_metric_columns(df) == ["accuracy"]


def test_pick_primary_metric_prefers_score_val() -> None:
    assert pick_primary_metric(["f1", "score_val", "accuracy"]) == "score_val"


def test_aggregate_metric_stats() -> None:
    df = pd.DataFrame({"accuracy": [0.8, 0.9, 1.0]})
    stats = aggregate_metric_stats(df, "accuracy")
    assert stats["accuracy.max"] == 1.0
    assert stats["accuracy.min"] == 0.8


def test_prepare_work_frame_automl() -> None:
    df = pd.DataFrame(
        [
            {"dataset_name": "ds", "task_type": "binary", "run_id": "k1", "model": "m1", "accuracy": 0.9},
            {"dataset_name": "ds", "task_type": "binary", "run_id": "k1", "model": "m2", "accuracy": 0.8},
        ]
    )
    work, dataset_col, entity_col, metrics, primary = prepare_work_frame(
        df,
        settings=_settings("automl"),
        batch_id="20260101T000000Z",
        source_uri="s3://b/k",
    )
    assert dataset_col == "dataset_name"
    assert entity_col == "model"
    assert primary == "accuracy"
    assert work["_task_type_key"].iloc[0] == "classification"
    assert work["_benchmark_key"].iloc[0] == "20260101T000000Z"
