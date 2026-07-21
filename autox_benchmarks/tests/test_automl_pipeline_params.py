"""Unit tests for automl_benchmark.pipeline_params."""

from __future__ import annotations

from pathlib import Path

import pytest

from automl_benchmark.pipeline_params import (
    build_pipeline_arguments,
    is_timeseries_dataset,
)
from automl_benchmark.settings import BenchmarkSettings


def _settings(
    tabular: Path,
    timeseries: Path,
    *,
    top_n: int = 3,
) -> BenchmarkSettings:
    return BenchmarkSettings(
        config_dir=tabular.parent.parent,
        pipeline_yaml=tabular,
        timeseries_pipeline_yaml=timeseries,
        train_data_bucket_name="bucket",
        train_data_secret_name="secret",
        artifact_s3_root_tabular="tabular-prefix",
        artifact_s3_root_timeseries="ts-prefix",
        benchmark_s3_prefix="benchmarks/ml",
        upload_benchmark_results=False,
        experiment_name="exp",
        top_n=top_n,
        poll_interval_seconds=30.0,
        timeout_seconds=3600.0,
        enable_caching=False,
        run_name_prefix="bench",
    )


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("binary", False),
        ("regression", False),
        ("timeseries", True),
        ("TIMESERIES", True),
    ],
)
def test_is_timeseries_dataset(task_type: str, expected: bool) -> None:
    assert is_timeseries_dataset({"task_type": task_type}) is expected


def test_build_tabular_arguments(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path)
    ds = {
        "train_data_file_key": "datasets/classification/a.csv",
        "label_column": "y",
        "task_type": "binary",
    }
    args = build_pipeline_arguments(ds, settings)
    assert args["train_data_secret_name"] == "secret"
    assert args["train_data_bucket_name"] == "bucket"
    assert args["train_data_file_key"] == "datasets/classification/a.csv"
    assert args["label_column"] == "y"
    assert args["task_type"] == "binary"
    assert args["top_n"] == 3


def test_build_timeseries_arguments(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path)
    ds = {
        "train_data_file_key": "datasets/timeseries/x.csv",
        "task_type": "timeseries",
        "id_column": "item_id",
        "timestamp_column": "ts",
        "target": "value",
        "prediction_length": 7,
        "known_covariates_names": ["cov_a"],
    }
    args = build_pipeline_arguments(ds, settings)
    assert args["target"] == "value"
    assert args["id_column"] == "item_id"
    assert args["timestamp_column"] == "ts"
    assert args["prediction_length"] == 7
    assert args["known_covariates_names"] == ["cov_a"]
    assert "label_column" not in args


def test_timeseries_target_fallback_to_label_column(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path)
    ds = {
        "train_data_file_key": "k.csv",
        "task_type": "timeseries",
        "id_column": "id",
        "timestamp_column": "ts",
        "label_column": "fallback_target",
    }
    args = build_pipeline_arguments(ds, settings)
    assert args["target"] == "fallback_target"


def test_timeseries_missing_target_raises(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path)
    ds = {
        "train_data_file_key": "k.csv",
        "task_type": "timeseries",
        "id_column": "id",
        "timestamp_column": "ts",
    }
    with pytest.raises(ValueError, match="target"):
        build_pipeline_arguments(ds, settings)


def test_pipeline_arguments_override(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path, top_n=2)
    ds = {
        "train_data_file_key": "k.csv",
        "label_column": "y",
        "task_type": "binary",
        "pipeline_arguments": {"top_n": 9, "custom_flag": True},
    }
    args = build_pipeline_arguments(ds, settings)
    assert args["top_n"] == 9
    assert args["custom_flag"] is True


def test_pipeline_params_alias(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    settings = _settings(tabular_pipeline_path, timeseries_pipeline_path)
    ds = {
        "train_data_file_key": "k.csv",
        "label_column": "y",
        "task_type": "binary",
        "pipeline_params": {"top_n": 5},
    }
    assert build_pipeline_arguments(ds, settings)["top_n"] == 5


