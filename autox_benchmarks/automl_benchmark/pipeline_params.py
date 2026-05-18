"""Map dataset manifest entries to pipeline argument dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from automl_benchmark.settings import BenchmarkSettings


def is_timeseries_dataset(dataset: dict[str, Any]) -> bool:
    t = dataset.get("task_type")
    if t is None:
        return False
    return str(t).strip().lower() == "timeseries"


def pipeline_file_for_dataset(dataset: dict[str, Any], settings: BenchmarkSettings) -> Path:
    if is_timeseries_dataset(dataset):
        return settings.timeseries_pipeline_yaml
    return settings.pipeline_yaml


def build_pipeline_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    if is_timeseries_dataset(dataset):
        return _build_timeseries_arguments(dataset, settings)
    return _build_tabular_arguments(dataset, settings)


def _build_tabular_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    return {
        "train_data_secret_name": settings.train_data_secret_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "train_data_file_key": str(dataset["train_data_file_key"]),
        "label_column": str(dataset["label_column"]),
        "task_type": str(dataset["task_type"]),
        "top_n": settings.top_n,
    }


def _build_timeseries_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    target = dataset.get("target") or dataset.get("label_column")
    if not target:
        raise ValueError("timeseries datasets require 'target' or 'label_column'")
    args: dict[str, Any] = {
        "train_data_secret_name": settings.train_data_secret_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "train_data_file_key": str(dataset["train_data_file_key"]),
        "target": str(target),
        "id_column": str(dataset["id_column"]),
        "timestamp_column": str(dataset["timestamp_column"]),
        "top_n": settings.top_n,
    }
    kc = dataset.get("known_covariates_names")
    if isinstance(kc, list) and kc:
        args["known_covariates_names"] = [str(x) for x in kc]
    pl = dataset.get("prediction_length")
    if pl is not None and str(pl).strip() != "":
        args["prediction_length"] = int(pl)
    return args
