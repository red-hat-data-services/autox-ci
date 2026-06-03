"""Tests for pipeline argument filtering (pass-through to KFP)."""

from __future__ import annotations

from benchmark_common.pipeline_run import filter_pipeline_arguments, get_pipeline_supported_params
from tests.conftest import TABULAR_PIPELINE


def test_get_pipeline_supported_params_includes_tabular_inputs() -> None:
    supported = get_pipeline_supported_params(TABULAR_PIPELINE)
    assert supported is not None
    assert "train_data_file_key" in supported
    assert "task_type" in supported
    assert "top_n" in supported


def test_filter_keeps_undeclared_arguments() -> None:
    args = {
        "train_data_secret_name": "s",
        "train_data_bucket_name": "b",
        "train_data_file_key": "k",
        "label_column": "y",
        "task_type": "binary",
        "top_n": 2,
        "not_in_pipeline_ir": "surprise",
    }
    out = filter_pipeline_arguments(args, TABULAR_PIPELINE)
    assert out["not_in_pipeline_ir"] == "surprise"
    assert out == args
