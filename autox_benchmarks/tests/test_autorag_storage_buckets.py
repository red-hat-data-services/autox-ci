"""Tests for AutoRAG S3 bucket resolution."""

from __future__ import annotations

import pytest

from autorag_benchmark.storage_buckets import (
    resolve_dataset_upload_bucket,
    resolve_pattern_artifacts_bucket,
)


def test_resolve_pattern_artifacts_bucket_from_config() -> None:
    cfg = {"storage": {"test_data_bucket_name": "test-bucket"}}
    assert resolve_pattern_artifacts_bucket(cfg) == "test-bucket"


def test_resolve_pattern_artifacts_bucket_explicit_override() -> None:
    cfg = {"storage": {"test_data_bucket_name": "test-bucket"}}
    assert resolve_pattern_artifacts_bucket(cfg, explicit="cli-bucket") == "cli-bucket"


def test_resolve_pattern_artifacts_bucket_missing_fails() -> None:
    with pytest.raises(ValueError, match="BENCHMARK_TEST_DATA_BUCKET_NAME"):
        resolve_pattern_artifacts_bucket({})


def test_resolve_dataset_upload_bucket_from_config() -> None:
    cfg = {"storage": {"input_data_bucket_name": "input-bucket"}}
    assert resolve_dataset_upload_bucket(cfg) == "input-bucket"
