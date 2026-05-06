"""Resolve train/data locations for root RHOAI tests (upload vs existing S3)."""

from __future__ import annotations

from typing import Any

from autox_tests.lib.config_loaders import (
    AutomlTabularTestConfig,
    AutomlTimeseriesTestConfig,
    AutoragOptimizationTestConfig,
)


def join_s3_key(prefix: str | None, key: str) -> str:
    """Combine an optional prefix and a key into one object key (no leading slash)."""
    k = key.strip().lstrip("/")
    if not prefix or not prefix.strip():
        return k
    p = prefix.strip().strip("/")
    return f"{p}/{k}" if k else p


def resolve_automl_dataset_s3_location(
    config: AutomlTabularTestConfig | AutomlTimeseriesTestConfig,
    uploaded_map: dict[str, dict[str, str]],
    rhoai_automl_config: dict[str, Any],
    test_data_source: dict[str, str | None],
) -> tuple[str, str]:
    """Return ``(bucket, object_key)`` for AutoGluon train data."""
    if config.data_mode == "existing_s3":
        bucket = (
            config.dataset_bucket
            or test_data_source.get("bucket")
            or rhoai_automl_config["s3_bucket_data"]
        )
        if not config.dataset_key:
            raise ValueError(f"Config {config.id!r} data_mode=existing_s3 requires dataset_key")
        key = config.dataset_key.strip()
        prefix = test_data_source.get("prefix")
        if prefix and not config.dataset_bucket:
            key = join_s3_key(prefix, key)
        return bucket, key
    loc = uploaded_map.get(config.dataset_path)
    if not loc:
        raise KeyError(
            f"No uploaded dataset for dataset_path={config.dataset_path!r}; "
            "check data_mode=upload and session upload fixture."
        )
    return loc["bucket"], loc["key"]


def resolve_autorag_s3_locations(
    config: AutoragOptimizationTestConfig,
    uploaded_map: dict[str, dict[str, str]],
    connection: dict[str, Any],
    test_data_source: dict[str, str | None],
) -> dict[str, str]:
    """Return bucket/key fields for the AutoRAG optimization pipeline submission."""
    if config.data_mode == "upload":
        loc = uploaded_map.get(config.id)
        if not loc:
            raise KeyError(f"No upload mapping for AutoRAG config id={config.id!r}")
        return {
            "test_data_bucket_name": loc["test_data_bucket_name"],
            "test_data_key": loc["test_data_key"],
            "input_data_bucket_name": loc["input_data_bucket_name"],
            "input_data_key": loc["input_data_key"],
        }
    if config.data_mode == "existing_s3":
        default_bucket = test_data_source.get("bucket")
        tb = config.test_data_bucket or default_bucket or connection.get("test_data_bucket_name")
        ib = config.input_data_bucket or default_bucket or connection.get("input_data_bucket_name")
        if not config.test_data_key or not config.input_data_key:
            raise ValueError(f"Config {config.id!r} data_mode=existing_s3 requires test_data_key and input_data_key")
        tk = config.test_data_key.strip()
        ik = config.input_data_key.strip()
        prefix = test_data_source.get("prefix")
        if prefix:
            if not config.test_data_bucket:
                tk = join_s3_key(prefix, tk)
            if not config.input_data_bucket:
                ik = join_s3_key(prefix, ik)
        if not tb or not ib:
            raise ValueError(
                f"Config {config.id!r}: set test_data_bucket / input_data_bucket in JSON "
                "or TEST_DATA_SOURCE_BUCKET / RHOAI_TEST_DATA_BUCKET in the environment"
            )
        return {
            "test_data_bucket_name": tb,
            "test_data_key": tk,
            "input_data_bucket_name": ib,
            "input_data_key": ik,
        }
    raise ValueError(f"Unknown data_mode {config.data_mode!r} for config {config.id!r}")
