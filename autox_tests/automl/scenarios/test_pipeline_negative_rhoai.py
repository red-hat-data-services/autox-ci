"""Negative OpenShift AI / KFP tests: invalid parameters must not yield successful runs."""

from __future__ import annotations

from typing import Any

import pytest

from autox_tests.lib.config_loaders import (
    get_automl_tabular_baseline_config,
    get_automl_timeseries_baseline_config,
    get_autorag_baseline_config_from_run,
)
from autox_tests.lib.data_resolution import resolve_automl_dataset_s3_location, resolve_autorag_s3_locations
from autox_tests.lib.integration_failures import require_autorag_env, require_rhoai_automl_env
from autox_tests.lib.kfp_negative import assert_pipeline_does_not_succeed
from autox_tests.lib.settings import (
    autorag_pipeline_arguments,
    get_autorag_connection_config,
    get_rhoai_automl_config,
    get_test_data_source_defaults,
)

# Unknown parameter name must not collide with real pipeline inputs.
_UNKNOWN_PIPELINE_PARAM = "__rhoai_negative_unknown_param__"

_TABULAR_DISPLAY = "autogluon-tabular-training-pipeline"
_TIMESERIES_DISPLAY = "autogluon-timeseries-training-pipeline"
_AUTORAG_DISPLAY = "documents-rag-optimization-pipeline"

# Invalid values for parameters that are not S3/data-path inputs (train_data_*, buckets, secrets).
_TABULAR_INVALID_NON_DATA_PARAMS: list[tuple[str, Any]] = [
    ("task_type", "not_a_valid_tabular_task_type"),
    ("top_n", 0),
    ("label_column", "___rhoai_negative_missing_label___"),
]

_TIMESERIES_INVALID_NON_DATA_PARAMS: list[tuple[str, Any]] = [
    ("target", "___rhoai_negative_unknown_target___"),
    ("id_column", "___rhoai_negative_unknown_id___"),
    ("timestamp_column", "___rhoai_negative_unknown_ts___"),
    ("prediction_length", 0),
    ("top_n", 0),
]

_AUTORAG_INVALID_NON_DATA_PARAMS: list[tuple[str, Any]] = [
    ("optimization_metric", "not_a_supported_metric"),
    ("optimization_max_rag_patterns", 0),
    ("embeddings_models", ["__rhoai_negative_invalid_embedding__"]),
    ("generation_models", ["__rhoai_negative_invalid_generation__"]),
]


@pytest.mark.integration
@pytest.mark.openshift_ai
@pytest.mark.pipeline_negative
@pytest.mark.tabular
class TestAutomlTabularNegativeRhoaiKfp:
    """Invalid tabular AutoML pipeline arguments."""

    def test_unknown_parameter_does_not_succeed(
        self,
        rhoai_automl_project: str | None,
        uploaded_automl_tabular_datasets: dict[str, dict[str, str]],
        kfp_client_automl: Any,
        automl_tabular_pipeline_package: str,
        automl_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_rhoai_automl_env()
        if not rhoai_automl_project or not kfp_client_automl:
            pytest.fail(
                "AutoML session setup did not complete: missing project namespace and/or KFP client."
            )
        cfg = get_rhoai_automl_config()
        assert cfg is not None
        test_config = get_automl_tabular_baseline_config()
        bucket, key = resolve_automl_dataset_s3_location(
            test_config,
            uploaded_automl_tabular_datasets,
            cfg,
            get_test_data_source_defaults(),
        )
        arguments = test_config.get_pipeline_arguments(bucket, key, cfg["s3_secret_name"])
        arguments[_UNKNOWN_PIPELINE_PARAM] = "1"
        assert_pipeline_does_not_succeed(
            kfp_client_automl,
            automl_tabular_pipeline_package,
            arguments,
            f"{automl_run_name}-neg-unknown-param",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_TABULAR_DISPLAY,
        )

    @pytest.mark.parametrize(
        ("param_name", "invalid_value"),
        _TABULAR_INVALID_NON_DATA_PARAMS,
        ids=[name for name, _ in _TABULAR_INVALID_NON_DATA_PARAMS],
    )
    def test_invalid_non_data_parameter_does_not_succeed(
        self,
        param_name: str,
        invalid_value: Any,
        rhoai_automl_project: str | None,
        uploaded_automl_tabular_datasets: dict[str, dict[str, str]],
        kfp_client_automl: Any,
        automl_tabular_pipeline_package: str,
        automl_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_rhoai_automl_env()
        if not rhoai_automl_project or not kfp_client_automl:
            pytest.fail(
                "AutoML session setup did not complete: missing project namespace and/or KFP client."
            )
        cfg = get_rhoai_automl_config()
        assert cfg is not None
        test_config = get_automl_tabular_baseline_config()
        bucket, key = resolve_automl_dataset_s3_location(
            test_config,
            uploaded_automl_tabular_datasets,
            cfg,
            get_test_data_source_defaults(),
        )
        arguments = test_config.get_pipeline_arguments(bucket, key, cfg["s3_secret_name"])
        arguments[param_name] = invalid_value
        assert_pipeline_does_not_succeed(
            kfp_client_automl,
            automl_tabular_pipeline_package,
            arguments,
            f"{automl_run_name}-neg-{param_name}",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_TABULAR_DISPLAY,
        )


@pytest.mark.integration
@pytest.mark.openshift_ai
@pytest.mark.pipeline_negative
@pytest.mark.timeseries
class TestAutomlTimeseriesNegativeRhoaiKfp:
    """Invalid time series AutoML pipeline arguments."""

    def test_unknown_parameter_does_not_succeed(
        self,
        rhoai_automl_project: str | None,
        uploaded_automl_timeseries_datasets: dict[str, dict[str, str]],
        kfp_client_automl: Any,
        automl_timeseries_pipeline_package: str,
        automl_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_rhoai_automl_env()
        if not rhoai_automl_project or not kfp_client_automl:
            pytest.fail(
                "AutoML session setup did not complete: missing project namespace and/or KFP client."
            )
        cfg = get_rhoai_automl_config()
        assert cfg is not None
        test_config = get_automl_timeseries_baseline_config()
        bucket, key = resolve_automl_dataset_s3_location(
            test_config,
            uploaded_automl_timeseries_datasets,
            cfg,
            get_test_data_source_defaults(),
        )
        arguments = test_config.get_pipeline_arguments(bucket, key, cfg["s3_secret_name"])
        arguments[_UNKNOWN_PIPELINE_PARAM] = "1"
        assert_pipeline_does_not_succeed(
            kfp_client_automl,
            automl_timeseries_pipeline_package,
            arguments,
            f"{automl_run_name}-ts-neg-unknown-param",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_TIMESERIES_DISPLAY,
        )

    @pytest.mark.parametrize(
        ("param_name", "invalid_value"),
        _TIMESERIES_INVALID_NON_DATA_PARAMS,
        ids=[name for name, _ in _TIMESERIES_INVALID_NON_DATA_PARAMS],
    )
    def test_invalid_non_data_parameter_does_not_succeed(
        self,
        param_name: str,
        invalid_value: Any,
        rhoai_automl_project: str | None,
        uploaded_automl_timeseries_datasets: dict[str, dict[str, str]],
        kfp_client_automl: Any,
        automl_timeseries_pipeline_package: str,
        automl_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_rhoai_automl_env()
        if not rhoai_automl_project or not kfp_client_automl:
            pytest.fail(
                "AutoML session setup did not complete: missing project namespace and/or KFP client."
            )
        cfg = get_rhoai_automl_config()
        assert cfg is not None
        test_config = get_automl_timeseries_baseline_config()
        bucket, key = resolve_automl_dataset_s3_location(
            test_config,
            uploaded_automl_timeseries_datasets,
            cfg,
            get_test_data_source_defaults(),
        )
        arguments = test_config.get_pipeline_arguments(bucket, key, cfg["s3_secret_name"])
        arguments[param_name] = invalid_value
        assert_pipeline_does_not_succeed(
            kfp_client_automl,
            automl_timeseries_pipeline_package,
            arguments,
            f"{automl_run_name}-ts-neg-{param_name}",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_TIMESERIES_DISPLAY,
        )


@pytest.mark.integration
@pytest.mark.openshift_ai
@pytest.mark.pipeline_negative
@pytest.mark.autorag
class TestAutoragNegativeRhoaiKfp:
    """Invalid AutoRAG optimization pipeline arguments."""

    def test_unknown_parameter_does_not_succeed(
        self,
        kfp_client_autorag: Any,
        autorag_pipeline_package: str,
        uploaded_autorag_by_config_id: dict[str, dict[str, str]],
        autorag_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_autorag_env()
        if not kfp_client_autorag:
            pytest.fail("AutoRAG session setup did not produce a KFP client.")
        base = get_autorag_baseline_config_from_run()
        if base is None:
            pytest.skip(
                "No AutoRAG scenario selected (e.g. RHOAI_TEST_CONFIG_TAGS excludes all configs); "
                "cannot resolve uploads for negative tests."
            )
        conn = get_autorag_connection_config()
        assert conn is not None
        locations = resolve_autorag_s3_locations(
            base,
            uploaded_autorag_by_config_id,
            conn,
            get_test_data_source_defaults(),
        )
        merged = {**conn, **locations}
        arguments = autorag_pipeline_arguments(merged)
        arguments.update(base.argument_overrides)
        arguments[_UNKNOWN_PIPELINE_PARAM] = "1"
        assert_pipeline_does_not_succeed(
            kfp_client_autorag,
            autorag_pipeline_package,
            arguments,
            f"{autorag_run_name}-neg-unknown-param",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_AUTORAG_DISPLAY,
            enable_caching=False,
        )

    @pytest.mark.parametrize(
        ("param_name", "invalid_value"),
        _AUTORAG_INVALID_NON_DATA_PARAMS,
        ids=[name for name, _ in _AUTORAG_INVALID_NON_DATA_PARAMS],
    )
    def test_invalid_non_data_parameter_does_not_succeed(
        self,
        param_name: str,
        invalid_value: Any,
        kfp_client_autorag: Any,
        autorag_pipeline_package: str,
        uploaded_autorag_by_config_id: dict[str, dict[str, str]],
        autorag_run_name: str,
        pipeline_negative_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        require_autorag_env()
        if not kfp_client_autorag:
            pytest.fail("AutoRAG session setup did not produce a KFP client.")
        base = get_autorag_baseline_config_from_run()
        if base is None:
            pytest.skip(
                "No AutoRAG scenario selected (e.g. RHOAI_TEST_CONFIG_TAGS excludes all configs); "
                "cannot resolve uploads for negative tests."
            )
        conn = get_autorag_connection_config()
        assert conn is not None
        locations = resolve_autorag_s3_locations(
            base,
            uploaded_autorag_by_config_id,
            conn,
            get_test_data_source_defaults(),
        )
        merged = {**conn, **locations}
        arguments = autorag_pipeline_arguments(merged)
        arguments.update(base.argument_overrides)
        arguments[param_name] = invalid_value
        assert_pipeline_does_not_succeed(
            kfp_client_autorag,
            autorag_pipeline_package,
            arguments,
            f"{autorag_run_name}-neg-{param_name}",
            timeout_seconds=pipeline_negative_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=_AUTORAG_DISPLAY,
            enable_caching=False,
        )
