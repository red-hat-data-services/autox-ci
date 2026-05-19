"""Parametrized functional tests for AutoGluon timeseries training pipeline on RHOAI.

Test scenarios are defined in configs/timeseries_test_configs.json. Data is pre-loaded
in S3; tests reference existing S3 keys without uploading local files. Filter by tags
with AUTOML_FUNCTIONAL_TESTS_TAGS (e.g. smoke, timeseries, negative).

Passing criteria for positive scenarios:
- Pipeline run finishes with SUCCEEDED status within timeout
- At least 1 model with metrics exists in S3 (MASE metric present)
- Leaderboard HTML artifact exists in S3
- Test dataset CSV artifact exists in S3

Passing criteria for negative scenarios:
- Pipeline run finishes with FAILED status within capped timeout
- At least one of the expected_failing_task names appears in the run's failed tasks
"""

import logging
import os
import random
import time

import pytest

from .configs.configs import (
    AutoMLTimeseriesFunctionalConfig,
    get_timeseries_configs_for_run,
)
from .conftest import get_automl_functional_config
from .utils import (
    TS_PRIMARY_METRIC,
    _collect_failure_details,
    _get_failed_task_names,
    _get_run_state,
    _run_failed,
    _run_pipeline_and_wait,
    _run_succeeded,
    collect_model_metrics_and_sizes,
    download_and_execute_automl_notebook,
    find_leaderboard_html,
    find_test_dataset_csv,
    run_deployment_test,
)

logger = logging.getLogger(__name__)

AUTOML_FUNCTIONAL_CONFIG = get_automl_functional_config()

TIMESERIES_POSITIVE_CONFIGS = get_timeseries_configs_for_run(pass_type="positive")
TIMESERIES_NEGATIVE_CONFIGS = get_timeseries_configs_for_run(pass_type="negative")

PIPELINE_DISPLAY_NAME = "autogluon-timeseries-training-pipeline"
_EXPECTED_FAIL_TIMEOUT_CAP = 600

DEPLOY_AFTER_TRAINING: bool = os.environ.get(
    "RHOAI_DEPLOY_AFTER_TRAINING", ""
).strip().lower() in ("1", "true", "yes")


@pytest.mark.timeseries
@pytest.mark.positive
@pytest.mark.skipif(
    AUTOML_FUNCTIONAL_CONFIG is None,
    reason="AutoML functional test env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, AUTOML_TRAIN_DATA_*; see .env)",
)
class TestAutoMLTimeseriesFunctional:
    """Positive functional tests for AutoGluon timeseries training pipeline."""

    @pytest.mark.parametrize(
        "test_config",
        TIMESERIES_POSITIVE_CONFIGS,
        ids=[c.id for c in TIMESERIES_POSITIVE_CONFIGS],
    )
    def test_timeseries_pipeline_run_succeeds(
        self,
        test_config: AutoMLTimeseriesFunctionalConfig,
        automl_functional_config,
        kfp_client_automl_functional,
        compiled_timeseries_pipeline_path,
        pipeline_run_timeout,
        s3_client_automl_functional,
        s3_cleanup_tracker,
        temp_kubeconfig_path,
    ):
        """Submit pipeline, assert SUCCEEDED, validate artifacts in S3."""
        if not kfp_client_automl_functional:
            pytest.fail("AutoML functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(automl_functional_config)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_automl_functional,
            compiled_timeseries_pipeline_path,
            arguments,
            pipeline_run_timeout,
        )
        elapsed = time.monotonic() - start

        state = _get_run_state(detail)
        logger.info(
            "[%s] run_id=%s state=%s elapsed=%.1fs",
            test_config.id,
            run_id,
            state,
            elapsed,
        )

        if not _run_succeeded(detail):
            failure_info = _collect_failure_details(
                kfp_client_automl_functional, run_id, config=automl_functional_config
            )
            pytest.fail(
                f"[{test_config.id}] Pipeline run {run_id} expected SUCCEEDED but got "
                f"{state}{failure_info}"
            )

        bucket = automl_functional_config.get("s3_bucket_artifacts")
        deployment_result: dict = {}
        if s3_client_automl_functional and bucket:
            prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
            s3_cleanup_tracker.track_artifact_prefix(bucket, prefix)

            model_entries = collect_model_metrics_and_sizes(
                s3_client_automl_functional, bucket, prefix
            )
            leaderboard_key, leaderboard_html = find_leaderboard_html(
                s3_client_automl_functional, bucket, prefix
            )
            test_dataset_key = find_test_dataset_csv(
                s3_client_automl_functional, bucket, prefix
            )

            total_predictor_size_mb = sum(
                e["total_predictor_size_mb"] for e in model_entries
            )
            for entry in model_entries:
                logger.info(
                    "[%s] model=%s predictor_size_mb=%.2f metrics=%s",
                    test_config.id,
                    entry["model_name"],
                    entry["total_predictor_size_mb"],
                    entry["metrics"],
                )
            logger.info(
                "[%s] total_predictor_size_mb=%.2f models=%d",
                test_config.id,
                total_predictor_size_mb,
                len(model_entries),
            )
            logger.info(
                "[%s] leaderboard_key=%s test_dataset_key=%s",
                test_config.id,
                leaderboard_key,
                test_dataset_key,
            )

            assert len(model_entries) >= 1, (
                f"[{test_config.id}] Expected at least 1 model with metrics under {prefix}; "
                f"found {len(model_entries)}"
            )

            for entry in model_entries:
                metrics = entry["metrics"]
                assert TS_PRIMARY_METRIC in metrics, (
                    f"[{test_config.id}] Model '{entry['model_name']}' metrics missing "
                    f"'{TS_PRIMARY_METRIC}': {list(metrics.keys())}"
                )

            assert leaderboard_key is not None, (
                f"[{test_config.id}] No leaderboard HTML artifact found under {prefix}"
            )
            assert leaderboard_html is not None, (
                f"[{test_config.id}] Leaderboard HTML found at {leaderboard_key} but could not be read"
            )

            assert test_dataset_key is not None, (
                f"[{test_config.id}] No sampled_test_dataset artifact found under {prefix}"
            )

            notebook_entries = [e for e in model_entries if e["notebook_key"]]
            if notebook_entries:
                chosen = random.choice(notebook_entries)
                download_and_execute_automl_notebook(
                    s3_client_automl_functional, bucket, chosen["notebook_key"]
                )

            if DEPLOY_AFTER_TRAINING and model_entries:
                ts_env_vars: dict[str, str] = {}
                if test_config.id_column != "item_id":
                    ts_env_vars["AUTOGLUON_TS_ID_COLUMN"] = test_config.id_column
                if test_config.timestamp_column != "timestamp":
                    ts_env_vars["AUTOGLUON_TS_TIMESTAMP_COLUMN"] = (
                        test_config.timestamp_column
                    )
                deployment_result = run_deployment_test(
                    scenario_id=test_config.id,
                    model_entries=model_entries,
                    s3_client=s3_client_automl_functional,
                    artifacts_bucket=bucket,
                    run_prefix=prefix,
                    run_id=run_id,
                    automl_functional_config=automl_functional_config,
                    temp_kubeconfig_path=temp_kubeconfig_path,
                    instances=test_config.inference_sample or None,
                    isvc_env_vars=ts_env_vars or None,
                )
                logger.info(
                    "[%s] deployment: isvc=%s ready=%s url=%s scored=%s predictions_count=%d",
                    test_config.id,
                    deployment_result.get("isvc_name"),
                    deployment_result.get("isvc_ready"),
                    deployment_result.get("isvc_url"),
                    deployment_result.get("scored"),
                    len(deployment_result.get("predictions") or []),
                )

        if (
            DEPLOY_AFTER_TRAINING
            and deployment_result
            and not deployment_result.get("skipped")
        ):
            assert deployment_result.get("scored"), (
                f"[{test_config.id}] KServe scoring failed: {deployment_result.get('score_error')}"
            )
            predictions = deployment_result.get("predictions")
            assert isinstance(predictions, list) and len(predictions) > 0, (
                f"[{test_config.id}] Predictions must be a non-empty list, got: {predictions!r}"
            )


@pytest.mark.timeseries
@pytest.mark.negative
@pytest.mark.skipif(
    AUTOML_FUNCTIONAL_CONFIG is None,
    reason="AutoML functional test env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, AUTOML_TRAIN_DATA_*; see .env)",
)
class TestAutoMLTimeseriesFunctionalNegative:
    """Negative functional tests for AutoGluon timeseries training pipeline."""

    @pytest.mark.parametrize(
        "test_config",
        TIMESERIES_NEGATIVE_CONFIGS,
        ids=[c.id for c in TIMESERIES_NEGATIVE_CONFIGS],
    )
    def test_timeseries_fault_scenario(
        self,
        test_config: AutoMLTimeseriesFunctionalConfig,
        automl_functional_config,
        kfp_client_automl_functional,
        compiled_timeseries_pipeline_path,
        pipeline_run_timeout,
        s3_client_automl_functional,
        s3_cleanup_tracker,
    ):
        """Submit pipeline with injected fault; assert FAILED within capped timeout."""
        if not kfp_client_automl_functional:
            pytest.fail("AutoML functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(automl_functional_config)
        timeout = min(pipeline_run_timeout, _EXPECTED_FAIL_TIMEOUT_CAP)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_automl_functional,
            compiled_timeseries_pipeline_path,
            arguments,
            timeout,
        )
        elapsed = time.monotonic() - start

        state = _get_run_state(detail)
        failed_task_names = _get_failed_task_names(kfp_client_automl_functional, run_id)

        logger.info(
            "[%s] run_id=%s state=%s elapsed=%.1fs fault=%r category=%r failed_tasks=%s expected=%s",
            test_config.id,
            run_id,
            state,
            round(elapsed, 1),
            test_config.injected_fault,
            test_config.fault_category,
            failed_task_names,
            test_config.expected_failing_task,
        )
        logger.info(
            _collect_failure_details(
                kfp_client_automl_functional, run_id, config=automl_functional_config
            )
        )

        assert _run_failed(detail), (
            f"[{test_config.id}] Pipeline run {run_id} expected FAILED but got {state}. "
            f"Injected fault: {test_config.injected_fault}"
        )

        if test_config.expected_failing_task:
            matched = any(
                t in failed_task_names for t in test_config.expected_failing_task
            )
            assert matched, (
                f"[{test_config.id}] Expected one of {test_config.expected_failing_task} to fail; "
                f"actual failed tasks: {failed_task_names}"
            )
