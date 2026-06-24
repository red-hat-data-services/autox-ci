"""Parametrized functional tests for AutoGluon tabular training pipeline on RHOAI.

Test scenarios are defined in configs/tabular_test_configs.json. Data is pre-loaded
in S3; tests reference existing S3 keys without uploading local files. Filter by tags
with AUTOML_FUNCTIONAL_TESTS_TAGS (e.g. smoke, regression, negative).

Passing criteria for positive scenarios:
- Pipeline run finishes with SUCCEEDED status within timeout
- At least 1 model with metrics exists in S3
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

from .configs.configs import AutoMLTabularFunctionalConfig, get_tabular_configs_for_run
from autox_tests.lib.k8s_utils import add_kubeconfig_to_config

from .conftest import get_automl_functional_config
from autox_tests.lib.kfp_run_state import _get_run_state, _run_failed, _run_succeeded
from .utils import (
    TASK_PRIMARY_METRICS_TABULAR,
    _collect_failure_details,
    _get_failed_task_names,
    _run_pipeline_and_wait,
    collect_model_metrics_and_sizes,
    column_sample_to_instances,
    column_sample_to_v2_inputs,
    download_and_execute_automl_notebook,
    find_leaderboard_html,
    find_test_dataset_csv,
    run_deployment_test,
)

logger = logging.getLogger(__name__)

AUTOML_FUNCTIONAL_CONFIG = get_automl_functional_config()

TABULAR_POSITIVE_CONFIGS = get_tabular_configs_for_run(pass_type="positive")
TABULAR_NEGATIVE_CONFIGS = get_tabular_configs_for_run(pass_type="negative")

_EXPECTED_FAIL_TIMEOUT_CAP = 600

DEPLOY_AFTER_TRAINING: bool = os.environ.get(
    "RHOAI_DEPLOY_AFTER_TRAINING", ""
).strip().lower() in ("1", "true", "yes")


@pytest.mark.tabular
@pytest.mark.positive
@pytest.mark.skipif(
    AUTOML_FUNCTIONAL_CONFIG is None,
    reason="AutoML env incomplete (RHOAI_URL, RHOAI_TOKEN, RHOAI_PROJECT_NAME, S3, RHOAI_TRAIN_DATA_*; see .env.ml.example)",
)
class TestAutoMLTabularFunctional:
    """Positive functional tests for AutoGluon tabular training pipeline."""

    @pytest.mark.parametrize(
        "test_config",
        TABULAR_POSITIVE_CONFIGS,
        ids=[c.id for c in TABULAR_POSITIVE_CONFIGS],
    )
    def test_tabular_pipeline_run_succeeds(
        self,
        test_config: AutoMLTabularFunctionalConfig,
        automl_functional_config,
        kfp_client_automl_functional,
        tabular_pipeline_run_target,
        pipeline_run_timeout,
        s3_client_automl_functional,
        s3_cleanup_tracker,
        rhoai_cluster_kubeconfig,
    ):
        """Submit pipeline, assert SUCCEEDED, validate artifacts in S3."""
        if not kfp_client_automl_functional:
            pytest.fail("AutoML functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(automl_functional_config)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_automl_functional,
            tabular_pipeline_run_target,
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

        bucket = automl_functional_config.get("s3_bucket_artifacts")
        if s3_client_automl_functional and bucket:
            prefix = f"{tabular_pipeline_run_target.artifact_prefix}/{run_id}"
            s3_cleanup_tracker.track_artifact_prefix(bucket, prefix)

        if not _run_succeeded(detail):
            failure_info = _collect_failure_details(
                kfp_client_automl_functional,
                run_id,
                config=add_kubeconfig_to_config(
                    automl_functional_config, rhoai_cluster_kubeconfig
                ),
            )
            pytest.fail(
                f"[{test_config.id}] Pipeline run {run_id} expected SUCCEEDED but got "
                f"{state}{failure_info}"
            )

        deployment_result: dict = {}
        if s3_client_automl_functional and bucket:

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

            primary_metric_key = TASK_PRIMARY_METRICS_TABULAR.get(test_config.task_type)
            if primary_metric_key:
                for entry in model_entries:
                    metrics = entry["metrics"]
                    assert primary_metric_key in metrics, (
                        f"[{test_config.id}] Model '{entry['model_name']}' metrics missing "
                        f"'{primary_metric_key}': {list(metrics.keys())}"
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
                instances = (
                    column_sample_to_instances(test_config.inference_sample)
                    if test_config.inference_sample
                    else None
                )
                v2_inputs = (
                    column_sample_to_v2_inputs(test_config.inference_sample)
                    if test_config.inference_sample
                    else None
                )
                deployment_result = run_deployment_test(
                    scenario_id=test_config.id,
                    model_entries=model_entries,
                    s3_client=s3_client_automl_functional,
                    artifacts_bucket=bucket,
                    run_prefix=prefix,
                    run_id=run_id,
                    automl_functional_config=automl_functional_config,
                    temp_kubeconfig_path=rhoai_cluster_kubeconfig,
                    instances=instances,
                    v2_inputs=v2_inputs,
                )
                logger.info(
                    "[%s] deployment: isvc=%s ready=%s url=%s",
                    test_config.id,
                    deployment_result.get("isvc_name"),
                    deployment_result.get("isvc_ready"),
                    deployment_result.get("isvc_url"),
                )
                logger.info(
                    "[%s] v1 response: scored=%s body=%s error=%s",
                    test_config.id,
                    deployment_result.get("scored"),
                    deployment_result.get("v1_response"),
                    deployment_result.get("score_error"),
                )
                logger.info(
                    "[%s] v2 response: status=%s body=%s",
                    test_config.id,
                    deployment_result.get("v2_status_code"),
                    deployment_result.get("v2_response"),
                )

        if (
            DEPLOY_AFTER_TRAINING
            and deployment_result
            and not deployment_result.get("skipped")
        ):
            assert deployment_result.get("scored"), (
                f"[{test_config.id}] KServe v1 scoring failed: {deployment_result.get('score_error')}"
            )
            predictions = deployment_result.get("predictions")
            assert isinstance(predictions, list) and len(predictions) > 0, (
                f"[{test_config.id}] v1 predictions must be a non-empty list, got: {predictions!r}"
            )
            if deployment_result.get("v2_status_code") is not None:
                assert deployment_result["v2_status_code"] == 200, (
                    f"[{test_config.id}] KServe v2 scoring expected HTTP 200, "
                    f"got {deployment_result['v2_status_code']}: {deployment_result.get('v2_error')}"
                )
                v2_outputs = deployment_result.get("v2_outputs")
                assert isinstance(v2_outputs, list) and len(v2_outputs) > 0, (
                    f"[{test_config.id}] v2 outputs must be a non-empty list, got: {v2_outputs!r}"
                )


@pytest.mark.tabular
@pytest.mark.negative
@pytest.mark.skipif(
    AUTOML_FUNCTIONAL_CONFIG is None,
    reason="AutoML env incomplete (RHOAI_URL, RHOAI_TOKEN, RHOAI_PROJECT_NAME, S3, RHOAI_TRAIN_DATA_*; see .env.ml.example)",
)
class TestAutoMLTabularFunctionalNegative:
    """Negative functional tests for AutoGluon tabular training pipeline."""

    @pytest.mark.parametrize(
        "test_config",
        TABULAR_NEGATIVE_CONFIGS,
        ids=[c.id for c in TABULAR_NEGATIVE_CONFIGS],
    )
    def test_tabular_fault_scenario(
        self,
        test_config: AutoMLTabularFunctionalConfig,
        automl_functional_config,
        kfp_client_automl_functional,
        tabular_pipeline_run_target,
        pipeline_run_timeout,
        s3_client_automl_functional,
        s3_cleanup_tracker,
        rhoai_cluster_kubeconfig,
    ):
        """Submit pipeline with injected fault; assert FAILED within capped timeout."""
        if not kfp_client_automl_functional:
            pytest.fail("AutoML functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(automl_functional_config)
        timeout = min(pipeline_run_timeout, _EXPECTED_FAIL_TIMEOUT_CAP)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_automl_functional,
            tabular_pipeline_run_target,
            arguments,
            timeout,
        )
        elapsed = time.monotonic() - start

        state = _get_run_state(detail)
        failed_task_names = _get_failed_task_names(kfp_client_automl_functional, run_id)

        bucket = automl_functional_config.get("s3_bucket_artifacts")
        if s3_client_automl_functional and bucket:
            prefix = f"{tabular_pipeline_run_target.artifact_prefix}/{run_id}"
            s3_cleanup_tracker.track_artifact_prefix(bucket, prefix)

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
        failure_details = _collect_failure_details(
            kfp_client_automl_functional,
            run_id,
            config=add_kubeconfig_to_config(
                automl_functional_config, rhoai_cluster_kubeconfig
            ),
        )
        logger.info(failure_details)

        if "POD LOGS FOR FAILED PODS:" not in failure_details:
            logger.warning("Pod logs not collected for run %s — check k8s connectivity", run_id)

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
