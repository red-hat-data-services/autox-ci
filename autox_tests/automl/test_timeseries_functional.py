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

import json
import logging
import os
import random
import time

import pytest

from .configs.configs import (
    AutoMLTimeseriesFunctionalConfig,
    get_timeseries_configs_for_run,
)
from .conftest import AUTOML_FUNCTIONAL_CONFIG
from .utils import (
    TS_PRIMARY_METRIC,
    _K8S_CALL_TIMEOUT,
    _KSERVE_GROUP,
    _KSERVE_SR_PLURAL,
    _KSERVE_SR_VERSION,
    collect_failure_details,
    collect_model_metrics_and_sizes,
    create_connection_rbac,
    create_connection_sa,
    create_inference_service,
    create_kserve_s3_secret,
    delete_inference_service,
    download_and_execute_automl_notebook,
    ensure_deployment_storage_annotations,
    ensure_serving_runtime,
    fetch_hardware_profile_resource_version,
    fetch_pod_logs_str,
    find_leaderboard_html,
    find_test_dataset_csv,
    find_top_model_predictor_prefix,
    get_failed_task_names,
    get_run_state,
    list_s3_objects,
    load_k8s_config,
    log_isvc_events,
    make_isvc_name,
    resolve_isvc_external_url,
    run_failed,
    run_pipeline_and_wait,
    run_succeeded,
    score_inference_service,
    wait_for_isvc_ready,
)

logger = logging.getLogger(__name__)

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
        run_id, detail = run_pipeline_and_wait(
            kfp_client_automl_functional,
            compiled_timeseries_pipeline_path,
            arguments,
            pipeline_run_timeout,
        )
        elapsed = time.monotonic() - start

        state = get_run_state(detail)
        logger.info(
            "[%s] run_id=%s state=%s elapsed=%.1fs",
            test_config.id,
            run_id,
            state,
            elapsed,
        )

        if not run_succeeded(detail):
            failure_info = collect_failure_details(
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
                deployment_result = self._run_deployment_test(
                    test_config=test_config,
                    model_entries=model_entries,
                    s3_client=s3_client_automl_functional,
                    artifacts_bucket=bucket,
                    run_prefix=prefix,
                    run_id=run_id,
                    automl_functional_config=automl_functional_config,
                    temp_kubeconfig_path=temp_kubeconfig_path,
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

    def _run_deployment_test(
        self,
        *,
        test_config: "AutoMLTimeseriesFunctionalConfig",
        model_entries: list[dict],
        s3_client,
        artifacts_bucket: str,
        run_prefix: str,
        run_id: str,
        automl_functional_config: dict,
        temp_kubeconfig_path: str | None,
    ) -> dict:
        """Deploy the top-1 timeseries model via KServe and validate readiness + scoring.

        Controlled by ``RHOAI_DEPLOY_AFTER_TRAINING=true``.
        Same env vars as tabular deployment: RHOAI_SERVING_IMAGE, RHOAI_CREATE_SERVING_RUNTIME,
        RHOAI_SERVING_RUNTIME_NAME, RHOAI_KSERVE_STORAGE_KEY, RHOAI_INFERENCE_TIMEOUT,
        RHOAI_HARDWARE_PROFILE_NAME, RHOAI_HARDWARE_PROFILE_NAMESPACE,
        RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION.
        """
        try:
            from kubernetes import client
        except ImportError:
            logger.warning("kubernetes package not installed; skipping deployment test")
            return {"skipped": True, "reason": "kubernetes package not installed"}

        namespace = automl_functional_config["rhoai_project"]
        token = automl_functional_config.get("rhoai_token")
        serving_image = os.environ.get("RHOAI_SERVING_IMAGE", "").strip()
        create_runtime = os.environ.get(
            "RHOAI_CREATE_SERVING_RUNTIME", ""
        ).strip().lower() in ("1", "true", "yes")
        hardware_profile_name = os.environ.get(
            "RHOAI_HARDWARE_PROFILE_NAME", "default-profile"
        ).strip()
        hardware_profile_namespace = os.environ.get(
            "RHOAI_HARDWARE_PROFILE_NAMESPACE", "redhat-ods-applications"
        ).strip()
        predictor_cpu = os.environ.get("RHOAI_PREDICTOR_CPU", "2").strip()
        predictor_memory = os.environ.get("RHOAI_PREDICTOR_MEMORY", "4Gi").strip()

        top_model = model_entries[0]
        model_name = top_model["model_name"]
        isvc_name = make_isvc_name(test_config.id, run_id)
        existing_runtime_name = os.environ.get("RHOAI_SERVING_RUNTIME_NAME", "").strip()
        serving_runtime_name = existing_runtime_name or isvc_name

        result: dict = {
            "model_name": model_name,
            "serving_runtime": serving_runtime_name,
            "storage_key": None,
            "isvc_name": isvc_name,
            "isvc_ready": False,
            "isvc_url": None,
            "scored": False,
            "predictions": None,
            "score_error": None,
        }

        predictor_prefix = find_top_model_predictor_prefix(
            s3_client, artifacts_bucket, run_prefix, model_name
        )
        if predictor_prefix is None:
            result["score_error"] = (
                f"Predictor prefix not found for model {model_name!r}"
            )
            return result

        predictor_objects = list_s3_objects(
            s3_client, artifacts_bucket, predictor_prefix
        )
        predictor_keys = {obj["Key"].split("/")[-1] for obj in predictor_objects}
        if "predictor.pkl" not in predictor_keys:
            result["score_error"] = (
                f"predictor.pkl not found under s3://{artifacts_bucket}/{predictor_prefix} "
                f"(found: {sorted(predictor_keys) or 'nothing'})"
            )
            return result

        storage_path = predictor_prefix
        isvc_created = False
        temp_secret_name: str | None = None
        temp_sa_name: str | None = None
        temp_rbac_name: str | None = None
        temp_runtime_name: str | None = None

        try:
            load_k8s_config(temp_kubeconfig_path)
            v1 = client.CoreV1Api()
            rbac_v1 = client.RbacAuthorizationV1Api()
            apps_v1 = client.AppsV1Api()
            co = client.CustomObjectsApi()

            existing_storage_key = os.environ.get(
                "RHOAI_KSERVE_STORAGE_KEY", ""
            ).strip()
            if existing_storage_key:
                storage_key = existing_storage_key
                result["storage_key"] = storage_key
            else:
                temp_secret_name = f"kserve-s3-{isvc_name[:40]}"
                create_kserve_s3_secret(
                    v1,
                    namespace,
                    temp_secret_name,
                    artifacts_bucket,
                    automl_functional_config,
                )
                storage_key = temp_secret_name
                result["storage_key"] = storage_key
                temp_sa_name = create_connection_sa(v1, namespace, temp_secret_name)
                temp_rbac_name = create_connection_rbac(
                    rbac_v1, namespace, temp_sa_name, temp_secret_name
                )
                logger.info(
                    "Waiting 15s for controller informer to index secret and SA..."
                )
                time.sleep(15)

            if create_runtime:
                if not serving_image:
                    logger.warning(
                        "RHOAI_CREATE_SERVING_RUNTIME=true but RHOAI_SERVING_IMAGE not set"
                    )
                else:
                    newly_created = ensure_serving_runtime(
                        co, namespace, serving_runtime_name, serving_image
                    )
                    if newly_created and not existing_runtime_name:
                        temp_runtime_name = serving_runtime_name
                        logger.info(
                            "Waiting 30s for KServe controller to index ServingRuntime..."
                        )
                        time.sleep(30)

            hw_rv = os.environ.get(
                "RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION", ""
            ).strip()
            if not hw_rv:
                hw_rv = fetch_hardware_profile_resource_version(
                    co, hardware_profile_namespace, hardware_profile_name
                )
            if not hw_rv:
                raise RuntimeError(
                    f"Could not resolve hardware-profile-resource-version for {hardware_profile_name!r} "
                    f"in {hardware_profile_namespace!r}. Set RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION to override."
                )

            # In RHOAI 3.4, the ID and timestamp columns must be configured 
            # via environment variables during deployment creation.
            ts_env_vars: dict[str, str] = {}
            if test_config.id_column != "item_id":
                ts_env_vars["AUTOGLUON_TS_ID_COLUMN"] = test_config.id_column
            if test_config.timestamp_column != "timestamp":
                ts_env_vars["AUTOGLUON_TS_TIMESTAMP_COLUMN"] = (
                    test_config.timestamp_column
                )

            create_inference_service(
                co,
                namespace,
                isvc_name,
                serving_runtime_name,
                storage_path,
                storage_key,
                hardware_profile_name=hardware_profile_name,
                hardware_profile_namespace=hardware_profile_namespace,
                hardware_profile_resource_version=hw_rv,
                predictor_cpu=predictor_cpu,
                predictor_memory=predictor_memory,
                env_vars=ts_env_vars or None,
            )
            isvc_created = True
            logger.info(
                "Created InferenceService %r in namespace %r", isvc_name, namespace
            )

            ensure_deployment_storage_annotations(
                apps_v1,
                namespace,
                isvc_name,
                storage_key=storage_key,
                artifacts_bucket=artifacts_bucket,
                storage_path=storage_path,
                wait_seconds=60,
            )
            log_isvc_events(v1, namespace, isvc_name)

            inference_timeout = int(os.environ.get("RHOAI_INFERENCE_TIMEOUT", "300"))
            isvc_ready, blocking_reason = wait_for_isvc_ready(
                co, namespace, isvc_name, timeout_seconds=inference_timeout
            )
            result["isvc_ready"] = isvc_ready

            if blocking_reason or not isvc_ready:
                pod_logs = fetch_pod_logs_str(
                    v1, namespace, f"serving.kserve.io/inferenceservice={isvc_name}"
                )
                logger.error("Predictor pod logs for %r:\n%s", isvc_name, pod_logs)
                if blocking_reason:
                    result["score_error"] = (
                        f"ISVC {isvc_name!r} blocking condition: {blocking_reason}"
                    )
                    return result

            external_url = resolve_isvc_external_url(co, namespace, isvc_name)
            result["isvc_url"] = external_url

            if not external_url:
                result["score_error"] = (
                    f"No external Route found for ISVC {isvc_name!r} after {inference_timeout}s"
                )
                return result

            if test_config.inference_sample:
                instances = test_config.inference_sample
                try:
                    response = score_inference_service(
                        external_url, isvc_name, instances, token
                    )
                    result["scored"] = True
                    result["predictions"] = response.get("predictions")
                    logger.info(
                        "Scoring succeeded for %r: %s",
                        isvc_name,
                        json.dumps(response, default=str),
                    )
                except Exception as score_err:
                    pod_logs = fetch_pod_logs_str(
                        v1, namespace, f"serving.kserve.io/inferenceservice={isvc_name}"
                    )
                    result["score_error"] = f"{score_err}\n{pod_logs}"
            else:
                logger.info(
                    "No inference_sample in config %r — skipping scoring",
                    test_config.id,
                )

        except Exception as deploy_err:
            logger.error(
                "Deployment test failed for %r: %s",
                test_config.id,
                deploy_err,
                exc_info=True,
            )
            result["score_error"] = str(deploy_err)

        finally:
            if isvc_created:
                try:
                    load_k8s_config(temp_kubeconfig_path)
                    delete_inference_service(
                        client.CustomObjectsApi(), namespace, isvc_name
                    )
                    logger.info("Deleted InferenceService %r", isvc_name)
                except Exception as e:
                    logger.warning("Failed to delete ISVC %r: %s", isvc_name, e)
            if temp_runtime_name:
                try:
                    client.CustomObjectsApi().delete_namespaced_custom_object(
                        group=_KSERVE_GROUP,
                        version=_KSERVE_SR_VERSION,
                        namespace=namespace,
                        plural=_KSERVE_SR_PLURAL,
                        name=temp_runtime_name,
                        _request_timeout=_K8S_CALL_TIMEOUT,
                    )
                    logger.info(
                        "Deleted temporary ServingRuntime %r", temp_runtime_name
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to delete ServingRuntime %r: %s", temp_runtime_name, e
                    )
            if temp_secret_name:
                try:
                    v1.delete_namespaced_secret(
                        temp_secret_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                    )
                    logger.info("Deleted temporary S3 secret %r", temp_secret_name)
                except Exception as e:
                    logger.warning(
                        "Failed to delete secret %r: %s", temp_secret_name, e
                    )
            if temp_rbac_name:
                try:
                    rbac_v1.delete_namespaced_role_binding(
                        temp_rbac_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                    )
                    rbac_v1.delete_namespaced_role(
                        temp_rbac_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                    )
                except Exception as e:
                    logger.warning("Failed to delete RBAC %r: %s", temp_rbac_name, e)
            if temp_sa_name:
                try:
                    v1.delete_namespaced_service_account(
                        temp_sa_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
                    )
                except Exception as e:
                    logger.warning("Failed to delete SA %r: %s", temp_sa_name, e)

        return result


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
        run_id, detail = run_pipeline_and_wait(
            kfp_client_automl_functional,
            compiled_timeseries_pipeline_path,
            arguments,
            timeout,
        )
        elapsed = time.monotonic() - start

        state = get_run_state(detail)
        failed_task_names = get_failed_task_names(kfp_client_automl_functional, run_id)

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
            collect_failure_details(
                kfp_client_automl_functional, run_id, config=automl_functional_config
            )
        )

        assert run_failed(detail), (
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
