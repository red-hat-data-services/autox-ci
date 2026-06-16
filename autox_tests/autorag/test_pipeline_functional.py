"""Parametrized functional tests for Documents RAG Optimization pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and pipeline
parameters. When not set, tests are skipped. See .env.example for required variables.

Test scenarios are defined in test_configs.json and loaded via test_configs.py. Each
scenario specifies pipeline parameter overrides and an expected result (pass or fail).
Filter scenarios by tags with RHOAI_TEST_CONFIG_TAGS (e.g. smoke, milvus-lite).

Passing criteria for expected-pass tests (from RHAIENG-4142):
- Pipeline run finishes with status success
- At least 1 pattern is generated
- All desired artifacts exist (indexing notebook, inference notebook, evaluation_results.json, pattern.json)
- Notebooks can be run and complete successfully (via papermill)
"""

import logging
import random

import pytest

from .conftest import (
    add_kubeconfig_to_config,
    get_functional_config,
)
from autox_tests.autorag.configs.configs import (
    AutoRAGTestConfig,
    get_test_configs_for_run,
)

from .utils import (
    _collect_failure_details,
    _download_and_execute_notebooks,
    _get_run_state,
    _run_failed,
    _run_pipeline_and_wait,
    _run_succeeded,
    _validate_artifacts_in_s3,
)

logger = logging.getLogger(__name__)


# Module-level constants for skipif and parametrize
DOCRAG_FUNCTIONAL_CONFIG = get_functional_config()
POSITIVE_CONFIGS_FOR_RUN = get_test_configs_for_run(pass_type="positive")
NEGATIVE_CONFIGS_FOR_RUN = get_test_configs_for_run(pass_type="negative")

# Shorter timeout for expected-fail tests (failures should surface quickly)
_EXPECTED_FAIL_TIMEOUT_CAP = 600


@pytest.mark.autorag
@pytest.mark.skipif(
    DOCRAG_FUNCTIONAL_CONFIG is None,
    reason="RHOAI functional test env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, pipeline params; see .env.example)",
)
class TestAutoRAGFunctional:
    """Functional tests for the Documents RAG Optimization pipeline."""

    @pytest.mark.negative
    @pytest.mark.parametrize(
        "test_scenario_config", NEGATIVE_CONFIGS_FOR_RUN, ids=[c.id for c in NEGATIVE_CONFIGS_FOR_RUN]
    )
    def test_autorag_pipelines_functional_negative(
        self,
        test_scenario_config: AutoRAGTestConfig,
        functional_env_config,
        kfp_client_functional,
        autorag_pipeline_run_target,
        pipeline_run_timeout,
        s3_client_functional,
        rhoai_cluster_kubeconfig,
    ):
        """Verify pipeline fails as expected for negative test scenarios."""
        if not kfp_client_functional:
            pytest.fail("Functional test prerequisites not available")

        arguments = test_scenario_config.get_pipeline_arguments(functional_env_config)

        timeout = min(pipeline_run_timeout, _EXPECTED_FAIL_TIMEOUT_CAP)

        run_id, detail = _run_pipeline_and_wait(
            kfp_client_functional,
            autorag_pipeline_run_target,
            arguments,
            timeout,
        )

        state = _get_run_state(detail)

        assert _run_failed(detail), (
            f"[{test_scenario_config.id}] Pipeline run {run_id} expected state FAILED but got {state}"
        )

        # Log failure details for observability even on expected failures
        logger.info(
            _collect_failure_details(
                kfp_client_functional,
                run_id,
                config=add_kubeconfig_to_config(
                    functional_env_config, rhoai_cluster_kubeconfig
                ),
            )
        )

    @pytest.mark.positive
    @pytest.mark.parametrize(
        "test_scenario_config", POSITIVE_CONFIGS_FOR_RUN, ids=[c.id for c in POSITIVE_CONFIGS_FOR_RUN]
    )
    def test_autorag_pipeline_functional_positive(
        self,
        test_scenario_config: AutoRAGTestConfig,
        functional_env_config,
        kfp_client_functional,
        autorag_pipeline_run_target,
        pipeline_run_timeout,
        s3_client_functional,
        rhoai_cluster_kubeconfig,
    ):
        """Run pipeline for one test config; validate based on expected result.

        For expected-pass scenarios: assert success, validate artifacts, execute notebooks.
        For expected-fail scenarios: assert the pipeline run fails (not succeeds).
        """
        if not kfp_client_functional:
            pytest.fail("Functional test prerequisites not available")

        arguments = test_scenario_config.get_pipeline_arguments(functional_env_config)

        timeout = pipeline_run_timeout

        run_id, detail = _run_pipeline_and_wait(
            kfp_client_functional,
            autorag_pipeline_run_target,
            arguments,
            timeout,
        )

        if not _run_succeeded(detail):
            failure_info = _collect_failure_details(
                kfp_client_functional,
                run_id,
                config=add_kubeconfig_to_config(
                    functional_env_config, rhoai_cluster_kubeconfig
                ),
            )
            pytest.fail(
                f"[{test_scenario_config.id}] Pipeline run {run_id} was expected to PASS but failed; "
                f"state={_get_run_state(detail)}"
                f"{failure_info}"
            )

        # Artifact validation (requires S3 config)
        if not s3_client_functional or not functional_env_config.get("s3_bucket_artifacts"):
            return

        artifact_bucket = functional_env_config["s3_bucket_artifacts"]
        prefix = f"{autorag_pipeline_run_target.artifact_prefix}/{run_id}"
        artifacts = _validate_artifacts_in_s3(s3_client_functional, artifact_bucket, prefix)

        assert len(artifacts["pattern_keys"]) >= 1, (
            f"[{test_scenario_config.id}] Expected at least 1 pattern artifact under {prefix}; "
            f"found {artifacts['pattern_keys']}"
        )
        assert len(artifacts["indexing_notebook_keys"]) >= 1, (
            f"[{test_scenario_config.id}] Expected at least 1 indexing notebook under {prefix}; "
            f"found {artifacts['indexing_notebook_keys']}"
        )
        assert len(artifacts["inference_notebook_keys"]) >= 1, (
            f"[{test_scenario_config.id}] Expected at least 1 inference notebook under {prefix}; "
            f"found {artifacts['inference_notebook_keys']}"
        )
        assert len(artifacts["evaluation_results_keys"]) >= 1, (
            f"[{test_scenario_config.id}] Expected evaluation_results.json under {prefix}; "
            f"found {artifacts['evaluation_results_keys']}"
        )

        # Notebook execution validation
        random_indexing_notebook = random.choice(artifacts["indexing_notebook_keys"])
        random_inference_notebook = random.choice(artifacts["inference_notebook_keys"])
        all_notebook_keys = [random_indexing_notebook, random_inference_notebook]
        _download_and_execute_notebooks(s3_client_functional, artifact_bucket, all_notebook_keys)
