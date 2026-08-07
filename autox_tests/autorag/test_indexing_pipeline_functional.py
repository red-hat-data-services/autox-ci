"""Parametrized functional tests for Documents Indexing pipeline on RHOAI.

Test scenarios are defined in configs/indexing_test_configs.json. Data is pre-loaded
in S3; tests reference existing S3 keys without uploading local files. Filter by tags
with TESTS_TAGS (e.g. smoke, remote::milvus, negative).

Passing criteria for positive scenarios:
- Pipeline run finishes with SUCCEEDED status within timeout

Passing criteria for negative scenarios:
- Pipeline run finishes with FAILED status within capped timeout
- At least one of the expected_failing_task names appears in the run's failed tasks
"""

import logging
import time

import pytest

from autox_tests.lib.k8s_utils import add_kubeconfig_to_config

from .conftest import get_indexing_functional_config
from .configs.configs import IndexingTestConfig, get_indexing_configs_for_run
from autox_tests.lib.kfp_run_state import _get_run_state, _run_failed, _run_succeeded
from .utils import (
    _collect_failure_details,
    _run_pipeline_and_wait,
)

logger = logging.getLogger(__name__)

INDEXING_FUNCTIONAL_CONFIG = get_indexing_functional_config()
INDEXING_POSITIVE_CONFIGS = get_indexing_configs_for_run(pass_type="positive")
INDEXING_NEGATIVE_CONFIGS = get_indexing_configs_for_run(pass_type="negative")

_EXPECTED_FAIL_TIMEOUT_CAP = 600


def _get_failed_task_names(client, run_id: str) -> list[str]:
    """Return display names of tasks that reached a FAILED/ERROR state in a run."""
    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)
        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None
        if not task_list:
            return []
        from autox_tests.lib.kfp_run_state import _normalize_state
        return [
            getattr(t, "display_name", None) or getattr(t, "task_id", "?")
            for t in task_list
            if _normalize_state(getattr(t, "state", None)) in ("FAILED", "ERROR", "SYSTEM_ERROR")
        ]
    except Exception as e:
        logger.warning("Could not fetch task details for run %s: %s", run_id, e)
        return []


@pytest.mark.autorag
@pytest.mark.indexing
@pytest.mark.skipif(
    INDEXING_FUNCTIONAL_CONFIG is None,
    reason=(
        "Indexing pipeline env incomplete "
        "(RHOAI_URL or RHOAI_KFP_URL, RHOAI_TOKEN, INPUT_DATA_BUCKET_NAME, "
        "OGX_SECRET_NAME, AUTORAG_INDEXING_EMBEDDING_MODEL_ID; see .env.rag.example)"
    ),
)
class TestAutoRAGIndexingFunctional:
    """Functional tests for the Documents Indexing pipeline."""

    @pytest.mark.negative
    @pytest.mark.parametrize(
        "test_config",
        INDEXING_NEGATIVE_CONFIGS,
        ids=[c.id for c in INDEXING_NEGATIVE_CONFIGS],
    )
    def test_indexing_pipeline_functional_negative(
        self,
        test_config: IndexingTestConfig,
        indexing_functional_env_config,
        kfp_client_indexing_functional,
        indexing_pipeline_run_target,
        pipeline_run_timeout,
        s3_cleanup_tracker,
        rhoai_cluster_kubeconfig,
    ):
        """Submit pipeline with injected fault; assert FAILED within capped timeout."""
        if not kfp_client_indexing_functional:
            pytest.fail("Indexing pipeline functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(indexing_functional_env_config)
        timeout = min(pipeline_run_timeout, _EXPECTED_FAIL_TIMEOUT_CAP)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_indexing_functional,
            indexing_pipeline_run_target,
            arguments,
            timeout,
        )
        elapsed = time.monotonic() - start
        state = _get_run_state(detail)
        failed_task_names = _get_failed_task_names(kfp_client_indexing_functional, run_id)

        logger.info(
            "[%s] run_id=%s state=%s elapsed=%.1fs failed_tasks=%s expected=%s",
            test_config.id,
            run_id,
            state,
            round(elapsed, 1),
            failed_task_names,
            test_config.expected_failing_task,
        )

        failure_details = _collect_failure_details(
            kfp_client_indexing_functional,
            run_id,
            config=add_kubeconfig_to_config(
                indexing_functional_env_config, rhoai_cluster_kubeconfig
            ),
        )
        logger.info(failure_details)

        if "POD LOGS FOR FAILED PODS:" not in failure_details:
            logger.warning("Pod logs not collected for run %s — check k8s connectivity", run_id)

        assert _run_failed(detail), (
            f"[{test_config.id}] Pipeline run {run_id} expected FAILED but got {state}"
        )

        if test_config.expected_failing_task:
            matched = any(t in failed_task_names for t in test_config.expected_failing_task)
            assert matched, (
                f"[{test_config.id}] Expected one of {test_config.expected_failing_task} to fail; "
                f"actual failed tasks: {failed_task_names}"
            )

    @pytest.mark.positive
    @pytest.mark.parametrize(
        "test_config",
        INDEXING_POSITIVE_CONFIGS,
        ids=[c.id for c in INDEXING_POSITIVE_CONFIGS],
    )
    def test_indexing_pipeline_functional_positive(
        self,
        test_config: IndexingTestConfig,
        indexing_functional_env_config,
        kfp_client_indexing_functional,
        indexing_pipeline_run_target,
        pipeline_run_timeout,
        s3_cleanup_tracker,
        rhoai_cluster_kubeconfig,
    ):
        """Submit indexing pipeline run; assert SUCCEEDED within timeout.

        The indexing pipeline writes chunks into an OGX vector store rather than S3,
        so success is validated by the SUCCEEDED run state only.
        """
        if not kfp_client_indexing_functional:
            pytest.fail("Indexing pipeline functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(indexing_functional_env_config)

        start = time.monotonic()
        run_id, detail = _run_pipeline_and_wait(
            kfp_client_indexing_functional,
            indexing_pipeline_run_target,
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
            round(elapsed, 1),
        )

        if not _run_succeeded(detail):
            failure_info = _collect_failure_details(
                kfp_client_indexing_functional,
                run_id,
                config=add_kubeconfig_to_config(
                    indexing_functional_env_config, rhoai_cluster_kubeconfig
                ),
            )
            pytest.fail(
                f"[{test_config.id}] Pipeline run {run_id} expected SUCCEEDED but got "
                f"{state}{failure_info}"
            )
