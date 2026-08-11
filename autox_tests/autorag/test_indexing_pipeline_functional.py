"""Parametrized functional tests for Documents Indexing pipeline on RHOAI.

Test scenarios are defined in configs/indexing_test_configs.json. Data is pre-loaded
in S3; tests reference existing S3 keys without uploading local files. Filter by tags
with TESTS_TAGS (e.g. smoke, remote::milvus, negative).

Passing criteria for positive scenarios:
- Pipeline run finishes with SUCCEEDED status within timeout
- indexing_report.json artifact is present in S3 (when RHOAI_TEST_ARTIFACTS_BUCKET is set)
- All discovered documents were indexed without failures
- At least one chunk was produced
- Report settings match the submitted pipeline arguments

Passing criteria for negative scenarios:
- Pipeline run finishes with FAILED status within capped timeout
- At least one of the expected_failing_task names appears in the run's failed tasks
"""

import logging
import os
import time

import pytest

from autox_tests.lib.k8s_utils import add_kubeconfig_to_config
from autox_tests.lib.s3_data import list_s3_objects, read_s3_json

from .conftest import get_indexing_functional_config
from .configs.configs import IndexingTestConfig, get_indexing_configs_for_run
from autox_tests.lib.kfp_run_state import _get_run_state, _run_failed, _run_succeeded
from .utils import (
    _collect_failure_details,
    _get_failed_task_names,
    _run_pipeline_and_wait,
)

logger = logging.getLogger(__name__)

INDEXING_FUNCTIONAL_CONFIG = get_indexing_functional_config()
INDEXING_POSITIVE_CONFIGS = get_indexing_configs_for_run(pass_type="positive")
INDEXING_NEGATIVE_CONFIGS = get_indexing_configs_for_run(pass_type="negative")

_EXPECTED_FAIL_TIMEOUT_CAP = 600


def _fetch_indexing_report(
    s3_client,
    env_config: dict,
    pipeline_run_target,
    run_id: str,
    test_id: str,
) -> dict | None:
    """Download and parse indexing_report.json from S3; return None and warn if unavailable."""
    bucket = (env_config or {}).get("s3_bucket_artifacts")
    if not s3_client or not bucket:
        logger.warning(
            "[%s] S3 client or artifact bucket not configured — skipping artifact assertions",
            test_id,
        )
        return None

    prefix = f"{pipeline_run_target.artifact_prefix}/{run_id}"
    try:
        objects = list_s3_objects(s3_client, bucket, prefix)
    except Exception as exc:
        pytest.fail(
            f"[{test_id}] Failed to list S3 artifacts under s3://{bucket}/{prefix}: {exc}"
        )

    report_key = next(
        (obj["Key"] for obj in objects if "indexing_report" in obj["Key"].lower()),
        None,
    )
    if report_key is None:
        pytest.fail(
            f"[{test_id}] indexing_report artifact not found in s3://{bucket}/{prefix}; "
            f"found keys: {[o['Key'] for o in objects]}"
        )

    report = read_s3_json(s3_client, bucket, report_key)
    if report is None:
        pytest.fail(f"[{test_id}] Failed to download/parse s3://{bucket}/{report_key}")
    return report


def _assert_indexing_report(report: dict, test_config: "IndexingTestConfig") -> None:
    """Assert correctness of the indexing_report.json produced by the pipeline."""
    tid = test_config.id
    vsb = report.get("settings", {}).get("vector_store_binding", {})
    chk = report.get("settings", {}).get("chunking", {})
    emb = report.get("settings", {}).get("embedding", {})

    logger.info(
        "[%s] indexing_report: total_documents=%s completed=%s failed=%s total_chunks=%s "
        "vector_store_id=%r provider_id=%r embedding_model=%r "
        "chunk_size=%s chunk_overlap=%s",
        tid,
        report.get("total_documents"),
        report.get("completed"),
        report.get("failed"),
        report.get("total_chunks"),
        vsb.get("vector_store_id"),
        vsb.get("provider_id"),
        emb.get("model_id"),
        chk.get("chunk_size"),
        chk.get("chunk_overlap"),
    )

    assert report.get("total_documents", 0) > 0, (
        f"[{tid}] total_documents is 0 — documents_discovery or text_extraction produced no output"
    )
    failed = report.get("failed")
    assert failed is not None, (
        f"[{tid}] indexing_report missing 'failed' field — report may be malformed"
    )
    assert failed == 0, (
        f"[{tid}] {failed} document(s) failed indexing: "
        + str([e for e in report.get("documents", []) if e.get("status") == "failed"])
    )
    assert report.get("total_chunks", 0) > 0, (
        f"[{tid}] total_chunks is 0 — chunker produced no output for any document"
    )

    assert vsb.get("vector_store_id"), (
        f"[{tid}] vector_store_id is empty — OGX collection was not created"
    )
    assert vsb.get("provider_id") == test_config.vector_io_provider_id, (
        f"[{tid}] provider_id mismatch: got {vsb.get('provider_id')!r}, "
        f"expected {test_config.vector_io_provider_id!r}"
    )

    expected_model_id = test_config.embedding_model_id
    if expected_model_id == "env":
        expected_model_id = os.getenv("AUTORAG_INDEXING_EMBEDDING_MODEL_ID")
        if expected_model_id is None:
            pytest.skip(
                f"[{tid}] AUTORAG_INDEXING_EMBEDDING_MODEL_ID not set — cannot assert embedding model"
            )
    assert emb.get("model_id") == expected_model_id, (
        f"[{tid}] embedding model_id mismatch: got {emb.get('model_id')!r}, "
        f"expected {expected_model_id!r}"
    )

    if test_config.chunk_size is not None:
        assert chk.get("chunk_size") == test_config.chunk_size, (
            f"[{tid}] chunk_size mismatch: got {chk.get('chunk_size')!r}, "
            f"expected {test_config.chunk_size!r}"
        )

    if test_config.chunk_overlap is not None:
        assert chk.get("chunk_overlap") == test_config.chunk_overlap, (
            f"[{tid}] chunk_overlap mismatch: got {chk.get('chunk_overlap')!r}, "
            f"expected {test_config.chunk_overlap!r}"
        )


@pytest.mark.autorag
@pytest.mark.indexing
@pytest.mark.skipif(
    INDEXING_FUNCTIONAL_CONFIG is None,
    reason=(
        "Indexing pipeline env incomplete "
        "(RHOAI_URL or RHOAI_KFP_URL, RHOAI_TOKEN, INPUT_DATA_BUCKET_NAME, "
        "OGX_SECRET_NAME; see .env.rag.example)"
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
        bucket = (indexing_functional_env_config or {}).get("s3_bucket_artifacts")
        if bucket:
            s3_cleanup_tracker.track_artifact_prefix(
                bucket, f"{indexing_pipeline_run_target.artifact_prefix}/{run_id}"
            )
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
        s3_client_indexing_functional,
        rhoai_cluster_kubeconfig,
    ):
        """Submit indexing pipeline run; assert SUCCEEDED and validate indexing_report artifact.

        Positive passing criteria:
        - Pipeline run finishes with SUCCEEDED status within timeout
        - indexing_report.json artifact is present in S3
        - All discovered documents were indexed (failed == 0)
        - At least one chunk was produced (total_chunks > 0)
        - Settings in the report match the pipeline arguments (provider, embedding model,
          chunking params when explicitly set)
        - A vector store collection ID was assigned by OGX
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
        bucket = (indexing_functional_env_config or {}).get("s3_bucket_artifacts")
        if bucket:
            s3_cleanup_tracker.track_artifact_prefix(
                bucket, f"{indexing_pipeline_run_target.artifact_prefix}/{run_id}"
            )
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

        report = _fetch_indexing_report(
            s3_client_indexing_functional,
            indexing_functional_env_config,
            indexing_pipeline_run_target,
            run_id,
            test_config.id,
        )
        if report is not None:
            _assert_indexing_report(report, test_config)
