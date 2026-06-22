"""Pytest fixtures for AutoRAG functional tests (Documents RAG Optimization)."""

import json
import logging
import os
from pathlib import Path

import pytest

from autox_tests.conftest import make_kfp_client_for_session
from autox_tests.lib.clients import make_s3_client
from autox_tests.lib.env import load_tests_env
from autox_tests.lib.managed_pipelines import (
    resolve_managed_pipeline_target,
    use_managed_pipelines_from_env,
)
from autox_tests.lib.pipeline_yaml_sources import PIPELINE_YAML_AUTORAG_ENV
from autox_tests.lib.s3_data import S3CleanupTracker
from autox_tests.lib.settings import (
    AUTORAG_UPLOAD_TEST_DATASETS_ENV,
    get_rhoai_namespace_setup_config,
    should_create_dspa_from_env,
)

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Load env vars from ``.env.rag`` before collection."""
    load_tests_env("autorag")


def _parse_json_list(env_name):
    """Parse an env var as a JSON array; return None if unset, raise on invalid JSON."""
    raw = os.environ.get(env_name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{env_name} must be a JSON array, got {type(value).__name__}")
    return value


def get_functional_config():
    """Build functional test config from environment; None if not configured."""
    load_tests_env("autorag")
    base = get_rhoai_namespace_setup_config()
    if base is None:
        return None

    kfp_url = (
        os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST") or ""
    ).strip()
    if not kfp_url and not should_create_dspa_from_env():
        logger.info(
            "Set RHOAI_KFP_URL or enable DSPA auto-setup (default when KFP URL is unset)."
        )
        return None

    default_secret = (os.environ.get("RHOAI_TRAIN_S3_SECRET_NAME") or "").strip() or (
        os.environ.get("RHOAI_TEST_S3_SECRET_NAME") or base.get("s3_secret_name") or ""
    ).strip()
    t_secret = (os.environ.get("TEST_DATA_SECRET_NAME") or default_secret).strip()
    i_secret = (os.environ.get("INPUT_DATA_SECRET_NAME") or default_secret).strip()
    t_bucket = (os.environ.get("TEST_DATA_BUCKET_NAME") or "").strip()
    i_bucket = (os.environ.get("INPUT_DATA_BUCKET_NAME") or "").strip()
    ogx_secret = (os.environ.get("OGX_SECRET_NAME") or "").strip()

    if not all(
        [base.get("rhoai_token"), t_secret, t_bucket, i_secret, i_bucket, ogx_secret]
    ):
        return None

    if not use_managed_pipelines_from_env():
        if not (os.environ.get(PIPELINE_YAML_AUTORAG_ENV) or "").strip():
            return None

    endpoint = (os.environ.get("ARTIFACTS_AWS_S3_ENDPOINT") or "").strip() or base.get(
        "s3_endpoint"
    )
    access = (os.environ.get("ARTIFACTS_AWS_ACCESS_KEY_ID") or "").strip() or base.get(
        "s3_access_key"
    )
    secret = (
        os.environ.get("ARTIFACTS_AWS_SECRET_ACCESS_KEY") or ""
    ).strip() or base.get("s3_secret_key")
    region = (
        (os.environ.get("ARTIFACTS_AWS_DEFAULT_REGION") or "").strip()
        or base.get("s3_region")
        or "us-east-1"
    )
    bucket_artifacts = (os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET") or "").strip()

    return {
        **base,
        "rhoai_kfp_url": kfp_url.rstrip("/") if kfp_url else None,
        "test_data_secret_name": t_secret,
        "test_data_bucket_name": t_bucket,
        "input_data_secret_name": i_secret,
        "input_data_bucket_name": i_bucket,
        "ogx_secret_name": ogx_secret,
        "s3_endpoint": endpoint,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_region": region,
        "s3_bucket_artifacts": bucket_artifacts or None,
    }


@pytest.fixture(scope="session")
def functional_env_config():
    """Session-scoped functional test config fixture."""
    return get_functional_config()


@pytest.fixture(scope="session")
def kfp_client_functional(
    functional_env_config,
    datascience_pipelines_application,
    rhoai_cluster_kubeconfig,
):
    """KFP client: ``RHOAI_KFP_URL`` or route from auto-created / existing DSPA."""
    if functional_env_config is None:
        return None
    try:
        return make_kfp_client_for_session(
            namespace_config=functional_env_config,
            configured_kfp_url=functional_env_config.get("rhoai_kfp_url"),
            datascience_pipelines_application=datascience_pipelines_application,
            kubeconfig_path=rhoai_cluster_kubeconfig,
        )
    except RuntimeError as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def s3_client_functional(functional_env_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    if functional_env_config is None:
        return None
    return make_s3_client(functional_env_config)


@pytest.fixture(scope="session")
def autorag_pipeline_run_target(kfp_client_functional, tmp_path_factory):
    """AutoRAG pipeline: managed KFP registration or legacy ``AUTORAG_PIPELINE_PATH`` package."""
    if not kfp_client_functional:
        pytest.skip(
            "KFP client not available — skipping pipeline run target resolution"
        )
    try:
        return resolve_managed_pipeline_target(
            kfp_client_functional,
            kind="autorag",
            path_env_var=PIPELINE_YAML_AUTORAG_ENV,
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-autorag"),
            cache_file_name="documents-rag-optimization-pipeline.yaml",
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TimeoutError,
        EnvironmentError,
    ) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def s3_cleanup_tracker():
    """Session-scoped S3 cleanup tracker shared across all AutoRAG scenarios."""
    return S3CleanupTracker()


@pytest.fixture(scope="session", autouse=True)
def s3_teardown(s3_client_functional, s3_cleanup_tracker):
    """Session-scoped teardown: delete pipeline artifacts from S3.

    Set ``AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true`` to skip deletion and
    inspect artifacts manually after the session.
    """
    yield
    if s3_client_functional is None:
        return

    keep_artifacts = os.environ.get(
        "AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS", ""
    ).strip().lower() in ("1", "true", "yes")
    if keep_artifacts:
        logger.info(
            "AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS is set — skipping artifact cleanup"
        )
        return

    from autox_tests.lib.s3_data import delete_s3_objects, list_s3_objects

    logger.info("Starting S3 artifact cleanup...")
    for bucket, prefixes in s3_cleanup_tracker.artifact_prefixes.items():
        for prefix in prefixes:
            objects = list_s3_objects(s3_client_functional, bucket, prefix)
            if objects:
                keys = [o["Key"] for o in objects]
                count = delete_s3_objects(s3_client_functional, bucket, keys)
                logger.info(
                    "Deleted %d artifact objects from s3://%s/%s", count, bucket, prefix
                )
    logger.info("S3 artifact cleanup complete.")


@pytest.fixture(scope="session", autouse=True)
def upload_datasets_if_requested(functional_env_config, s3_client_functional):
    """Upload test datasets to S3 at session start when ``AUTORAG_UPLOAD_TEST_DATASETS`` is set.

    When set to ``1``, ``true``, or ``yes``, input and test datasets referenced in
    test_configs.json are uploaded from the local ``data/`` directory to S3 before any
    tests run. When unset, datasets are assumed to already be present in S3.

    Input data keys are uploaded to ``INPUT_DATA_BUCKET_NAME``; test data keys to
    ``TEST_DATA_BUCKET_NAME``. All uploaded objects are deleted from S3 after the session.
    """
    uploaded: list[tuple[str, str]] = []  # (bucket, s3_key)

    raw = os.environ.get(AUTORAG_UPLOAD_TEST_DATASETS_ENV, "").strip().lower()
    if raw in ("1", "true", "yes"):
        if functional_env_config is None or s3_client_functional is None:
            pytest.skip(
                f"{AUTORAG_UPLOAD_TEST_DATASETS_ENV} is set but S3 client is not configured — "
                "set AWS_* and INPUT_DATA_BUCKET_NAME / TEST_DATA_BUCKET_NAME env vars"
            )
        else:
            from .configs.configs import get_all_dataset_keys
            from .utils import upload_test_datasets

            local_data_dir = Path(__file__).parent / "data"
            input_bucket = functional_env_config["input_data_bucket_name"]
            test_bucket = functional_env_config["test_data_bucket_name"]
            input_keys, test_keys = get_all_dataset_keys()

            for key in upload_test_datasets(
                s3_client_functional,
                input_bucket,
                input_keys,
                local_data_dir,
            ):
                uploaded.append((input_bucket, key))

            for key in upload_test_datasets(
                s3_client_functional,
                test_bucket,
                test_keys,
                local_data_dir,
            ):
                uploaded.append((test_bucket, key))

    yield

    if uploaded:
        from autox_tests.lib.s3_data import delete_s3_objects
        from collections import defaultdict

        by_bucket: dict[str, list[str]] = defaultdict(list)
        for bucket, key in uploaded:
            by_bucket[bucket].append(key)

        total_deleted = 0
        total_keys = len(uploaded)
        for bucket, keys in by_bucket.items():
            deleted = delete_s3_objects(s3_client_functional, bucket, keys)
            total_deleted += deleted

        if total_deleted < total_keys:
            logger.warning(
                "Dataset teardown: only %d of %d object(s) deleted — buckets may be dirty",
                total_deleted,
                total_keys,
            )
        else:
            logger.info(
                "Dataset teardown complete: %d file(s) removed from S3",
                total_deleted,
            )


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
