"""Pytest fixtures for AutoRAG functional tests (Documents RAG Optimization)."""

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from autox_tests.conftest import make_kfp_client_for_session
from autox_tests.lib.clients import make_s3_client
from autox_tests.lib.env import load_tests_env
from autox_tests.lib.managed_pipelines import (
    resolve_indexing_pipeline_target,
    resolve_managed_pipeline_target,
    use_managed_pipelines_from_env,
)
from autox_tests.lib.pipeline_yaml_sources import (
    PIPELINE_YAML_AUTORAG_ENV,
)
from autox_tests.lib.s3_data import S3CleanupTracker
from autox_tests.lib.settings import (
    get_rhoai_namespace_setup_config,
    should_create_dspa_from_env,
)

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Load env vars from ``.env.rag`` before collection."""
    load_tests_env("autorag")


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
    maas_secret = (os.environ.get("MAAS_SECRET_NAME") or "").strip()
    vector_db_secret = (os.environ.get("VECTOR_DB_SECRET_NAME") or "").strip()

    if not all(
        [
            base.get("rhoai_token"),
            t_secret,
            t_bucket,
            i_secret,
            i_bucket,
            maas_secret,
            vector_db_secret,
        ]
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
        "maas_secret_name": maas_secret,
        "vector_db_secret_name": vector_db_secret,
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
def indexing_pipeline_run_target(kfp_client_indexing_functional, tmp_path_factory):
    """Indexing pipeline: managed KFP registration or legacy ``AUTORAG_INDEXING_PIPELINE_PATH`` package."""
    if not kfp_client_indexing_functional:
        pytest.skip(
            "KFP client not available — skipping indexing pipeline run target resolution"
        )
    try:
        return resolve_indexing_pipeline_target(
            kfp_client_indexing_functional,
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-autorag-indexing"),
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
def s3_client_functional(functional_env_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    if functional_env_config is None:
        return None
    return make_s3_client(functional_env_config)


def get_indexing_functional_config():
    """Build indexing pipeline functional test config from environment; None if not configured.

    Lighter than ``get_functional_config()`` — does not require test_data_* env vars since
    the indexing pipeline only reads from input data, not a separate test dataset.
    """
    load_tests_env("autorag")
    base = get_rhoai_namespace_setup_config()
    if base is None:
        return None

    kfp_url = (
        os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST") or ""
    ).strip()
    if not kfp_url and not should_create_dspa_from_env():
        return None

    default_secret = (os.environ.get("RHOAI_TRAIN_S3_SECRET_NAME") or "").strip() or (
        os.environ.get("RHOAI_TEST_S3_SECRET_NAME") or base.get("s3_secret_name") or ""
    ).strip()
    i_secret = (os.environ.get("INPUT_DATA_SECRET_NAME") or default_secret).strip()
    i_bucket = (os.environ.get("INPUT_DATA_BUCKET_NAME") or "").strip()
    maas_secret = (os.environ.get("MAAS_SECRET_NAME") or "").strip()
    vector_db_secret = (os.environ.get("VECTOR_DB_SECRET_NAME") or "").strip()

    if not all(
        [base.get("rhoai_token"), i_secret, i_bucket, maas_secret, vector_db_secret]
    ):
        return None

    endpoint = (os.environ.get("ARTIFACTS_AWS_S3_ENDPOINT") or "").strip() or base.get("s3_endpoint")
    access = (os.environ.get("ARTIFACTS_AWS_ACCESS_KEY_ID") or "").strip() or base.get("s3_access_key")
    secret = (os.environ.get("ARTIFACTS_AWS_SECRET_ACCESS_KEY") or "").strip() or base.get("s3_secret_key")
    region = (
        (os.environ.get("ARTIFACTS_AWS_DEFAULT_REGION") or "").strip()
        or base.get("s3_region")
        or "us-east-1"
    )
    bucket_artifacts = (os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET") or "").strip()

    return {
        **base,
        "rhoai_kfp_url": kfp_url.rstrip("/") if kfp_url else None,
        "input_data_secret_name": i_secret,
        "input_data_bucket_name": i_bucket,
        "maas_secret_name": maas_secret,
        "vector_db_secret_name": vector_db_secret,
        "s3_endpoint": endpoint,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_region": region,
        "s3_bucket_artifacts": bucket_artifacts or None,
    }


@pytest.fixture(scope="session")
def indexing_functional_env_config():
    """Session-scoped indexing pipeline functional test config fixture."""
    return get_indexing_functional_config()


@pytest.fixture(scope="session")
def s3_client_indexing_functional(indexing_functional_env_config):
    """Session-scoped S3 client for indexing pipeline artifact checks (optional)."""
    if indexing_functional_env_config is None:
        return None
    return make_s3_client(indexing_functional_env_config)


@pytest.fixture(scope="session")
def kfp_client_indexing_functional(
    indexing_functional_env_config,
    datascience_pipelines_application,
    rhoai_cluster_kubeconfig,
):
    """KFP client for indexing pipeline tests."""
    if indexing_functional_env_config is None:
        return None
    try:
        return make_kfp_client_for_session(
            namespace_config=indexing_functional_env_config,
            configured_kfp_url=indexing_functional_env_config.get("rhoai_kfp_url"),
            datascience_pipelines_application=datascience_pipelines_application,
            kubeconfig_path=rhoai_cluster_kubeconfig,
        )
    except RuntimeError as e:
        pytest.fail(str(e))


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
def s3_teardown(request, s3_cleanup_tracker):
    """Session-scoped teardown: delete pipeline artifacts from S3.

    Set ``AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true`` to skip deletion and
    inspect artifacts manually after the session.
    """
    from .configs.configs import AutoRAGTestConfig

    s3_client_functional = (
        request.getfixturevalue("s3_client_functional")
        if _has_selected_config(request, AutoRAGTestConfig) else None
    )
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
def s3_teardown_indexing(request, s3_cleanup_tracker):
    """Session-scoped teardown: delete indexing pipeline artifacts from S3.

    Mirrors s3_teardown but uses the indexing-pipeline S3 client so cleanup
    works even when the optimization pipeline is not configured.
    Set ``AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true`` to skip deletion.
    """
    from .configs.configs import IndexingTestConfig

    s3_client_indexing_functional = (
        request.getfixturevalue("s3_client_indexing_functional")
        if _has_selected_config(request, IndexingTestConfig) else None
    )
    yield
    if s3_client_indexing_functional is None:
        return

    keep_artifacts = os.environ.get(
        "AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS", ""
    ).strip().lower() in ("1", "true", "yes")
    if keep_artifacts:
        return

    from autox_tests.lib.s3_data import delete_s3_objects, list_s3_objects

    for bucket, prefixes in s3_cleanup_tracker.artifact_prefixes.items():
        for prefix in prefixes:
            objects = list_s3_objects(s3_client_indexing_functional, bucket, prefix)
            if objects:
                keys = [o["Key"] for o in objects]
                count = delete_s3_objects(s3_client_indexing_functional, bucket, keys)
                logger.info(
                    "Deleted %d indexing artifact objects from s3://%s/%s",
                    count,
                    bucket,
                    prefix,
                )


def _has_selected_config(request, config_type):
    """Keep functional clients out of TLS-only and local configuration runs."""
    return any(
        isinstance(value, config_type)
        for item in request.session.items
        for value in getattr(getattr(item, "callspec", None), "params", {}).values()
    )


@pytest.fixture(scope="session", autouse=True)
def upload_datasets(request: pytest.FixtureRequest):
    """Upload selected optimization/indexing datasets and clean up after the session.

    Documents go to INPUT_DATA_BUCKET_NAME; optimization benchmarks go to
    TEST_DATA_BUCKET_NAME. Shared keys are uploaded once per bucket. Local config
    tests require no datasets. Cleanup also runs when setup or a test fails.
    """
    from .configs.configs import AutoRAGTestConfig, IndexingTestConfig
    from .utils import upload_test_datasets

    configs = []
    for item in request.session.items:
        callspec = getattr(item, "callspec", None)
        if callspec:
            configs.extend(callspec.params.values())

    datasets: dict[str, tuple[Any, set[str]]] = {}
    for config_type, env_fixture, client_fixture in (
        (AutoRAGTestConfig, "functional_env_config", "s3_client_functional"),
        (IndexingTestConfig, "indexing_functional_env_config", "s3_client_indexing_functional"),
    ):
        selected = [config for config in configs if isinstance(config, config_type)]
        if not selected:
            continue
        env_config = request.getfixturevalue(env_fixture)
        if env_config is None:
            # Pipeline tests skip when their environment is incomplete.
            continue
        client = request.getfixturevalue(client_fixture)
        if client is None:
            pytest.skip("S3 client is required to upload AutoRAG datasets — set AWS_* env vars")

        bucket_keys = [
            (
                env_config["input_data_bucket_name"],
                {key for config in selected for key in (config.input_data_keys or [])},
            )
        ]
        if config_type is AutoRAGTestConfig:
            bucket_keys.append(
                (
                    env_config["test_data_bucket_name"],
                    {config.test_data_key for config in selected if config.test_data_key},
                )
            )
        for bucket, keys in bucket_keys:
            if keys:
                datasets.setdefault(bucket, (client, set()))[1].update(keys)

    for bucket, (client, keys) in datasets.items():
        uploaded_keys: list[str] = []

        def cleanup(client=client, bucket=bucket, uploaded_keys=uploaded_keys):
            from autox_tests.lib.s3_data import delete_s3_objects

            deleted = delete_s3_objects(client, bucket, uploaded_keys)
            if deleted < len(uploaded_keys):
                logger.warning(
                    "Dataset teardown: only %d of %d object(s) deleted from s3://%s",
                    deleted, len(uploaded_keys), bucket,
                )
            logger.info(
                "Dataset teardown: %d file(s) removed from s3://%s", deleted, bucket
            )

        request.addfinalizer(cleanup)
        upload_test_datasets(
            client, bucket, list(keys), Path(__file__).parent / "data",
            uploaded_keys=uploaded_keys,
        )


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
