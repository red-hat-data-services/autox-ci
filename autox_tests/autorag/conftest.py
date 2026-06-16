"""Pytest fixtures for AutoRAG functional tests (Documents RAG Optimization)."""

import json
import logging
import os

import pytest

from autox_tests.conftest import make_kfp_client_for_session
from autox_tests.lib.clients import make_s3_client
from autox_tests.lib.env import load_tests_env
from autox_tests.lib.managed_pipelines import (
    PipelineRunTarget,
    resolve_managed_pipeline_target,
    use_managed_pipelines_from_env,
)
from autox_tests.lib.pipeline_yaml_sources import PIPELINE_YAML_AUTORAG_ENV
from autox_tests.lib.settings import (
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

    kfp_url = (os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST") or "").strip()
    if not kfp_url and not should_create_dspa_from_env():
        logger.info("Set RHOAI_KFP_URL or enable DSPA auto-setup (default when KFP URL is unset).")
        return None

    default_secret = (
        (os.environ.get("RHOAI_TRAIN_S3_SECRET_NAME") or "").strip()
        or (os.environ.get("RHOAI_TEST_S3_SECRET_NAME") or base.get("s3_secret_name") or "").strip()
    )
    t_secret = (os.environ.get("TEST_DATA_SECRET_NAME") or default_secret).strip()
    i_secret = (os.environ.get("INPUT_DATA_SECRET_NAME") or default_secret).strip()
    t_bucket = (os.environ.get("TEST_DATA_BUCKET_NAME") or "").strip()
    i_bucket = (os.environ.get("INPUT_DATA_BUCKET_NAME") or "").strip()
    ogx_secret = (os.environ.get("OGX_SECRET_NAME") or "").strip()

    if not all([base.get("rhoai_token"), t_secret, t_bucket, i_secret, i_bucket, ogx_secret]):
        return None

    if not use_managed_pipelines_from_env():
        if not (os.environ.get(PIPELINE_YAML_AUTORAG_ENV) or "").strip():
            return None

    endpoint = (
        (os.environ.get("ARTIFACTS_AWS_S3_ENDPOINT") or "").strip()
        or base.get("s3_endpoint")
    )
    access = (
        (os.environ.get("ARTIFACTS_AWS_ACCESS_KEY_ID") or "").strip()
        or base.get("s3_access_key")
    )
    secret = (
        (os.environ.get("ARTIFACTS_AWS_SECRET_ACCESS_KEY") or "").strip()
        or base.get("s3_secret_key")
    )
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


def add_kubeconfig_to_config(
    config: dict, kubeconfig_path: str | None
) -> dict:
    """Add temp_kubeconfig_path to config for _collect_failure_details."""
    if kubeconfig_path is None:
        return config
    return {**config, "temp_kubeconfig_path": kubeconfig_path}


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
        pytest.skip("KFP client not available — skipping pipeline run target resolution")
    try:
        return resolve_managed_pipeline_target(
            kfp_client_functional,
            kind="autorag",
            path_env_var=PIPELINE_YAML_AUTORAG_ENV,
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-autorag"),
            cache_file_name="documents-rag-optimization-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, EnvironmentError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
