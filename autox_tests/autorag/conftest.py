"""Pytest fixtures for AutoRAG functional tests (Documents RAG Optimization)."""

import json
import logging
import os

import pytest

from autox_tests.lib.clients import make_kfp_client, make_s3_client
from autox_tests.lib.env import load_tests_env

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Load env vars from ``autox_tests/.env`` before collection."""
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
    """Build functional test config from environment; None if not configured.

    Relaxed guards compared to integration config (does not require
    vector_io_provider_id or input_data_key since those are
    overridden per-scenario). Adds milvus provider IDs and constrained model lists.
    """
    load_tests_env("autorag")

    kfp_url = os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST")
    token = os.environ.get("RHOAI_TOKEN") or os.environ.get("KFP_TOKEN")
    project = os.environ.get("RHOAI_PROJECT_NAME") or os.environ.get("KFP_NAMESPACE")
    t_secret = os.environ.get("TEST_DATA_SECRET_NAME")
    t_bucket = os.environ.get("TEST_DATA_BUCKET_NAME")
    i_secret = os.environ.get("INPUT_DATA_SECRET_NAME")
    i_bucket = os.environ.get("INPUT_DATA_BUCKET_NAME")
    ogx_secret = os.environ.get("OGX_SECRET_NAME")

    if not all([kfp_url, token, t_secret, t_bucket, i_secret, i_bucket, ogx_secret]):
        return None

    endpoint = os.environ.get("ARTIFACTS_AWS_S3_ENDPOINT")
    access = os.environ.get("ARTIFACTS_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("ARTIFACTS_AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("ARTIFACTS_AWS_DEFAULT_REGION", "us-east-1")
    bucket_artifacts = os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET")

    base = {
        "rhoai_kfp_url": kfp_url.strip().rstrip("/"),
        "rhoai_token": token.strip(),
        "rhoai_project": (project or "").strip(),
        "test_data_secret_name": t_secret.strip(),
        "test_data_bucket_name": t_bucket.strip(),
        "input_data_secret_name": i_secret.strip(),
        "input_data_bucket_name": i_bucket.strip(),
        "ogx_secret_name": ogx_secret.strip(),
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }
    if not base["rhoai_project"]:
        logger.info("Missing RHOAI_PROJECT_NAME. Functional config cannot be created.")
        return None

    return base


@pytest.fixture(scope="session")
def functional_env_config():
    """Session-scoped functional test config fixture."""
    return get_functional_config()


@pytest.fixture(scope="session")
def kfp_client_functional(functional_env_config):
    """Session-scoped KFP client for functional tests."""
    return make_kfp_client(functional_env_config)


@pytest.fixture(scope="session")
def s3_client_functional(functional_env_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    return make_s3_client(functional_env_config)


@pytest.fixture(scope="session")
def compiled_pipeline_path(tmp_path_factory):
    """Resolve AutoRAG pipeline YAML: local path, URL, or GitHub default.

    Set ``AUTORAG_PIPELINE_PATH`` to a local file, an ``https://`` URL, or leave
    unset to download from the default pipelines-components GitHub repo.
    """
    from autox_tests.lib.pipeline_yaml_sources import resolve_precompiled_pipeline_yaml

    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var="AUTORAG_PIPELINE_PATH",
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml"),
            cache_file_name="documents-rag-optimization-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
