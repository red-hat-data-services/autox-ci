import json
import logging
import os

import pytest

from .utils import (
    make_kfp_client,
    make_s3_client,
)

logger = logging.getLogger(__name__)


def _parse_json_list(env_name):
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
    llama_stack_vector_io_provider_id or input_data_key since those are
    overridden per-scenario). Adds milvus provider IDs and constrained model lists.
    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(".env"))

    kfp_url = os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST")
    token = os.environ.get("RHOAI_TOKEN") or os.environ.get("KFP_TOKEN")
    project = os.environ.get("RHOAI_PROJECT_NAME") or os.environ.get("KFP_NAMESPACE")
    t_secret = os.environ.get("TEST_DATA_SECRET_NAME")
    t_bucket = os.environ.get("TEST_DATA_BUCKET_NAME")
    i_secret = os.environ.get("INPUT_DATA_SECRET_NAME")
    i_bucket = os.environ.get("INPUT_DATA_BUCKET_NAME")
    llama_secret = os.environ.get("LLAMA_STACK_SECRET_NAME")

    if not all([kfp_url, token, t_secret, t_bucket, i_secret, i_bucket, llama_secret]):
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
        "llama_stack_secret_name": llama_secret.strip(),
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }
    if base is None:
        logger.info("Missing required environmental variables. Functional config cannot be created.")
        return None

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
def compiled_pipeline_path():
    """Compile the Documents RAG Optimization pipeline to a temp YAML file."""
    pipeline_path = os.getenv("AUTORAG_PIPELINE_PATH")
    if pipeline_path is None:
        raise EnvironmentError("AUTORAG_PIPELINE_PATH environment variable not set.")

    return pipeline_path


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
