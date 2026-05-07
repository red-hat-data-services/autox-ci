"""Pytest fixtures for AutoML functional tests (tabular + timeseries)."""

import logging
import os
from pathlib import Path

import pytest

from autox_tests.lib.env import load_tests_env
from autox_tests.lib.pipeline_yaml_sources import resolve_precompiled_pipeline_yaml

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"


def pytest_configure(config: pytest.Config) -> None:
    load_tests_env()


def get_automl_functional_config():
    """Build AutoML functional test config from environment; None if not configured."""
    load_tests_env()

    kfp_url = os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST")
    token = os.environ.get("RHOAI_TOKEN") or os.environ.get("KFP_TOKEN")
    project = os.environ.get("RHOAI_PROJECT_NAME") or os.environ.get("KFP_NAMESPACE")
    train_secret = os.environ.get("AUTOML_TRAIN_DATA_SECRET_NAME") or os.environ.get("RHOAI_TEST_S3_SECRET_NAME")
    train_bucket = os.environ.get("AUTOML_TRAIN_DATA_BUCKET_NAME") or os.environ.get("RHOAI_TEST_DATA_BUCKET")

    if not all([kfp_url, token, train_secret, train_bucket]):
        return None

    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    bucket_artifacts = os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET")

    base = {
        "rhoai_kfp_url": kfp_url.strip().rstrip("/"),
        "rhoai_token": token.strip(),
        "rhoai_project": (project or "").strip(),
        "train_data_secret_name": train_secret.strip(),
        "train_data_bucket_name": train_bucket.strip(),
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
        "deploy_after_training": os.environ.get("RHOAI_DEPLOY_AFTER_TRAINING", "false").strip().lower() in ("1", "true", "yes"),
        "serving_image": (os.environ.get("RHOAI_SERVING_IMAGE") or "").strip() or None,
        "serving_runtime_name": os.environ.get("RHOAI_SERVING_RUNTIME_NAME", "kserve-autogluonserver").strip(),
        "create_serving_runtime": os.environ.get("RHOAI_CREATE_SERVING_RUNTIME", "false").strip().lower() in ("1", "true", "yes"),
        "inference_timeout": int(os.environ.get("RHOAI_INFERENCE_TIMEOUT", "600")),
        "notebook_runner_image": (os.environ.get("RHOAI_NOTEBOOK_RUNNER_IMAGE") or "").strip() or None,
    }

    if not base["rhoai_project"]:
        logger.info("Missing RHOAI_PROJECT_NAME. AutoML functional config cannot be created.")
        return None

    return base


AUTOML_FUNCTIONAL_CONFIG = get_automl_functional_config()


@pytest.fixture(scope="session")
def automl_functional_config():
    return AUTOML_FUNCTIONAL_CONFIG


@pytest.fixture(scope="session")
def kfp_client_automl_functional(automl_functional_config):
    if automl_functional_config is None:
        return None
    from .utils import make_kfp_client
    return make_kfp_client(automl_functional_config)


@pytest.fixture(scope="session")
def s3_client_automl_functional(automl_functional_config):
    if automl_functional_config is None or not automl_functional_config.get("s3_endpoint"):
        return None
    from .utils import make_s3_client
    return make_s3_client(automl_functional_config)


@pytest.fixture(scope="session")
def compiled_tabular_pipeline_path(tmp_path_factory):
    """Resolve tabular AutoML pipeline YAML: local path or URL."""
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var="AUTOML_TABULAR_PIPELINE_PATH",
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-tabular"),
            cache_file_name="autogluon-tabular-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def compiled_timeseries_pipeline_path(tmp_path_factory):
    """Resolve timeseries AutoML pipeline YAML: local path or URL."""
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var="AUTOML_TIMESERIES_PIPELINE_PATH",
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-timeseries"),
            cache_file_name="autogluon-timeseries-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def uploaded_tabular_datasets(automl_functional_config, s3_client_automl_functional):
    """Upload tabular CSV datasets to S3 for functional tests."""
    if automl_functional_config is None or s3_client_automl_functional is None:
        return {}
    from .configs.configs import get_tabular_configs_for_run
    return _upload_datasets(
        s3_client_automl_functional,
        automl_functional_config["train_data_bucket_name"],
        get_tabular_configs_for_run(),
    )


@pytest.fixture(scope="session")
def uploaded_timeseries_datasets(automl_functional_config, s3_client_automl_functional):
    """Upload timeseries CSV datasets to S3 for functional tests."""
    if automl_functional_config is None or s3_client_automl_functional is None:
        return {}
    from .configs.configs import get_timeseries_configs_for_run
    return _upload_datasets(
        s3_client_automl_functional,
        automl_functional_config["train_data_bucket_name"],
        get_timeseries_configs_for_run(),
    )


def _upload_datasets(s3_client, bucket, configs):
    result = {}
    seen = set()
    for config in configs:
        rel_path = config.dataset_path
        s3_key = config.train_data_file_key
        if rel_path in seen:
            continue
        seen.add(rel_path)
        full_path = _DATA_DIR.parent / rel_path
        if not full_path.is_file():
            logger.warning("Dataset file not found: %s (skipping upload)", full_path)
            continue
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=full_path.read_bytes(),
                ContentType="text/csv",
            )
            result[rel_path] = {"bucket": bucket, "key": s3_key}
        except Exception as e:
            logger.error("Failed to upload %s to s3://%s/%s: %s", rel_path, bucket, s3_key, e)
    return result


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
