"""Pytest fixtures for AutoML functional tests (tabular + timeseries)."""

import logging
import os
import tempfile
from pathlib import Path

import pytest

from autox_tests.lib.env import load_tests_env
from autox_tests.lib.pipeline_yaml_sources import (
    PIPELINE_YAML_TABULAR_ENV,
    PIPELINE_YAML_TIMESERIES_ENV,
    resolve_precompiled_pipeline_yaml,
)

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Load env vars from ``.env.ml`` before collection."""
    load_tests_env("automl")


def get_automl_functional_config():
    """Build AutoML functional test config from environment; None if not configured."""
    load_tests_env("automl")

    rhoai_url = os.environ.get("RHOAI_URL")
    kfp_url = os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST")
    token = os.environ.get("RHOAI_TOKEN") or os.environ.get("KFP_TOKEN")
    project = os.environ.get("RHOAI_PROJECT_NAME") or os.environ.get("KFP_NAMESPACE")
    train_secret = os.environ.get("RHOAI_TEST_S3_SECRET_NAME")
    train_bucket = os.environ.get("AUTOML_TRAIN_DATA_BUCKET_NAME") or os.environ.get(
        "RHOAI_TEST_DATA_BUCKET"
    )

    if not all([kfp_url, token, train_secret, train_bucket]):
        return None

    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    bucket_artifacts = os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET")

    base = {
        "rhoai_url": rhoai_url.strip().rstrip("/") if rhoai_url else None,
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
    }

    if not base["rhoai_project"]:
        logger.info(
            "Missing RHOAI_PROJECT_NAME. AutoML functional config cannot be created."
        )
        return None

    return base


@pytest.fixture(scope="session")
def automl_functional_config():
    """Session-scoped AutoML functional test config dict (None if env is incomplete)."""
    return get_automl_functional_config()


@pytest.fixture(scope="session")
def kfp_client_automl_functional(automl_functional_config):
    """Session-scoped KFP client for AutoML functional tests."""
    if automl_functional_config is None:
        return None
    from autox_tests.lib.clients import make_kfp_client

    return make_kfp_client(automl_functional_config)


@pytest.fixture(scope="session")
def s3_client_automl_functional(automl_functional_config):
    """Session-scoped S3 client for AutoML functional tests (None if S3 not configured)."""
    if automl_functional_config is None or not automl_functional_config.get(
        "s3_endpoint"
    ):
        return None
    from autox_tests.lib.clients import make_s3_client

    return make_s3_client(automl_functional_config)


@pytest.fixture(scope="session")
def compiled_tabular_pipeline_path(tmp_path_factory):
    """Resolve tabular AutoML pipeline YAML: local path, URL, or GitHub default.

    Set ``AUTOML_TABULAR_PIPELINE_PATH`` to a local file or ``https://`` URL.
    """
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var=PIPELINE_YAML_TABULAR_ENV,
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-tabular"),
            cache_file_name="autogluon-tabular-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def compiled_timeseries_pipeline_path(tmp_path_factory):
    """Resolve timeseries AutoML pipeline YAML: local path, URL, or GitHub default.

    Set ``AUTOML_TIMESERIES_PIPELINE_PATH`` to a local file or ``https://`` URL.
    """
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var=PIPELINE_YAML_TIMESERIES_ENV,
            cache_dir=tmp_path_factory.mktemp("pipeline-yaml-timeseries"),
            cache_file_name="autogluon-timeseries-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Max seconds to wait for a pipeline run (``RHOAI_PIPELINE_RUN_TIMEOUT``)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


def _build_temp_kubeconfig(server_url: str, token: str, namespace: str) -> str:
    """Write a minimal bearer-token kubeconfig to a temp file; return its path."""
    import yaml

    kubeconfig = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "cluster": {"server": server_url, "insecure-skip-tls-verify": True},
                "name": "cluster",
            }
        ],
        "users": [{"user": {"token": token}, "name": "user"}],
        "contexts": [
            {
                "context": {
                    "cluster": "cluster",
                    "user": "user",
                    "namespace": namespace,
                },
                "name": "ctx",
            }
        ],
        "current-context": "ctx",
    }
    fd, path = tempfile.mkstemp(suffix=".kubeconfig", prefix="automl-test-")
    with os.fdopen(fd, "w") as f:
        yaml.dump(kubeconfig, f)
    return path


@pytest.fixture(scope="session")
def temp_kubeconfig_path(automl_functional_config):
    """Session-scoped temp kubeconfig built from RHOAI_URL + RHOAI_TOKEN.

    Yields the path to the temp file, or None when config is not set or RHOAI_URL is missing.
    """
    if automl_functional_config is None or not automl_functional_config.get(
        "rhoai_url"
    ):
        yield None
        return
    path = _build_temp_kubeconfig(
        server_url=automl_functional_config["rhoai_url"],
        token=automl_functional_config["rhoai_token"],
        namespace=automl_functional_config["rhoai_project"],
    )
    try:
        yield path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


class S3CleanupTracker:
    """Accumulates S3 artifact prefixes to delete during session teardown."""

    def __init__(self):
        """Initialize with an empty tracking dict."""
        self.artifact_prefixes: dict[str, list[str]] = {}  # bucket -> [prefixes]

    def track_artifact_prefix(self, bucket: str, prefix: str) -> None:
        """Record a pipeline artifact prefix for teardown cleanup."""
        self.artifact_prefixes.setdefault(bucket, []).append(prefix)


@pytest.fixture(scope="session")
def s3_cleanup_tracker():
    """Session-scoped S3 cleanup tracker shared across all AutoML scenarios."""
    return S3CleanupTracker()


@pytest.fixture(scope="session", autouse=True)
def upload_datasets_if_requested(automl_functional_config, s3_client_automl_functional):
    """Upload test datasets to S3 at session start when ``AUTOML_UPLOAD_TEST_DATASETS`` is set.

    When set to ``1``, ``true``, or ``yes``, datasets referenced in tabular_test_configs.json
    and timeseries_test_configs.json are uploaded from the local ``data/`` directory to S3
    before any tests run. When unset, datasets are assumed to already be present in S3.
    """
    from autox_tests.lib.settings import AUTOML_UPLOAD_TEST_DATASETS_ENV

    uploaded_keys: list[str] = []
    bucket: str | None = None

    raw = os.environ.get(AUTOML_UPLOAD_TEST_DATASETS_ENV, "").strip().lower()
    if raw in ("1", "true", "yes"):
        if automl_functional_config is None or s3_client_automl_functional is None:
            pytest.skip(
                f"{AUTOML_UPLOAD_TEST_DATASETS_ENV} is set but S3 client is not configured — "
                "set AWS_* and RHOAI_TRAIN_DATA_BUCKET env vars"
            )
        else:
            from .configs.configs import get_all_train_data_file_keys
            from .utils import upload_test_datasets

            local_data_dir = Path(__file__).parent / "data"
            bucket = automl_functional_config["train_data_bucket_name"]
            uploaded_keys = upload_test_datasets(
                s3_client_automl_functional,
                bucket,
                get_all_train_data_file_keys(),
                local_data_dir,
            )

    yield

    if uploaded_keys and bucket:
        from autox_tests.lib.s3_data import delete_s3_objects

        deleted = delete_s3_objects(s3_client_automl_functional, bucket, uploaded_keys)
        if deleted < len(uploaded_keys):
            logger.warning(
                "Dataset teardown: only %d of %d object(s) deleted from s3://%s — bucket may be dirty",
                deleted,
                len(uploaded_keys),
                bucket,
            )
        else:
            logger.info(
                "Dataset teardown complete: %d file(s) removed from s3://%s",
                deleted,
                bucket,
            )


@pytest.fixture(scope="session", autouse=True)
def s3_teardown(s3_client_automl_functional, s3_cleanup_tracker):
    """Session-scoped teardown: delete pipeline artifacts from S3."""
    yield
    if s3_client_automl_functional is None:
        return

    keep_artifacts = os.environ.get(
        "AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS", ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if keep_artifacts:
        logger.info(
            "AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS is set — skipping artifact cleanup"
        )
        return

    from autox_tests.lib.s3_data import delete_s3_objects, list_s3_objects

    logger.info("Starting S3 artifact cleanup...")
    for bucket, prefixes in s3_cleanup_tracker.artifact_prefixes.items():
        for prefix in prefixes:
            objects = list_s3_objects(s3_client_automl_functional, bucket, prefix)
            if objects:
                keys = [o["Key"] for o in objects]
                count = delete_s3_objects(s3_client_automl_functional, bucket, keys)
                logger.info(
                    "Deleted %d artifact objects from s3://%s/%s", count, bucket, prefix
                )
    logger.info("S3 artifact cleanup complete.")
