"""Pytest fixtures for AutoML functional tests (tabular + timeseries)."""

import logging
import os
from pathlib import Path
from typing import Literal

import pytest

from autox_tests.conftest import make_kfp_client_for_session
from autox_tests.lib.env import load_tests_env
from autox_tests.lib.managed_pipelines import (
    PipelineRunTarget,
    resolve_managed_pipeline_target,
    use_managed_pipelines_from_env,
)
from autox_tests.lib.pipeline_yaml_sources import (
    PIPELINE_YAML_TABULAR_ENV,
    PIPELINE_YAML_TIMESERIES_ENV,
)
from autox_tests.lib.settings import (
    AUTOML_UPLOAD_TEST_DATASETS_ENV,
    RHOAI_TRAIN_DATA_BUCKET_ENV,
    RHOAI_TRAIN_S3_SECRET_NAME_ENV,
    S3_BUCKET_DATA_ENV,
    S3_SECRET_NAME_ENV,
    get_rhoai_namespace_setup_config,
    should_create_dspa_from_env,
)

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Load env vars from ``.env.ml`` before collection."""
    load_tests_env("automl")


def get_automl_functional_config():
    """Build AutoML functional test config from environment; None if not configured."""
    load_tests_env("automl")
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

    train_secret_name = (os.environ.get(RHOAI_TRAIN_S3_SECRET_NAME_ENV) or "").strip()
    test_s3_secret = (os.environ.get(S3_SECRET_NAME_ENV) or "").strip()
    default_secret = (base.get("s3_secret_name") or "").strip()
    train_secret = train_secret_name or test_s3_secret or default_secret

    train_data_bucket = (os.environ.get(RHOAI_TRAIN_DATA_BUCKET_ENV) or "").strip()
    test_data_bucket = (os.environ.get(S3_BUCKET_DATA_ENV) or "").strip()
    legacy_train_bucket = (
        os.environ.get("AUTOML_TRAIN_DATA_BUCKET_NAME") or ""
    ).strip()
    train_bucket = train_data_bucket or test_data_bucket or legacy_train_bucket
    if not train_secret or not train_bucket:
        return None

    if not use_managed_pipelines_from_env():
        if not (os.environ.get(PIPELINE_YAML_TABULAR_ENV) or "").strip():
            return None
        if not (os.environ.get(PIPELINE_YAML_TIMESERIES_ENV) or "").strip():
            return None

    bucket_artifacts = (
        os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET") or train_bucket
    ).strip()

    return {
        **base,
        "rhoai_kfp_url": kfp_url.rstrip("/") if kfp_url else None,
        "train_data_secret_name": train_secret,
        "train_data_bucket_name": train_bucket,
        "s3_bucket_artifacts": bucket_artifacts,
    }


@pytest.fixture(scope="session")
def automl_functional_config():
    """Session-scoped AutoML functional test config dict (None if env is incomplete)."""
    return get_automl_functional_config()


@pytest.fixture(scope="session")
def kfp_client_automl_functional(
    automl_functional_config,
    datascience_pipelines_application,
    rhoai_cluster_kubeconfig,
):
    """KFP client: ``RHOAI_KFP_URL`` or route from auto-created / existing DSPA."""
    if automl_functional_config is None:
        return None
    try:
        return make_kfp_client_for_session(
            namespace_config=automl_functional_config,
            configured_kfp_url=automl_functional_config.get("rhoai_kfp_url"),
            datascience_pipelines_application=datascience_pipelines_application,
            kubeconfig_path=rhoai_cluster_kubeconfig,
        )
    except RuntimeError as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def s3_client_automl_functional(automl_functional_config):
    """Session-scoped S3 client for AutoML functional tests (None if S3 not configured)."""
    if automl_functional_config is None or not automl_functional_config.get(
        "s3_endpoint"
    ):
        return None
    from autox_tests.lib.clients import make_s3_client

    return make_s3_client(automl_functional_config)


def _resolve_automl_pipeline_target(
    kfp_client_automl_functional,
    tmp_path_factory,
    *,
    kind: Literal["tabular", "timeseries"],
    path_env_var: str,
    cache_subdir: str,
    cache_file_name: str,
) -> PipelineRunTarget:
    if not kfp_client_automl_functional:
        pytest.skip(
            "KFP client not available — skipping pipeline run target resolution"
        )
    try:
        return resolve_managed_pipeline_target(
            kfp_client_automl_functional,
            kind=kind,
            path_env_var=path_env_var,
            cache_dir=tmp_path_factory.mktemp(cache_subdir),
            cache_file_name=cache_file_name,
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
def tabular_pipeline_run_target(kfp_client_automl_functional, tmp_path_factory):
    """Tabular pipeline: managed KFP registration or local/URL ``pipeline.yaml`` package."""
    return _resolve_automl_pipeline_target(
        kfp_client_automl_functional,
        tmp_path_factory,
        kind="tabular",
        path_env_var=PIPELINE_YAML_TABULAR_ENV,
        cache_subdir="pipeline-yaml-tabular",
        cache_file_name="autogluon-tabular-pipeline.yaml",
    )


@pytest.fixture(scope="session")
def timeseries_pipeline_run_target(kfp_client_automl_functional, tmp_path_factory):
    """Timeseries pipeline: managed KFP registration or local/URL ``pipeline.yaml`` package."""
    return _resolve_automl_pipeline_target(
        kfp_client_automl_functional,
        tmp_path_factory,
        kind="timeseries",
        path_env_var=PIPELINE_YAML_TIMESERIES_ENV,
        cache_subdir="pipeline-yaml-timeseries",
        cache_file_name="autogluon-timeseries-pipeline.yaml",
    )


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Max seconds to wait for a pipeline run (``RHOAI_PIPELINE_RUN_TIMEOUT``)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


@pytest.fixture(scope="session")
def temp_kubeconfig_path(automl_functional_config, rhoai_cluster_kubeconfig):
    """Reuse cluster kubeconfig for KServe deployment tests."""
    if automl_functional_config is None:
        yield None
        return
    yield rhoai_cluster_kubeconfig


class S3CleanupTracker:
    """Accumulates S3 artifact prefixes to delete during session teardown."""

    def __init__(self):
        """Initialize with an empty tracking dict."""
        self.artifact_prefixes: dict[str, list[str]] = {}

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
        from .utils import delete_s3_objects

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

    from .utils import delete_s3_objects, list_s3_objects

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
