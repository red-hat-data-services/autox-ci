"""Shared fixtures for OpenShift AI KFP integration tests under ``tests/scenarios/``."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from autox_tests.lib.dspa_support import (
    create_datascience_pipelines_application,
    get_dspa_route_kfp_base_url,
    wait_for_dspa_ready,
)
from autox_tests.lib.env import load_tests_env, resolve_suite_asset_path
from autox_tests.lib.pytest_terminal import emit_terminal_line
from autox_tests.lib.pipeline_yaml_sources import (
    PIPELINE_YAML_AUTORAG_ENV,
    PIPELINE_YAML_TABULAR_ENV,
    PIPELINE_YAML_TIMESERIES_ENV,
    resolve_precompiled_pipeline_yaml,
)
from autox_tests.lib.rhoai_support import build_temp_kubeconfig, ensure_rhoai_project_and_s3_secret
from autox_tests.lib.s3_data import ensure_s3_bucket_exists
from autox_tests.lib.settings import (
    get_autorag_connection_config,
    get_default_upload_bucket_name,
    get_dspa_config_from_env,
    get_rhoai_automl_config,
    get_rhoai_integration_https_verify,
    get_rhoai_namespace_setup_config,
    get_s3_boto_config_from_env,
    get_s3_create_bucket_if_missing,
    rhoai_negative_pipeline_family_allowed,
)


def pytest_configure(config: pytest.Config) -> None:
    """Load ``tests/.env`` before collection (env vars already set take precedence)."""
    load_tests_env()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect tests that should not run given ``RHOAI_TEST_CONFIG_TAGS``.

    * **Positive** ``test_*_rhoai.py`` classes: if JSON tag filtering yields no scenarios for that
      pipeline, deselect (avoids empty parametrization / ``NOTSET`` ids).
    * **Negative** ``test_pipeline_negative_rhoai.py`` classes: deselect families whose tag
      (``tabular`` / ``timeseries`` / ``autorag``) is not in ``RHOAI_TEST_CONFIG_TAGS`` when set.

    Deselection (not ``skip``) keeps ``--collect-only`` aligned with what would execute.
    """
    from autox_tests.lib.config_loaders import (
        get_automl_tabular_configs_for_run,
        get_automl_timeseries_configs_for_run,
        get_autorag_configs_for_run,
    )

    positive_empty = {
        "TestAutomlTabularRhoaiKfp": not get_automl_tabular_configs_for_run(),
        "TestAutomlTimeseriesRhoaiKfp": not get_automl_timeseries_configs_for_run(),
        "TestAutoragRhoaiKfp": not get_autorag_configs_for_run(),
    }
    negative_class_family = {
        "TestAutomlTabularNegativeRhoaiKfp": "tabular",
        "TestAutomlTimeseriesNegativeRhoaiKfp": "timeseries",
        "TestAutoragNegativeRhoaiKfp": "autorag",
    }

    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []

    for item in items:
        cls = getattr(item, "cls", None)
        name = cls.__name__ if cls is not None else None

        if name in positive_empty and positive_empty[name]:
            deselected.append(item)
            continue

        if name in negative_class_family:
            if not rhoai_negative_pipeline_family_allowed(negative_class_family[name]):
                deselected.append(item)
                continue

        kept.append(item)

    items[:] = kept
    if deselected:
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope="session")
def rhoai_automl_config() -> dict[str, Any] | None:
    """RHOAI + S3 settings for AutoGluon tabular/timeseries pipeline runs."""
    return get_rhoai_automl_config()


@pytest.fixture(scope="session")
def autorag_config() -> dict[str, Any] | None:
    """KFP + Llama + k8s secret refs; train data paths come from JSON (upload / existing_s3)."""
    return get_autorag_connection_config()


@pytest.fixture(scope="session")
def rhoai_namespace_setup_config() -> dict[str, Any] | None:
    """``RHOAI_URL`` + ``AWS_*`` + project + secret name (no KFP URL or data bucket required)."""
    return get_rhoai_namespace_setup_config()


@pytest.fixture(scope="session")
def temp_kubeconfig_path(rhoai_namespace_setup_config: dict[str, Any] | None) -> str | None:
    """Minimal kubeconfig for OpenShift API access (S3 secret + optional ProjectRequest)."""
    if rhoai_namespace_setup_config is None:
        return None
    path = build_temp_kubeconfig(
        rhoai_namespace_setup_config["rhoai_url"],
        rhoai_namespace_setup_config["rhoai_token"],
        rhoai_namespace_setup_config["rhoai_project"],
        insecure_skip_tls_verify=rhoai_namespace_setup_config.get("kube_insecure_skip_tls", True),
        certificate_authority_data=rhoai_namespace_setup_config.get("kube_certificate_authority_data"),
    )
    try:
        yield path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


@pytest.fixture(scope="session")
def rhoai_project_and_s3_secret(
    rhoai_namespace_setup_config: dict[str, Any] | None,
    temp_kubeconfig_path: str | None,
) -> str | None:
    """Create OpenShift project (if needed) and apply ``RHOAI_TEST_S3_SECRET_NAME`` in ``RHOAI_PROJECT_NAME``."""
    if rhoai_namespace_setup_config is None:
        return None
    return ensure_rhoai_project_and_s3_secret(rhoai_namespace_setup_config, temp_kubeconfig_path)


@pytest.fixture(scope="session")
def rhoai_automl_project(
    rhoai_automl_config: dict[str, Any] | None,
    rhoai_project_and_s3_secret: str | None,
) -> str | None:
    """Namespace + S3 connection secret for AutoGluon runs (same cluster prep as AutoRAG when env is shared)."""
    if rhoai_automl_config is None:
        return None
    return rhoai_project_and_s3_secret


@pytest.fixture(scope="session")
def datascience_pipelines_application(
    request: pytest.FixtureRequest,
    rhoai_namespace_setup_config: dict[str, Any] | None,
    rhoai_project_and_s3_secret: str | None,
    temp_kubeconfig_path: str | None,
) -> dict[str, Any] | None:
    """Optionally create a DataSciencePipelinesApplication when ``RHOAI_CREATE_DSPA`` is set."""
    dspa_cfg = get_dspa_config_from_env()
    if (
        rhoai_namespace_setup_config is None
        or rhoai_project_and_s3_secret is None
        or not dspa_cfg
        or not dspa_cfg.get("create")
    ):
        return None
    try:
        import kubernetes  # noqa: F401
    except ImportError:
        pytest.fail(
            "kubernetes Python client is required when RHOAI_CREATE_DSPA is enabled. "
            "Install with: pip install kubernetes  (or pip install -e '.[test_rhoai]')."
        )

    project = rhoai_project_and_s3_secret
    bucket = (
        os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET") or os.environ.get("RHOAI_TEST_DATA_BUCKET") or ""
    ).strip()
    secret_name = rhoai_namespace_setup_config.get("s3_secret_name")
    endpoint = rhoai_namespace_setup_config.get("s3_endpoint")
    region = rhoai_namespace_setup_config.get("s3_region")
    endpoint_for_dspa = (dspa_cfg.get("object_storage_endpoint") or "").strip() or (endpoint or "").strip()

    def _dspa_progress(msg: str) -> None:
        emit_terminal_line(request.config, msg)

    emit_terminal_line(request.config, f"Starting DSPA setup for namespace {project!r}...")
    created, err = create_datascience_pipelines_application(
        project,
        dspa_cfg,
        kubeconfig_path=temp_kubeconfig_path,
        object_storage_secret_name=secret_name if bucket else None,
        object_storage_endpoint=endpoint_for_dspa if bucket else None,
        object_storage_region=region if bucket else None,
        object_storage_bucket=bucket if bucket else None,
        progress=_dspa_progress,
    )
    if created is None and err:
        logging.getLogger(__name__).error("DSPA creation failed: %s", err)
        pytest.fail(f"DataSciencePipelinesApplication creation failed: {err}")

    if created is not None:
        dspa_name = (created.get("metadata") or {}).get("name", "dspa")
        namespace = (created.get("metadata") or {}).get("namespace", project)
        ready_timeout = int(dspa_cfg.get("ready_wait_timeout", 600))
        buffer_seconds = int(dspa_cfg.get("ready_buffer_seconds", 30))
        if not wait_for_dspa_ready(
            namespace,
            dspa_name,
            dspa_cfg,
            kubeconfig_path=temp_kubeconfig_path,
            timeout_seconds=ready_timeout,
            progress=_dspa_progress,
        ):
            logging.getLogger(__name__).warning(
                "DSPA %s/%s did not become Ready within %s s; continuing anyway",
                namespace,
                dspa_name,
                ready_timeout,
            )
        emit_terminal_line(
            request.config,
            f"Post-ready buffer: sleeping {buffer_seconds}s before session tests continue...",
        )
        time.sleep(buffer_seconds)
    return created


@pytest.fixture(scope="session")
def s3_client(rhoai_automl_config: dict[str, Any] | None):
    """S3 client for uploads (AutoML train files, AutoRAG documents). Uses AutoML env or ``AWS_*``."""
    try:
        import boto3
    except ImportError:
        pytest.fail(
            "boto3 is required for S3 uploads in integration tests. "
            "Install with: pip install boto3  (or pip install -e '.[test_rhoai]')."
        )
    verify_tls = get_rhoai_integration_https_verify()
    if rhoai_automl_config is not None:
        c = rhoai_automl_config
        return boto3.client(
            "s3",
            endpoint_url=c["s3_endpoint"],
            aws_access_key_id=c["s3_access_key"],
            aws_secret_access_key=c["s3_secret_key"],
            region_name=c["s3_region"],
            verify=verify_tls,
        )
    cfg = get_s3_boto_config_from_env()
    if cfg is None:
        return None
    return boto3.client(
        "s3",
        endpoint_url=cfg["s3_endpoint"],
        aws_access_key_id=cfg["s3_access_key"],
        aws_secret_access_key=cfg["s3_secret_key"],
        region_name=cfg["s3_region"],
        verify=verify_tls,
    )


def _upload_unique_datasets(
    rhoai_automl_config: dict[str, Any] | None,
    s3_client: Any,
    dataset_paths: list[str],
) -> dict[str, dict[str, str]]:
    """Upload each distinct path under ``kfp-integration-test/``; return path -> bucket/key."""
    if rhoai_automl_config is None or s3_client is None:
        return {}
    bucket = rhoai_automl_config["s3_bucket_data"]
    if get_s3_create_bucket_if_missing():
        ensure_s3_bucket_exists(
            s3_client,
            bucket,
            region=str(rhoai_automl_config.get("s3_region") or "").strip() or None,
        )
    prefix = "kfp-integration-test"
    result: dict[str, dict[str, str]] = {}
    for rel_path in dataset_paths:
        if rel_path in result:
            continue
        full_path = resolve_suite_asset_path(rel_path)
        if not full_path.is_file():
            pytest.fail(f"Test dataset file is missing from the repository: {full_path}")
        body = full_path.read_bytes()
        key = f"{prefix}/{rel_path}"
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="text/csv",
            )
        except Exception as e:
            pytest.fail(f"Failed to upload test dataset {rel_path!r} to S3 bucket {bucket!r}: {e}")
        result[rel_path] = {"bucket": bucket, "key": key}
    return result


@pytest.fixture(scope="session")
def uploaded_automl_tabular_datasets(
    rhoai_automl_config: dict[str, Any] | None,
    s3_client: Any,
) -> dict[str, dict[str, str]]:
    """Upload tabular CSVs referenced in ``tests/config/automl_tabular_test_configs.json``."""
    from autox_tests.lib.config_loaders import get_automl_tabular_dataset_paths

    paths = get_automl_tabular_dataset_paths()
    return _upload_unique_datasets(rhoai_automl_config, s3_client, paths)


@pytest.fixture(scope="session")
def uploaded_automl_timeseries_datasets(
    rhoai_automl_config: dict[str, Any] | None,
    s3_client: Any,
) -> dict[str, dict[str, str]]:
    """Upload time series CSVs referenced in ``tests/config/automl_timeseries_test_configs.json``."""
    from autox_tests.lib.config_loaders import get_automl_timeseries_dataset_paths

    paths = get_automl_timeseries_dataset_paths()
    return _upload_unique_datasets(rhoai_automl_config, s3_client, paths)


@pytest.fixture(scope="session")
def uploaded_autorag_by_config_id(
    s3_client: Any,
) -> dict[str, dict[str, str]]:
    """Upload local document trees + benchmark JSON per ``data_mode=upload`` AutoRAG config."""
    from autox_tests.lib.config_loaders import get_autorag_configs_for_run
    from autox_tests.lib.s3_data import upload_file_to_s3, upload_tree_to_s3_prefix

    conn = get_autorag_connection_config()
    if conn is None or not s3_client:
        return {}
    bucket = get_default_upload_bucket_name()
    if not bucket:
        return {}
    if get_s3_create_bucket_if_missing():
        ensure_s3_bucket_exists(
            s3_client,
            bucket,
            region=str(conn.get("s3_region") or "").strip() or None,
        )
    result: dict[str, dict[str, str]] = {}
    for c in get_autorag_configs_for_run():
        if c.data_mode != "upload":
            continue
        assert c.documents_directory_path and c.benchmark_dataset_path
        base = f"kfp-integration-test-autorag/{c.id}"
        doc_dir = resolve_suite_asset_path(c.documents_directory_path)
        bench = resolve_suite_asset_path(c.benchmark_dataset_path)
        if not doc_dir.is_dir():
            pytest.fail(f"AutoRAG test directory is missing: {doc_dir}")
        if not bench.is_file():
            pytest.fail(f"AutoRAG benchmark JSON is missing: {bench}")
        input_prefix = f"{base}/input_docs/"
        test_key = f"{base}/benchmark.json"
        upload_tree_to_s3_prefix(s3_client, bucket=bucket, key_prefix=input_prefix, local_root=doc_dir)
        upload_file_to_s3(s3_client, bucket=bucket, key=test_key, local_path=bench)
        result[c.id] = {
            "test_data_bucket_name": bucket,
            "test_data_key": test_key,
            "input_data_bucket_name": bucket,
            "input_data_key": input_prefix,
        }
    return result


@pytest.fixture(scope="session")
def kfp_client_automl(
    rhoai_automl_config: dict[str, Any] | None,
    datascience_pipelines_application: dict[str, Any] | None,
    temp_kubeconfig_path: str | None,
) -> Any:
    """KFP client for AutoML runs (``RHOAI_KFP_URL`` or DSPA route when ``RHOAI_CREATE_DSPA``)."""
    if rhoai_automl_config is None:
        return None
    import kfp

    host: str | None = None
    dspa_cfg = get_dspa_config_from_env()
    if datascience_pipelines_application is not None and dspa_cfg and dspa_cfg.get("create"):
        ns = (datascience_pipelines_application.get("metadata") or {}).get("namespace")
        if ns:
            host = get_dspa_route_kfp_base_url(
                ns,
                route_name_prefix=str(dspa_cfg.get("route_name_prefix", "ds-pipeline")),
                timeout_seconds=int(dspa_cfg.get("route_wait_timeout", 300)),
                kubeconfig_path=temp_kubeconfig_path,
            )
    if host is None:
        raw = rhoai_automl_config.get("rhoai_kfp_url")
        host = str(raw).strip() if raw else None
    if not host:
        pytest.fail(
            "Could not determine Kubeflow Pipelines API URL for AutoML tests.\n"
            "- Set RHOAI_KFP_URL to the Data Science Pipelines HTTPS route, or\n"
            "- Set RHOAI_CREATE_DSPA=true and ensure the ds-pipeline route is created in the project.\n"
            "If the route is slow to appear, increase RHOAI_DSPA_ROUTE_WAIT_TIMEOUT. "
            "See tests/.env.example."
        )
    host = str(host).rstrip("/") + "/"
    client_kw: dict[str, Any] = {
        "host": host,
        "namespace": rhoai_automl_config["rhoai_project"],
        "existing_token": rhoai_automl_config.get("rhoai_token"),
    }
    if not get_rhoai_integration_https_verify():
        client_kw["verify_ssl"] = False
    return kfp.Client(**client_kw)


@pytest.fixture(scope="session")
def kfp_client_autorag(
    autorag_config: dict[str, Any] | None,
    datascience_pipelines_application: dict[str, Any] | None,
    temp_kubeconfig_path: str | None,
) -> Any:
    """KFP client for AutoRAG (``RHOAI_KFP_URL`` or DSPA route when ``RHOAI_CREATE_DSPA``)."""
    if autorag_config is None:
        return None
    import kfp

    host: str | None = None
    dspa_cfg = get_dspa_config_from_env()
    if datascience_pipelines_application is not None and dspa_cfg and dspa_cfg.get("create"):
        ns = (datascience_pipelines_application.get("metadata") or {}).get("namespace")
        if ns:
            host = get_dspa_route_kfp_base_url(
                ns,
                route_name_prefix=str(dspa_cfg.get("route_name_prefix", "ds-pipeline")),
                timeout_seconds=int(dspa_cfg.get("route_wait_timeout", 300)),
                kubeconfig_path=temp_kubeconfig_path,
            )
    if host is None:
        raw = autorag_config.get("rhoai_kfp_url")
        host = str(raw).strip() if raw else None
    if not host:
        pytest.fail(
            "Could not determine Kubeflow Pipelines API URL for AutoRAG tests.\n"
            "- Set RHOAI_KFP_URL (or KFP_HOST), or\n"
            "- Set RHOAI_CREATE_DSPA=true and ensure the pipeline route is available.\n"
            "If the route is slow to appear, increase RHOAI_DSPA_ROUTE_WAIT_TIMEOUT. "
            "See tests/.env.example."
        )
    host = str(host).rstrip("/") + "/"
    client_kw = {
        "host": host,
        "namespace": autorag_config["rhoai_project"],
        "existing_token": autorag_config.get("rhoai_token"),
    }
    if not get_rhoai_integration_https_verify():
        client_kw["verify_ssl"] = False
    return kfp.Client(**client_kw)


@pytest.fixture(scope="session")
def pipeline_yaml_cache_dir() -> Iterator[Path]:
    """Temporary directory for downloaded ``pipeline.yaml`` files (session scope)."""
    d = Path(tempfile.mkdtemp(prefix="rhoai-pipeline-yaml-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def pipeline_run_timeout() -> int:
    """Max seconds to wait for a pipeline run (``RHOAI_PIPELINE_RUN_TIMEOUT``)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


@pytest.fixture
def pipeline_poll_interval_seconds() -> int:
    """Seconds between KFP progress polls (``RHOAI_KFP_POLL_INTERVAL_SECONDS``)."""
    return int(os.environ.get("RHOAI_KFP_POLL_INTERVAL_SECONDS", "25"))


@pytest.fixture
def pipeline_negative_run_timeout(pipeline_run_timeout: int) -> int:
    """Max seconds to wait for negative pipeline runs (``RHOAI_PIPELINE_NEGATIVE_RUN_TIMEOUT``)."""
    raw = os.environ.get("RHOAI_PIPELINE_NEGATIVE_RUN_TIMEOUT")
    if raw is not None and str(raw).strip():
        return int(raw)
    return pipeline_run_timeout


def _make_run_name(prefix: str) -> str:
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{hex_part}-{time_part}"


@pytest.fixture
def automl_run_name() -> str:
    """Unique KFP run name for AutoGluon pipeline submissions."""
    return _make_run_name("rhoai-automl")


@pytest.fixture
def autorag_run_name() -> str:
    """Unique KFP run name for AutoRAG pipeline submissions."""
    return _make_run_name("rhoai-autorag")


@pytest.fixture(scope="session")
def automl_tabular_pipeline_package(pipeline_yaml_cache_dir: Path) -> str:
    """Path to tabular AutoML ``pipeline.yaml`` (local path or URL)."""
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var=PIPELINE_YAML_TABULAR_ENV,
            cache_dir=pipeline_yaml_cache_dir,
            cache_file_name="autogluon-tabular-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def automl_timeseries_pipeline_package(pipeline_yaml_cache_dir: Path) -> str:
    """Path to time series AutoML ``pipeline.yaml`` (local path or URL)."""
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var=PIPELINE_YAML_TIMESERIES_ENV,
            cache_dir=pipeline_yaml_cache_dir,
            cache_file_name="autogluon-timeseries-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))


@pytest.fixture(scope="session")
def autorag_pipeline_package(pipeline_yaml_cache_dir: Path) -> str:
    """Path to AutoRAG ``pipeline.yaml`` (local path or URL)."""
    try:
        return resolve_precompiled_pipeline_yaml(
            path_env_var=PIPELINE_YAML_AUTORAG_ENV,
            cache_dir=pipeline_yaml_cache_dir,
            cache_file_name="documents-rag-optimization-pipeline.yaml",
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.fail(str(e))
