"""Shared session fixtures for functional suites under ``autox_tests/automl`` and ``autorag``."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from autox_tests.lib.dspa_support import (
    create_datascience_pipelines_application,
    get_dspa_route_kfp_base_url,
    verify_kfp_api_health,
    wait_for_dspa_ready,
)
from autox_tests.lib.rhoai_support import (
    build_temp_kubeconfig,
    ensure_rhoai_project_and_s3_secret,
)
from autox_tests.lib.settings import (
    RHOAI_TRAIN_DATA_BUCKET_ENV,
    S3_BUCKET_ARTIFACTS_ENV,
    S3_BUCKET_DATA_ENV,
    get_dspa_config_from_env,
    get_rhoai_integration_https_verify,
    get_rhoai_namespace_setup_config,
    should_create_dspa_from_env,
)

logger = logging.getLogger(__name__)


def _ensure_datascience_pipelines_application(
    *,
    namespace: str,
    namespace_config: dict[str, Any],
    kubeconfig_path: str | None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Create or reuse a DSPA when :func:`get_dspa_config_from_env` is active."""
    dspa_cfg = get_dspa_config_from_env()
    if not dspa_cfg or not dspa_cfg.get("create"):
        return None

    bucket = (
        (os.environ.get(S3_BUCKET_ARTIFACTS_ENV) or "").strip()
        or (os.environ.get(RHOAI_TRAIN_DATA_BUCKET_ENV) or "").strip()
        or (os.environ.get(S3_BUCKET_DATA_ENV) or "").strip()
        or (os.environ.get("AUTOML_TRAIN_DATA_BUCKET_NAME") or "").strip()
    )
    secret_name = namespace_config.get("s3_secret_name")
    endpoint = namespace_config.get("s3_endpoint")
    region = namespace_config.get("s3_region")
    incluster_endpoint = (dspa_cfg.get("object_storage_endpoint") or "").strip()
    endpoint_for_dspa = incluster_endpoint or (endpoint or "").strip()

    if bucket and not endpoint_for_dspa:
        raise RuntimeError(
            "DSPA external S3 requires AWS_S3_ENDPOINT when a data bucket is configured. "
            "Set AWS_S3_ENDPOINT in the env file."
        )

    if progress:
        progress("Creating DataSciencePipelinesApplication...")

    created, err = create_datascience_pipelines_application(
        namespace,
        dspa_cfg,
        kubeconfig_path=kubeconfig_path,
        object_storage_secret_name=secret_name if bucket else None,
        object_storage_endpoint=endpoint_for_dspa if bucket else None,
        object_storage_region=region if bucket else None,
        object_storage_bucket=bucket if bucket else None,
        progress=progress,
    )
    if created is None and err:
        raise RuntimeError(f"DataSciencePipelinesApplication creation failed: {err}")

    if created is not None:
        dspa_name = (created.get("metadata") or {}).get("name", "dspa")
        ready_timeout = int(dspa_cfg.get("ready_wait_timeout", 600))
        buffer_seconds = int(dspa_cfg.get("ready_buffer_seconds", 30))
        if not wait_for_dspa_ready(
            namespace,
            dspa_name,
            dspa_cfg,
            kubeconfig_path=kubeconfig_path,
            timeout_seconds=ready_timeout,
            progress=progress,
        ):
            logger.warning(
                "DSPA did not become Ready within %s s; continuing anyway",
                ready_timeout,
            )

        if progress:
            progress(
                f"Post-ready buffer: sleeping {buffer_seconds}s before tests continue..."
            )
        time.sleep(buffer_seconds)

        # Verify KFP API is responsive after buffer
        route_prefix = dspa_cfg.get("route_name_prefix", "ds-pipeline")
        kfp_url = get_dspa_route_kfp_base_url(
            namespace,
            route_name_prefix=route_prefix,
            timeout_seconds=60,
            kubeconfig_path=kubeconfig_path,
        )
        if kfp_url:
            verify_ssl = namespace_config.get("kfp_verify_ssl", False)
            if not verify_kfp_api_health(kfp_url, timeout_seconds=30, verify_ssl=verify_ssl):
                logger.warning("KFP API not responding after DSPA Ready; tests may fail")
                if progress:
                    progress("Warning: KFP API health check failed - tests may be unstable")
        else:
            logger.warning("Could not resolve KFP route URL for health check")

    return created


def _resolve_kfp_api_host(
    *,
    namespace: str,
    namespace_config: dict[str, Any],
    datascience_pipelines_application: dict[str, Any] | None,
    kubeconfig_path: str | None,
    configured_kfp_url: str | None,
) -> str:
    """Return KFP API base URL (with trailing slash) from route discovery or env."""
    host: str | None = None

    if datascience_pipelines_application is not None:
        dspa_cfg = get_dspa_config_from_env() or {}
        ns = (datascience_pipelines_application.get("metadata") or {}).get(
            "namespace"
        ) or namespace
        host = get_dspa_route_kfp_base_url(
            ns,
            route_name_prefix=str(dspa_cfg.get("route_name_prefix", "ds-pipeline")),
            timeout_seconds=int(dspa_cfg.get("route_wait_timeout", 300)),
            kubeconfig_path=kubeconfig_path,
        )

    if host is None and configured_kfp_url:
        host = configured_kfp_url.strip().rstrip("/")

    if not host:
        raise RuntimeError(
            "Could not determine Kubeflow Pipelines API URL.\n"
            "- Set RHOAI_KFP_URL to an existing pipeline server route, or\n"
            "- Omit RHOAI_KFP_URL and let tests create a DSPA (default when RHOAI_CREATE_DSPA "
            "is not false).\n"
            "If the route is slow to appear, increase RHOAI_DSPA_ROUTE_WAIT_TIMEOUT."
        )
    return host.rstrip("/") + "/"


@pytest.fixture(scope="session")
def rhoai_namespace_setup_config() -> dict[str, Any] | None:
    """``RHOAI_URL``, token, project, and S3 credentials from the environment."""
    return get_rhoai_namespace_setup_config()


@pytest.fixture(scope="session")
def rhoai_cluster_kubeconfig(
    rhoai_namespace_setup_config: dict[str, Any] | None,
) -> Generator[str | None, None, None]:
    """Minimal kubeconfig for OpenShift API (namespace, secrets, DSPA, routes)."""
    if rhoai_namespace_setup_config is None or not rhoai_namespace_setup_config.get("rhoai_url"):
        yield None
        return
    path = build_temp_kubeconfig(
        rhoai_namespace_setup_config["rhoai_url"],
        rhoai_namespace_setup_config["rhoai_token"],
        rhoai_namespace_setup_config["rhoai_project"],
        insecure_skip_tls_verify=rhoai_namespace_setup_config.get(
            "kube_insecure_skip_tls", True
        ),
        certificate_authority_data=rhoai_namespace_setup_config.get(
            "kube_certificate_authority_data"
        ),
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
    rhoai_cluster_kubeconfig: str | None,
) -> str | None:
    """Ensure OpenShift namespace and ``RHOAI_TEST_S3_SECRET_NAME`` exist."""
    if rhoai_namespace_setup_config is None or rhoai_cluster_kubeconfig is None:
        return None
    return ensure_rhoai_project_and_s3_secret(
        rhoai_namespace_setup_config, rhoai_cluster_kubeconfig
    )


@pytest.fixture(scope="session")
def datascience_pipelines_application(
    request: pytest.FixtureRequest,
    rhoai_namespace_setup_config: dict[str, Any] | None,
    rhoai_project_and_s3_secret: str | None,
    rhoai_cluster_kubeconfig: str | None,
) -> dict[str, Any] | None:
    """Create DSPA with managed pipelines when auto-setup is enabled (default without ``RHOAI_KFP_URL``)."""
    if (
        rhoai_namespace_setup_config is None
        or rhoai_project_and_s3_secret is None
        or rhoai_cluster_kubeconfig is None
        or not should_create_dspa_from_env()
    ):
        return None
    try:
        import kubernetes  # noqa: F401
    except ImportError:
        pytest.fail(
            "kubernetes Python client is required for DSPA auto-setup. "
            "Install with: uv sync --extra test_automl"
        )

    def _progress(msg: str) -> None:
        print(f"\n{msg}", flush=True)
        logger.info(msg)

    _progress(
        "Auto-setup: creating DSPA (leave RHOAI_KFP_URL unset or set RHOAI_CREATE_DSPA=true)"
    )

    return _ensure_datascience_pipelines_application(
        namespace=rhoai_project_and_s3_secret,
        namespace_config=rhoai_namespace_setup_config,
        kubeconfig_path=rhoai_cluster_kubeconfig,
        progress=_progress,
    )


def make_kfp_client_for_session(
    *,
    namespace_config: dict[str, Any],
    configured_kfp_url: str | None,
    datascience_pipelines_application: dict[str, Any] | None,
    kubeconfig_path: str | None,
) -> Any:
    """Build a ``kfp.Client`` using DSPA route discovery or ``RHOAI_KFP_URL``."""
    import kfp

    host = _resolve_kfp_api_host(
        namespace=namespace_config["rhoai_project"],
        namespace_config=namespace_config,
        datascience_pipelines_application=datascience_pipelines_application,
        kubeconfig_path=kubeconfig_path,
        configured_kfp_url=configured_kfp_url,
    )
    client_kw: dict[str, Any] = {
        "host": host,
        "namespace": namespace_config["rhoai_project"],
        "existing_token": namespace_config.get("rhoai_token"),
    }
    if not get_rhoai_integration_https_verify():
        client_kw["verify_ssl"] = False
    return kfp.Client(**client_kw)
