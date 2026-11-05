"""Shared environment-driven cluster and storage settings for AutoML and AutoRAG tests."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from autox_tests.lib.env import load_tests_env

logger = logging.getLogger(__name__)

# Shared cluster and storage settings.

RHOAI_URL_ENV = "RHOAI_URL"
RHOAI_KFP_URL_ENV = "RHOAI_KFP_URL"
RHOAI_KFP_URL_ENV_ALT = "KFP_HOST"
RHOAI_TOKEN_ENV = "RHOAI_TOKEN"
RHOAI_PROJECT_ENV = "RHOAI_PROJECT_NAME"
S3_ENDPOINT_ENV = "AWS_S3_ENDPOINT"
S3_ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
S3_SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
S3_REGION_ENV = "AWS_DEFAULT_REGION"
S3_BUCKET_DATA_ENV = "RHOAI_TEST_DATA_BUCKET"
S3_BUCKET_ARTIFACTS_ENV = "RHOAI_TEST_ARTIFACTS_BUCKET"
S3_SECRET_NAME_ENV = "RHOAI_TEST_S3_SECRET_NAME"
# AutoML functional tests (rhoai-3.5-ea.2 naming; preferred over legacy TEST_* names)
RHOAI_TRAIN_DATA_BUCKET_ENV = "RHOAI_TRAIN_DATA_BUCKET"
RHOAI_TRAIN_S3_SECRET_NAME_ENV = "RHOAI_TRAIN_S3_SECRET_NAME"
S3_SKIP_SECRET_SETUP_ENV = "RHOAI_SKIP_S3_SECRET_SETUP"
S3_SECRET_OVERWRITE_KEYS_ENV = "RHOAI_S3_SECRET_OVERWRITE_KEYS"

# OpenShift API TLS for the Kubernetes client (``RHOAI_URL`` / kubeconfig)
RHOAI_OPENSHIFT_CA_BUNDLE_PATH_ENV = "RHOAI_OPENSHIFT_CA_BUNDLE_PATH"
RHOAI_OPENSHIFT_CA_DATA_ENV = "RHOAI_OPENSHIFT_CA_DATA"
RHOAI_OPENSHIFT_API_INSECURE_TLS_ENV = "RHOAI_OPENSHIFT_API_INSECURE_TLS"

# HTTPS clients used by integration tests (boto3 S3, ``kfp.Client``): verify TLS certs (default: true).
RHOAI_HTTPS_VERIFY_ENV = "RHOAI_HTTPS_VERIFY"

# Optional: create DataSciencePipelinesApplication in the test namespace (requires DSPO / RHOAI)
RHOAI_CREATE_DSPA_ENV = "RHOAI_CREATE_DSPA"
RHOAI_DSPA_API_GROUP_ENV = "RHOAI_DSPA_API_GROUP"
RHOAI_DSPA_API_VERSION_ENV = "RHOAI_DSPA_API_VERSION"
RHOAI_DSPA_PLURAL_ENV = "RHOAI_DSPA_PLURAL"
RHOAI_DSPA_ROUTE_NAME_PREFIX_ENV = "RHOAI_DSPA_ROUTE_NAME_PREFIX"
RHOAI_DSPA_ROUTE_WAIT_TIMEOUT_ENV = "RHOAI_DSPA_ROUTE_WAIT_TIMEOUT"
RHOAI_DSPA_READY_WAIT_TIMEOUT_ENV = "RHOAI_DSPA_READY_WAIT_TIMEOUT"
RHOAI_DSPA_READY_BUFFER_SECONDS_ENV = "RHOAI_DSPA_READY_BUFFER_SECONDS"
RHOAI_DSPA_NAME_ENV = "RHOAI_DSPA_NAME"
RHOAI_DSPA_DSP_VERSION_ENV = "RHOAI_DSPA_DSP_VERSION"
RHOAI_DSPA_MANAGED_PIPELINES_IMAGE_ENV = "RHOAI_DSPA_MANAGED_PIPELINES_IMAGE"
RHOAI_DSPA_MANAGED_PIPELINE_NAMES_ENV = "RHOAI_DSPA_MANAGED_PIPELINE_NAMES"
# Optional DSPA object-storage endpoint override (e.g. in-cluster URL vs public ``AWS_S3_ENDPOINT``).
INCLUSTER_AWS_S3_ENDPOINT_ENV = "INCLUSTER_AWS_S3_ENDPOINT"


def get_rhoai_integration_https_verify() -> bool:
    """Return whether to verify TLS certificates for S3 (boto3) and Kubeflow Pipelines (``kfp.Client``).

    Set ``RHOAI_HTTPS_VERIFY=false`` (or ``0`` / ``no`` / ``off``) when endpoints use self-signed or
    enterprise CAs that are not trusted by the local store (e.g. lab MinIO/S3 gateways).

    Precedence:

    1. If ``RHOAI_HTTPS_VERIFY`` is non-empty, falsy tokens disable verification.
    2. Else if ``KFP_VERIFY_SSL`` is non-empty (legacy, used by some pipeline ``tests/conftest.py``),
       the same falsy tokens disable verification.
    3. Else ``True`` (verify).

    Variables already set in the process take precedence over ``autox_tests/.env``
    when :func:`autox_tests.lib.env.load_tests_env` runs.
    """
    load_tests_env()
    for key in (RHOAI_HTTPS_VERIFY_ENV, "KFP_VERIFY_SSL"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        return raw.lower() not in ("0", "false", "no", "off")
    return True


def _kube_tls_for_namespace_config() -> tuple[bool, str | None]:
    """Return ``(insecure_skip_tls_verify, certificate_authority_data)`` for kubeconfig.

    - If ``RHOAI_OPENSHIFT_CA_BUNDLE_PATH`` or ``RHOAI_OPENSHIFT_CA_DATA`` is set, TLS is verified
      using that PEM (enterprise / custom CA, or imported self-signed cluster CA).
    - Otherwise the default is **insecure** skip verify, which matches many lab clusters whose API
      uses a self-signed cert and no local CA file.
    - To require verification without supplying a CA, set ``RHOAI_OPENSHIFT_API_INSECURE_TLS=false``
      **and** provide a CA path or data (otherwise this raises ``ValueError``).
    """
    load_tests_env()
    ca_path = (os.environ.get(RHOAI_OPENSHIFT_CA_BUNDLE_PATH_ENV) or "").strip()
    ca_data = (os.environ.get(RHOAI_OPENSHIFT_CA_DATA_ENV) or "").strip()
    insecure_raw = (
        (os.environ.get(RHOAI_OPENSHIFT_API_INSECURE_TLS_ENV) or "").strip().lower()
    )

    if ca_path:
        p = Path(ca_path).expanduser()
        if not p.is_file():
            raise ValueError(
                f"{RHOAI_OPENSHIFT_CA_BUNDLE_PATH_ENV} is not a readable file: {ca_path}"
            )
        pem = p.read_bytes()
        return (False, base64.standard_b64encode(pem).decode("ascii"))
    if ca_data:
        return (False, ca_data.replace("\n", "").replace(" ", ""))

    if insecure_raw in ("0", "false", "no", "off"):
        raise ValueError(
            f"{RHOAI_OPENSHIFT_API_INSECURE_TLS_ENV}=false requires "
            f"{RHOAI_OPENSHIFT_CA_BUNDLE_PATH_ENV} or {RHOAI_OPENSHIFT_CA_DATA_ENV} "
            "(OpenShift API server CA PEM)"
        )
    return True, None


_DSPA_CREATE_FALSE = frozenset({"0", "false", "no", "off"})
_DSPA_CREATE_TRUE = frozenset({"1", "true", "yes", "on"})


def should_create_dspa_from_env() -> bool:
    """Whether the test session should create a DataSciencePipelinesApplication.

    Default: **yes** when ``RHOAI_KFP_URL`` is unset (one-command bootstrap from ``.env``).
    Set ``RHOAI_CREATE_DSPA=false`` to disable, or set ``RHOAI_KFP_URL`` to use an existing server.
    """
    load_tests_env()
    raw = (os.environ.get(RHOAI_CREATE_DSPA_ENV) or "").strip().lower().strip("'\"")
    if raw in _DSPA_CREATE_FALSE:
        return False
    if raw in _DSPA_CREATE_TRUE:
        return True
    kfp_url = (os.environ.get(RHOAI_KFP_URL_ENV) or "").strip() or (
        os.environ.get(RHOAI_KFP_URL_ENV_ALT) or ""
    ).strip()
    return not kfp_url


def parse_timeout_seconds_from_env(
    env_var: str,
    default: int,
    max_seconds: int | None = None,
) -> int:
    """Parse an integer timeout (seconds) from the environment with validation.

    Args:
        env_var: Environment variable name to read
        default: Default value if env var is unset or empty
        max_seconds: Maximum allowed timeout (None for no limit). Values above this
                     trigger a warning and are capped to max_seconds.

    Returns:
        Validated timeout in seconds

    Raises:
        ValueError: If the value is not a valid integer or is negative
    """
    raw = (os.environ.get(env_var) or "").strip() or str(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var}={raw!r}: expected an integer (seconds)") from exc

    if value <= 0:
        raise ValueError(
            f"{env_var}={value}: timeout must be a positive integer (seconds)"
        )

    if max_seconds is not None and value > max_seconds:
        logger.warning(
            "%s=%d exceeds recommended maximum %ds; capping to %ds",
            env_var,
            value,
            max_seconds,
            max_seconds,
        )
        return max_seconds

    return value


def _build_managed_pipelines_spec_from_env() -> dict[str, Any]:
    """Build ``spec.apiServer.managedPipelines`` from env.

    Returns ``{}`` when no image or pipeline names are set; the DSPA CR still gets
    ``managedPipelines: {}`` so the operator can apply its default managed pipeline set.
    """
    image = (os.environ.get(RHOAI_DSPA_MANAGED_PIPELINES_IMAGE_ENV) or "").strip()
    names_raw = (os.environ.get(RHOAI_DSPA_MANAGED_PIPELINE_NAMES_ENV) or "").strip()
    if not image and not names_raw:
        return {}
    spec: dict[str, Any] = {}
    if image:
        spec["image"] = image
    if names_raw:
        spec["pipelines"] = [
            {"name": n.strip()} for n in names_raw.split(",") if n.strip()
        ]
    return spec


def get_dspa_config_from_env() -> dict[str, Any] | None:
    """Return DSPA creation options when auto-setup is enabled; else ``None``."""
    if not should_create_dspa_from_env():
        return None
    load_tests_env()

    dsp_version = (os.environ.get(RHOAI_DSPA_DSP_VERSION_ENV) or "").strip() or "v2"
    dspa_name = (os.environ.get(RHOAI_DSPA_NAME_ENV) or "dspa").strip()
    return {
        "create": True,
        "api_group": os.environ.get(RHOAI_DSPA_API_GROUP_ENV)
        or "datasciencepipelinesapplications.opendatahub.io",
        "api_version": os.environ.get(RHOAI_DSPA_API_VERSION_ENV) or "v1",
        "plural": os.environ.get(RHOAI_DSPA_PLURAL_ENV)
        or "datasciencepipelinesapplications",
        "route_name_prefix": os.environ.get(RHOAI_DSPA_ROUTE_NAME_PREFIX_ENV)
        or "ds-pipeline",
        "route_wait_timeout": parse_timeout_seconds_from_env(
            RHOAI_DSPA_ROUTE_WAIT_TIMEOUT_ENV, 300, max_seconds=600
        ),
        "ready_wait_timeout": parse_timeout_seconds_from_env(
            RHOAI_DSPA_READY_WAIT_TIMEOUT_ENV, 600, max_seconds=1800
        ),
        "ready_buffer_seconds": parse_timeout_seconds_from_env(
            RHOAI_DSPA_READY_BUFFER_SECONDS_ENV, 30, max_seconds=300
        ),
        "dsp_version": dsp_version,
        "resource_name": dspa_name,
        "managed_pipelines": _build_managed_pipelines_spec_from_env(),
        "object_storage_endpoint": (
            (os.environ.get(INCLUSTER_AWS_S3_ENDPOINT_ENV) or "").strip() or None
        ),
    }


def get_rhoai_namespace_setup_config() -> dict[str, Any] | None:
    """Return OpenShift API URL, token, project, and S3 fields needed to create the namespace and S3 secret.

    Shared by AutoML and AutoRAG. Does not require ``RHOAI_KFP_URL`` or a data bucket;
    each suite adds its pipeline and dataset settings.

    ``RHOAI_URL`` is optional when ``RHOAI_KFP_URL`` is set — namespace and secret setup is skipped
    and the KFP client connects directly to the existing pipeline server.
    """
    load_tests_env()
    url = os.environ.get(RHOAI_URL_ENV)
    token = os.environ.get(RHOAI_TOKEN_ENV)
    project = os.environ.get(RHOAI_PROJECT_ENV)
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV, "us-east-1")
    secret_name = (
        (os.environ.get(RHOAI_TRAIN_S3_SECRET_NAME_ENV) or "").strip()
        or (os.environ.get(S3_SECRET_NAME_ENV) or "").strip()
        or "s3-connection"
    )

    kfp_url = (os.environ.get(RHOAI_KFP_URL_ENV) or "").strip()
    if not kfp_url and not url:
        return None
    if not all([token, endpoint, access, secret]):
        return None
    insecure, ca_b64 = _kube_tls_for_namespace_config()
    return {
        "rhoai_url": url.rstrip("/") if url else None,
        "rhoai_token": token.strip(),
        "rhoai_project": (project or "kfp-integration-test").strip(),
        "s3_endpoint": endpoint,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_region": region,
        "s3_secret_name": secret_name,
        "kube_insecure_skip_tls": insecure,
        "kube_certificate_authority_data": ca_b64,
    }


def should_skip_s3_secret_setup() -> bool:
    """Return whether pytest must not create or modify the S3 connection secret."""
    load_tests_env()
    raw = (os.environ.get(S3_SKIP_SECRET_SETUP_ENV) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def should_overwrite_s3_secret_keys() -> bool:
    """Replace credential keys in an existing secret (default: keep UI/dashboard keys)."""
    load_tests_env()
    raw = (os.environ.get(S3_SECRET_OVERWRITE_KEYS_ENV) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")
