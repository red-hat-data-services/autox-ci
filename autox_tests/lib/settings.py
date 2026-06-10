"""Environment-driven settings for root OpenShift AI KFP tests.

Call :func:`load_tests_env` from ``tests/lib/env.py`` before reading configuration
(``tests/scenarios/conftest.py`` does this at session start via ``pytest_configure``).
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from autox_tests.lib.env import load_tests_env

logger = logging.getLogger(__name__)

# --- AutoML (tabular + timeseries) shared storage / cluster ---

RHOAI_URL_ENV = "RHOAI_URL"
RHOAI_KFP_URL_ENV = "RHOAI_KFP_URL"
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
S3_CREATE_BUCKET_IF_MISSING_ENV = "RHOAI_TEST_S3_CREATE_BUCKET_IF_MISSING"

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

# Optional defaults for ``data_mode=existing_s3`` in JSON (AutoML + AutoRAG)
TEST_DATA_SOURCE_BUCKET_ENV = "TEST_DATA_SOURCE_BUCKET"
TEST_DATA_SOURCE_PREFIX_ENV = "TEST_DATA_SOURCE_PREFIX"

# --- Documents RAG optimization ---

RHOAI_KFP_URL_ENV_ALT = "KFP_HOST"
RHOAI_TOKEN_ENV_ALT = "KFP_TOKEN"
RHOAI_PROJECT_ENV_ALT = "KFP_NAMESPACE"
TEST_DATA_BUCKET_ENV = "TEST_DATA_BUCKET_NAME"
TEST_DATA_KEY_ENV = "TEST_DATA_KEY"
INPUT_DATA_BUCKET_ENV = "INPUT_DATA_BUCKET_NAME"
INPUT_DATA_KEY_ENV = "INPUT_DATA_KEY"
OGX_SECRET_ENV = "OGX_SECRET_NAME"
VECTOR_IO_PROVIDER_ENV = "VECTOR_IO_PROVIDER_ID"

# Tag filter (shared with per-pipeline integration tests)
TEST_CONFIG_TAGS_ENV = "RHOAI_TEST_CONFIG_TAGS"


def get_rhoai_test_config_tag_filter() -> frozenset[str] | None:
    """Return lowercase tags from ``RHOAI_TEST_CONFIG_TAGS``, or ``None`` if unset/empty (no filter)."""
    load_tests_env()
    raw = os.environ.get(TEST_CONFIG_TAGS_ENV)
    if not raw or not str(raw).strip():
        return None
    allowed = {t.strip().lower() for t in str(raw).split(",") if t.strip()}
    return frozenset(allowed) if allowed else None


def rhoai_negative_pipeline_family_allowed(family: str) -> bool:
    """Return whether negative pipeline tests should run for this family given tag filter.

    ``family`` must be one of ``tabular``, ``timeseries``, ``autorag`` (same strings as in JSON
    scenario ``tags``). When ``RHOAI_TEST_CONFIG_TAGS`` is unset or empty, all families are allowed.
    When set, a family runs only if that tag appears in the comma-separated list (case-insensitive),
    consistent with JSON scenario tag filtering in ``tests.lib.config_loaders``.
    """
    filt = get_rhoai_test_config_tag_filter()
    if filt is None:
        return True
    return family.strip().lower() in filt


def get_rhoai_integration_https_verify() -> bool:
    """Return whether to verify TLS certificates for S3 (boto3) and Kubeflow Pipelines (``kfp.Client``).

    Set ``RHOAI_HTTPS_VERIFY=false`` (or ``0`` / ``no`` / ``off``) when endpoints use self-signed or
    enterprise CAs that are not trusted by the local store (e.g. lab MinIO/S3 gateways).

    Precedence:

    1. If ``RHOAI_HTTPS_VERIFY`` is non-empty, falsy tokens disable verification.
    2. Else if ``KFP_VERIFY_SSL`` is non-empty (legacy, used by some pipeline ``tests/conftest.py``),
       the same falsy tokens disable verification.
    3. Else ``True`` (verify).

    Environment variables already set in the process are not overridden by ``tests/.env`` until
    :func:`tests.lib.env.load_tests_env` runs.
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
        raise ValueError(
            f"{env_var}={raw!r}: expected an integer (seconds)"
        ) from exc

    if value <= 0:
        raise ValueError(f"{env_var}={value}: timeout must be a positive integer (seconds)")

    if max_seconds is not None and value > max_seconds:
        logger.warning(
            "%s=%d exceeds recommended maximum %ds; capping to %ds",
            env_var, value, max_seconds, max_seconds
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

    Shared by AutoML and AutoRAG. Does **not** require ``RHOAI_KFP_URL`` or ``RHOAI_TEST_DATA_BUCKET``;
    use :func:`get_rhoai_automl_config` when submitting AutoGluon pipelines (those need KFP + data bucket).
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

    if not all([url, token, endpoint, access, secret]):
        return None
    insecure, ca_b64 = _kube_tls_for_namespace_config()
    return {
        "rhoai_url": url.rstrip("/"),
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


def get_rhoai_automl_config() -> dict[str, Any] | None:
    """Return config for AutoML pipelines (namespace setup + KFP + data bucket); None if incomplete.

    ``RHOAI_KFP_URL`` may be omitted when ``RHOAI_CREATE_DSPA=true``; the session fixture then
    discovers the API URL from the OpenShift Route created for the pipeline server.
    """
    base = get_rhoai_namespace_setup_config()
    if base is None:
        return None
    load_tests_env()
    kfp_url = os.environ.get(RHOAI_KFP_URL_ENV)
    bucket_data = os.environ.get(S3_BUCKET_DATA_ENV)
    bucket_artifacts = os.environ.get(S3_BUCKET_ARTIFACTS_ENV)
    if not bucket_data:
        return None
    if not kfp_url and not should_create_dspa_from_env():
        return None
    return {
        **base,
        "rhoai_kfp_url": kfp_url.strip().rstrip("/") if kfp_url else None,
        "s3_bucket_data": bucket_data,
        "s3_bucket_artifacts": bucket_artifacts or bucket_data,
    }


def get_test_data_source_defaults() -> dict[str, str | None]:
    """Return optional ``TEST_DATA_SOURCE_BUCKET`` and prefix for JSON ``existing_s3`` modes."""
    load_tests_env()
    b = os.environ.get(TEST_DATA_SOURCE_BUCKET_ENV)
    p = os.environ.get(TEST_DATA_SOURCE_PREFIX_ENV)
    return {
        "bucket": b.strip() if b else None,
        "prefix": p.strip().strip("/") if p else None,
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


def get_s3_create_bucket_if_missing() -> bool:
    """Return whether uploads may call :func:`tests.lib.s3_data.ensure_s3_bucket_exists` (default: True)."""
    load_tests_env()
    raw = (os.environ.get(S3_CREATE_BUCKET_IF_MISSING_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def get_s3_boto_config_from_env() -> dict[str, Any] | None:
    """Return S3 client settings from ``AWS_*`` env vars (independent of AutoML block)."""
    load_tests_env()
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV, "us-east-1")
    if not all([endpoint, access, secret]):
        return None
    return {
        "s3_endpoint": endpoint,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_region": region,
    }


def get_default_upload_bucket_name() -> str | None:
    """Default bucket for uploading test fixtures (AutoRAG / shared uploads)."""
    load_tests_env()
    raw = (
        os.environ.get(S3_BUCKET_DATA_ENV)
        or os.environ.get(TEST_DATA_SOURCE_BUCKET_ENV)
        or ""
    ).strip()
    return raw or None


def get_autorag_connection_config() -> dict[str, Any] | None:
    """Return KFP + OGX + k8s S3 secret name + optional S3 creds.

    ``test_data_secret_name`` and ``input_data_secret_name`` both use ``RHOAI_TEST_S3_SECRET_NAME``
    (same as AutoML). Per-run bucket/object paths come from JSON (``upload`` or ``existing_s3``).
    Optional ``TEST_DATA_*`` / ``INPUT_DATA_*`` bucket env values are fallbacks for ``existing_s3``.
    """
    load_tests_env()
    kfp_url = os.environ.get(RHOAI_KFP_URL_ENV) or os.environ.get(RHOAI_KFP_URL_ENV_ALT)
    token = os.environ.get(RHOAI_TOKEN_ENV) or os.environ.get(RHOAI_TOKEN_ENV_ALT)
    project = os.environ.get(RHOAI_PROJECT_ENV) or os.environ.get(RHOAI_PROJECT_ENV_ALT)
    # Same k8s secret as AutoML (``RHOAI_TEST_S3_SECRET_NAME``): used for both pipeline S3 secret params.
    s3_secret = (os.environ.get(S3_SECRET_NAME_ENV) or "s3-connection").strip()
    ogx_secret = os.environ.get(OGX_SECRET_ENV)
    vector_io = os.environ.get(VECTOR_IO_PROVIDER_ENV)

    if not all([token, s3_secret, ogx_secret, vector_io]):
        return None
    if not kfp_url and not should_create_dspa_from_env():
        return None

    t_bucket = (os.environ.get(TEST_DATA_BUCKET_ENV) or "").strip()
    t_key = (os.environ.get(TEST_DATA_KEY_ENV) or "").strip()
    i_bucket = (os.environ.get(INPUT_DATA_BUCKET_ENV) or "").strip()
    i_key = (os.environ.get(INPUT_DATA_KEY_ENV) or "").strip()

    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV, "us-east-1")
    bucket_artifacts = os.environ.get(S3_BUCKET_ARTIFACTS_ENV)

    return {
        "rhoai_kfp_url": kfp_url.strip().rstrip("/") if kfp_url else None,
        "rhoai_token": token.strip(),
        "rhoai_project": (project or "kfp-integration-test").strip(),
        "test_data_secret_name": s3_secret,
        "test_data_bucket_name": t_bucket,
        "test_data_key": t_key,
        "input_data_secret_name": s3_secret,
        "input_data_bucket_name": i_bucket,
        "input_data_key": i_key,
        "ogx_secret_name": ogx_secret.strip(),
        "vector_io_provider_id": vector_io.strip(),
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }


def get_autorag_config() -> dict[str, Any] | None:
    """Return connection config only when **all** of ``TEST_DATA_*`` and ``INPUT_DATA_*`` bucket keys are set in env.

    Prefer :func:`get_autorag_connection_config` for the root ``tests/`` suite (JSON-driven data).
    """
    c = get_autorag_connection_config()
    if c is None:
        return None
    if not (
        c["test_data_bucket_name"]
        and c["test_data_key"]
        and c["input_data_bucket_name"]
        and c["input_data_key"]
    ):
        return None
    return c


def describe_rhoai_automl_config_failure() -> str | None:
    """Return ``None`` if :func:`get_rhoai_automl_config` is usable; else a detailed message."""
    load_tests_env()
    required_ns: list[tuple[str, str]] = [
        (RHOAI_URL_ENV, "OpenShift/Kubernetes API URL (for namespace + secrets)"),
        (RHOAI_TOKEN_ENV, "API bearer token"),
        (S3_ENDPOINT_ENV, "S3 endpoint URL"),
        (S3_ACCESS_KEY_ENV, "S3 access key"),
        (S3_SECRET_KEY_ENV, "S3 secret key"),
    ]
    missing_ns = [
        f"  - {name} ({why})"
        for name, why in required_ns
        if not (os.environ.get(name) or "").strip()
    ]
    if missing_ns:
        return (
            "Missing environment variables for cluster namespace and S3 secret setup:\n"
            + "\n".join(missing_ns)
        )

    try:
        ns_probe = get_rhoai_namespace_setup_config()
    except ValueError as e:
        return (
            "Invalid OpenShift API TLS settings "
            f"({RHOAI_OPENSHIFT_CA_BUNDLE_PATH_ENV} / {RHOAI_OPENSHIFT_CA_DATA_ENV} / "
            f"{RHOAI_OPENSHIFT_API_INSECURE_TLS_ENV}):\n{e}"
        )

    if ns_probe is None:
        return "Could not build namespace configuration (unexpected); verify RHOAI_* and AWS_* variables."

    if not (os.environ.get(S3_BUCKET_DATA_ENV) or "").strip():
        return (
            f"Missing {S3_BUCKET_DATA_ENV} (bucket for training data uploads and pipeline inputs).\n"
            "See tests/.env.example."
        )

    dspa = get_dspa_config_from_env()
    kfp_set = bool((os.environ.get(RHOAI_KFP_URL_ENV) or "").strip())
    if not kfp_set and not (dspa and dspa.get("create")):
        return (
            "Kubeflow Pipelines API URL is not configured:\n"
            f"  - Set {RHOAI_KFP_URL_ENV} to the Data Science Pipelines route, **or**\n"
            f"  - Set {RHOAI_CREATE_DSPA_ENV}=true so the suite can create a DSPA and use the ds-pipeline route.\n"
            "See tests/.env.example."
        )

    if get_rhoai_automl_config() is None:
        return "AutoML configuration is still incomplete after validation (internal check); see tests/.env.example."
    return None


def describe_autorag_connection_config_failure() -> str | None:
    """Return ``None`` if :func:`get_autorag_connection_config` works; else explain gaps."""
    load_tests_env()
    kfp_url = (
        os.environ.get(RHOAI_KFP_URL_ENV) or os.environ.get(RHOAI_KFP_URL_ENV_ALT) or ""
    ).strip()
    token = (
        os.environ.get(RHOAI_TOKEN_ENV) or os.environ.get(RHOAI_TOKEN_ENV_ALT) or ""
    ).strip()
    ogx_secret = (os.environ.get(OGX_SECRET_ENV) or "").strip()
    vector_io = (os.environ.get(VECTOR_IO_PROVIDER_ENV) or "").strip()
    dspa = get_dspa_config_from_env()

    lines: list[str] = []
    if not token:
        lines.append(
            f"  - {RHOAI_TOKEN_ENV} or {RHOAI_TOKEN_ENV_ALT} (KFP / cluster token)"
        )
    if not ogx_secret:
        lines.append(
            f"  - {OGX_SECRET_ENV} (Kubernetes secret with OGX client settings)"
        )
    if not vector_io:
        lines.append(
            f"  - {VECTOR_IO_PROVIDER_ENV} (registered vector I/O provider id)"
        )
    if not kfp_url and not (dspa and dspa.get("create")):
        lines.append(
            f"  - {RHOAI_KFP_URL_ENV} or {RHOAI_KFP_URL_ENV_ALT} (pipeline API URL), "
            f"or {RHOAI_CREATE_DSPA_ENV}=true"
        )
    if lines:
        return "AutoRAG integration requires:\n" + "\n".join(lines)

    if get_autorag_connection_config() is None:
        return "AutoRAG connection config is incomplete (unexpected); see tests/.env.example."
    return None


def describe_autorag_integration_failure() -> str | None:
    """Return ``None`` if the full AutoRAG test preconditions hold; else a detailed message."""
    conn_err = describe_autorag_connection_config_failure()
    if conn_err is not None:
        return conn_err

    from autox_tests.lib.config_loaders import get_autorag_configs_for_run

    conn = get_autorag_connection_config()
    assert conn is not None  # follows describe_autorag_connection_config_failure

    configs = get_autorag_configs_for_run()
    needs_upload = any(c.data_mode == "upload" for c in configs)
    needs_existing = any(c.data_mode == "existing_s3" for c in configs)

    s3cfg = get_s3_boto_config_from_env()
    if needs_upload or needs_existing:
        if not s3cfg:
            return (
                "Selected JSON configs use upload or existing_s3 data modes; set all of:\n"
                f"  - {S3_ENDPOINT_ENV}\n"
                f"  - {S3_ACCESS_KEY_ENV}\n"
                f"  - {S3_SECRET_KEY_ENV}"
            )

    if needs_upload:
        if not get_default_upload_bucket_name():
            return (
                "upload data mode requires a default bucket:\n"
                f"  - {S3_BUCKET_DATA_ENV} or {TEST_DATA_SOURCE_BUCKET_ENV}"
            )

    tds = get_test_data_source_defaults()
    if needs_existing:
        for c in configs:
            if c.data_mode != "existing_s3":
                continue
            tb = (
                c.test_data_bucket
                or tds.get("bucket")
                or conn.get("test_data_bucket_name")
            )
            ib = (
                c.input_data_bucket
                or tds.get("bucket")
                or conn.get("input_data_bucket_name")
            )
            if not c.test_data_key or not c.input_data_key:
                return (
                    f"Config {c.id!r} (existing_s3): set test_data_key and input_data_key in "
                    "tests/config/autorag_test_configs.json"
                )
            if not tb or not ib:
                return (
                    f"Config {c.id!r} (existing_s3): set test_data_bucket and input_data_bucket in JSON, "
                    f"or set {TEST_DATA_SOURCE_BUCKET_ENV} / {S3_BUCKET_DATA_ENV} for defaults."
                )

    return None


def autorag_integration_ready() -> tuple[bool, str]:
    """Return whether root AutoRAG tests can run, and a short reason string if not."""
    msg = describe_autorag_integration_failure()
    if msg is None:
        return True, ""
    return False, msg


def autorag_pipeline_arguments(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build KFP pipeline arguments for AutoRAG from integration config."""
    return {
        "test_data_secret_name": cfg["test_data_secret_name"],
        "test_data_bucket_name": cfg["test_data_bucket_name"],
        "test_data_key": cfg["test_data_key"],
        "input_data_secret_name": cfg["input_data_secret_name"],
        "input_data_bucket_name": cfg["input_data_bucket_name"],
        "input_data_key": cfg["input_data_key"],
        "ogx_secret_name": cfg["ogx_secret_name"],
        "vector_io_provider_id": cfg["vector_io_provider_id"],
    }
