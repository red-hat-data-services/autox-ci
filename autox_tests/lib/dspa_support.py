"""DataSciencePipelinesApplication (DSPA) helpers for root OpenShift AI tests."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

_ROUTE_GROUP = "route.openshift.io"
_ROUTE_VERSION = "v1"
_ROUTE_PLURAL = "routes"


def _brief_dsp_conditions(conditions: list[Any]) -> str:
    """Format DSPA status conditions for progress (type and status only; no API messages)."""
    if not conditions:
        return "no conditions in status yet"
    parts: list[str] = []
    for c in conditions[:8]:
        t = c.get("type") or "?"
        st = c.get("status") or "?"
        parts.append(f"{t}={st}")
    return "; ".join(parts)


def _parse_object_storage_endpoint(endpoint: str) -> tuple[str, str, str]:
    """Split an S3 API endpoint into ``(scheme, hostname, port)`` for DSPA ``externalStorage``."""
    raw = (endpoint or "").strip()
    if not raw:
        raise ValueError("S3 endpoint is empty")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    if scheme not in ("http", "https"):
        scheme = "https"
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"Could not parse host from S3 endpoint {endpoint!r}")
    port = str(parsed.port) if parsed.port else ""
    return scheme, host, port


def create_datascience_pipelines_application(
    namespace: str,
    dspa_cfg: dict[str, Any],
    *,
    kubeconfig_path: str | None,
    object_storage_endpoint: str | None = None,
    object_storage_region: str | None = None,
    object_storage_secret_name: str | None = None,
    object_storage_bucket: str | None = None,
    resource_name: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Create a DataSciencePipelinesApplication CR; on 409 return existing CR.

    Returns ``(cr, None)`` on success, or ``(None, error_message)`` on failure.

    If ``progress`` is set, it is called with short human-readable status lines (for terminal output).
    """
    # Validate required parameters
    if not namespace or not isinstance(namespace, str):
        return (None, "namespace must be a non-empty string")
    if not dspa_cfg or not isinstance(dspa_cfg, dict):
        return (None, "dspa_cfg must be a non-empty dict")

    required_keys = ["api_group", "api_version", "plural"]
    missing = [k for k in required_keys if k not in dspa_cfg]
    if missing:
        return (None, f"dspa_cfg missing required keys: {missing}")

    if progress:
        progress("Loading Kubernetes config (kubeconfig or in-cluster)...")
    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception as e:
        try:
            config.load_incluster_config()
        except Exception as e2:
            return (
                None,
                f"Could not load kubeconfig or in-cluster config: {e!r}; in-cluster: {e2!r}",
            )

    if object_storage_secret_name and object_storage_bucket and object_storage_endpoint:
        # Validate endpoint is not empty before parsing
        if not object_storage_endpoint.strip():
            return (
                None,
                "object_storage_endpoint is empty; provide a valid S3 API URL or omit for internal storage",
            )

        # Parse endpoint and validate structure
        try:
            scheme, host, port = _parse_object_storage_endpoint(object_storage_endpoint)
        except ValueError as e:
            return (None, f"Invalid S3 endpoint: {e}")

        if not host:
            return (
                None,
                f"Invalid object storage endpoint {object_storage_endpoint!r}: could not determine hostname "
                "(use a full URL, e.g. http://minio.minio.svc:9000)",
            )
        external_storage: dict[str, Any] = {
            "basePath": "",
            "bucket": object_storage_bucket,
            "host": host,
            "region": object_storage_region or "",
            "s3CredentialsSecret": {
                "accessKey": "AWS_ACCESS_KEY_ID",
                "secretKey": "AWS_SECRET_ACCESS_KEY",
                "secretName": object_storage_secret_name,
            },
            "scheme": scheme,
        }
        if port:
            external_storage["port"] = port
        disable_hc = (
            (os.environ.get("RHOAI_DSPA_OBJECT_STORAGE_DISABLE_HEALTH_CHECK") or "")
            .strip()
            .lower()
        )
        if disable_hc in ("1", "true", "yes", "on"):
            object_storage = {
                "disableHealthCheck": True,
                "externalStorage": external_storage,
            }
        else:
            object_storage = {"externalStorage": external_storage}
        if progress:
            progress("Configuring external object storage for DSPA")
    else:
        object_storage = {"internal": {}}
        if progress:
            progress("Using internal/default object storage for DSPA")

    spec: dict[str, Any] = {
        "objectStorage": object_storage,
        "podToPodTLS": False,
    }
    dsp_version = (dspa_cfg.get("dsp_version") or "").strip()
    if dsp_version:
        spec["dspVersion"] = dsp_version

    dspa_name = (resource_name or dspa_cfg.get("resource_name") or "dspa").strip()

    managed_pipelines = dspa_cfg.get("managed_pipelines")
    if managed_pipelines is not None:
        api_server: dict[str, Any] = {
            "enableSamplePipeline": False,
            "managedPipelines": managed_pipelines,
        }
        if progress:
            progress("Managed pipelines enabled in DSPA spec")
        spec["apiServer"] = api_server
    elif progress:
        progress("DSPA spec has no managedPipelines block")

    body = {
        "apiVersion": f"{dspa_cfg['api_group']}/{dspa_cfg['api_version']}",
        "kind": "DataSciencePipelinesApplication",
        "metadata": {
            "name": dspa_name,
            "namespace": namespace,
        },
        "spec": spec,
    }
    # Create API client once and reuse for create and potential get
    co = client.CustomObjectsApi()

    try:
        if progress:
            progress("Creating DataSciencePipelinesApplication...")
        created = co.create_namespaced_custom_object(
            group=dspa_cfg["api_group"],
            version=dspa_cfg["api_version"],
            namespace=namespace,
            plural=dspa_cfg["plural"],
            body=body,
        )
        if progress:
            progress(
                "DataSciencePipelinesApplication created; cluster is reconciling the CR."
            )
        return (created, None)
    except ApiException as e:
        if e.status == 409:
            if progress:
                progress("DSPA already exists (409); reusing existing CR")
            try:
                existing = co.get_namespaced_custom_object(
                    group=dspa_cfg["api_group"],
                    version=dspa_cfg["api_version"],
                    namespace=namespace,
                    plural=dspa_cfg["plural"],
                    name=dspa_name,
                )
                if progress:
                    progress("Using existing DataSciencePipelinesApplication.")
                return (existing, None)
            except ApiException as get_e:
                return (None, f"DSPA already exists but get failed: {get_e!r}")
        detail = getattr(e, "body", None)
        if isinstance(detail, str) and detail:
            try:
                import json

                detail = json.loads(detail)
            except Exception:
                pass
        msg_parts = [
            f"DSPA creation failed: HTTP {getattr(e, 'status', '?')}",
            f"reason={getattr(e, 'reason', '')}",
        ]
        if detail and isinstance(detail, dict):
            for key in ("message", "reason", "details"):
                if key in detail and detail[key]:
                    msg_parts.append(f"{key}={detail[key]}")
        else:
            msg_parts.append(f"body={detail!r}")
        return (None, "; ".join(msg_parts))
    except Exception as e:
        return (None, f"DSPA creation failed: {type(e).__name__}: {e!r}")


def wait_for_dspa_ready(
    namespace: str,
    dspa_name: str,
    dspa_cfg: dict[str, Any],
    *,
    kubeconfig_path: str | None,
    timeout_seconds: int = 600,
    progress: Callable[[str], None] | None = None,
    progress_interval_seconds: float = 30.0,
) -> bool:
    """Poll DSPA until status.conditions has type=Ready and status=True."""
    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            return False
    co = client.CustomObjectsApi()
    start = time.monotonic()
    deadline = start + timeout_seconds
    last_progress = start - progress_interval_seconds
    if progress:
        progress(f"Waiting for DSPA Ready=True (timeout {timeout_seconds}s)...")
    while time.monotonic() < deadline:
        try:
            cr = co.get_namespaced_custom_object(
                group=dspa_cfg["api_group"],
                version=dspa_cfg["api_version"],
                namespace=namespace,
                plural=dspa_cfg["plural"],
                name=dspa_name,
            )
        except ApiException:
            now = time.monotonic()
            if progress and now - last_progress >= progress_interval_seconds:
                last_progress = now
                progress("DSPA API get failed (will retry in 10s)...")
            time.sleep(10)
            continue
        status = cr.get("status") or {}
        conditions = status.get("conditions") or []
        for c in conditions:
            if (c.get("type") or "") == "Ready" and (c.get("status") or "") == "True":
                if progress:
                    progress("DSPA reports Ready=True")
                return True
        now = time.monotonic()
        if progress and now - last_progress >= progress_interval_seconds:
            last_progress = now
            elapsed = now - start
            brief = _brief_dsp_conditions(conditions)
            progress(
                f"Still waiting for DSPA Ready=True: {elapsed:.0f}s / {timeout_seconds}s ({brief})"
            )
        time.sleep(10)
    if progress:
        progress(f"Timed out after {timeout_seconds}s waiting for DSPA Ready=True")
    return False


def get_dspa_route_kfp_base_url(
    namespace: str,
    *,
    route_name_prefix: str = "ds-pipeline",
    timeout_seconds: int = 300,
    kubeconfig_path: str | None,
) -> str | None:
    """Return ``https://host/`` for the OpenShift Route exposing the pipeline API."""
    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            return None
    co = client.CustomObjectsApi()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            resp = co.list_namespaced_custom_object(
                group=_ROUTE_GROUP,
                version=_ROUTE_VERSION,
                namespace=namespace,
                plural=_ROUTE_PLURAL,
            )
        except ApiException:
            time.sleep(5)
            continue
        items = resp.get("items") or []
        route = None
        for r in items:
            name = (r.get("metadata") or {}).get("name") or ""
            if name.startswith(route_name_prefix):
                route = r
                break
        if not route and items:
            route = items[0]
        if route:
            host = (route.get("spec") or {}).get("host")
            if not host and (route.get("status") or {}).get("ingress"):
                host = route["status"]["ingress"][0].get("host")
            if host:
                return f"https://{host}".rstrip("/") + "/"
        time.sleep(5)
    return None


def verify_kfp_api_health(
    kfp_url: str,
    *,
    timeout_seconds: int = 30,
    verify_ssl: bool = False,
) -> bool:
    """Verify KFP API is responsive by checking /apis/ endpoint.

    Returns True if the API responds with HTTP 200, False otherwise.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests module not available; skipping KFP health check")
        return True

    endpoint = f"{kfp_url.rstrip('/')}/apis/"
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            resp = requests.get(endpoint, verify=verify_ssl, timeout=10)
            if resp.status_code == 200:
                logger.info("KFP API health check passed: %s", endpoint)
                return True
            logger.debug("KFP API returned status %d (retrying)", resp.status_code)
        except requests.exceptions.RequestException as e:
            logger.debug("KFP API health check failed: %s (retrying)", e)
        time.sleep(2)

    logger.warning("KFP API health check timed out after %ds: %s", timeout_seconds, endpoint)
    return False
