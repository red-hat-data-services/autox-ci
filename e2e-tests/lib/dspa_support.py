"""DataSciencePipelinesApplication (DSPA) helpers for root OpenShift AI tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from kubernetes import client, config
from kubernetes.client.rest import ApiException

_ROUTE_GROUP = "route.openshift.io"
_ROUTE_VERSION = "v1"
_ROUTE_PLURAL = "routes"


def _brief_dsp_conditions(conditions: list[Any]) -> str:
    if not conditions:
        return "no conditions in status yet"
    parts: list[str] = []
    for c in conditions[:8]:
        t = c.get("type") or "?"
        st = c.get("status") or "?"
        parts.append(f"{t}={st}")
    return "; ".join(parts)


def _parse_object_storage_endpoint(endpoint: str) -> tuple[str, str, str]:
    """Split an S3 API endpoint into ``(scheme, hostname, port)`` for DSPA ``externalStorage``.

    Accepts a full URL (``http://`` or ``https://``) or a ``host[:port]`` string (treated as HTTPS).
    """
    raw = (endpoint or "").strip()
    if not raw:
        return ("https", "", "")
    if "://" not in raw:
        raw = f"https://{raw}"
    p = urlparse(raw)
    scheme = (p.scheme or "https").lower()
    if scheme not in ("http", "https"):
        scheme = "https"
    host = p.hostname or ""
    port = str(p.port) if p.port else ""
    return (scheme, host, port)


def create_datascience_pipelines_application(
    namespace: str,
    dspa_cfg: dict[str, Any],
    *,
    kubeconfig_path: str | None,
    object_storage_endpoint: str | None = None,
    object_storage_region: str | None = None,
    object_storage_secret_name: str | None = None,
    object_storage_bucket: str | None = None,
    resource_name: str = "dspa",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Create a DataSciencePipelinesApplication CR; on 409 return existing CR.

    Returns ``(cr, None)`` on success, or ``(None, error_message)`` on failure.

    If ``progress`` is set, it is called with short human-readable status lines (for terminal output).
    """
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
        scheme, host, port = _parse_object_storage_endpoint(object_storage_endpoint)
        if not host:
            return (
                None,
                f"Invalid object storage endpoint {object_storage_endpoint!r}: could not determine hostname "
                "(use a full URL, e.g. http://minio.minio.svc:9000)",
            )
        object_storage: dict[str, Any] = {
            "externalStorage": {
                "basePath": "",
                "bucket": object_storage_bucket,
                "host": host,
                "port": port,
                "region": object_storage_region or "",
                "s3CredentialsSecret": {
                    "accessKey": "AWS_ACCESS_KEY_ID",
                    "secretKey": "AWS_SECRET_ACCESS_KEY",
                    "secretName": object_storage_secret_name,
                },
                "scheme": scheme,
            }
        }
        if progress:
            port_s = f":{port}" if port else ""
            progress(
                f"DSPA will use external S3 storage ({scheme}://{host}{port_s}, bucket={object_storage_bucket!r})."
            )
    else:
        object_storage = {"internal": {}}
        if progress:
            progress("DSPA will use internal/default object storage (no external S3 block in CR).")

    body = {
        "apiVersion": f"{dspa_cfg['api_group']}/{dspa_cfg['api_version']}",
        "kind": "DataSciencePipelinesApplication",
        "metadata": {
            "name": resource_name,
            "namespace": namespace,
        },
        "spec": {
            "objectStorage": object_storage,
            "podToPodTLS": False,
        },
    }
    dspa_name = body["metadata"]["name"]
    try:
        if progress:
            progress(
                f"Creating DataSciencePipelinesApplication {dspa_name!r} in namespace {namespace!r}..."
            )
        co = client.CustomObjectsApi()
        created = co.create_namespaced_custom_object(
            group=dspa_cfg["api_group"],
            version=dspa_cfg["api_version"],
            namespace=namespace,
            plural=dspa_cfg["plural"],
            body=body,
        )
        if progress:
            progress("DataSciencePipelinesApplication created; cluster is reconciling the CR.")
        return (created, None)
    except ApiException as e:
        if e.status == 409:
            if progress:
                progress(
                    f"DSPA {dspa_name!r} already exists (409); fetching existing CR from namespace {namespace!r}..."
                )
            try:
                co = client.CustomObjectsApi()
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
        progress(
            f"Waiting for DSPA {dspa_name!r} Ready=True in namespace {namespace!r} "
            f"(timeout {timeout_seconds}s)..."
        )
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
                    progress(f"DSPA {dspa_name!r} reports Ready=True.")
                return True
        now = time.monotonic()
        if progress and now - last_progress >= progress_interval_seconds:
            last_progress = now
            elapsed = now - start
            brief = _brief_dsp_conditions(conditions)
            progress(
                f"Still waiting for Ready... {elapsed:.0f}s / {timeout_seconds}s — {brief}"
            )
        time.sleep(10)
    if progress:
        progress(f"Timed out after {timeout_seconds}s; Ready=True not observed.")
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
