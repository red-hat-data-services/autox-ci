"""OpenShift / Kubernetes helpers for RHOAI root tests (namespace + S3 connection secret)."""

from __future__ import annotations

import base64
import json
import os
import tempfile

import pytest

from autox_tests.lib.settings import (
    should_overwrite_s3_secret_keys,
    should_skip_s3_secret_setup,
)


def build_temp_kubeconfig(
    server_url: str,
    token: str,
    namespace: str = "default",
    *,
    insecure_skip_tls_verify: bool = True,
    certificate_authority_data: str | None = None,
) -> str:
    """Write a minimal kubeconfig to a temp file; return its path.

    If ``certificate_authority_data`` is set (base64-encoded PEM, same as kubeconfig
    ``certificate-authority-data``), the cluster uses TLS verification with that CA.
    Otherwise ``insecure_skip_tls_verify`` controls ``insecure-skip-tls-verify`` (typical for
    self-signed API certificates when you do not install the cluster CA locally).
    """
    import yaml

    server_url = (server_url or "").rstrip("/")
    cluster: dict = {"server": server_url}
    ca = (certificate_authority_data or "").strip()
    if ca:
        cluster["certificate-authority-data"] = ca.replace("\n", "").replace(" ", "")
    else:
        cluster["insecure-skip-tls-verify"] = insecure_skip_tls_verify
    cfg = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "rhoai",
                "cluster": cluster,
            }
        ],
        "users": [{"name": "rhoai", "user": {"token": token or ""}}],
        "contexts": [
            {
                "name": "rhoai",
                "context": {
                    "cluster": "rhoai",
                    "user": "rhoai",
                    "namespace": namespace or "default",
                },
            }
        ],
        "current-context": "rhoai",
    }
    fd, path = tempfile.mkstemp(suffix=".kubeconfig", prefix="rhoai-root-tests-")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    except Exception:
        # Clean up temp file on write failure
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _decode_jwt_sub(token: str) -> str | None:
    """Extract the ``sub`` claim from a JWT without signature verification.

    Note: Does not validate token - only extracts payload for ServiceAccount detection.
    Returns None on any parsing error without exposing token in exception.
    """
    try:
        # Validate basic structure before processing to avoid token exposure in exceptions
        if not token or not isinstance(token, str):
            return None

        parts = token.strip().split(".")
        if len(parts) != 3:
            return None

        # Safe: only the payload is processed, not the full token
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = base64.urlsafe_b64decode(payload_b64)
        data = json.loads(payload)
        return data.get("sub")
    except (ValueError, json.JSONDecodeError, KeyError, UnicodeDecodeError):
        # Return None without re-raising - avoids token appearing in stack traces
        return None


def _parse_service_account_sub(sub: str) -> tuple[str, str] | None:
    """Parse ``system:serviceaccount:<ns>:<name>`` into ``(namespace, name)``; None otherwise."""
    if not sub or not isinstance(sub, str):
        return None
    prefix = "system:serviceaccount:"
    if not sub.startswith(prefix):
        return None
    rest = sub[len(prefix) :].strip()
    parts = rest.split(":")
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())


def _ensure_admin_role_for_sa_in_namespace(
    rbac_v1,
    namespace: str,
    sa_namespace: str,
    sa_name: str,
    binding_name: str = "rhoai-root-tests-admin",
) -> None:
    """Create or replace an admin RoleBinding for a ServiceAccount in the target namespace."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    role_ref = client.V1RoleRef(
        api_group="rbac.authorization.k8s.io",
        kind="ClusterRole",
        name="admin",
    )
    subject = client.RbacV1Subject(
        kind="ServiceAccount",
        name=sa_name,
        namespace=sa_namespace,
    )
    body = client.V1RoleBinding(
        api_version="rbac.authorization.k8s.io/v1",
        kind="RoleBinding",
        metadata=client.V1ObjectMeta(name=binding_name),
        role_ref=role_ref,
        subjects=[subject],
    )
    try:
        rbac_v1.create_namespaced_role_binding(namespace, body)
    except ApiException as e:
        if e.status == 409:
            rbac_v1.replace_namespaced_role_binding(binding_name, namespace, body)
        else:
            raise


def _s3_connection_secret_metadata(secret_name: str, existing: object | None = None):
    """Metadata so the RHOAI dashboard keeps recognizing an S3 data connection."""
    from kubernetes import client

    labels = {
        "opendatahub.io/managed": "true",
        "opendatahub.io/dashboard": "true",
    }
    annotations = {
        "opendatahub.io/connection-type": "s3",
        "opendatahub.io/connection-type-protocol": "s3",
        "opendatahub.io/connection-type-ref": "s3",
        "openshift.io/display-name": secret_name,
    }
    if existing is not None:
        labels = {**(getattr(existing, "labels", None) or {}), **labels}
        annotations = {**(getattr(existing, "annotations", None) or {}), **annotations}
    return client.V1ObjectMeta(
        name=secret_name,
        labels=labels,
        annotations=annotations,
    )


def ensure_rhoai_project_and_s3_secret(
    rhoai_namespace_config: dict,
    temp_kubeconfig_path: str | None,
) -> str:
    """Create OpenShift project/namespace if needed and apply the S3 connection secret.

    ``rhoai_namespace_config`` must include ``rhoai_project``, ``rhoai_token``, ``s3_secret_name``,
    and ``s3_endpoint`` / ``s3_access_key`` / ``s3_secret_key`` / ``s3_region`` (same shape as
    :func:`tests.lib.settings.get_rhoai_namespace_setup_config`).
    """
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        pytest.fail(
            "kubernetes Python client is required for OpenShift AI integration tests. "
            "Install with: pip install kubernetes  (or pip install -e '.[test_rhoai]')."
        )

    project_name = rhoai_namespace_config["rhoai_project"]
    secret_name = rhoai_namespace_config["s3_secret_name"]
    token = rhoai_namespace_config.get("rhoai_token")

    try:
        if temp_kubeconfig_path:
            config.load_kube_config(config_file=temp_kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            pytest.fail(
                "Could not load kubeconfig (see RHOAI_URL / temp kubeconfig) or in-cluster config. "
                "For local runs, set RHOAI_URL and RHOAI_TOKEN so a kubeconfig can be built."
            )

    v1 = client.CoreV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()

    project_request_group = "project.openshift.io"
    project_request_version = "v1"
    project_request_plural = "projectrequests"
    project_just_created = False
    try:
        co = client.CustomObjectsApi()
        body = {
            "apiVersion": f"{project_request_group}/{project_request_version}",
            "kind": "ProjectRequest",
            "metadata": {"name": project_name},
        }
        co.create_cluster_custom_object(
            group=project_request_group,
            version=project_request_version,
            plural=project_request_plural,
            body=body,
        )
        project_just_created = True
    except ApiException as e:
        if e.status == 409:
            pass
        elif e.status in (404, 403):
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=project_name)
            )
            try:
                v1.create_namespace(namespace)
                project_just_created = True
            except ApiException as e2:
                if e2.status != 409:
                    raise
        else:
            raise

    if should_skip_s3_secret_setup():
        return project_name

    existing_meta = None
    try:
        existing_meta = v1.read_namespaced_secret(secret_name, project_name).metadata
    except ApiException as e:
        if e.status != 404:
            raise

    metadata = _s3_connection_secret_metadata(secret_name, existing_meta)
    string_data = {
        "AWS_ACCESS_KEY_ID": rhoai_namespace_config["s3_access_key"],
        "AWS_SECRET_ACCESS_KEY": rhoai_namespace_config["s3_secret_key"],
        "AWS_S3_ENDPOINT": rhoai_namespace_config["s3_endpoint"],
        "AWS_DEFAULT_REGION": rhoai_namespace_config["s3_region"],
    }
    secret = client.V1Secret(
        metadata=metadata,
        type="Opaque",
        string_data=string_data,
    )

    def _metadata_patch_body() -> dict:
        """Build patch body that updates only labels/annotations, not credentials."""
        return {
            "metadata": {
                "name": metadata.name,
                "labels": metadata.labels or {},
                "annotations": metadata.annotations or {},
            }
        }

    def _upsert_s3_secret() -> None:
        """Create or update S3 secret; respect RHOAI_S3_SECRET_OVERWRITE_KEYS setting."""
        should_overwrite = should_overwrite_s3_secret_keys()

        if existing_meta is not None:
            # Secret exists - either replace with new credentials or patch metadata only
            if should_overwrite:
                v1.replace_namespaced_secret(secret_name, project_name, secret)
            else:
                # Patch labels/annotations only; keep existing credentials from UI/dashboard
                v1.patch_namespaced_secret(secret_name, project_name, _metadata_patch_body())
        else:
            # Secret doesn't exist - create it
            try:
                v1.create_namespaced_secret(project_name, secret)
            except ApiException as e:
                if e.status == 409:
                    # Race condition: another process created it between our read and create
                    if should_overwrite:
                        v1.replace_namespaced_secret(secret_name, project_name, secret)
                    else:
                        v1.patch_namespaced_secret(secret_name, project_name, _metadata_patch_body())
                else:
                    raise

    try:
        _upsert_s3_secret()
    except ApiException as e:
        if e.status != 403:
            raise
        if not (project_just_created and token):
            pytest.fail(
                f"Cannot create secret in namespace {project_name!r}. "
                f"Grant the ServiceAccount 'edit' or 'admin' in that namespace."
            )
        sub = _decode_jwt_sub(token)
        sa_identity = _parse_service_account_sub(sub) if sub else None
        if not sa_identity:
            pytest.fail(
                f"Cannot create secret in namespace {project_name!r} and could not determine "
                f"ServiceAccount from token."
            )
        sa_namespace, sa_name = sa_identity
        try:
            _ensure_admin_role_for_sa_in_namespace(
                rbac_v1, project_name, sa_namespace, sa_name
            )
        except ApiException as rb_e:
            if rb_e.status == 403:
                pytest.fail(
                    "ServiceAccount cannot create RoleBindings in the new project. "
                    "Grant the SA a role that allows creating rolebindings or create the secret manually."
                )
            raise
        try:
            _upsert_s3_secret()
        except ApiException as e2:
            if e2.status == 403:
                pytest.fail(
                    f"Cannot create secret in namespace {project_name!r} even after creating "
                    f"admin RoleBinding."
                )
            raise

    return project_name


def ensure_automl_project_and_s3_secret(
    rhoai_automl_config: dict,
    temp_kubeconfig_path: str | None,
) -> str:
    """Backward-compatible name for :func:`ensure_rhoai_project_and_s3_secret`."""
    return ensure_rhoai_project_and_s3_secret(rhoai_automl_config, temp_kubeconfig_path)
