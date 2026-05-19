"""Shared utilities for AutoML functional tests."""

import json
import logging
import os
import re
import secrets
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KServe / Kubernetes constants
# ---------------------------------------------------------------------------

_KSERVE_GROUP = "serving.kserve.io"
_KSERVE_ISVC_VERSION = "v1beta1"
_KSERVE_SR_VERSION = "v1alpha1"
_KSERVE_ISVC_PLURAL = "inferenceservices"
_KSERVE_SR_PLURAL = "servingruntimes"

_K8S_CALL_TIMEOUT = 30  # seconds per Kubernetes API call
_HW_PROFILE_FETCH_ATTEMPTS = 6
_HW_PROFILE_FETCH_DELAY_SECONDS = 3.0

# Expected primary metric key per task type (tabular pipeline).
TASK_PRIMARY_METRICS_TABULAR: dict[str, str] = {
    "regression": "r2",
    "binary": "accuracy",
    "multiclass": "accuracy",
}

# Expected primary metric key for timeseries models (AutoGluon uses MASE by default).
TS_PRIMARY_METRIC = "MASE"


def make_kfp_client(config):
    """Create a KFP client from a config dict; returns None if config is None."""
    if config is None:
        return None
    import kfp

    host = config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return kfp.Client(
        host=host,
        namespace=config["rhoai_project"],
        existing_token=config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )


def make_s3_client(config):
    """Create a boto3 S3 client from a config dict; returns None if not configured."""
    if config is None or not config.get("s3_endpoint"):
        return None
    try:
        import boto3
    except ImportError:
        return None
    return boto3.client(
        "s3",
        endpoint_url=config["s3_endpoint"],
        aws_access_key_id=config["s3_access_key"],
        aws_secret_access_key=config["s3_secret_key"],
        region_name=config["s3_region"],
    )


def make_run_name(prefix: str) -> str:
    """Return a unique run name: ``<prefix>-<6 hex chars>-<YYYYMMDD-HHMMSS>``."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{hex_part}-{time_part}"


def run_pipeline_and_wait(client, compiled_path, arguments, timeout):
    """Submit a pipeline run and block until completion; return ``(run_id, detail)``."""
    run_name = make_run_name("automl-func")
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
        enable_caching=False,
    )
    run_id = run.run_id
    detail = client.wait_for_run_completion(run_id, timeout=timeout)
    return run_id, detail


def normalize_state(state):
    """Normalize a state value (str or enum) to an uppercase string."""
    if state is None:
        return None
    return str(getattr(state, "name", state)).upper()


def get_run_state(detail):
    """Extract the run state string from a KFP run detail object."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    return normalize_state(state)


def run_succeeded(detail):
    """Return True if the run finished with SUCCEEDED state."""
    return get_run_state(detail) == "SUCCEEDED"


def run_failed(detail):
    """Return True if the run finished with FAILED state."""
    return get_run_state(detail) == "FAILED"


def validate_artifacts_in_s3(s3_client, bucket, prefix):
    """List and categorize S3 artifacts under prefix.

    Returns dict with keys: model_keys, leaderboard_keys, notebook_keys, all_keys.
    """
    result = {
        "model_keys": [],
        "leaderboard_keys": [],
        "notebook_keys": [],
        "all_keys": [],
    }
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                result["all_keys"].append(key)
                lower_key = key.lower()
                if key.endswith(".pkl") or "model" in lower_key:
                    result["model_keys"].append(key)
                if "leaderboard" in lower_key or key.endswith(".html"):
                    result["leaderboard_keys"].append(key)
                if key.endswith(".ipynb"):
                    result["notebook_keys"].append(key)
    except Exception as e:
        raise AssertionError(
            f"Failed to list S3 artifacts under s3://{bucket}/{prefix}: {e}"
        ) from e
    return result


def collect_failure_details(client, run_id, config=None):
    """Collect failure details from a failed pipeline run."""
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]

    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)

        run_error = getattr(run_obj, "error", None)
        if run_error:
            error_msg = getattr(run_error, "message", str(run_error))
            lines.append(f"\nRUN ERROR: {error_msg}")

        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None

        if task_list:
            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(
                    task, "task_id", "?"
                )
                state = getattr(task, "state", None)
                state_str = normalize_state(state) or "NOT_STARTED"

                if name in ("root", "executor") or name.endswith("-driver"):
                    continue

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")
                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")
                else:
                    lines.append(f"  TASK: {name} -- {state_str}")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details: {e}]")

    lines.append("=" * 80)
    return "\n".join(lines)


def get_failed_task_names(client, run_id: str) -> list[str]:
    """Return display names of user-visible FAILED/ERROR tasks from a pipeline run."""
    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)
        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None
        if not task_list:
            return []
        failed = []
        for task in task_list:
            name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
            state = getattr(task, "state", None)
            state_str = normalize_state(state) or ""
            if name in ("root", "executor") or name.endswith("-driver"):
                continue
            if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                failed.append(name)
        return failed
    except Exception as exc:
        logger.warning("Could not get failed task names for run %s: %s", run_id, exc)
        return []


def list_s3_objects(s3_client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns list of {Key, Size, ...} dicts."""
    paginator = s3_client.get_paginator("list_objects_v2")
    return [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents") or []
    ]


def read_s3_json(s3_client, bucket: str, key: str) -> dict | None:
    """Read and parse a JSON file from S3; returns None on failure."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to read s3://%s/%s: %s", bucket, key, e)
        return None


def delete_s3_objects(s3_client, bucket: str, keys: list[str]) -> int:
    """Delete objects from S3 in batches. Returns count of deleted objects."""
    deleted = 0
    batch_size = 1000
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        delete_req = {"Objects": [{"Key": k} for k in batch], "Quiet": True}
        try:
            s3_client.delete_objects(Bucket=bucket, Delete=delete_req)
            deleted += len(batch)
        except Exception as e:
            logger.warning(
                "Failed to delete %d objects from s3://%s: %s", len(batch), bucket, e
            )
    return deleted


def collect_model_metrics_and_sizes(
    s3_client, bucket: str, run_prefix: str
) -> list[dict]:
    """Scan S3 artifacts for metrics.json files and compute per-model predictor size.

    Returns list of dicts with keys:
        {model_name, metrics, artifact_key, total_predictor_size_bytes,
         total_predictor_size_mb, notebook_key}
    """
    objects = list_s3_objects(s3_client, bucket, run_prefix)

    metrics_by_model: dict[str, dict] = {}
    for obj in objects:
        key = obj["Key"]
        if key.endswith("metrics.json") and "/metrics/metrics.json" in key:
            data = read_s3_json(s3_client, bucket, key)
            if data is not None:
                # Path: .../ModelName/metrics/metrics.json → model name is 3 parts up
                parts = key.rsplit("/", 3)
                model_name = parts[-3] if len(parts) >= 3 else "unknown"
                metrics_by_model[model_name] = {
                    "model_name": model_name,
                    "metrics": data,
                    "artifact_key": key,
                    "total_predictor_size_bytes": 0,
                    "notebook_key": None,
                }

    for obj in objects:
        key = obj["Key"]
        size = obj.get("Size", 0)
        for model_name, entry in metrics_by_model.items():
            if f"/{model_name}/predictor/" in key:
                entry["total_predictor_size_bytes"] += size
            if (
                key.endswith("automl_predictor_notebook.ipynb")
                and f"/{model_name}/notebooks/" in key
            ):
                entry["notebook_key"] = key

    for entry in metrics_by_model.values():
        entry["total_predictor_size_mb"] = round(
            entry["total_predictor_size_bytes"] / (1024 * 1024), 2
        )

    return list(metrics_by_model.values())


def find_leaderboard_html(
    s3_client, bucket: str, run_prefix: str
) -> tuple[str | None, str | None]:
    """Find the leaderboard HTML artifact (html_artifact in key path) in S3.

    Returns (s3_key, html_content) on success or (None, None) if not found.
    """
    objects = list_s3_objects(s3_client, bucket, run_prefix)
    for obj in objects:
        key = obj["Key"]
        if "html_artifact" in key:
            try:
                resp = s3_client.get_object(Bucket=bucket, Key=key)
                content = resp["Body"].read().decode("utf-8")
                return key, content
            except Exception as exc:
                logger.warning(
                    "Failed to read leaderboard HTML s3://%s/%s: %s", bucket, key, exc
                )
                return key, None
    return None, None


def find_test_dataset_csv(s3_client, bucket: str, run_prefix: str) -> str | None:
    """Find the sampled_test_dataset artifact produced by the data loader component in S3."""
    objects = list_s3_objects(s3_client, bucket, run_prefix)
    for obj in objects:
        if "sampled_test_dataset" in obj["Key"]:
            return obj["Key"]
    return None


# ---------------------------------------------------------------------------
# KServe deployment helpers
# ---------------------------------------------------------------------------


def make_isvc_name(scenario_id: str, run_id: str) -> str:
    """Return a valid Kubernetes name for an InferenceService (≤36 chars, DNS label safe).

    Layout: ``automl-`` (7) + clean (≤20) + ``-`` (1) + run_id[:8] (8) = ≤36.
    The cap at 36 chars keeps the odh-model-controller-generated
    ``{name}-kube-rbac-proxy-sar-config`` volume name within the 63-char limit.
    """
    clean = re.sub(r"[^a-z0-9]+", "-", scenario_id.lower()).strip("-")[:20]
    return f"automl-{clean}-{run_id[:8]}"


def load_k8s_config(kubeconfig_path: str | None) -> None:
    """Load kubernetes config from a file or fall back to in-cluster config."""
    from kubernetes import config

    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        config.load_incluster_config()


def find_top_model_predictor_prefix(
    s3_client, bucket: str, run_prefix: str, model_name: str
) -> str | None:
    """Find the S3 prefix (with trailing slash) for a model's predictor directory."""
    objects = list_s3_objects(s3_client, bucket, run_prefix)
    needle = f"/{model_name}/predictor/"
    for obj in objects:
        key = obj["Key"]
        idx = key.find(needle)
        if idx != -1:
            return key[: idx + len(needle)]
    return None


def create_kserve_s3_secret(
    v1, namespace: str, secret_name: str, bucket: str, config: dict
) -> None:
    """Create (or replace) an RHOAI Data Connection secret for KServe storage initializer."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    string_data: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": config["s3_access_key"],
        "AWS_SECRET_ACCESS_KEY": config["s3_secret_key"],
        "AWS_S3_ENDPOINT": config["s3_endpoint"],
        "AWS_S3_BUCKET": bucket,
    }
    secret = client.V1Secret(
        metadata=client.V1ObjectMeta(
            name=secret_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
            annotations={
                "opendatahub.io/connection-type": "s3",
                "opendatahub.io/connection-type-protocol": "s3",
                "opendatahub.io/connection-type-ref": "s3",
                "openshift.io/display-name": secret_name,
            },
        ),
        type="Opaque",
        string_data=string_data,
    )
    try:
        v1.create_namespaced_secret(
            namespace, secret, _request_timeout=_K8S_CALL_TIMEOUT
        )
    except ApiException as e:
        if e.status == 409:
            v1.replace_namespaced_secret(
                secret_name, namespace, secret, _request_timeout=_K8S_CALL_TIMEOUT
            )
        else:
            raise


def create_connection_sa(v1, namespace: str, secret_name: str) -> str:
    """Create the companion ServiceAccount required by odh-model-controller for a Data Connection.

    Returns the ServiceAccount name (``{secret_name}-sa``).
    """
    from kubernetes import client as k8s_client
    from kubernetes.client.rest import ApiException

    sa_name = f"{secret_name}-sa"
    sa = k8s_client.V1ServiceAccount(
        metadata=k8s_client.V1ObjectMeta(name=sa_name, namespace=namespace),
        secrets=[k8s_client.V1ObjectReference(name=secret_name)],
    )
    try:
        v1.create_namespaced_service_account(
            namespace, sa, _request_timeout=_K8S_CALL_TIMEOUT
        )
        logger.info("Created ServiceAccount %r in namespace %r", sa_name, namespace)
    except ApiException as exc:
        if exc.status == 409:
            logger.info("ServiceAccount %r already exists — reusing", sa_name)
        else:
            raise
    return sa_name


def create_connection_rbac(
    rbac_v1, namespace: str, sa_name: str, secret_name: str
) -> str:
    """Create Role + RoleBinding so the SA can GET the Data Connection secret.

    Returns the RoleBinding name (same as ``sa_name``).
    """
    from kubernetes import client as k8s_client
    from kubernetes.client.rest import ApiException

    role_name = sa_name
    role = k8s_client.V1Role(
        metadata=k8s_client.V1ObjectMeta(
            name=role_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
        ),
        rules=[
            k8s_client.V1PolicyRule(
                api_groups=[""],
                resources=["secrets"],
                verbs=["get"],
                resource_names=[secret_name],
            )
        ],
    )
    try:
        rbac_v1.create_namespaced_role(
            namespace, role, _request_timeout=_K8S_CALL_TIMEOUT
        )
    except ApiException as exc:
        if exc.status != 409:
            raise

    rb_name = sa_name
    role_binding = k8s_client.V1RoleBinding(
        metadata=k8s_client.V1ObjectMeta(
            name=rb_name,
            namespace=namespace,
            labels={
                "opendatahub.io/managed": "true",
                "opendatahub.io/dashboard": "true",
            },
        ),
        subjects=[
            k8s_client.RbacV1Subject(
                kind="ServiceAccount", name=sa_name, namespace=namespace
            )
        ],
        role_ref=k8s_client.V1RoleRef(
            api_group="rbac.authorization.k8s.io", kind="Role", name=role_name
        ),
    )
    try:
        rbac_v1.create_namespaced_role_binding(
            namespace, role_binding, _request_timeout=_K8S_CALL_TIMEOUT
        )
    except ApiException as exc:
        if exc.status != 409:
            raise
    return rb_name


def ensure_serving_runtime(
    co, namespace: str, runtime_name: str, serving_image: str
) -> bool:
    """Create the AutoGluon ServingRuntime if it does not exist.

    Returns True if newly created, False if it already existed.
    """
    from kubernetes.client.rest import ApiException

    try:
        co.get_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_SR_VERSION,
            namespace=namespace,
            plural=_KSERVE_SR_PLURAL,
            name=runtime_name,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        logger.info(
            "ServingRuntime %r already exists — skipping creation", runtime_name
        )
        return False
    except ApiException as e:
        if e.status != 404:
            raise

    runtime = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_SR_VERSION}",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": namespace,
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": "AutoGluon ServingRuntime for KServe",
                "openshift.io/display-name": "AutoGluon ServingRuntime for KServe",
            },
        },
        "spec": {
            "annotations": {
                "prometheus.kserve.io/path": "/metrics",
                "prometheus.kserve.io/port": "8080",
            },
            "supportedModelFormats": [{"name": "autogluon", "version": "1"}],
            "protocolVersions": ["v1", "v2"],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": serving_image,
                    "args": [
                        "--model_name={{.Name}}",
                        "--model_dir=/mnt/models",
                        "--http_port=8080",
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "privileged": False,
                        "runAsNonRoot": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {
                        "requests": {"cpu": "1", "memory": "2Gi"},
                        "limits": {"cpu": "1", "memory": "2Gi"},
                    },
                }
            ],
        },
    }
    co.create_namespaced_custom_object(
        group=_KSERVE_GROUP,
        version=_KSERVE_SR_VERSION,
        namespace=namespace,
        plural=_KSERVE_SR_PLURAL,
        body=runtime,
        _request_timeout=_K8S_CALL_TIMEOUT,
    )
    logger.info("Created ServingRuntime %r in %r", runtime_name, namespace)
    return True


def ensure_deployment_storage_annotations(
    apps_v1,
    namespace: str,
    isvc_name: str,
    storage_key: str,
    artifacts_bucket: str,
    storage_path: str,
    wait_seconds: int = 60,
) -> bool:
    """Wait for the predictor Deployment and ensure it has storage initializer annotations.

    If odh-model-controller did not set them, patches the Deployment directly.
    Returns True if the Deployment was found, False if it never appeared.
    """
    from kubernetes.client.rest import ApiException

    deployment_name = f"{isvc_name}-predictor"
    deadline = time.monotonic() + wait_seconds
    dep = None
    while time.monotonic() < deadline:
        try:
            dep = apps_v1.read_namespaced_deployment(
                deployment_name, namespace, _request_timeout=_K8S_CALL_TIMEOUT
            )
            break
        except ApiException as exc:
            if exc.status == 404:
                time.sleep(5)
            else:
                break
        except Exception:
            break

    if dep is None:
        logger.warning(
            "Deployment %r did not appear within %ds", deployment_name, wait_seconds
        )
        return False

    ann = dep.metadata.annotations or {}
    if ann.get("internal.serving.kserve.io/storage-spec-key") and ann.get(
        "internal.serving.kserve.io/storage-initializer-sourceuri"
    ):
        return True

    storage_uri = f"s3://{artifacts_bucket}/{storage_path.rstrip('/')}"
    storage_annotations = {
        "internal.serving.kserve.io/storage-initializer-sourceuri": storage_uri,
        "internal.serving.kserve.io/storage-spec-key": storage_key,
        "internal.serving.kserve.io/storage-spec": "true",
    }
    patch_body = {
        "metadata": {"annotations": storage_annotations},
        "spec": {"template": {"metadata": {"annotations": storage_annotations}}},
    }
    try:
        apps_v1.patch_namespaced_deployment(
            deployment_name, namespace, patch_body, _request_timeout=_K8S_CALL_TIMEOUT
        )
        logger.info(
            "Patched Deployment %r with storage annotations (uri=%r)",
            deployment_name,
            storage_uri,
        )
    except Exception as exc:
        logger.warning("Failed to patch Deployment %r: %s", deployment_name, exc)
    return True


def log_isvc_events(v1, namespace: str, isvc_name: str) -> None:
    """Read and log Kubernetes Events for an InferenceService."""
    try:
        events = v1.list_namespaced_event(
            namespace,
            field_selector=f"involvedObject.name={isvc_name},involvedObject.kind=InferenceService",
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        for evt in events.items or []:
            logger.info(
                "ISVC %r event: type=%s reason=%r message=%r",
                isvc_name,
                evt.type,
                evt.reason or "",
                evt.message or "",
            )
    except Exception as exc:
        logger.warning("Could not read events for ISVC %r: %s", isvc_name, exc)


def create_inference_service(
    co,
    namespace: str,
    isvc_name: str,
    runtime_name: str,
    storage_path: str,
    storage_key: str,
    hardware_profile_name: str = "default-profile",
    hardware_profile_namespace: str = "redhat-ods-applications",
    hardware_profile_resource_version: str = "",
    predictor_cpu: str = "2",
    predictor_memory: str = "4Gi",
    env_vars: dict[str, str] | None = None,
) -> None:
    """Create a KServe InferenceService in RawDeployment mode with an external Route."""
    from kubernetes.client.rest import ApiException

    annotations = {
        "serving.kserve.io/stop": "false",
        "serving.kserve.io/deploymentMode": "RawDeployment",
        "security.opendatahub.io/enable-auth": "true",
        "openshift.io/display-name": isvc_name,
        "openshift.io/description": "",
        "opendatahub.io/connections": storage_key,
        "opendatahub.io/connection-path": storage_path.rstrip("/"),
        "opendatahub.io/model-type": "predictive",
        "opendatahub.io/hardware-profile-name": hardware_profile_name,
        "opendatahub.io/hardware-profile-namespace": hardware_profile_namespace,
    }
    if hardware_profile_resource_version:
        annotations["opendatahub.io/hardware-profile-resource-version"] = (
            hardware_profile_resource_version
        )

    isvc = {
        "apiVersion": f"{_KSERVE_GROUP}/{_KSERVE_ISVC_VERSION}",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "labels": {
                "networking.kserve.io/visibility": "exposed",
                "opendatahub.io/dashboard": "true",
            },
            "annotations": annotations,
        },
        "spec": {
            "predictor": {
                "automountServiceAccountToken": False,
                "serviceAccountName": f"{storage_key}-sa",
                "deploymentStrategy": {"type": "RollingUpdate"},
                "maxReplicas": 1,
                "minReplicas": 1,
                "model": {
                    "modelFormat": {"name": "autogluon", "version": "1"},
                    "name": "",
                    "runtime": runtime_name,
                    "resources": {
                        "requests": {"cpu": predictor_cpu, "memory": predictor_memory},
                        "limits": {"cpu": predictor_cpu, "memory": predictor_memory},
                    },
                    "storage": {
                        "key": storage_key,
                        "path": storage_path.rstrip("/"),
                    },
                    **(
                        {"env": [{"name": k, "value": v} for k, v in env_vars.items()]}
                        if env_vars
                        else {}
                    ),
                },
            }
        },
    }
    try:
        co.create_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            body=isvc,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
    except ApiException as e:
        if e.status != 409:
            raise


def wait_for_isvc_ready(
    co,
    namespace: str,
    isvc_name: str,
    timeout_seconds: int = 300,
    poll_interval: int = 30,
) -> tuple[bool, str | None]:
    """Poll an InferenceService until Ready=True, a terminal failure, or timeout.

    Returns ``(is_ready, blocking_reason)``.
    """
    from kubernetes.client.rest import ApiException

    _BLOCKING_REASONS = frozenset(
        {
            "ServingRuntimeNotFound",
            "NoSupportedRuntime",
            "InvalidStorageSpec",
            "RuntimeNotRecognized",
            "UnsupportedProtocol",
        }
    )

    start = time.monotonic()
    last_cond_fingerprint: frozenset = frozenset()

    while True:
        elapsed = time.monotonic() - start
        try:
            isvc = co.get_namespaced_custom_object(
                group=_KSERVE_GROUP,
                version=_KSERVE_ISVC_VERSION,
                namespace=namespace,
                plural=_KSERVE_ISVC_PLURAL,
                name=isvc_name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
        except ApiException as exc:
            logger.warning(
                "ISVC %r: GET failed (elapsed %.0fs, HTTP %s)",
                isvc_name,
                elapsed,
                exc.status,
            )
        except Exception as exc:
            logger.warning(
                "ISVC %r: GET failed (elapsed %.0fs): %s", isvc_name, elapsed, exc
            )
        else:
            status = isvc.get("status") or {}
            conditions = status.get("conditions") or []

            cond_fingerprint = frozenset(
                (c.get("type"), c.get("status"), c.get("reason", ""))
                for c in conditions
            )
            if cond_fingerprint != last_cond_fingerprint:
                for cond in conditions:
                    logger.info(
                        "ISVC %r condition %s=%s%s",
                        isvc_name,
                        cond.get("type", "?"),
                        cond.get("status", "?"),
                        (f" | reason={cond['reason']}" if cond.get("reason") else "")
                        + (f" | {cond['message']!r}" if cond.get("message") else ""),
                    )
                last_cond_fingerprint = cond_fingerprint

            for cond in conditions:
                if cond.get("status") == "False":
                    reason = cond.get("reason", "")
                    if reason in _BLOCKING_REASONS:
                        blocking = f"{cond.get('type')}=False reason={reason}: {cond.get('message', '')}"
                        logger.error(
                            "ISVC %r: terminal failure — %s", isvc_name, blocking
                        )
                        return False, blocking

            cond_map = {c.get("type"): c.get("status") for c in conditions}
            if cond_map.get("Ready") == "True":
                logger.info("ISVC %r: Ready=True after %.0fs", isvc_name, elapsed)
                return True, None

        if elapsed >= timeout_seconds:
            logger.warning("ISVC %r: timed out after %.0fs", isvc_name, elapsed)
            return False, None

        sleep_secs = min(poll_interval, timeout_seconds - elapsed)
        time.sleep(sleep_secs)


def resolve_isvc_external_url(co, namespace: str, isvc_name: str) -> str | None:
    """Return the external HTTPS URL for an InferenceService, or None if not yet available."""
    from kubernetes.client.rest import ApiException

    try:
        isvc = co.get_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            name=isvc_name,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
        status_url = (isvc.get("status") or {}).get("url", "")
        if status_url.startswith("https://") and ".svc.cluster.local" not in status_url:
            return status_url
    except ApiException:
        pass

    for route_name in (isvc_name, f"{isvc_name}-predictor"):
        try:
            route = co.get_namespaced_custom_object(
                group="route.openshift.io",
                version="v1",
                namespace=namespace,
                plural="routes",
                name=route_name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
            spec = route.get("spec") or {}
            host = spec.get("host")
            if not host:
                ingress = (route.get("status") or {}).get("ingress") or []
                host = ingress[0].get("host") if ingress else None
            if host:
                return f"https://{host}"
        except ApiException as exc:
            if exc.status != 404:
                pass
    return None


def column_sample_to_instances(sample: list[dict]) -> list[dict]:
    """Convert column-oriented [{col: [val, ...]}] to per-row instance dicts with list values.

    AutoGluon KServe server expects ``{col: [val], ...}`` per instance (list-wrapped scalars).
    """
    if not sample:
        return []
    col_data = sample[0]
    n_rows = len(next(iter(col_data.values()), []))
    return [
        {col: [values[i]] for col, values in col_data.items()} for i in range(n_rows)
    ]


def score_inference_service(
    isvc_url: str,
    model_name: str,
    instances: list[dict],
    token: str | None,
    max_retries: int = 5,
    retry_interval_seconds: int = 30,
) -> dict:
    """Send a KServe v1 predict request with retry on 5xx transient errors.

    Returns the parsed JSON response dict.
    Raises RuntimeError after all retries are exhausted.
    """
    predict_url = f"{isvc_url.rstrip('/')}/v1/models/{model_name}:predict"
    payload = json.dumps({"instances": instances}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    last_error: str = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                predict_url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (500, 502, 503, 504) and attempt < max_retries - 1:
                logger.warning(
                    "Scoring attempt %d/%d got %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    last_error,
                    retry_interval_seconds,
                )
                time.sleep(retry_interval_seconds)
                continue
            raise RuntimeError(last_error) from exc
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(retry_interval_seconds)
                continue
            raise RuntimeError(last_error) from exc
    raise RuntimeError(
        f"All {max_retries} scoring attempts failed. Last error: {last_error}"
    )


def fetch_hardware_profile_resource_version(co, namespace: str, name: str) -> str:
    """Fetch ``metadata.resourceVersion`` for a HardwareProfile CR with retries.

    Returns non-empty string or "" if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(_HW_PROFILE_FETCH_ATTEMPTS):
        try:
            obj = co.get_namespaced_custom_object(
                group="infrastructure.opendatahub.io",
                version="v1",
                namespace=namespace,
                plural="hardwareprofiles",
                name=name,
                _request_timeout=_K8S_CALL_TIMEOUT,
            )
            rv = (obj.get("metadata") or {}).get("resourceVersion", "")
            if rv:
                return rv
        except Exception as exc:
            last_exc = exc
        if attempt < _HW_PROFILE_FETCH_ATTEMPTS - 1:
            time.sleep(_HW_PROFILE_FETCH_DELAY_SECONDS)

    logger.error(
        "Could not fetch resourceVersion for HardwareProfile %r in %r after %d attempts. Last error: %s",
        name,
        namespace,
        _HW_PROFILE_FETCH_ATTEMPTS,
        last_exc,
    )
    return ""


def delete_inference_service(co, namespace: str, isvc_name: str) -> None:
    """Delete an InferenceService, silently ignoring 404."""
    from kubernetes.client.rest import ApiException

    try:
        co.delete_namespaced_custom_object(
            group=_KSERVE_GROUP,
            version=_KSERVE_ISVC_VERSION,
            namespace=namespace,
            plural=_KSERVE_ISVC_PLURAL,
            name=isvc_name,
            _request_timeout=_K8S_CALL_TIMEOUT,
        )
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete InferenceService %r: %s", isvc_name, e)


def fetch_pod_logs_str(
    v1, namespace: str, label_selector: str, tail_lines: int = 100
) -> str:
    """Fetch logs from all pods matching *label_selector* and return a formatted string.

    For containers in CrashLoopBackOff or Error state, also fetches the previous
    terminated container's logs so crash output is visible even after a restart.
    """
    lines = []
    _CRASH_REASONS = frozenset({"CrashLoopBackOff", "Error", "OOMKilled"})
    try:
        pod_list = v1.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector, _request_timeout=30
        )
        if not pod_list.items:
            return f"[No pods found with label selector {label_selector!r} in namespace {namespace!r}]"
        lines.append(f"Pod logs for {label_selector!r} ({len(pod_list.items)} pod(s)):")
        for pod in pod_list.items:
            pod_name = pod.metadata.name
            phase = pod.status.phase if pod.status else "unknown"
            lines.append(f"\n--- Pod: {pod_name} (phase: {phase}) ---")
            containers = (
                [c.name for c in (pod.spec.containers or [])] if pod.spec else []
            )
            container_statuses = (
                {cs.name: cs for cs in (pod.status.container_statuses or [])}
                if pod.status
                else {}
            )
            for container_name in containers:
                cs = container_statuses.get(container_name)
                waiting_reason = ""
                if cs and cs.state and cs.state.waiting:
                    waiting_reason = cs.state.waiting.reason or ""
                is_crashed = waiting_reason in _CRASH_REASONS
                try:
                    log = v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        container=container_name,
                        tail_lines=tail_lines,
                        _request_timeout=60,
                    )
                    lines.append(f"[container: {container_name}]")
                    lines.append(log if log else "(empty)")
                except Exception as e:
                    lines.append(
                        f"[container: {container_name}] error fetching current logs: {e}"
                    )
                if is_crashed:
                    try:
                        prev_log = v1.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            container=container_name,
                            tail_lines=tail_lines,
                            previous=True,
                            _request_timeout=60,
                        )
                        lines.append(f"[container: {container_name} — previous run]")
                        lines.append(prev_log if prev_log else "(empty)")
                    except Exception:
                        pass
    except Exception as e:
        return f"[Could not fetch pod logs for {label_selector!r}: {e}]"
    return "\n".join(lines)


_AUTOML_NOTEBOOK_ENV_PREFIXES = ("AWS_",)
_SYSTEM_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "USER",
        "LOGNAME",
        "SHELL",
    }
)


def download_and_execute_automl_notebook(
    s3_client, bucket: str, notebook_key: str
) -> None:
    """Download an AutoML predictor notebook from S3 and execute it locally via papermill.

    Raises:
        AssertionError: If the notebook fails to execute.
    """
    try:
        import papermill as pm
    except ImportError as e:
        raise AssertionError(
            "papermill is not installed; cannot execute notebook"
        ) from e

    with tempfile.TemporaryDirectory(prefix="automl-notebook-") as tmpdir:
        filename = Path(notebook_key).name
        input_path = Path(tmpdir) / f"input_{filename}"
        output_path = Path(tmpdir) / f"output_{filename}"

        s3_client.download_file(bucket, notebook_key, str(input_path))

        original_cwd = os.getcwd()
        original_environ = os.environ.copy()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            os.chdir(output_path.parent)

            filtered_env = {
                k: v
                for k, v in original_environ.items()
                if k in _SYSTEM_ENV_KEYS
                or any(k.startswith(p) for p in _AUTOML_NOTEBOOK_ENV_PREFIXES)
            }
            os.environ.clear()
            os.environ.update(filtered_env)

            pm.execute_notebook(
                str(input_path), str(output_path), kernel_name="python3"
            )
        except pm.PapermillExecutionError as e:
            raise AssertionError(
                f"AutoML notebook {filename} (key={notebook_key}) failed: {e}"
            ) from e
        except Exception as e:
            raise AssertionError(
                f"AutoML notebook {filename} (key={notebook_key}) execution error: {e}"
            ) from e
        finally:
            os.environ.clear()
            os.environ.update(original_environ)
            os.chdir(original_cwd)
