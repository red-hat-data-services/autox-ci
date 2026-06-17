"""Kubernetes API utilities shared across AutoML and AutoRAG functional tests."""

from __future__ import annotations

import os
import re
from typing import Any

RHOAI_POD_LOG_TAIL_LINES_ENV = "RHOAI_POD_LOG_TAIL_LINES"
# Intentionally lower than the old 100-line default to keep CI log output concise.
# Override with RHOAI_POD_LOG_TAIL_LINES; set to 0 for full output.
_DEFAULT_POD_LOG_TAIL_LINES = 30

_CRASH_REASONS = frozenset({"CrashLoopBackOff", "Error", "OOMKilled"})
# Tekton PipelineRun wrapper tasks end with a random 5-char suffix.
_PIPELINE_RUN_WRAPPER_RE = re.compile(r"-[a-z0-9]{5}$", re.IGNORECASE)


def pod_log_tail_lines() -> int | None:
    """Return tail line count for failed-pod logs (default 30).

    Override with ``RHOAI_POD_LOG_TAIL_LINES``. Set to ``0`` for full logs.
    """
    raw = (os.environ.get(RHOAI_POD_LOG_TAIL_LINES_ENV) or "").strip()
    if not raw:
        return _DEFAULT_POD_LOG_TAIL_LINES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_POD_LOG_TAIL_LINES
    return value if value > 0 else None


def derive_k8s_api_url(kfp_url: str) -> str | None:
    """Derive OpenShift API server URL from a KFP route URL.

    Standard OCP: https://<route>.apps.<cluster-domain> -> https://api.<cluster-domain>:6443
    ROSA:         https://<route>.apps.rosa.<cluster-domain> -> https://api.<cluster-domain>:443

    Override entirely with K8S_API_URL env var, or just the port with K8S_API_PORT.

    Args:
        kfp_url: KFP route URL (e.g., https://ds-pipeline.apps.cluster.example.com)

    Returns:
        Kubernetes API URL or None if URL format is invalid.
    """
    override = os.environ.get("K8S_API_URL")
    if override:
        return override.strip().rstrip("/")

    from urllib.parse import urlparse

    hostname = urlparse(kfp_url).hostname or ""
    apps_idx = hostname.find(".apps.")
    if apps_idx < 0:
        return None
    base_domain = hostname[apps_idx + len(".apps.") :]
    is_rosa = base_domain.startswith("rosa.")
    if is_rosa:
        base_domain = base_domain[len("rosa.") :]
    default_port = 443 if is_rosa else 6443
    port = os.environ.get("K8S_API_PORT", str(default_port)).strip()
    return f"https://api.{base_domain}:{port}"


def make_k8s_core_api(token: str, kfp_url: str):
    """Create a Kubernetes CoreV1Api client authenticated with a bearer token.

    Args:
        token: Bearer token for authentication
        kfp_url: KFP route URL used to derive the k8s API URL

    Returns:
        kubernetes.client.CoreV1Api instance

    Raises:
        RuntimeError: If K8S API URL cannot be derived from KFP URL
    """
    from kubernetes import client as k8s_client

    api_url = derive_k8s_api_url(kfp_url)
    if not api_url:
        raise RuntimeError(f"Cannot derive K8S API URL from KFP URL: {kfp_url}")

    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")

    configuration = k8s_client.Configuration()
    configuration.host = api_url
    configuration.api_key = {"authorization": f"Bearer {token}"}
    configuration.verify_ssl = verify_ssl
    if not verify_ssl:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    return k8s_client.CoreV1Api(api_client=k8s_client.ApiClient(configuration))


def merge_kubeconfig_into_config(config: dict, kubeconfig_path: str | None) -> dict:
    """Merge temp_kubeconfig_path into a functional test config dict.

    Args:
        config: Base functional config dict.
        kubeconfig_path: Path to kubeconfig file or None.

    Returns:
        New dict with temp_kubeconfig_path merged in (or original if path is None).
    """
    if kubeconfig_path is None:
        return config
    return {**config, "temp_kubeconfig_path": kubeconfig_path}


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


def make_k8s_core_api_from_config(config: dict[str, Any]):
    """Build a CoreV1Api client from a functional test config dict.

    Prefers ``temp_kubeconfig_path``; falls back to bearer token + KFP URL.
    """
    temp_kubeconfig_path = config.get("temp_kubeconfig_path")
    if temp_kubeconfig_path is not None:
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config

        # Load kubeconfig into an isolated Configuration (does not mutate global config)
        configuration = k8s_client.Configuration()
        k8s_config.load_kube_config(
            config_file=temp_kubeconfig_path, client_configuration=configuration
        )

        verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
        configuration.verify_ssl = verify_ssl not in ("0", "false", "no")

        if not configuration.verify_ssl:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        return k8s_client.CoreV1Api(api_client=k8s_client.ApiClient(configuration))

    token = config.get("rhoai_token")
    kfp_url = config.get("rhoai_kfp_url")
    if not token or not kfp_url:
        return None
    return make_k8s_core_api(token, kfp_url)


_RUN_ID_LABEL_KEYS = (
    "pipeline/runid",
    "pipeline/runId",
)


def _looks_like_pipeline_run_wrapper_task(task_name: str) -> bool:
    """Return True for Tekton PipelineRun wrapper tasks (not user components).

    Tekton PipelineRuns create wrapper tasks with predictable naming: the task name
    is the PipelineRun name itself. These tasks always end with a 5-char random suffix
    and typically contain the run ID or 'pipelinerun' in the name.

    NOTE: This is a heuristic fallback. Prefer label-based detection when possible
    (see _list_run_pods which queries tekton.dev/pipelineRun labels directly).
    """
    if not _PIPELINE_RUN_WRAPPER_RE.search(task_name):
        return False
    lowered = task_name.lower()
    # Match tasks explicitly containing 'pipelinerun' or 'pipeline-run'
    # Avoid matching user tasks that happen to contain 'pipeline' (e.g., 'data-pipeline-transform')
    # This may still have false positives/negatives; label queries are more reliable.
    return "pipelinerun" in lowered or "pipeline-run" in lowered


def _extract_pipeline_run_name(failed_task_names: list[str]) -> str | None:
    """Return the Tekton PipelineRun name from failed KFP task names, if present."""
    for name in failed_task_names:
        if _looks_like_pipeline_run_wrapper_task(name):
            return name
    return None


def _list_pods_for_selector(v1, namespace: str, selector: str, logger=None):
    try:
        return v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=selector,
            _request_timeout=30,
        ).items
    except Exception as e:
        if logger:
            logger.debug("Pod list failed for selector %r: %s", selector, e)
        return []


def _pod_phase_failed(pod) -> bool:
    """Return True if pod has failed or has containers with non-zero exit codes.

    Detects both explicit Failed/Unknown phase and containers that crashed
    (e.g., CrashLoopBackOff, OOMKilled) which may still show phase as Running.
    """
    if not pod.status:
        return False
    if pod.status.phase in ("Failed", "Unknown"):
        return True
    for cs in pod.status.container_statuses or []:
        if cs.state and cs.state.terminated and cs.state.terminated.exit_code != 0:
            return True
    return False


def _list_run_pods(
    v1, namespace: str, run_id: str, failed_task_names: list[str] | None = None, logger=None
):
    """Return failed pods belonging to a pipeline run.

    Strategy:
    1. Query by standard pipeline run ID labels (pipeline/runid, pipeline/runId)
    2. If failed tasks suggest a Tekton PipelineRun wrapper, also query tekton.dev/pipelineRun
    3. If still no pods found, try extracting PipelineRun name from ANY pod's labels
       and query again (handles cases where task names are unreliable)
    """
    failed_task_names = failed_task_names or []
    pipeline_run_name = _extract_pipeline_run_name(failed_task_names)
    seen: dict[str, object] = {}

    def _add(pods) -> None:
        for pod in pods:
            if _pod_phase_failed(pod):
                seen[pod.metadata.name] = pod

    # Strategy 1: Standard run ID labels
    for run_label_key in _RUN_ID_LABEL_KEYS:
        pods = _list_pods_for_selector(v1, namespace, f"{run_label_key}={run_id}", logger)
        if pods:
            _add(pods)

    # Strategy 2: Tekton PipelineRun label (if wrapper task name detected)
    if pipeline_run_name:
        _add(
            _list_pods_for_selector(
                v1, namespace, f"tekton.dev/pipelineRun={pipeline_run_name}", logger
            )
        )

    # Strategy 3: Fallback - extract PipelineRun name from pod labels (authoritative)
    if not seen:
        # Query all pods for this run (not just failed) to find PipelineRun name
        all_pods = []
        for key in _RUN_ID_LABEL_KEYS:
            all_pods = _list_pods_for_selector(v1, namespace, f"{key}={run_id}", logger)
            if all_pods:
                break
        for pod in all_pods:
            labels = pod.metadata.labels or {}
            pr_name = labels.get("tekton.dev/pipelineRun")
            if pr_name:
                # Found authoritative PipelineRun name, query for failed pods
                tekton_pods = _list_pods_for_selector(
                    v1, namespace, f"tekton.dev/pipelineRun={pr_name}", logger
                )
                _add(tekton_pods)
                break  # Only need to find one PipelineRun name

    return list(seen.values())


def _containers_to_log(pod) -> list[str]:
    """Return all container names in a pod."""
    if not pod.spec:
        return []
    return [c.name for c in (pod.spec.containers or [])]


def append_failed_task_pod_logs(
    lines: list[str],
    run_id: str,
    config: dict[str, Any],
    failed_task_names: list[str] | None = None,
    *,
    tail_lines: int | None = -1,
    logger=None,
) -> None:
    """Fetch logs from failed pods in a pipeline run and append them to *lines*.

    Raises:
        Exception: Kubernetes API errors are propagated to caller for logging.
    """
    if tail_lines == -1:
        tail_lines = pod_log_tail_lines()

    namespace = config.get("rhoai_project")
    if not namespace:
        lines.append("\n[Missing rhoai_project (namespace); skipping pod log fetch]")
        return

    lines.append(f"\n[Querying pod logs in namespace: {namespace}]")

    v1 = make_k8s_core_api_from_config(config)
    if v1 is None:
        lines.append(
            "\n[Missing temp_kubeconfig_path or (rhoai_token + rhoai_kfp_url); "
            "skipping pod log fetch]"
        )
        return

    run_pods = _list_run_pods(v1, namespace, run_id, failed_task_names or [], logger)
    if not run_pods:
        pipeline_run_name = _extract_pipeline_run_name(failed_task_names or [])
        hint = f"selectors tried: pipeline/runid={run_id!r}" + (
            f", tekton.dev/pipelineRun={pipeline_run_name!r}"
            if pipeline_run_name
            else ""
        )
        lines.append(f"\n[No failed pods found for run {run_id!r}; {hint}]")
        return

    tail_desc = f"last {tail_lines} lines" if tail_lines else "full log"
    lines.append("\n\nPOD LOGS FOR FAILED PODS:")
    lines.append("-" * 80)
    lines.append(f"Failed pods: {len(run_pods)} ({tail_desc} per container)")
    pod_logs = fetch_logs_from_pods(v1, namespace, run_pods, tail_lines=tail_lines)
    lines.append(pod_logs)


def append_failed_task_pod_logs_safe(
    lines: list[str],
    run_id: str,
    config: dict[str, Any],
    failed_task_names: list[str] | None = None,
    *,
    tail_lines: int | None = -1,
    logger=None,
) -> None:
    """Safe wrapper for append_failed_task_pod_logs with kubernetes availability check.

    Handles kubernetes package availability and exception logging internally.

    Args:
        lines: List to append log output to
        run_id: Pipeline run ID
        config: Functional config dict
        failed_task_names: List of failed task names from KFP API
        tail_lines: Number of log lines to fetch (-1 uses env default)
        logger: Optional logger for exception logging
    """
    try:
        import kubernetes  # noqa: F401
    except ImportError:
        lines.append(
            "\n[kubernetes package not available for pod log fetch. "
            "Install with: pip install kubernetes]"
        )
        return

    try:
        append_failed_task_pod_logs(
            lines,
            run_id,
            config,
            failed_task_names,
            tail_lines=tail_lines,
        )
    except Exception as e:
        lines.append(f"\n[Could not fetch pod logs: {e}]")
        if logger:
            logger.exception("Failed to fetch pod logs for run %s", run_id)


def fetch_logs_from_pods(
    v1, namespace: str, pods: list, tail_lines: int | None = None
) -> str:
    """Fetch logs from a list of pod objects and return a formatted string.

    Args:
        v1: Kubernetes CoreV1Api client instance
        namespace: Kubernetes namespace
        pods: List of pod objects (V1Pod instances)
        tail_lines: Number of log lines to fetch per container

    Returns:
        Formatted string with pod logs
    """
    if not pods:
        return "[No pods to fetch logs from]"
    lines = []
    for pod in pods:
        pod_name = pod.metadata.name
        phase = pod.status.phase if pod.status else "unknown"
        lines.append(f"\n--- Pod: {pod_name} (phase: {phase}) ---")
        lines.extend(
            _fetch_container_logs_for_pod(
                v1, namespace, pod_name, pod, tail_lines=tail_lines
            )
        )
    return "\n".join(lines)


def _fetch_container_logs_for_pod(
    v1, namespace: str, pod_name: str, pod, *, tail_lines: int | None
) -> list[str]:
    """Return formatted log lines for all containers in a pod."""
    lines: list[str] = []
    containers = _containers_to_log(pod)
    if not containers:
        lines.append("[no containers]")
        return lines

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
        log_kwargs: dict[str, Any] = {
            "name": pod_name,
            "namespace": namespace,
            "container": container_name,
            "_request_timeout": 60,
        }
        if tail_lines:
            log_kwargs["tail_lines"] = tail_lines
        try:
            log = v1.read_namespaced_pod_log(**log_kwargs)
            lines.append(f"[container: {container_name}]")
            lines.append(log if log else "(empty)")
        except Exception as e:
            lines.append(
                f"[container: {container_name}] error fetching current logs: {e}"
            )
        if is_crashed:
            try:
                prev_kwargs = {**log_kwargs, "previous": True}
                prev_log = v1.read_namespaced_pod_log(**prev_kwargs)
                lines.append(f"[container: {container_name} — previous run]")
                lines.append(prev_log if prev_log else "(empty)")
            except Exception:
                pass
    return lines


def fetch_logs_by_selector(
    v1, namespace: str, label_selector: str, tail_lines: int | None = None
) -> str:
    """Fetch logs from pods matching a label selector and return a formatted string.

    For containers in CrashLoopBackOff or Error state, also fetches the previous
    terminated container's logs so crash output is visible even after a restart.

    Args:
        v1: Kubernetes CoreV1Api client instance
        namespace: Kubernetes namespace to search for pods
        label_selector: Label selector to filter pods (e.g., "pipeline/runid=xyz")
        tail_lines: Number of log lines to fetch per container

    Returns:
        Formatted string with pod logs or error message
    """
    lines = []
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
            lines.extend(
                _fetch_container_logs_for_pod(
                    v1, namespace, pod_name, pod, tail_lines=tail_lines
                )
            )
    except Exception as e:
        return f"[Could not fetch pod logs for {label_selector!r}: {e}]"
    return "\n".join(lines)
