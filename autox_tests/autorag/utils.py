"""Shared utilities for AutoRAG functional tests."""

import logging
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from autox_tests.lib.kfp_run_state import _normalize_state
from autox_tests.lib.s3_data import upload_file_to_s3

logger = logging.getLogger(__name__)


def _make_docrag_run_name():
    """Return a run name: docrag-func-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"autorag-func-test-{hex_part}-{time_part}"


def _run_pipeline_and_wait(client, pipeline_target, arguments, timeout):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    from autox_tests.lib.managed_pipelines import submit_pipeline_run_and_wait

    run_name = _make_docrag_run_name()
    return submit_pipeline_run_and_wait(
        client,
        pipeline_target,
        arguments,
        run_name=run_name,
        timeout=timeout,
    )



def _collect_failure_details(client, run_id, config=None):
    """Collect failure details from a failed pipeline run via the Kubernetes API.

    Uses the Kubernetes client to find pods with label ``pipeline/runid=<run_id>``,
    identifies failed pods, and fetches their logs.  Task-level metadata is still
    pulled from the KFP v2 API for context.

    Args:
        client: KFP client instance (used for run-level / task-level metadata).
        run_id: The pipeline run ID.
        config: Functional config dict with ``rhoai_token``, ``rhoai_kfp_url``,
            and ``rhoai_project`` keys used for Kubernetes authentication.

    Returns:
        Formatted string with failure details and pod logs.
    """
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]

    # --- Run-level and task-level details from KFP v2 API ---
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
            _INTERNAL_SUFFIXES = ("-driver",)
            _INTERNAL_NAMES = ("root", "executor")

            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                state = getattr(task, "state", None)
                state_str = _normalize_state(state) or "NOT_STARTED"

                if name in _INTERNAL_NAMES or any(name.endswith(s) for s in _INTERNAL_SUFFIXES):
                    continue

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")

                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")

                    start = getattr(task, "start_time", None)
                    end = getattr(task, "end_time", None)
                    if start and end:
                        lines.append(f"  Duration: {start} -> {end}")
                else:
                    lines.append(f"  TASK: {name} — {state_str}")
        else:
            lines.append("\n[No task_details in run response]")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details from KFP API: {e}]")

    # --- Pod logs via Kubernetes API (label-based pod discovery) ---
    try:
        namespace = config.get("rhoai_project") if config else None
        token = config.get("rhoai_token") if config else None
        kfp_url = config.get("rhoai_kfp_url") if config else None
        _append_failed_pod_logs(run_id, namespace, lines, token=token, kfp_url=kfp_url)
    except Exception as e:
        lines.append(f"\n[Could not fetch pod logs: {e}]")

    lines.append("=" * 80)
    return "\n".join(lines)


def _derive_k8s_api_url(kfp_url):
    """Derive OpenShift API server URL from a KFP route URL.

    Standard OCP: https://<route>.apps.<cluster-domain> -> https://api.<cluster-domain>:6443
    ROSA:         https://<route>.apps.rosa.<cluster-domain> -> https://api.<cluster-domain>:443

    Override entirely with K8S_API_URL env var, or just the port with K8S_API_PORT.
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


def _make_k8s_core_api(token, kfp_url):
    """Create a Kubernetes CoreV1Api client authenticated with a bearer token."""
    from kubernetes import client as k8s_client

    api_url = _derive_k8s_api_url(kfp_url)
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


def _is_pod_failed(pod):
    """Return True if a pod is in a failed state."""
    phase = pod.status.phase or ""
    if phase.lower() == "failed":
        return True
    for cs in pod.status.container_statuses or []:
        terminated = cs.state.terminated if cs.state else None
        if terminated and terminated.exit_code != 0:
            return True
    return False


def _append_failed_pod_logs(run_id, namespace, lines, token=None, kfp_url=None):
    """Find failed pods for a pipeline run by label and append their logs.

    Lists pods matching ``pipeline/runid=<run_id>`` in the given namespace,
    filters for failed pods, and fetches logs from each container.
    """
    if not token or not kfp_url:
        lines.append("\n[Missing RHOAI_TOKEN or RHOAI_KFP_URL; skipping pod log fetch]")
        return

    try:
        import kubernetes  # noqa: F401
    except ImportError:
        lines.append("\n[kubernetes package not installed; skipping pod log fetch]")
        return

    ns = namespace or "default"
    api = _make_k8s_core_api(token, kfp_url)

    pod_list = api.list_namespaced_pod(
        namespace=ns,
        label_selector=f"pipeline/runid={run_id}",
        _request_timeout=30,
    )

    if not pod_list.items:
        lines.append(f"\n[No pods found with label pipeline/runid={run_id} in namespace {ns}]")
        return

    failed_pods = [p for p in pod_list.items if _is_pod_failed(p)]

    if not failed_pods:
        all_phases = ", ".join(f"{p.metadata.name}={p.status.phase}" for p in pod_list.items)
        lines.append(f"\n[No failed pods among {len(pod_list.items)} pods: {all_phases}]")
        return

    lines.append(f"\nFound {len(failed_pods)} failed pod(s) out of {len(pod_list.items)} total")

    for pod in failed_pods:
        pod_name = pod.metadata.name
        lines.append(f"\n--- Failed pod: {pod_name} (phase: {pod.status.phase}) ---")

        containers = [c.name for c in (pod.spec.containers or [])]
        for container_name in containers:
            try:
                log = api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=ns,
                    container=container_name,
                    tail_lines=100,
                    _request_timeout=60,
                )
                lines.append(f"[container: {container_name}]")
                lines.append(log if log else "(empty)")
            except Exception as e:
                lines.append(f"[container: {container_name}] error: {e}")


def _validate_artifacts_in_s3(s3_client, bucket, prefix):
    """List and categorize S3 artifacts under prefix.

    Returns:
        Dict with keys: "pattern_keys", "indexing_notebook_keys", "inference_notebook_keys",
        "evaluation_results_keys", "leaderboard_keys", "responses_body_keys", "all_keys".

    Raises:
        AssertionError: If S3 listing fails.
    """
    result = {
        "pattern_keys": [],
        "indexing_notebook_keys": [],
        "inference_notebook_keys": [],
        "evaluation_results_keys": [],
        "leaderboard_keys": [],
        "responses_body_keys": [],
        "all_keys": [],
    }
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                result["all_keys"].append(key)
                lower_key = key.lower()
                if key.endswith("pattern.json") or "rag_patterns" in lower_key:
                    result["pattern_keys"].append(key)
                if key.endswith(".ipynb") and "indexing" in lower_key:
                    result["indexing_notebook_keys"].append(key)
                if key.endswith(".ipynb") and "inference" in lower_key:
                    result["inference_notebook_keys"].append(key)
                if "evaluation_results.json" in key:
                    result["evaluation_results_keys"].append(key)
                if "leaderboard" in lower_key or key.endswith(".html"):
                    result["leaderboard_keys"].append(key)
                if "v1_responses_body.json" in key:
                    result["responses_body_keys"].append(key)
    except Exception as e:
        raise AssertionError(f"Failed to list S3 artifacts under s3://{bucket}/{prefix}: {e}") from e
    return result


_NOTEBOOK_ENV_PREFIXES = ("OGX_CLIENT_", "AWS_")
_SYSTEM_ENV_KEYS = frozenset({"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL"})


def _inject_and_run(notebook_path: Path, output_path: Path) -> None:
    """Inject mocked input() function into the notebook and execute it."""
    import nbformat
    import papermill as pm

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mock_code = 'def input(prompt=""):\n    return "Sample query?"'
    nb.cells.insert(0, nbformat.v4.new_code_cell(mock_code))

    injected_path = notebook_path.with_name(f"injected_{notebook_path.name}")
    with open(injected_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    original_cwd = os.getcwd()
    original_environ = os.environ.copy()
    try:
        safe_cwd = output_path.parent
        safe_cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(safe_cwd)

        filtered_env = {
            k: v
            for k, v in original_environ.items()
            if k in _SYSTEM_ENV_KEYS or any(k.startswith(p) for p in _NOTEBOOK_ENV_PREFIXES)
        }
        os.environ.clear()
        os.environ.update(filtered_env)

        pm.execute_notebook(str(injected_path), str(output_path), kernel_name="python3")
    finally:
        os.environ.clear()
        os.environ.update(original_environ)
        os.chdir(original_cwd)
        injected_path.unlink(missing_ok=True)


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def upload_test_datasets(
    s3_client,
    bucket: str,
    s3_keys: list[str],
    local_data_dir: Path,
) -> list[str]:
    """Upload local files or directories to S3 for each key in s3_keys.

    Keys with a file extension are matched by filename (basename). Keys without an
    extension are treated as S3 directory prefixes: the matching local directory is found
    by its name, and all files within it are uploaded under that prefix.

    When multiple local files or directories share the same name, the one whose parent
    directory name shares the longest common prefix with the S3 key's parent component
    is chosen. Keys with no local match are skipped (covers intentional negative-test keys).

    Returns the list of S3 keys that were actually uploaded.
    """
    if not local_data_dir.is_dir():
        raise FileNotFoundError(
            f"AUTORAG_UPLOAD_TEST_DATASETS is set but local data directory does not exist: {local_data_dir}"
        )

    local_files: dict[str, list[Path]] = {}
    local_dirs: dict[str, list[Path]] = {}
    for entry in local_data_dir.rglob("*"):
        if entry.is_file():
            local_files.setdefault(entry.name, []).append(entry)
        elif entry.is_dir():
            local_dirs.setdefault(entry.name, []).append(entry)

    uploaded_keys: list[str] = []
    failed_uploads: list[str] = []

    for s3_key in sorted(set(s3_keys)):
        key_path = Path(s3_key)
        parent_hint = key_path.parent.name
        leaf = key_path.name

        if key_path.suffix:
            candidates = sorted(local_files.get(leaf, []))
            if not candidates:
                logger.debug("No local file for key %r — skipping upload", s3_key)
                continue
            scores = [_common_prefix_len(p.parent.name, parent_hint) for p in candidates]
            best_score = max(scores)
            if len(candidates) > 1:
                if best_score == 0:
                    logger.warning(
                        "Cannot discriminate among %d candidates for key %r — picking %s",
                        len(candidates), s3_key, candidates[0],
                    )
                else:
                    logger.warning(
                        "Multiple local files named %r; picking best match for parent %r",
                        leaf, parent_hint,
                    )
            local_path = candidates[scores.index(best_score)]
            try:
                logger.info("Uploading %s → s3://%s/%s", local_path, bucket, s3_key)
                upload_file_to_s3(s3_client, bucket=bucket, key=s3_key, local_path=local_path)
                uploaded_keys.append(s3_key)
            except Exception as exc:
                logger.error(
                    "Failed to upload %s → s3://%s/%s: %s", local_path, bucket, s3_key, exc
                )
                failed_uploads.append(s3_key)
        else:
            candidates = sorted(local_dirs.get(leaf, []))
            if not candidates:
                logger.debug("No local directory for key %r — skipping upload", s3_key)
                continue
            scores = [_common_prefix_len(d.parent.name, parent_hint) for d in candidates]
            best_score = max(scores)
            if len(candidates) > 1:
                if best_score == 0:
                    logger.warning(
                        "Cannot discriminate among %d candidates for key %r — picking %s",
                        len(candidates), s3_key, candidates[0],
                    )
                else:
                    logger.warning(
                        "Multiple local directories named %r; picking best match for parent %r",
                        leaf, parent_hint,
                    )
            local_dir = candidates[scores.index(best_score)]
            for f in sorted(local_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(local_dir)
                file_s3_key = f"{s3_key}/{rel}"
                try:
                    logger.info("Uploading %s → s3://%s/%s", f, bucket, file_s3_key)
                    upload_file_to_s3(s3_client, bucket=bucket, key=file_s3_key, local_path=f)
                    uploaded_keys.append(file_s3_key)
                except Exception as exc:
                    logger.error(
                        "Failed to upload %s → s3://%s/%s: %s", f, bucket, file_s3_key, exc
                    )
                    failed_uploads.append(file_s3_key)

    logger.info(
        "Dataset upload complete: %d file(s) uploaded to s3://%s",
        len(uploaded_keys),
        bucket,
    )
    if failed_uploads:
        raise RuntimeError(
            f"Failed to upload {len(failed_uploads)} dataset file(s) to s3://{bucket}: {failed_uploads}"
        )
    return uploaded_keys


def _download_and_execute_notebooks(s3_client, bucket, notebook_keys):
    """Download notebooks from S3 and execute them via papermill.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        notebook_keys: List of S3 keys pointing to .ipynb files.

    Raises:
        AssertionError: If any notebook fails execution.
    """
    import papermill as pm

    errors = []
    with tempfile.TemporaryDirectory(prefix="autorag-pipeline-notebook-") as tmpdir:
        for key in notebook_keys:
            filename = Path(key).name
            input_path = Path(tmpdir) / f"input_{filename}"
            output_path = Path(tmpdir) / f"output_{filename}"

            s3_client.download_file(bucket, key, str(input_path))

            try:
                _inject_and_run(input_path, output_path)
            except pm.PapermillExecutionError as e:
                errors.append(f"Notebook {filename} (key={key}) failed: {e}")
            except Exception as e:
                errors.append(f"Notebook {filename} (key={key}) execution error: {e}")

    if errors:
        raise AssertionError("Notebook execution failures:\n" + "\n".join(errors))
