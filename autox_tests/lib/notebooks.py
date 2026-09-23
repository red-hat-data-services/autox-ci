"""Shared notebook execution helpers for AutoML and AutoRAG test suites."""

import json
import logging
import os
import secrets
import time
from typing import Any

from autox_tests.lib.k8s_utils import make_k8s_core_api_from_config

RHOAI_NOTEBOOK_RUNNER_IMAGE_ENV = "RHOAI_NOTEBOOK_RUNNER_IMAGE"
RHOAI_NOTEBOOK_JOB_TIMEOUT_ENV = "RHOAI_NOTEBOOK_JOB_TIMEOUT"
RHOAI_NOTEBOOK_CPU_ENV = "RHOAI_NOTEBOOK_CPU"
RHOAI_NOTEBOOK_MEMORY_ENV = "RHOAI_NOTEBOOK_MEMORY"
RHOAI_NOTEBOOK_PIP_SECRET_ENV = "RHOAI_NOTEBOOK_PIP_SECRET_NAME"
RHOAI_NOTEBOOK_DOCLING_SECRET_ENV = "RHOAI_NOTEBOOK_DOCLING_SECRET_NAME"
_DEFAULT_NOTEBOOK_JOB_TIMEOUT_SECONDS = 900
_DEFAULT_NOTEBOOK_CPU = "2"
_DEFAULT_NOTEBOOK_MEMORY = "4Gi"
_NOTEBOOK_JOB_POLL_SECONDS = 15

logger = logging.getLogger(__name__)


_NOTEBOOK_JOB_PROGRAM = r"""import json
import os
from pathlib import Path

import boto3
import nbformat
import papermill as pm

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("AWS_S3_ENDPOINT"),
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    verify=os.environ.get("S3_SSL_VERIFY", "true").strip().lower()
    not in ("0", "false", "no"),
)
print(f"Running notebooks with image: {os.environ['NOTEBOOK_RUNNER_IMAGE']}", flush=True)

docling_prefix = os.environ.get("DOCLING_ARTIFACTS_S3_PREFIX", "").strip().lstrip("/")
docling_bucket = os.environ.get("DOCLING_ARTIFACTS_S3_BUCKET", "").strip()
docling_path = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip()
if docling_prefix or docling_bucket or docling_path:
    if not (docling_prefix and docling_bucket and docling_path):
        raise RuntimeError(
            "Docling artifact configuration requires DOCLING_ARTIFACTS_S3_BUCKET, "
            "DOCLING_ARTIFACTS_S3_PREFIX, and DOCLING_ARTIFACTS_PATH"
        )
    destination = Path(docling_path)
    downloaded = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=docling_bucket, Prefix=docling_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key.removeprefix(docling_prefix)
            if not relative:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(docling_bucket, key, str(target))
            downloaded += 1
    if not downloaded:
        raise RuntimeError(f"No Docling artifacts found at s3://{docling_bucket}/{docling_prefix}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(f"Downloaded {downloaded} Docling artifacts to {destination}", flush=True)

for index, notebook_key in enumerate(json.loads(os.environ["NOTEBOOK_S3_KEYS"])):
    workdir = Path("/tmp/notebooks") / str(index)
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "input.ipynb"
    output_path = workdir / "output.ipynb"
    s3.download_file(os.environ["NOTEBOOK_S3_BUCKET"], notebook_key, str(input_path))

    # Only disconnected environments configure a pip Secret to provide the
    # package index. Disable the generated notebook's hard-coded extra index so
    # pip uses the index configured by that Secret instead. Match AutoGluon
    # install cells rather than one exact cell, as tabular and time-series
    # templates differ slightly.
    if os.environ.get("NOTEBOOK_HAS_PIP_SECRET", "false").lower() == "true":
        with input_path.open(encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if (
                cell.cell_type == "code"
                and "PIP_EXTRA_INDEX_URL" in cell.source
                and "%pip install autogluon." in cell.source
            ):
                cell.source = cell.source.replace(
                    "PIP_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL_DEV"
                )
        with input_path.open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"Patched AutoGluon install cell in {notebook_key}", flush=True)

    # Some generated indexing notebooks contain a malformed text-extraction cell:
    # the assignment and function call are concatenated, and the progress f-string
    # uses escaped braces. Replace only that exact broken cell before execution.
    if notebook_key.lower().endswith("indexing.ipynb"):
        with input_path.open(encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if (
                cell.cell_type == "code"
                and 'extracted_text_dir = step_output_dir / "extracted_text"extraction_result' in cell.source
            ):
                cell.source = '''extracted_text_dir = step_output_dir / "extracted_text"

extraction_result = extract_text(
    documents=result.to_dict()["documents"],
    bucket=result.bucket,
    output_dir=extracted_text_dir,
    docling_artifacts_path=os.getenv("DOCLING_ARTIFACTS_PATH"),
)

print(
    f"Extracted {extraction_result.processed_count}/{extraction_result.total_documents} documents "
    f"({extraction_result.error_count} errors)"
)'''
                with input_path.open("w", encoding="utf-8") as f:
                    nbformat.write(notebook, f)
                print(f"Patched malformed text-extraction cell in {notebook_key}", flush=True)
                break

    if os.environ.get("NOTEBOOK_INJECT_MOCK_INPUT", "false").lower() == "true":
        with input_path.open(encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_code_cell('def input(prompt=""):\n    return "Sample query?"'))
        with input_path.open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    pm.execute_notebook(
        str(input_path),
        str(output_path),
        cwd=str(workdir),
        kernel_name=os.environ.get("NOTEBOOK_KERNEL_NAME", "python3"),
    )
"""


def notebook_runner_image() -> str | None:
    """Return the configured in-cluster notebook-runner image, if enabled."""
    return (os.environ.get(RHOAI_NOTEBOOK_RUNNER_IMAGE_ENV) or "").strip() or None


def _notebook_job_timeout_seconds() -> int:
    raw = (os.environ.get(RHOAI_NOTEBOOK_JOB_TIMEOUT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_NOTEBOOK_JOB_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RHOAI_NOTEBOOK_JOB_TIMEOUT_ENV}={raw!r}: expected a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError(f"{RHOAI_NOTEBOOK_JOB_TIMEOUT_ENV} must be a positive integer")
    return value


def _job_pod_logs(core_api: Any, namespace: str, job_name: str) -> str:
    """Return concise logs from pods owned by a notebook Job."""
    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
            _request_timeout=30,
        ).items
    except Exception as exc:
        return f"Unable to list Job pods: {exc}"

    logs: list[str] = []
    for pod in pods:
        name = getattr(getattr(pod, "metadata", None), "name", "unknown")
        pod_status = getattr(pod, "status", None)
        status_details = [f"phase={getattr(pod_status, 'phase', 'unknown')}"]
        for container_status in getattr(pod_status, "container_statuses", None) or []:
            state = getattr(container_status, "state", None)
            terminated = getattr(state, "terminated", None)
            waiting = getattr(state, "waiting", None)
            if terminated is not None:
                status_details.append(
                    f"container={container_status.name} terminated="
                    f"reason={terminated.reason or 'unknown'} "
                    f"exit_code={terminated.exit_code} signal={terminated.signal}"
                )
            elif waiting is not None:
                status_details.append(
                    f"container={container_status.name} waiting="
                    f"reason={waiting.reason or 'unknown'}"
                )
        logs.append(f"--- pod/{name} status: {'; '.join(status_details)} ---")
        try:
            output = core_api.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                tail_lines=100,
                _request_timeout=30,
            )
        except Exception as exc:
            output = f"Unable to read pod log: {exc}"
        logs.append(output)
    return "\n".join(logs) or "No pod logs were available."


def run_notebooks_as_k8s_job(
    *,
    bucket: str,
    notebook_keys: list[str],
    config: dict[str, Any],
    secret_names: list[str],
    inject_mock_input: bool = False,
) -> None:
    """Execute S3-hosted notebooks sequentially in one ephemeral Job pod.

    The Job is enabled only when ``RHOAI_NOTEBOOK_RUNNER_IMAGE`` is set. The
    configured secrets are imported as environment variables, keeping S3, MaaS,
    and vector-store credentials out of the test process and Job command line.
    """
    if not notebook_keys:
        raise ValueError("Notebook Job execution requires at least one notebook key")
    image = notebook_runner_image()
    if image is None:
        raise AssertionError(
            "Notebook execution was requested, but %s is not set"
            % RHOAI_NOTEBOOK_RUNNER_IMAGE_ENV
        )

    timeout = _notebook_job_timeout_seconds()
    namespace = str(config.get("rhoai_project") or "").strip()
    if not namespace:
        raise AssertionError("Notebook Job execution requires config.rhoai_project")

    core_api = make_k8s_core_api_from_config(config)
    if core_api is None:
        raise AssertionError(
            "Notebook Job execution requires a Kubernetes kubeconfig or RHOAI_TOKEN and RHOAI_KFP_URL"
        )

    from kubernetes import client as k8s_client

    job_name = f"notebook-test-{secrets.token_hex(5)}"
    cpu = (os.environ.get(RHOAI_NOTEBOOK_CPU_ENV) or _DEFAULT_NOTEBOOK_CPU).strip()
    memory = (os.environ.get(RHOAI_NOTEBOOK_MEMORY_ENV) or _DEFAULT_NOTEBOOK_MEMORY).strip()
    unique_secrets = list(dict.fromkeys(name for name in secret_names if name))
    pip_secret = (os.environ.get(RHOAI_NOTEBOOK_PIP_SECRET_ENV) or "").strip()
    if pip_secret:
        unique_secrets.append(pip_secret)
    docling_secret = (os.environ.get(RHOAI_NOTEBOOK_DOCLING_SECRET_ENV) or "").strip()
    if docling_secret:
        unique_secrets.append(docling_secret)
    unique_secrets = list(dict.fromkeys(unique_secrets))
    env_from = [
        k8s_client.V1EnvFromSource(
            secret_ref=k8s_client.V1SecretEnvSource(name=name, optional=False)
        )
        for name in unique_secrets
    ]
    artifact_bucket = (
        os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET") or bucket
    ).strip()
    container = k8s_client.V1Container(
        name="notebook-runner",
        image=image,
        image_pull_policy="IfNotPresent",
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": memory},
            limits={"cpu": cpu, "memory": memory},
        ),
        command=["python", "-c", _NOTEBOOK_JOB_PROGRAM],
        env=[
            k8s_client.V1EnvVar(name="NOTEBOOK_S3_BUCKET", value=bucket),
            # Generated notebooks must read artifacts from the test artifact bucket,
            # not the input/training bucket stored in the S3 Secret.
            k8s_client.V1EnvVar(name="AWS_S3_BUCKET", value=artifact_bucket),
            k8s_client.V1EnvVar(name="NOTEBOOK_S3_KEYS", value=json.dumps(notebook_keys)),
            k8s_client.V1EnvVar(name="NOTEBOOK_RUNNER_IMAGE", value=image),
            k8s_client.V1EnvVar(
                name="NOTEBOOK_HAS_PIP_SECRET", value="true" if pip_secret else "false"
            ),
            k8s_client.V1EnvVar(
                name="S3_SSL_VERIFY",
                value=os.environ.get("S3_SSL_VERIFY", "true"),
            ),
            k8s_client.V1EnvVar(
                name="NOTEBOOK_INJECT_MOCK_INPUT",
                value="true" if inject_mock_input else "false",
            ),
            k8s_client.V1EnvVar(
                name="NOTEBOOK_KERNEL_NAME",
                value=os.environ.get("RHOAI_NOTEBOOK_KERNEL_NAME", "python3"),
            ),
        ],
        env_from=env_from or None,
    )
    job = k8s_client.V1Job(
        metadata=k8s_client.V1ObjectMeta(
            name=job_name,
            labels={"app.kubernetes.io/component": "autox-notebook-test"},
        ),
        spec=k8s_client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=300,
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(
                    labels={"app.kubernetes.io/component": "autox-notebook-test"}
                ),
                spec=k8s_client.V1PodSpec(restart_policy="Never", containers=[container]),
            ),
        ),
    )
    batch_api = k8s_client.BatchV1Api(api_client=core_api.api_client)
    try:
        logger.info(
            "Starting notebook Job %s with image %s (cpu=%s, memory=%s)",
            job_name,
            image,
            cpu,
            memory,
        )
        batch_api.create_namespaced_job(namespace=namespace, body=job, _request_timeout=30)
    except Exception as exc:
        raise AssertionError(f"Failed to create notebook Job {job_name}: {exc}") from exc

    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            status = batch_api.read_namespaced_job_status(
                name=job_name, namespace=namespace, _request_timeout=30
            ).status
            if (status.succeeded or 0) >= 1:
                logger.info("Notebook Job %s completed", job_name)
                return
            if (status.failed or 0) >= 1:
                logs = _job_pod_logs(core_api, namespace, job_name)
                raise AssertionError(f"Notebook Job {job_name} failed:\n{logs}")
            time.sleep(_NOTEBOOK_JOB_POLL_SECONDS)
        logs = _job_pod_logs(core_api, namespace, job_name)
        raise AssertionError(f"Notebook Job {job_name} timed out after {timeout}s:\n{logs}")
    finally:
        try:
            batch_api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                propagation_policy="Background",
                _request_timeout=30,
            )
        except Exception as exc:
            logger.warning("Failed to delete notebook Job %s: %s", job_name, exc)
