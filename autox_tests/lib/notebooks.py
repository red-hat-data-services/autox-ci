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
_DEFAULT_NOTEBOOK_JOB_TIMEOUT_SECONDS = 900
_NOTEBOOK_JOB_POLL_SECONDS = 5

logger = logging.getLogger(__name__)


_NOTEBOOK_JOB_PROGRAM = r'''import json
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

for index, notebook_key in enumerate(json.loads(os.environ["NOTEBOOK_S3_KEYS"])):
    workdir = Path("/tmp/notebooks") / str(index)
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "input.ipynb"
    output_path = workdir / "output.ipynb"
    s3.download_file(os.environ["NOTEBOOK_S3_BUCKET"], notebook_key, str(input_path))

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
'''


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
        try:
            output = core_api.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                tail_lines=100,
                _request_timeout=30,
            )
        except Exception as exc:
            output = f"Unable to read pod log: {exc}"
        logs.append(f"--- pod/{name} ---\n{output}")
    return "\n".join(logs) or "No pod logs were available."


def run_notebook_as_k8s_job(
    *,
    bucket: str,
    notebook_key: str,
    config: dict[str, Any],
    secret_names: list[str],
    inject_mock_input: bool = False,
) -> None:
    """Execute one S3-hosted notebook in an ephemeral Job in the test namespace."""
    run_notebooks_as_k8s_job(
        bucket=bucket,
        notebook_keys=[notebook_key],
        config=config,
        secret_names=secret_names,
        inject_mock_input=inject_mock_input,
    )


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
        logger.info(
            "Skipping notebook execution; set %s to run notebooks in a Kubernetes Job",
            RHOAI_NOTEBOOK_RUNNER_IMAGE_ENV,
        )
        return

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
    unique_secrets = list(dict.fromkeys(name for name in secret_names if name))
    env_from = [
        k8s_client.V1EnvFromSource(
            secret_ref=k8s_client.V1SecretEnvSource(name=name, optional=False)
        )
        for name in unique_secrets
    ]
    container = k8s_client.V1Container(
        name="notebook-runner",
        image=image,
        image_pull_policy="IfNotPresent",
        command=["python", "-c", _NOTEBOOK_JOB_PROGRAM],
        env=[
            k8s_client.V1EnvVar(name="NOTEBOOK_S3_BUCKET", value=bucket),
            k8s_client.V1EnvVar(name="NOTEBOOK_S3_KEYS", value=json.dumps(notebook_keys)),
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
