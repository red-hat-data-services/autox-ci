"""Strict assertions for configured end-to-end product pipeline certificate flows."""

import hashlib
import json
import os
import re
from pathlib import Path


TLS_ERROR = re.compile(
    r"CERTIFICATE_VERIFY_FAILED|certificate verify failed|certificate verification failed|"
    r"hostname mismatch|certificate has expired|unable to get local issuer certificate|"
    r"self[- ]signed certificate|certificate signed by unknown authority",
    re.IGNORECASE,
)


def validate_flows(flows):
    """Require explicit CA faults and certificate-rejection evidence."""
    ids = set()
    for flow in flows:
        if not {"id", "family", "arguments", "expected"} <= flow.keys():
            raise ValueError("A flow needs id, family, arguments and expected")
        if flow["id"] in ids:
            raise ValueError("Flow IDs must be unique")
        ids.add(flow["id"])
        package = flow.get("package_path")
        managed = flow.get("pipeline_id") and flow.get("pipeline_version_id")
        has_explicit_target = any(
            flow.get(key)
            for key in ("package_path", "pipeline_id", "pipeline_version_id")
        )
        if has_explicit_target and bool(package) == bool(managed):
            raise ValueError(
                "Choose package_path or both pipeline_id and pipeline_version_id"
            )
        if (
            flow.get("fault") != "maas_wrong_ca"
            or flow["family"] != "autorag"
            or flow["expected"] != "tls_error"
        ):
            raise ValueError("TLS product flows require a negative AutoRAG CA fault")
        if flow["expected"] == "tls_error" and not flow.get("expected_failed_task"):
            raise ValueError("Negative flows require expected_failed_task")
    return flows


def pod_task_name(pod):
    """Identify KFP lightweight tasks even when Argo pod names hide task names."""
    keys = ("pipelines.kubeflow.org/task_name", "tekton.dev/pipelineTask")
    for values in (
        pod.metadata.labels or {},
        getattr(pod.metadata, "annotations", None) or {},
    ):
        for key in keys:
            if values.get(key):
                return values[key]
    for container in pod.spec.containers:
        if container.name != "main":
            continue
        arguments = getattr(container, "args", None) or []
        for index, argument in enumerate(arguments):
            name = None
            if argument == "--function_to_execute" and index + 1 < len(arguments):
                name = arguments[index + 1]
            elif isinstance(argument, str) and argument.startswith(
                "--function_to_execute="
            ):
                name = argument.split("=", 1)[1]
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                return name.replace("_", "-")
    return pod.metadata.name


def collect_run_evidence(core, namespace, run_id):
    """Capture metadata from all containers; return logs for assertions only, not persistence."""
    pods = core.list_namespaced_pod(
        namespace, label_selector=f"pipeline/runid={run_id}"
    ).items
    if not pods:
        raise AssertionError(
            f"No pipeline pods found for run {run_id}; cannot verify runtime evidence"
        )
    evidence = {"run_id": run_id, "pods": []}
    logs = []
    for pod in pods:
        task = pod_task_name(pod)
        statuses = {item.name: item for item in (pod.status.container_statuses or [])}
        item = {
            "pod": pod.metadata.name,
            "task": task,
            "phase": pod.status.phase,
            "containers": [],
        }
        for container in pod.spec.containers:
            status = statuses.get(container.name)
            item["containers"].append(
                {
                    "name": container.name,
                    "image": container.image,
                    "image_id": status.image_id if status else None,
                    "failed": bool(
                        status
                        and status.state.terminated
                        and status.state.terminated.exit_code
                    ),
                }
            )
            try:
                content = core.read_namespaced_pod_log(
                    pod.metadata.name, namespace, container=container.name
                )
            except Exception:
                content = ""
            logs.append(
                (task, container.name, item["containers"][-1]["failed"], content)
            )
        evidence["pods"].append(item)
    return evidence, logs


def assert_tls_failure(logs, task_fragment):
    matching = [
        content for task, _, failed, content in logs if task_fragment in task and failed
    ]
    assert matching, f"Expected failed task containing {task_fragment!r} was not found"
    assert any(TLS_ERROR.search(content) for content in matching), (
        "Expected task failed without evidence of certificate verification rejection"
    )


def write_evidence(identifier, evidence):
    path = Path(os.environ.get("AUTOX_TLS_REPORT_DIR", "tls-results"))
    path.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha256(identifier.encode()).hexdigest()[:20]
    (path / (safe_name + ".json")).write_text(
        json.dumps(evidence, indent=2, sort_keys=True)
    )
