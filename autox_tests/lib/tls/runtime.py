"""Kubernetes runtime runner and JSON result validation for certificate tests."""

import json
import os
import time
import uuid
from pathlib import Path

from kubernetes import client

from autox_tests.lib.tls.settings import CredentialValues, keep_resources, timeout_seconds

PROBE_SOURCE = Path(__file__).with_name("probe.py").read_text()
RESULT_PREFIX = "AUTOX_TLS_RESULT="


def parse_result(logs, case):
    lines = [
        line[len(RESULT_PREFIX) :]
        for line in logs.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise AssertionError(
            "Probe must emit exactly one structured result; inspect retained pod logs"
        )
    result = json.loads(lines[0])
    assert result["id"] == case["id"], (
        "Probe returned a result for a different scenario"
    )
    if result.get("outcome") == "not_applicable":
        import pytest

        assert case.get("engine") == "s3" and case.get("trust") in {
            "mounted",
            "missing",
        }, "Unexpected skip outside a default-trust S3 scenario"
        assert result.get("s3_default_trust", {}).get("trusted") is True, (
            "Missing independent default-trust evidence for S3 skip"
        )
        pytest.skip(result["skip_reason"])
    assert result["outcome"] == case["expected"], json.dumps(result, sort_keys=True)
    assert result["passed"] is True, json.dumps(result, sort_keys=True)
    return result


def run_job(core, namespace, image, case, material, *, profile=None, timeout=None):
    """Run one isolated probe, always remove owned resources, persist sanitized evidence."""
    timeout = timeout_seconds(180, "RHOAI_NOTEBOOK_JOB_TIMEOUT") if timeout is None else timeout
    profile = profile or {}
    identifier = "autox-tls-" + uuid.uuid4().hex[:12]
    labels = {"app.kubernetes.io/part-of": "autox-tls", "autox-tls-session": identifier}
    batch = client.BatchV1Api(api_client=core.api_client)
    created = []
    env = [client.V1EnvVar(name="TLS_SCENARIO", value=json.dumps(case))]
    secret_data = CredentialValues({**material, **profile.get("credentials", {})})
    for name in profile.get("credentials", {}):
        env.append(
            client.V1EnvVar(
                name=name,
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(name=identifier, key=name)
                ),
            )
        )
    if profile.get("ca_pem"):
        secret_data["service-ca.pem"] = profile["ca_pem"]
        env.append(
            client.V1EnvVar(
                name="TLS_SERVICE_CA_FILE",
                value="/opt/autox-tls/material/service-ca.pem",
            )
        )
    # Data selection can live in .env.rag without modifying a credentials Secret.
    for name, value in profile.get("environment", {}).items():
        env.append(client.V1EnvVar(name=name, value=value))
    volumes = [
        client.V1Volume(
            name="probe", config_map=client.V1ConfigMapVolumeSource(name=identifier)
        ),
        client.V1Volume(
            name="material", secret=client.V1SecretVolumeSource(secret_name=identifier)
        ),
    ]
    mounts = [
        client.V1VolumeMount(
            name="probe", mount_path="/opt/autox-tls/probe", read_only=True
        ),
        client.V1VolumeMount(
            name="material", mount_path="/opt/autox-tls/material", read_only=True
        ),
    ]
    if profile.get("ca_configmap"):
        volumes.append(
            client.V1Volume(
                name="service-ca",
                config_map=client.V1ConfigMapVolumeSource(
                    name=profile["ca_configmap"],
                    items=[
                        client.V1KeyToPath(
                            key=profile.get("ca_key", "ca-bundle.crt"), path="ca.pem"
                        )
                    ],
                ),
            )
        )
        mounts.append(
            client.V1VolumeMount(
                name="service-ca", mount_path="/opt/autox-tls/service", read_only=True
            )
        )
        env.append(
            client.V1EnvVar(
                name="TLS_SERVICE_CA_FILE", value="/opt/autox-tls/service/ca.pem"
            )
        )
    env_from = (
        [
            client.V1EnvFromSource(
                secret_ref=client.V1SecretEnvSource(name=profile["secret_name"])
            )
        ]
        if profile.get("secret_name")
        else []
    )
    pod_spec = client.V1PodSpec(
        restart_policy="Never",
        active_deadline_seconds=timeout,
        containers=[
            client.V1Container(
                name="probe",
                image=image,
                image_pull_policy="Always",
                env=env,
                env_from=env_from,
                command=["python", "/opt/autox-tls/probe/probe.py"],
                volume_mounts=mounts,
                resources=client.V1ResourceRequirements(
                    requests={"cpu": "100m", "memory": "512Mi"},
                    limits={"cpu": "2", "memory": "2Gi"},
                ),
            )
        ],
        volumes=volumes,
    )
    if os.environ.get("AUTOX_TLS_PULL_SECRET"):
        pod_spec.image_pull_secrets = [
            client.V1LocalObjectReference(name=os.environ["AUTOX_TLS_PULL_SECRET"])
        ]
    try:
        core.create_namespaced_config_map(
            namespace,
            client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=identifier, labels=labels),
                data={"probe.py": PROBE_SOURCE},
            ),
        )
        created.append("configmap")
        core.create_namespaced_secret(
            namespace,
            client.V1Secret(
                metadata=client.V1ObjectMeta(name=identifier, labels=labels),
                string_data=secret_data,
            ),
        )
        created.append("secret")
        batch.create_namespaced_job(
            namespace,
            client.V1Job(
                metadata=client.V1ObjectMeta(name=identifier, labels=labels),
                spec=client.V1JobSpec(
                    backoff_limit=0,
                    active_deadline_seconds=timeout,
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels=labels), spec=pod_spec
                    ),
                ),
            ),
        )
        created.append("job")
        deadline = time.monotonic() + timeout + 30
        pods = []
        while time.monotonic() < deadline:
            pods = core.list_namespaced_pod(
                namespace, label_selector=f"job-name={identifier}"
            ).items
            status = batch.read_namespaced_job_status(identifier, namespace).status
            job_failed = bool(status.failed) or any(
                condition.type == "Failed" and condition.status == "True"
                for condition in (status.conditions or [])
            )
            # The Job controller may not have accounted for a terminated pod
            # yet (for example while processing job-tracking finalizers). Its
            # counters must not hide the probe's actual terminal state.
            if job_failed or any(
                pod.status.phase in {"Succeeded", "Failed"} for pod in pods
            ):
                break
            time.sleep(3)
        else:
            states = [
                {
                    "pod": pod.metadata.name,
                    "phase": pod.status.phase,
                    "waiting": [
                        item.state.waiting.reason
                        for item in (pod.status.container_statuses or [])
                        if item.state.waiting
                    ],
                }
                for pod in pods
            ]
            raise AssertionError(
                f"Probe Job timed out (not an expected TLS rejection): {states}"
            )
        assert len(pods) == 1, "Expected exactly one probe pod"
        pod = pods[0]
        log_error_type = None
        try:
            logs = core.read_namespaced_pod_log(
                pod.metadata.name, namespace, container="probe"
            )
        except Exception as exc:
            # API errors and container messages can contain credentials.
            logs = ""
            log_error_type = type(exc).__name__
        # Store only structured evidence; raw SDK logs may contain credentials.
        lines = [line for line in logs.splitlines() if line.startswith(RESULT_PREFIX)]
        result = (
            json.loads(lines[0][len(RESULT_PREFIX) :])
            if len(lines) == 1
            else {"id": case["id"], "outcome": "no_result"}
        )
        result.update(
            {
                "image": image,
                "image_ids": [
                    item.image_id for item in (pod.status.container_statuses or [])
                ],
                "namespace": namespace,
                "pod": pod.metadata.name,
                "pod_phase": pod.status.phase,
                "job_succeeded": status.succeeded or 0,
                "job_failed": status.failed or 0,
            }
        )
        if log_error_type:
            result.update(
                {
                    "outcome": "logs_unavailable",
                    "passed": False,
                    "log_error_type": log_error_type,
                    "pod_conditions": [
                        {"type": item.type, "status": item.status, "reason": item.reason}
                        for item in (pod.status.conditions or [])
                    ],
                    "container_states": [
                        {
                            "name": item.name,
                            "waiting_reason": item.state.waiting.reason
                            if item.state and item.state.waiting else None,
                            "terminated_reason": item.state.terminated.reason
                            if item.state and item.state.terminated else None,
                            "exit_code": item.state.terminated.exit_code
                            if item.state and item.state.terminated else None,
                        }
                        for item in (
                            (pod.status.init_container_statuses or [])
                            + (pod.status.container_statuses or [])
                        )
                    ],
                }
            )
        destination = Path(os.environ.get("AUTOX_TLS_REPORT_DIR", "tls-results"))
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / f"{identifier}.json"
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        if log_error_type:
            raise AssertionError(
                "Probe startup/result retrieval failed: logs unavailable; "
                f"inspect sanitized report {report_path}"
            ) from None
        parsed = parse_result(logs, case)
        assert pod.status.phase == "Succeeded", (
            f"Probe report passed but Pod did not succeed (phase={pod.status.phase})"
        )
        assert not job_failed, "Probe report passed but Job failed"
        return parsed
    finally:
        if not keep_resources():
            for kind in reversed(created):
                try:
                    if kind == "job":
                        batch.delete_namespaced_job(
                            identifier, namespace, propagation_policy="Foreground"
                        )
                    elif kind == "secret":
                        core.delete_namespaced_secret(identifier, namespace)
                    else:
                        core.delete_namespaced_config_map(identifier, namespace)
                except client.ApiException as exc:
                    if exc.status != 404:
                        # A cleanup failure must be visible, but must not conceal a test failure.
                        import warnings

                        warnings.warn(
                            f"Unable to clean test-owned {kind}/{identifier}: HTTP {exc.status}",
                            stacklevel=2,
                        )
