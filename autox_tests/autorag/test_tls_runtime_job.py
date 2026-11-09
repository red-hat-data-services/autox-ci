"""Check runtime completion against pod state when Job counters lag behind."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kubernetes import client

from autox_tests.lib.tls import runtime

pytestmark = pytest.mark.config


@pytest.fixture
def probe_job(monkeypatch, tmp_path):
    core, batch = Mock(), Mock()
    monkeypatch.setattr(runtime.client, "BatchV1Api", lambda **kwargs: batch)
    monkeypatch.setattr(runtime, "keep_resources", lambda: False)
    monkeypatch.setenv("AUTOX_TLS_REPORT_DIR", str(tmp_path))
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    case = {"id": "M06-maas-env", "expected": "success"}
    report = {"id": case["id"], "outcome": "success", "passed": True}
    core.read_namespaced_pod_log.return_value = runtime.RESULT_PREFIX + json.dumps(report)
    batch.read_namespaced_job_status.return_value = client.V1Job(status=client.V1JobStatus())

    def pod(phase):
        return client.V1Pod(
            metadata=client.V1ObjectMeta(name="probe-pod"),
            status=client.V1PodStatus(phase=phase),
        )

    return SimpleNamespace(core=core, batch=batch, case=case, report=report, pod=pod, clock=clock, path=tmp_path)


def run(probe_job):
    return runtime.run_job(probe_job.core, "tests", "product:test", probe_job.case, {}, timeout=1)


def test_succeeded_pod_does_not_wait_for_job_accounting(probe_job):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod("Succeeded")])

    assert run(probe_job) == probe_job.report

    assert probe_job.clock.now == 0
    probe_job.batch.read_namespaced_job_status.assert_called_once()
    evidence = json.loads(next(probe_job.path.glob("*.json")).read_text())
    assert evidence["pod_phase"] == "Succeeded"
    assert evidence["job_succeeded"] == 0
    probe_job.batch.delete_namespaced_job.assert_called_once()
    probe_job.core.delete_namespaced_secret.assert_called_once()
    probe_job.core.delete_namespaced_config_map.assert_called_once()


def test_job_counter_does_not_replace_terminal_pod_evidence(probe_job):
    probe_job.batch.read_namespaced_job_status.return_value = client.V1Job(status=client.V1JobStatus(succeeded=1))
    probe_job.core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[probe_job.pod("Running")]),
        SimpleNamespace(items=[probe_job.pod("Succeeded")]),
    ]

    assert run(probe_job) == probe_job.report

    assert probe_job.batch.read_namespaced_job_status.call_count == 2


@pytest.mark.parametrize("phase", ["Succeeded", "Failed"])
def test_terminal_pod_still_requires_a_matching_probe_report(probe_job, phase):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod(phase)])
    probe_job.core.read_namespaced_pod_log.return_value = runtime.RESULT_PREFIX + json.dumps({
        **probe_job.report, "outcome": "error", "passed": False,
    })

    with pytest.raises(AssertionError, match='"passed": false'):
        run(probe_job)

    assert len(list(probe_job.path.glob("*.json"))) == 1
    probe_job.batch.delete_namespaced_job.assert_called_once()


def test_failed_pod_cannot_pass_with_a_success_report(probe_job):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod("Failed")])

    with pytest.raises(AssertionError, match="Pod did not succeed"):
        run(probe_job)


def test_succeeded_pod_without_report_cannot_pass(probe_job):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod("Succeeded")])
    probe_job.core.read_namespaced_pod_log.return_value = "process exited without structured evidence"

    with pytest.raises(AssertionError, match="exactly one structured result"):
        run(probe_job)


@pytest.mark.parametrize("condition_only", [False, True])
def test_explicit_job_failure_is_not_hidden_by_a_succeeded_pod(probe_job, condition_only):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod("Succeeded")])
    probe_job.batch.read_namespaced_job_status.return_value = client.V1Job(status=client.V1JobStatus(
        failed=None if condition_only else 1,
        conditions=[client.V1JobCondition(type="Failed", status="True")] if condition_only else None,
    ))

    with pytest.raises(AssertionError, match="Job failed"):
        run(probe_job)


def test_running_pod_still_times_out(probe_job):
    probe_job.core.list_namespaced_pod.return_value = SimpleNamespace(items=[probe_job.pod("Running")])

    with pytest.raises(AssertionError, match="Probe Job timed out"):
        run(probe_job)

    probe_job.core.read_namespaced_pod_log.assert_not_called()
    probe_job.batch.delete_namespaced_job.assert_called_once()
