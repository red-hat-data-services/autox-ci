"""Unit coverage for the failure-class classification behind the rerun policy (RHAI-10079)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autox_tests.lib import failure_classes
from autox_tests.lib.failure_classes import (
    EarlyFlakyFailure,
    MaasSlownessTimeout,
    ResourceShortageTimeout,
    classify_pipeline_run_timeout,
    failure_class_name,
    is_transient_s3_error,
    rerun_transient_failures,
    resource_shortage_signals_from_pod,
    resource_shortage_signals_from_pods,
)

pytestmark = pytest.mark.config


def _pod(
    name: str = "pod-a",
    *,
    phase: str = "Running",
    pod_reason: str | None = None,
    conditions: list | None = None,
    container_statuses: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        status=SimpleNamespace(
            phase=phase,
            reason=pod_reason,
            conditions=conditions or [],
            container_statuses=container_statuses or [],
        ),
    )


def _condition(
    type_: str = "PodScheduled",
    status: str = "False",
    reason: str = "Unschedulable",
    message: str = "0/3 nodes are available: 3 Insufficient memory.",
) -> SimpleNamespace:
    return SimpleNamespace(type=type_, status=status, reason=reason, message=message)


def _container_status(
    name: str = "runner",
    *,
    terminated: SimpleNamespace | None = None,
    waiting: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        state=SimpleNamespace(terminated=terminated, waiting=waiting),
    )


def _client(state: str | None = "RUNNING") -> SimpleNamespace:
    return SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(run=SimpleNamespace(state=state))
    )


def _client_raising() -> SimpleNamespace:
    def _boom(run_id):
        raise RuntimeError("kfp api unreachable")

    return SimpleNamespace(get_run=_boom)


class TestExceptionHierarchy:
    def test_timeout_classes_subclass_timeout_error(self) -> None:
        assert issubclass(MaasSlownessTimeout, TimeoutError)
        assert issubclass(ResourceShortageTimeout, TimeoutError)


class TestResourceShortageSignals:
    def test_unschedulable_pod(self) -> None:
        pod = _pod(phase="Pending", conditions=[_condition()])
        signals = resource_shortage_signals_from_pod(pod)
        assert len(signals) == 1
        assert "pod-a" in signals[0]
        assert "Pending/Unschedulable" in signals[0]

    def test_oomkilled_container(self) -> None:
        pod = _pod(
            container_statuses=[
                _container_status(
                    terminated=SimpleNamespace(reason="OOMKilled", exit_code=137)
                )
            ]
        )
        signals = resource_shortage_signals_from_pod(pod)
        assert len(signals) == 1
        assert "OOMKilled" in signals[0]

    def test_evicted_pod(self) -> None:
        pod = _pod(pod_reason="Evicted", phase="Failed")
        signals = resource_shortage_signals_from_pod(pod)
        assert len(signals) == 1
        assert "Evicted" in signals[0]

    def test_healthy_pod_has_no_signals(self) -> None:
        pod = _pod(
            container_statuses=[
                _container_status(waiting=SimpleNamespace(reason=None))
            ]
        )
        assert resource_shortage_signals_from_pod(pod) == []

    def test_crashloopbackoff_is_not_a_resource_signal(self) -> None:
        """A crashing container is a real failure, not a scheduling constraint."""
        pod = _pod(
            container_statuses=[
                _container_status(waiting=SimpleNamespace(reason="CrashLoopBackOff"))
            ]
        )
        assert resource_shortage_signals_from_pod(pod) == []

    def test_signals_aggregate_across_pods(self) -> None:
        pods = [
            _pod(name="a", phase="Pending", conditions=[_condition()]),
            _pod(
                name="b",
                container_statuses=[
                    _container_status(
                        terminated=SimpleNamespace(reason="OOMKilled", exit_code=137)
                    )
                ],
            ),
            _pod(name="c"),
        ]
        signals = resource_shortage_signals_from_pods(pods)
        assert len(signals) == 2
        assert any("pod a" in s for s in signals)
        assert any("pod b" in s for s in signals)

    def test_no_pods_yields_no_signals(self) -> None:
        assert resource_shortage_signals_from_pods([]) == []
        assert resource_shortage_signals_from_pods(None) == []


class TestClassifyPipelineRunTimeout:
    def _original(self) -> TimeoutError:
        return TimeoutError("Run r1 did not reach a terminal state")

    def test_resource_signals_win_even_when_running(self, monkeypatch) -> None:
        monkeypatch.setattr(
            failure_classes,
            "collect_resource_signals_for_run",
            lambda config, run_id: ["pod x: Pending/Unschedulable"],
        )
        result = classify_pipeline_run_timeout(
            self._original(),
            client=_client(state="RUNNING"),
            run_id="r1",
            timeout=600,
            kind="autorag",
            config={"rhoai_project": "ns"},
        )
        assert isinstance(result, ResourceShortageTimeout)
        assert "Pending/Unschedulable" in str(result)

    def test_autorag_still_running_is_maas_slowness(self) -> None:
        result = classify_pipeline_run_timeout(
            self._original(),
            client=_client(state="RUNNING"),
            run_id="r1",
            timeout=3600,
            kind="autorag",
            config=None,
        )
        assert isinstance(result, MaasSlownessTimeout)

    def test_automl_still_running_is_not_rerunnable(self) -> None:
        original = self._original()
        result = classify_pipeline_run_timeout(
            original,
            client=_client(state="RUNNING"),
            run_id="r1",
            timeout=3600,
            kind="automl",
            config=None,
        )
        assert result is original

    def test_unknown_state_returns_original(self) -> None:
        original = self._original()
        result = classify_pipeline_run_timeout(
            original,
            client=_client_raising(),
            run_id="r1",
            timeout=3600,
            kind="autorag",
            config=None,
        )
        assert result is original


class TestTransientS3Errors:
    def test_builtin_connection_error(self) -> None:
        assert is_transient_s3_error(ConnectionError("reset by peer"))

    def test_client_error_transient_status(self) -> None:
        from botocore.exceptions import ClientError

        exc = ClientError(
            {
                "Error": {"Code": "SlowDown", "Message": "slow"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            "ListObjectsV2",
        )
        assert is_transient_s3_error(exc)

    def test_client_error_permanent_status(self) -> None:
        from botocore.exceptions import ClientError

        exc = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": "missing"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "ListObjectsV2",
        )
        assert not is_transient_s3_error(exc)

    def test_botocore_connection_error(self) -> None:
        from botocore.exceptions import EndpointConnectionError

        assert is_transient_s3_error(
            EndpointConnectionError(endpoint_url="https://s3.example.invalid")
        )

    def test_unrelated_error_is_not_transient(self) -> None:
        assert not is_transient_s3_error(ValueError("bad"))


class TestReportLabelling:
    @pytest.mark.parametrize(
        ("exc", "label"),
        [
            (EarlyFlakyFailure("kfp 503"), "early_flaky"),
            (MaasSlownessTimeout("still running"), "maas_slowness"),
            (ResourceShortageTimeout("unschedulable"), "resource_shortage"),
            (AssertionError("real regression"), None),
        ],
    )
    def test_failure_class_name(self, exc: BaseException, label: str | None) -> None:
        assert failure_class_name(exc) == label

    def test_failure_class_name_none(self) -> None:
        assert failure_class_name(None) is None

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (EarlyFlakyFailure("x"), True),
            (MaasSlownessTimeout("x"), True),
            (ResourceShortageTimeout("x"), False),
            (AssertionError("x"), False),
        ],
    )
    def test_rerun_condition(self, exc: BaseException, expected: bool) -> None:
        assert rerun_transient_failures(exc) is expected
