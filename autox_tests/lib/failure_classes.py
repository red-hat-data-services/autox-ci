"""Failure-class-aware rerun support for the functional test suites (RHAI-10079).

Three failure classes drive the pytest-rerunfailures policy configured in
``run_tests.sh``:

=========================  ==============================  =====================
Class                      Exception                       Rerun expectation
=========================  ==============================  =====================
Early flaky                :class:`EarlyFlakyFailure`      Immediate rerun
MaaS slowness              :class:`MaasSlownessTimeout`    Rerun with backoff
Lack of resources          :class:`ResourceShortageTimeout`  Never (fail fast)
=========================  ==============================  =====================

The classes complement — never duplicate — the in-request retry layers that
already exist: ``KFP_API_RETRY_ATTEMPTS`` (read-only KFP calls),
``_submit_run_with_retry`` (run submission), and the KServe predict retry.
Exceptions are raised by the wait/polling helpers at the point where the
failure is observed, so the classification is derived from cluster state that
is only available there. Test reports expose the class via the
``failure_class`` user property set in the root ``conftest.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EarlyFlakyFailure(Exception):
    """Transient setup/API failure that a rerun can plausibly fix.

    Covers KFP 5xx/connection resets that escaped the in-request retry layers
    and transient S3 network errors. An immediate rerun usually succeeds.
    """


class MaasSlownessTimeout(TimeoutError):
    """An AutoRAG wait exceeded its budget because MaaS was slow.

    The MaaS inference endpoint (embedding/generation) responded slowly, so the
    run or Job was still progressing when its wait budget expired. A rerun
    after a backoff delay is likely to succeed.
    """


class ResourceShortageTimeout(TimeoutError):
    """Work was stuck or killed by insufficient cluster resources.

    Pods unschedulable, OOMKilled, or Evicted (node pressure / quota) on
    pipeline runs, notebook Jobs, or KServe predictors. Rerunning cannot fix
    this and must not mask the problem.
    """


def _pod_name(pod: Any) -> str:
    return getattr(getattr(pod, "metadata", None), "name", None) or "<unnamed>"


def resource_shortage_signals_from_pod(pod: Any) -> list[str]:
    """Return human-readable lack-of-resources signals found on one pod.

    Detects unschedulable pods (``PodScheduled=False/Unschedulable``), OOMKilled
    containers, and Evicted pods (node pressure or quota). CrashLoopBackOff is
    deliberately *not* treated as a resource signal: a crashing container is a
    real failure, not a scheduling constraint.
    """
    signals: list[str] = []
    name = _pod_name(pod)
    status = getattr(pod, "status", None)

    if getattr(status, "reason", None) == "Evicted":
        message = getattr(status, "message", None) or "no message"
        signals.append(f"pod {name}: Evicted ({message})")

    for condition in getattr(status, "conditions", None) or []:
        if (
            getattr(condition, "type", None) == "PodScheduled"
            and getattr(condition, "status", None) == "False"
            and getattr(condition, "reason", None) == "Unschedulable"
        ):
            message = getattr(condition, "message", None) or "no message"
            signals.append(f"pod {name}: Pending/Unschedulable ({message})")

    for container_status in getattr(status, "container_statuses", None) or []:
        terminated = getattr(
            getattr(container_status, "state", None), "terminated", None
        )
        if terminated is not None and getattr(terminated, "reason", None) == "OOMKilled":
            signals.append(
                f"pod {name}: container {getattr(container_status, 'name', '?')} "
                f"OOMKilled (exit_code={getattr(terminated, 'exit_code', '?')})"
            )
    return signals


def resource_shortage_signals_from_pods(pods: Any) -> list[str]:
    """Aggregate lack-of-resources signals across a collection of pods."""
    signals: list[str] = []
    for pod in pods or []:
        signals.extend(resource_shortage_signals_from_pod(pod))
    return signals


def collect_resource_signals_for_run(
    config: dict[str, Any] | None, run_id: str
) -> list[str]:
    """Best-effort inspection of a pipeline run's pods for resource signals.

    Returns ``[]`` when Kubernetes access is not configured or fails — the
    classification then falls back to the KFP run state alone.
    """
    if not config:
        return []
    namespace = str(config.get("rhoai_project") or "").strip()
    if not namespace:
        return []
    try:
        from autox_tests.lib.k8s_utils import (
            list_run_pods,
            make_k8s_core_api_from_config,
        )

        core_api = make_k8s_core_api_from_config(config)
        if core_api is None:
            return []
        return resource_shortage_signals_from_pods(
            list_run_pods(core_api, namespace, run_id)
        )
    except Exception as exc:
        logger.debug("Resource-signal collection failed for run %s: %s", run_id, exc)
        return []


def _run_state_str(run_detail: Any) -> str | None:
    """Extract the uppercase KFP run state string from a run detail object."""
    run_obj = getattr(run_detail, "run", run_detail)
    state = getattr(run_obj, "state", None)
    if state is None:
        return None
    return str(getattr(state, "name", state)).upper()


def classify_pipeline_run_timeout(
    original: TimeoutError,
    *,
    client: Any,
    run_id: str,
    timeout: int,
    kind: str,
    config: dict[str, Any] | None = None,
) -> TimeoutError:
    """Classify a pipeline-run ``TimeoutError`` for the rerun policy.

    * Pod-level resource-shortage signals (unschedulable, OOMKilled, Evicted)
      → :class:`ResourceShortageTimeout` — never rerun.
    * AutoRAG run still RUNNING with no resource signals →
      :class:`MaasSlownessTimeout` — rerun with backoff. AutoRAG pipelines are
      MaaS-bound (embedding/generation), so a run that outlives its budget
      while still progressing is the MaaS-slowness signature.
    * Anything else (AutoML still running, unknown state) → the original
      exception, which matches no ``--only-rerun`` class and is not rerun:
      slow AutoGluon training is not made better by rerunning.

    Reading the run state is best effort; on failure the original exception is
    returned unchanged.
    """
    run_state: str | None = None
    try:
        run_state = _run_state_str(client.get_run(run_id))
    except Exception as exc:
        logger.debug(
            "Could not read run state for %s during classification: %s", run_id, exc
        )

    base = f"Run {run_id} did not finish within {timeout}s (last state={run_state!r})."
    signals = collect_resource_signals_for_run(config, run_id)
    if signals:
        detail = "\n".join(f"  - {signal}" for signal in signals)
        return ResourceShortageTimeout(
            f"{base} Lack of cluster resources detected:\n{detail}\n"
            "A rerun cannot fix this; the failure is not masked as flaky."
        )
    if kind == "autorag" and run_state == "RUNNING":
        return MaasSlownessTimeout(
            f"{base} The run was still progressing when its budget expired; AutoRAG "
            "pipelines are MaaS-bound (embedding/generation), so this is treated "
            "as MaaS slowness and is rerun with a backoff delay."
        )
    return original


_S3_TRANSIENT_ERROR_CODES = frozenset(
    {"SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable"}
)
_S3_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_s3_error(exc: BaseException) -> bool:
    """Return True for S3 client errors an immediate rerun can plausibly fix."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        from botocore.exceptions import ClientError
        from botocore.exceptions import ConnectionError as BotoConnectionError
    except ImportError:  # pragma: no cover - boto3 is a project dependency
        return False
    if isinstance(exc, BotoConnectionError):
        return True
    if isinstance(exc, ClientError):
        response = getattr(exc, "response", None) or {}
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        code = (response.get("Error") or {}).get("Code", "")
        return status in _S3_TRANSIENT_STATUS_CODES or code in _S3_TRANSIENT_ERROR_CODES
    return False


_FAILURE_CLASS_BY_EXCEPTION: tuple[tuple[type[BaseException], str], ...] = (
    (EarlyFlakyFailure, "early_flaky"),
    (MaasSlownessTimeout, "maas_slowness"),
    (ResourceShortageTimeout, "resource_shortage"),
)


def failure_class_name(exc: BaseException | None) -> str | None:
    """Map an exception to its failure-class label, or None when unclassified."""
    if exc is None:
        return None
    for exc_type, label in _FAILURE_CLASS_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return label
    return None


def rerun_transient_failures(exception: BaseException, *args: Any) -> bool:
    """``@pytest.mark.flaky`` condition: rerun the two transient classes only.

    Per-test escape hatch for known-flaky scenarios::

        @pytest.mark.flaky(
            reruns=2, reruns_delay=60, condition=rerun_transient_failures
        )

    ``ResourceShortageTimeout`` and plain failures never rerun.
    """
    return isinstance(exception, (EarlyFlakyFailure, MaasSlownessTimeout))
