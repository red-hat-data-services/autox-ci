"""KFP pipeline run state helpers shared across test suites."""

import logging
import time

_logger = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset({"SUCCEEDED", "SKIPPED", "CANCELLED"})
_FAILED_STATES = frozenset({"FAILED", "ERROR", "SYSTEM_ERROR"})
_GET_FAILED_TASKS_RETRIES = 4
_GET_FAILED_TASKS_DELAY = 5.0


def _normalize_state(state) -> str | None:
    """Normalize a state value (str or enum) to an uppercase string like 'SUCCEEDED'."""
    if state is None:
        return None
    return str(getattr(state, "name", state)).upper()


def _get_run_state(detail) -> str | None:
    """Extract the run state string from a KFP run detail object."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    return _normalize_state(state)


def _run_succeeded(detail) -> bool:
    """Return True if the run finished with SUCCEEDED state."""
    return _get_run_state(detail) == "SUCCEEDED"


def _run_failed(detail) -> bool:
    """Return True if the run finished with FAILED state (not timeout or still running)."""
    return _get_run_state(detail) == "FAILED"


def _get_failed_task_names(client, run_id: str) -> list[str]:
    """Return display names of user-visible FAILED/ERROR tasks from a pipeline run.

    Retries when the run is in a terminal failed state but task-level states have not
    yet propagated — a race that occurs with Tekton-backed managed pipelines immediately
    after the run reaches FAILED.
    """
    last_exc: Exception | None = None
    for attempt in range(_GET_FAILED_TASKS_RETRIES):
        try:
            run_detail = client.get_run(run_id)
            run_obj = getattr(run_detail, "run", run_detail)
            run_state = _normalize_state(getattr(run_obj, "state", None)) or ""
            rd = getattr(run_obj, "run_details", None)
            task_list = getattr(rd, "task_details", None) if rd else None

            failed = []
            if task_list:
                for task in task_list:
                    name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                    state_str = _normalize_state(getattr(task, "state", None)) or ""
                    if name in ("root", "executor") or name.endswith("-driver"):
                        continue
                    if state_str in _FAILED_STATES:
                        failed.append(name)

            if failed or run_state in _TERMINAL_STATES or attempt == _GET_FAILED_TASKS_RETRIES - 1:
                return failed

            _logger.debug(
                "No failed tasks for run %s (attempt %d/%d, run_state=%s) — retrying in %.0fs",
                run_id, attempt + 1, _GET_FAILED_TASKS_RETRIES, run_state, _GET_FAILED_TASKS_DELAY,
            )
            time.sleep(_GET_FAILED_TASKS_DELAY)
        except Exception as exc:
            last_exc = exc
            _logger.warning(
                "Could not get failed task names for run %s (attempt %d/%d): %s",
                run_id, attempt + 1, _GET_FAILED_TASKS_RETRIES, exc,
            )
            if attempt < _GET_FAILED_TASKS_RETRIES - 1:
                time.sleep(_GET_FAILED_TASKS_DELAY)

    if last_exc:
        _logger.warning("All retries exhausted for run %s, last error: %s", run_id, last_exc)
    return []
