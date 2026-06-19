"""KFP pipeline run state helpers shared across test suites."""


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
