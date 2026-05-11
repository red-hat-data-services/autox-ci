"""Poll Kubeflow Pipeline runs and print overall and per-task status."""

from __future__ import annotations

import os
import re
import time
from typing import Any

_FINISH_STATES = frozenset({"succeeded", "failed", "skipped", "error"})

_UUID_ONLY_DISPLAY_NAME = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _should_omit_task(display_name: str, *, pipeline_display_name: str | None) -> bool:
    """Return True for KFP/Argo scaffolding (drivers, root DAG, executor, loops), not DSL components."""
    if not display_name:
        return True
    name = display_name.strip()
    if _UUID_ONLY_DISPLAY_NAME.match(name):
        return True
    key = name.lower()
    if key.endswith("-driver"):
        return True
    if key in ("root", "executor"):
        return True
    if key.startswith("for-loop"):
        return True
    if key.startswith("iteration-item"):
        return True
    if key.startswith("iteration-iterations"):
        return True
    if pipeline_display_name:
        if re.match(rf"^{re.escape(pipeline_display_name.lower())}-[a-z0-9]+$", key):
            return True
    return False


def _task_display_label(task: Any) -> str:
    """Return the human-readable label for a KFP task (display_name or task_id fallback)."""
    return (getattr(task, "display_name", None) or getattr(task, "task_id", None) or "").strip()


def _runtime_state_str(state: Any) -> str:
    """Convert a KFP runtime state (str or enum) to its string representation."""
    if state is None:
        return "unknown"
    if isinstance(state, str):
        return state
    return str(state)


def _format_component_task_lines(
    run: Any,
    *,
    pipeline_display_name: str | None,
) -> list[str]:
    """Build one indented line per user component task; skip infra rows."""
    details = getattr(run, "run_details", None)
    if not details:
        return [
            "  (no run_details yet; tasks appear as the pipeline schedules work)",
        ]
    tasks = getattr(details, "task_details", None) or []
    if not tasks:
        return ["  (no task_details yet)"]
    component_tasks = [t for t in tasks if not _should_omit_task(_task_display_label(t), pipeline_display_name=pipeline_display_name)]
    if not component_tasks:
        return [
            "  (no component tasks yet; only drivers/root/executor/loops in run_details)",
        ]
    lines: list[str] = []
    for task in sorted(
        component_tasks,
        key=lambda t: (_task_display_label(t).lower(), getattr(t, "task_id", "") or ""),
    ):
        name = _task_display_label(task) or "?"
        tid = getattr(task, "task_id", None) or ""
        st = _runtime_state_str(getattr(task, "state", None))
        pod = getattr(task, "pod_name", None)
        chunks = [f"  - {name}", f"state={st}"]
        if tid:
            chunks.append(f"task_id={tid}")
        if pod:
            chunks.append(f"pod={pod}")
        err = getattr(task, "error", None)
        if err is not None:
            msg = getattr(err, "message", None) or str(err)
            if msg:
                short = msg.replace("\n", " ").strip()
                if len(short) > 120:
                    short = short[:117] + "..."
                chunks.append(f"error={short}")
        lines.append(" ".join(chunks))
    return lines


def _write_progress_line(line: str) -> None:
    """Write one progress line so it is visible while pytest captures stdout/stderr.

    Uses the controlling terminal when available (``/dev/tty``), which pytest does not
    capture, so long-running integration tests show live status. Falls back to ``print``.
    """
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:
            tty.write(line + "\n")
            tty.flush()
    except OSError:
        print(line, flush=True)


def _emit_poll_block(
    *,
    run_id: str,
    state_raw: Any,
    elapsed_s: float,
    run: Any,
    pipeline_display_name: str | None,
    blank_before: bool,
) -> None:
    """Write one progress block: run-level state line plus per-task status lines."""
    if blank_before:
        _write_progress_line("")
    _write_progress_line(
        f"[rhoai-kfp] run_id={run_id} state={_runtime_state_str(state_raw)} elapsed_s={elapsed_s:.0f}"
    )
    for line in _format_component_task_lines(run, pipeline_display_name=pipeline_display_name):
        _write_progress_line(line)


def wait_for_run_with_progress(
    client: Any,
    run_id: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int = 25,
    pipeline_display_name: str | None = None,
) -> Any:
    """Poll ``get_run`` until the run reaches a terminal state or timeout.

    After each poll, emits run state (with elapsed wall time) and one line per **user**
    component task—omitting KFP/Argo scaffolding (``*-driver``, ``root``, ``executor``,
    loop tasks, UUID-only names, and the compiled root DAG pod named
    ``{pipeline_display_name}-<suffix>`` when ``pipeline_display_name`` is set).

    Output uses :func:`_write_progress_line` (``/dev/tty`` when available). Set
    ``RHOAI_KFP_PIPELINE_DISPLAY_NAME`` if you do not pass ``pipeline_display_name``.

    Args:
        client: ``kfp.Client`` instance.
        run_id: Pipeline run ID.
        timeout_seconds: Maximum wall time to wait.
        poll_interval_seconds: Seconds between ``get_run`` polls and progress snapshots.
        pipeline_display_name: Pipeline ``name=`` from ``@dsl.pipeline`` (filters the root
            DAG task). Falls back to env ``RHOAI_KFP_PIPELINE_DISPLAY_NAME``.

    Returns:
        Final ``V2beta1Run`` from the API.

    Raises:
        TimeoutError: If the run does not finish within ``timeout_seconds``.
    """
    display_name = (pipeline_display_name or os.environ.get("RHOAI_KFP_PIPELINE_DISPLAY_NAME") or "").strip() or None
    _write_progress_line(
        f"[rhoai-kfp] tracking run_id={run_id} poll_interval_s={poll_interval_seconds} "
        f"pipeline_filter={display_name or 'generic'}"
    )
    _write_progress_line("")
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    poll_index = 0
    while True:
        run = client.get_run(run_id)
        state_raw = getattr(run, "state", None)
        state = (state_raw or "").lower()
        elapsed = time.monotonic() - started
        _emit_poll_block(
            run_id=run_id,
            state_raw=state_raw,
            elapsed_s=elapsed,
            run=run,
            pipeline_display_name=display_name,
            blank_before=poll_index > 0,
        )
        poll_index += 1
        if state in _FINISH_STATES:
            _write_progress_line("")
            return run
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Run {run_id} did not finish within {timeout_seconds}s (last state={state_raw!r})"
            )
        time.sleep(max(1, poll_interval_seconds))


def run_succeeded(run: Any) -> bool:
    """Return True if the run object represents a successful completion."""
    state = getattr(run, "state", None)
    if isinstance(state, str):
        return state.upper() == "SUCCEEDED"
    return False


def run_failed_terminal(run: Any) -> bool:
    """Return True if the run ended in a failure-style terminal state (FAILED or ERROR)."""
    state = getattr(run, "state", None)
    if isinstance(state, str):
        return state.upper() in {"FAILED", "ERROR"}
    return False
