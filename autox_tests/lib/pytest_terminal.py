"""Emit progress lines to the real pytest terminal (not the stdout/stderr capture buffer)."""

from __future__ import annotations

import sys
from typing import Any


def emit_terminal_line(
    config: Any | None,
    msg: str,
    *,
    prefix: str = "[DSPA]",
) -> None:
    """Write one line to the terminal so it is visible while capture is enabled.

    Uses pytest's ``TerminalReporter`` when available (same mechanism as ``pytest -s`` bypass
    for internal messages). Falls back to the interpreter's original ``stderr``.
    """
    line = f"{prefix} {msg}" if prefix else msg
    try:
        if config is not None:
            pm = getattr(config, "pluginmanager", None)
            if pm is not None:
                tr = pm.get_plugin("terminalreporter")
                if tr is not None:
                    tr.write_line(line)
                    return
    except Exception:
        pass
    err = getattr(sys, "__stderr__", None) or sys.stderr
    err.write(line + "\n")
    err.flush()
