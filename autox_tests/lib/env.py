"""Load local env files from ``autox_tests/``.

Variables already present in the process environment are not overwritten
(`override=False`), so CI or shell exports take precedence over the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# This file lives in ``autox_tests/lib/``; the suite root is the parent of ``lib``.
_TESTS_DIR = Path(__file__).resolve().parents[1]


def load_tests_env(
    component: Literal["autorag", "automl"] | None = None,
) -> None:
    """Load the suite's env file or ``autox_tests/.env`` if no suite is given."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    if component == "autorag":
        env_path = _TESTS_DIR / ".env.rag"
    elif component == "automl":
        env_path = _TESTS_DIR / ".env.ml"
    else:
        env_path = _TESTS_DIR / ".env"
    load_dotenv(env_path, override=False)
