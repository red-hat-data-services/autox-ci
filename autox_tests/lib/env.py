"""Load a single `.env` file for the root `tests/` suite.

Variables already present in the process environment are not overwritten
(`override=False`), so CI or shell exports take precedence over the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# This file lives in ``tests/lib/``; the suite root is the parent of ``lib``.
_TESTS_DIR = Path(__file__).resolve().parents[1]


def load_tests_env(component: Literal["autorag", "automl"] | None = None) -> None:
    """Load ``tests/.env`` if present; never override existing environment variables."""
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


def tests_dir() -> Path:
    """Return the absolute path to the ``tests/`` directory."""
    return _TESTS_DIR


def repo_root() -> Path:
    """Return the VCS / project root (parent directory of the ``e2e-tests/`` suite folder)."""
    return _TESTS_DIR.parent


def resolve_suite_asset_path(rel: str) -> Path:
    """Resolve a path from JSON configs to an absolute filesystem path.

    Configs may use ``tests/data/...`` from the layout where this suite lived in a repo's
    ``tests/`` folder. Here the package maps to ``e2e-tests/`` with assets under ``data/``;
    paths starting with ``tests/`` are resolved under :func:`tests_dir` after stripping that
    prefix. Other paths are taken relative to :func:`tests_dir` (e.g. ``data/foo.csv``).
    """
    p = rel.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("tests/"):
        p = p[len("tests/") :]
    return tests_dir() / p
