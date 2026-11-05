"""Resolve KFP pipeline package paths: local ``pipeline.yaml`` or download from GitHub."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from autox_tests.lib.env import load_tests_env

PIPELINE_YAML_TABULAR_ENV = "AUTOML_TABULAR_PIPELINE_PATH"
PIPELINE_YAML_TIMESERIES_ENV = "AUTOML_TIMESERIES_PIPELINE_PATH"
PIPELINE_YAML_AUTORAG_ENV = "AUTORAG_PIPELINE_PATH"
PIPELINE_YAML_AUTORAG_INDEXING_ENV = "AUTORAG_INDEXING_PIPELINE_PATH"


def _download(url: str, dest: Path, timeout_seconds: float = 120.0) -> None:
    """Download a URL to a local file path, raising RuntimeError on HTTP failures."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "autox-ci-e2e-tests"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching pipeline YAML from {url!r}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch pipeline YAML from {url!r}: {e}") from e
    dest.write_bytes(body)


def resolve_precompiled_pipeline_yaml(
    *,
    path_env_var: str,
    cache_dir: Path,
    cache_file_name: str,
) -> str:
    """Return absolute path to a ``pipeline.yaml``.

    The environment variable ``path_env_var`` **must** be set to one of:

    1. **Local file path** — used directly (must exist).
    2. **URL** (``http://`` or ``https://``) — downloaded into ``cache_dir``.
       This includes GitHub raw URLs, e.g.
       ``https://raw.githubusercontent.com/org/repo/branch/path/pipeline.yaml``

    Raises :class:`EnvironmentError` when the variable is unset or empty.
    """
    load_tests_env()
    raw = (os.environ.get(path_env_var) or "").strip()

    if not raw:
        raise EnvironmentError(
            f"{path_env_var} is not set. Provide a local file path or a URL "
            f"(e.g. https://raw.githubusercontent.com/org/repo/branch/path/pipeline.yaml)."
        )

    if raw.startswith(("http://", "https://")):
        dest = cache_dir / cache_file_name
        if dest.is_file() and dest.stat().st_size > 0:
            return str(dest.resolve())
        _download(raw, dest)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError(f"Downloaded pipeline YAML is missing or empty: {dest}")
        return str(dest.resolve())

    p = Path(raw).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"{path_env_var} does not point to a file: {p}"
        )
    return str(p)
