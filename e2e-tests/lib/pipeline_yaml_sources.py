"""Resolve KFP pipeline package paths: local ``pipeline.yaml`` or download from GitHub."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from tests.lib.env import load_tests_env

# Precompiled IR paths under ``pipelines/training/`` in pipelines-components (rhoai-3.4 branch).
PIPELINE_TRAINING_TABULAR_REL = "automl/autogluon_tabular_training_pipeline/pipeline.yaml"
PIPELINE_TRAINING_TIMESERIES_REL = "automl/autogluon_timeseries_training_pipeline/pipeline.yaml"
PIPELINE_TRAINING_AUTORAG_REL = "autorag/documents_rag_optimization_pipeline/pipeline.yaml"

PIPELINE_YAML_TABULAR_ENV = "RHOAI_PIPELINE_YAML_TABULAR"
PIPELINE_YAML_TIMESERIES_ENV = "RHOAI_PIPELINE_YAML_TIMESERIES"
PIPELINE_YAML_AUTORAG_ENV = "RHOAI_PIPELINE_YAML_AUTORAG"

# Default: https://github.com/red-hat-data-services/pipelines-components/tree/rhoai-3.4/pipelines/training
PIPELINES_COMPONENTS_REPO_ENV = "RHOAI_PIPELINES_COMPONENTS_REPO"
PIPELINES_COMPONENTS_REF_ENV = "RHOAI_PIPELINES_COMPONENTS_REF"

_DEFAULT_REPO = "red-hat-data-services/pipelines-components"
_DEFAULT_REF = "rhoai-3.4"
_RAW_GITHUB = "https://raw.githubusercontent.com"


def _default_raw_url(repo_relative_under_training: str) -> str:
    load_tests_env()
    repo = (os.environ.get(PIPELINES_COMPONENTS_REPO_ENV) or "").strip() or _DEFAULT_REPO
    ref = (os.environ.get(PIPELINES_COMPONENTS_REF_ENV) or "").strip() or _DEFAULT_REF
    rel = repo_relative_under_training.lstrip("/")
    return f"{_RAW_GITHUB}/{repo}/{ref}/pipelines/training/{rel}"


def _download(url: str, dest: Path, timeout_seconds: float = 120.0) -> None:
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
    repo_relative_under_training: str,
    cache_dir: Path,
    cache_file_name: str,
) -> str:
    """Return absolute path to a ``pipeline.yaml``.

    If ``path_env_var`` is set to a non-empty path, that file is used (must exist).
    Otherwise the YAML is downloaded from the default pipelines-components raw URL
    (override repo/ref via ``RHOAI_PIPELINES_COMPONENTS_*``).
    """
    load_tests_env()
    raw = (os.environ.get(path_env_var) or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"{path_env_var} is set but does not point to a file: {p}"
            )
        return str(p)

    url = _default_raw_url(repo_relative_under_training)
    dest = cache_dir / cache_file_name
    if dest.is_file() and dest.stat().st_size > 0:
        return str(dest.resolve())
    _download(url, dest)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded pipeline YAML is missing or empty: {dest}")
    return str(dest.resolve())
