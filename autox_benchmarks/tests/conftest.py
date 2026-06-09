"""Shared fixtures for autox_benchmarks tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

# Optional: set to run integration tests when invoking the full suite (e.g. CI).
INTEGRATION_ENV_VAR = "BENCHMARK_RUN_INTEGRATION"


def integration_tests_enabled() -> bool:
    return os.environ.get(INTEGRATION_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def _invocation_includes_integration(config) -> bool:
    """True when pytest was asked to run tests/integration explicitly."""
    if integration_tests_enabled():
        return True
    root = Path(str(config.rootpath))
    for arg in config.args or []:
        raw = Path(str(arg))
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        norm = path.as_posix()
        if norm.endswith("/tests/integration") or "/tests/integration/" in f"{norm}/":
            return True
        if path.name == "integration" and path.parent.name == "tests":
            return True
    return False


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Keep integration out of default `pytest` / `pytest tests/` runs."""
    path_str = str(collection_path).replace("\\", "/")
    is_integration = "/tests/integration" in path_str or path_str.endswith("tests/integration")
    if not is_integration:
        return None
    if _invocation_includes_integration(config):
        return None
    return True


# autox_benchmarks/ (parent of tests/)
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_AUTOML = REPO_ROOT / "tests" / "fixtures" / "automl"
TABULAR_PIPELINE = REPO_ROOT / "pipelines" / "autogluon-tabular-training-pipeline.yaml"
TIMESERIES_PIPELINE = REPO_ROOT / "pipelines" / "autogluon-timeseries-training-pipeline.yaml"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tabular_pipeline_path() -> Path:
    assert TABULAR_PIPELINE.is_file(), f"Missing {TABULAR_PIPELINE}"
    return TABULAR_PIPELINE


@pytest.fixture
def timeseries_pipeline_path() -> Path:
    assert TIMESERIES_PIPELINE.is_file(), f"Missing {TIMESERIES_PIPELINE}"
    return TIMESERIES_PIPELINE


@pytest.fixture
def automl_fixture_dir() -> Path:
    return FIXTURES_AUTOML


@pytest.fixture
def automl_benchmark_yaml(automl_fixture_dir: Path) -> Path:
    path = automl_fixture_dir / "benchmark.yaml"
    assert path.is_file()
    return path


@pytest.fixture
def automl_env_file(automl_fixture_dir: Path) -> Path:
    path = automl_fixture_dir / "benchmark.env"
    assert path.is_file()
    return path


@pytest.fixture
def automl_manifest_path(automl_fixture_dir: Path) -> Path:
    path = automl_fixture_dir / "dataset_manifest.yaml"
    assert path.is_file()
    return path


def read_results_csv(path: Path) -> pd.DataFrame:
    assert path.is_file(), f"Expected CSV at {path}"
    return pd.read_csv(path)


def dry_run_arguments_from_row(row: pd.Series) -> dict:
    blob = row.get("metrics_blob")
    assert pd.notna(blob) and str(blob).strip(), "metrics_blob should hold dry-run arguments"
    return json.loads(str(blob))


@pytest.fixture
def automl_orchestrator(automl_benchmark_yaml: Path, automl_env_file: Path):
    from automl_benchmark.orchestrator import BenchmarkOrchestrator

    return BenchmarkOrchestrator(automl_benchmark_yaml, env_file=automl_env_file)


@pytest.fixture
def isolated_env(automl_benchmark_yaml, automl_env_file, monkeypatch):
    """Point default env discovery at test fixtures; clear package-path overrides."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("BENCHMARK_CONFIG_PATH", str(automl_benchmark_yaml))
    monkeypatch.setenv("BENCHMARK_ENV_FILE", str(automl_env_file))
    for key in (
        "BENCHMARK_TABULAR_PACKAGE_PATH",
        "TABULAR_PACKAGE_PATH",
        "BENCHMARK_TIMESERIES_PACKAGE_PATH",
        "TIMESERIES_PACKAGE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
