"""Smoke test for scripts/benchmark_orchestrator.py entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, read_results_csv


def test_script_dry_run_subprocess(
    automl_benchmark_yaml: Path,
    automl_env_file: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "script_out.csv"
    cmd = [
        sys.executable,
        "scripts/benchmark_orchestrator.py",
        "--config",
        str(automl_benchmark_yaml),
        "--env-file",
        str(automl_env_file),
        "--output",
        str(out),
        "--dry-run",
        "--dataset-filter",
        "tabular",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    df = read_results_csv(out)
    assert len(df) == 2
    assert (df["state"] == "DRY_RUN").all()
