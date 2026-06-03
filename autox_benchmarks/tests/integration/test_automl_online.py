"""Online integration tests for AutoML benchmark orchestrator (real KFP + S3).

Prerequisite: config/credentials.ini (or BENCHMARK_CREDENTIALS_PATH) with working
[kfp], [storage], [s3], and KFP token. Smoke training CSV is uploaded automatically
from tests/fixtures/automl/integration/breast-w_n200.csv when missing on the bucket.

Run:
  cd autox_benchmarks
  pytest tests/integration/ -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from automl_benchmark.orchestrator import BenchmarkOrchestrator
from benchmark_common.run_state import is_success_state, is_terminal_state
from tests.conftest import REPO_ROOT, read_results_csv
from tests.integration.conftest import INTEGRATION_BENCHMARK_YAML

pytestmark = pytest.mark.integration


class TestIntegrationPrerequisites:
    """Explicit first check; session setup also validates before any test runs."""

    def test_credentials_kfp_and_s3_ready(
        self,
        integration_context,
        integration_merged_config: tuple[dict[str, Any], Path],
    ) -> None:
        cfg, _ = integration_merged_config
        kfp = cfg.get("kfp") or {}
        assert str(kfp.get("host", "")).startswith("http")
        assert str(kfp.get("namespace", "")).strip()
        assert str(kfp.get("experiment_name", "")).strip()
        assert integration_context.kfp_client is not None
        storage = cfg.get("storage") or {}
        assert str(storage.get("train_data_bucket_name", "")).strip()


class TestSmokeBenchmarkRun:
    """End-to-end: one breast-w-smoke tabular run via the orchestrator (top_n=1)."""

    @pytest.fixture
    def orchestrator(self, integration_credentials_path: Path) -> BenchmarkOrchestrator:
        return BenchmarkOrchestrator(
            INTEGRATION_BENCHMARK_YAML,
            credentials_ini_path=integration_credentials_path,
        )

    def test_orchestrator_smoke_run_succeeds(
        self,
        orchestrator: BenchmarkOrchestrator,
        integration_output_csv: Path,
        integration_merged_config: tuple[dict[str, Any], Path],
    ) -> None:
        cfg, _ = integration_merged_config
        experiment = str((cfg.get("kfp") or {}).get("experiment_name", ""))
        assert experiment, "kfp.experiment_name must be set in credentials.ini"

        code = orchestrator.execute(
            output_csv=integration_output_csv,
            dry_run=False,
            fail_fast=True,
            dataset_filter="tabular",
            skip_identical_runs=False,
            tabular_package_path_cli=str(
                REPO_ROOT / "pipelines" / "autogluon-tabular-training-pipeline.yaml"
            ),
        )
        assert code == 0, "orchestrator returned non-zero exit code"

        df = read_results_csv(integration_output_csv)
        assert len(df) == 1, f"expected one smoke row, got {len(df)}"
        row = df.iloc[0]
        assert row["dataset_id"] == "breast-w-smoke"
        assert str(row.get("run_id", "")).strip(), "expected KFP run_id in results CSV"

        state = str(row.get("state", "")).upper()
        assert is_terminal_state(state), f"run not terminal: {state}"
        assert is_success_state(state), (
            f"smoke run failed: state={state}, error={row.get('error', '')}"
        )

    def test_cli_smoke_run_subprocess(
        self,
        integration_credentials_path: Path,
        integration_output_csv: Path,
    ) -> None:
        out = integration_output_csv.parent / "cli_smoke.csv"
        cmd = [
            sys.executable,
            "scripts/benchmark_orchestrator.py",
            "--config",
            str(INTEGRATION_BENCHMARK_YAML),
            "--credentials",
            str(integration_credentials_path),
            "--output",
            str(out),
            "--dataset-filter",
            "tabular",
            "--tabular-package-path",
            str(REPO_ROOT / "pipelines" / "autogluon-tabular-training-pipeline.yaml"),
            "--rerun-identical-experiments",
        ]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
            timeout=float(os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "7200")) + 120,
        )
        assert result.returncode == 0, (result.stderr or "") + (result.stdout or "")
        df = read_results_csv(out)
        assert len(df) == 1
        assert is_success_state(str(df.iloc[0]["state"]).upper())
