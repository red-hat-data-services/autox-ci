"""CLI and argument wiring tests for AutoML benchmark orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from automl_benchmark.cli import default_config_path, main


class TestDefaultConfigPath:
    def test_default_relative_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BENCHMARK_CONFIG_PATH", raising=False)
        assert default_config_path() == Path("config/benchmark.yaml")

    def test_env_override(self, automl_benchmark_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BENCHMARK_CONFIG_PATH", str(automl_benchmark_yaml))
        assert default_config_path() == automl_benchmark_yaml


class TestMainArgumentForwarding:
    """Ensure every CLI flag reaches BenchmarkOrchestrator.execute with expected values."""

    @pytest.fixture
    def base_argv(self, automl_benchmark_yaml: Path, automl_env_file: Path, tmp_path: Path) -> list[str]:
        return [
            "--config",
            str(automl_benchmark_yaml),
            "--env-file",
            str(automl_env_file),
            "--output",
            str(tmp_path / "out.csv"),
            "--dry-run",
        ]

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_minimal_dry_run(self, mock_execute, base_argv: list[str]) -> None:
        assert main(base_argv) == 0
        kwargs = mock_execute.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["fail_fast"] is False
        assert kwargs["dataset_filter"] == "all"
        assert kwargs["skip_identical_runs"] is True
        assert kwargs["tabular_package_path_cli"] is None
        assert kwargs["timeseries_package_path_cli"] is None

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_fail_fast_flag(self, mock_execute, base_argv: list[str]) -> None:
        argv = [*base_argv, "--fail-fast"]
        assert main(argv) == 0
        assert mock_execute.call_args.kwargs["fail_fast"] is True

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    @pytest.mark.parametrize("dataset_filter", ["all", "tabular", "timeseries"])
    def test_dataset_filter(
        self,
        mock_execute,
        base_argv: list[str],
        dataset_filter: str,
    ) -> None:
        argv = [*base_argv, "--dataset-filter", dataset_filter]
        assert main(argv) == 0
        assert mock_execute.call_args.kwargs["dataset_filter"] == dataset_filter

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_rerun_identical_experiments(self, mock_execute, base_argv: list[str]) -> None:
        argv = [*base_argv, "--rerun-identical-experiments"]
        assert main(argv) == 0
        assert mock_execute.call_args.kwargs["skip_identical_runs"] is False

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_tabular_package_path_cli(
        self,
        mock_execute,
        base_argv: list[str],
        tabular_pipeline_path: Path,
    ) -> None:
        argv = [*base_argv, "--tabular-package-path", str(tabular_pipeline_path)]
        assert main(argv) == 0
        assert mock_execute.call_args.kwargs["tabular_package_path_cli"] == str(tabular_pipeline_path)

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_timeseries_package_path_cli(
        self,
        mock_execute,
        base_argv: list[str],
        timeseries_pipeline_path: Path,
    ) -> None:
        argv = [*base_argv, "--timeseries-package-path", str(timeseries_pipeline_path)]
        assert main(argv) == 0
        assert mock_execute.call_args.kwargs["timeseries_package_path_cli"] == str(timeseries_pipeline_path)

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_env_tabular_package_path_default(
        self,
        mock_execute,
        base_argv: list[str],
        tabular_pipeline_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BENCHMARK_TABULAR_PACKAGE_PATH", str(tabular_pipeline_path))
        # base_argv without explicit --tabular-package-path should pick up env default
        assert main(base_argv) == 0
        assert mock_execute.call_args.kwargs["tabular_package_path_cli"] == str(tabular_pipeline_path)

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_env_config_and_env_file(
        self,
        mock_execute,
        automl_benchmark_yaml: Path,
        automl_env_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BENCHMARK_CONFIG_PATH", str(automl_benchmark_yaml))
        monkeypatch.setenv("BENCHMARK_ENV_FILE", str(automl_env_file))
        argv = ["--dry-run", "--output", str(tmp_path / "env_out.csv")]
        assert main(argv) == 0
        mock_execute.assert_called_once()

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=0)
    def test_verbose_does_not_change_exit_code(self, mock_execute, base_argv: list[str]) -> None:
        assert main([*base_argv, "-v"]) == 0


class TestMainErrors:
    def test_missing_config_returns_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_benchmark.yaml"
        assert main(["--config", str(missing), "--dry-run"]) == 1

    @patch("automl_benchmark.cli.BenchmarkOrchestrator.execute", return_value=1)
    def test_orchestrator_failure_propagates(
        self,
        _mock_execute,
        automl_benchmark_yaml: Path,
        automl_env_file: Path,
    ) -> None:
        argv = [
            "--config",
            str(automl_benchmark_yaml),
            "--env-file",
            str(automl_env_file),
            "--dry-run",
        ]
        assert main(argv) == 1
