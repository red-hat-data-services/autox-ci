"""Integration-style dry-run tests for automl_benchmark.orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from automl_benchmark.orchestrator import BenchmarkOrchestrator, _dataset_matches_filter, _validate_dataset_entry
from tests.conftest import dry_run_arguments_from_row, read_results_csv


class TestOrchestratorHelpers:
    @pytest.mark.parametrize(
        ("dataset_filter", "task_type", "expected"),
        [
            ("all", "binary", True),
            ("all", "timeseries", True),
            ("tabular", "binary", True),
            ("tabular", "timeseries", False),
            ("timeseries", "binary", False),
            ("timeseries", "timeseries", True),
        ],
    )
    def test_dataset_matches_filter(self, dataset_filter: str, task_type: str, expected: bool) -> None:
        ds = {"task_type": task_type}
        assert _dataset_matches_filter(ds, dataset_filter) is expected

    def test_validate_tabular_missing_fields(self) -> None:
        err = _validate_dataset_entry({"train_data_file_key": "k.csv"}, "bad")
        assert err is not None and "label_column" in err

    def test_validate_timeseries_missing_columns(self) -> None:
        ds = {"train_data_file_key": "k.csv", "task_type": "timeseries", "target": "y"}
        err = _validate_dataset_entry(ds, "ts-bad")
        assert err is not None and "id_column" in err


class TestDryRunExecution:
    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_dry_run_does_not_call_kfp(
        self,
        mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "dry.csv"
        assert automl_orchestrator.execute(output_csv=out, dry_run=True) == 0
        mock_kfp.assert_not_called()

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_dry_run_writes_all_valid_datasets(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "all.csv"
        assert automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="all") == 0
        df = read_results_csv(out)
        # breast-w, boston, sunspots (broken-tabular skipped)
        assert len(df) == 3
        assert set(df["state"]) == {"DRY_RUN"}
        assert df["run_id"].fillna("").astype(str).str.len().eq(0).all()

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_dataset_filter_tabular(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "tabular.csv"
        assert automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="tabular") == 0
        df = read_results_csv(out)
        assert len(df) == 2
        assert set(df["dataset_id"]) == {"breast-w-test", "boston-test"}

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_dataset_filter_timeseries(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "ts.csv"
        assert automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="timeseries") == 0
        df = read_results_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["dataset_id"] == "sunspots-test"
        assert df.iloc[0]["task_type"] == "timeseries"

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_pipeline_arguments_in_metrics_blob(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "args.csv"
        automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="tabular")
        df = read_results_csv(out)
        breast = df[df["dataset_id"] == "breast-w-test"].iloc[0]
        args = dry_run_arguments_from_row(breast)
        assert args["top_n"] == 4  # manifest pipeline_arguments override

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_tabular_and_timeseries_pipeline_names(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        caplog.set_level(logging.INFO)
        out = tmp_path / "pipelines.csv"
        automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="all")
        messages = " ".join(r.message for r in caplog.records)
        assert "autogluon-tabular-training-pipeline.yaml" in messages
        assert "autogluon-timeseries-training-pipeline.yaml" in messages

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_fail_fast_stops_on_invalid_dataset(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "ff.csv"
        code = automl_orchestrator.execute(output_csv=out, dry_run=True, fail_fast=True, dataset_filter="all")
        assert code == 1
        # fail_fast returns before write_results_csv when the invalid row is reached
        assert not out.is_file()

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_without_fail_fast_skips_invalid_and_continues(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "continue.csv"
        assert automl_orchestrator.execute(output_csv=out, dry_run=True, fail_fast=False) == 0
        df = read_results_csv(out)
        assert "broken-tabular" not in set(df["dataset_id"])

    @patch("automl_benchmark.orchestrator.upload_batch_aggregated")
    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_dry_run_skips_batch_s3_upload(
        self,
        _mock_kfp,
        mock_batch_upload,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "noupload.csv"
        automl_orchestrator.execute(output_csv=out, dry_run=True)
        mock_batch_upload.assert_not_called()

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_cli_tabular_package_path_override(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tabular_pipeline_path: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "cli_tab.csv"
        assert (
            automl_orchestrator.execute(
                output_csv=out,
                dry_run=True,
                dataset_filter="tabular",
                tabular_package_path_cli=str(tabular_pipeline_path),
            )
            == 0
        )
        _, settings, _, _ = automl_orchestrator.load_config_and_datasets(
            dataset_filter="tabular",
            tabular_package_path_cli=str(tabular_pipeline_path),
        )
        assert settings.pipeline_yaml.resolve() == tabular_pipeline_path.resolve()

    @patch("automl_benchmark.orchestrator.create_kfp_client")
    def test_timeseries_arguments_shape(
        self,
        _mock_kfp,
        automl_orchestrator: BenchmarkOrchestrator,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "ts_args.csv"
        automl_orchestrator.execute(output_csv=out, dry_run=True, dataset_filter="timeseries")
        row = read_results_csv(out).iloc[0]
        args = dry_run_arguments_from_row(row)
        assert args["id_column"] == "item_id"
        assert args["timestamp_column"] == "timestamp"
        assert args["target"] == "target"
        assert args["prediction_length"] == 12


class TestLoadConfig:
    def test_merged_credentials_and_upload_disabled(
        self,
        automl_orchestrator: BenchmarkOrchestrator,
    ) -> None:
        cfg, settings, datasets, config_dir = automl_orchestrator.load_config_and_datasets()
        assert str(cfg["kfp"]["host"]).startswith("https://")
        assert settings.train_data_bucket_name == "test-benchmark-bucket"
        assert settings.upload_benchmark_results is False
        assert settings.top_n == 2
        assert config_dir == automl_orchestrator.config_path.parent
        assert len(datasets) == 4

    def test_missing_credentials_ini_fails(self, automl_benchmark_yaml: Path, tmp_path: Path) -> None:
        orch = BenchmarkOrchestrator(automl_benchmark_yaml, credentials_ini_path=tmp_path / "missing.ini")
        with pytest.raises(FileNotFoundError):
            orch.load_config_and_datasets()

    def test_missing_pipeline_file_returns_one(
        self,
        automl_benchmark_yaml: Path,
        automl_credentials_ini: Path,
        tmp_path: Path,
    ) -> None:
        bad_yaml = tmp_path / "bad_benchmark.yaml"
        bad_yaml.write_text(
            "pipeline:\n  package_path: ../nope/tabular.yaml\n"
            "  timeseries_package_path: ../nope/ts.yaml\n"
            "dataset_manifest_path: dataset_manifest.yaml\n",
            encoding="utf-8",
        )
        manifest = automl_benchmark_yaml.parent / "dataset_manifest.yaml"
        orch = BenchmarkOrchestrator(bad_yaml, credentials_ini_path=automl_credentials_ini)
        out = tmp_path / "fail.csv"
        assert orch.execute(output_csv=out, dry_run=True) == 1
