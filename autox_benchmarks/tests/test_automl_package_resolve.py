"""Tests for AutoML compiled pipeline path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from benchmark_common.pipeline_package_resolve import resolve_automl_pipeline_package_paths


def _write_benchmark(config_dir: Path) -> Path:
    path = config_dir / "benchmark.yaml"
    path.write_text(
        yaml.dump(
            {
                "pipeline": {"compile": {}},
                "run": {"top_n": 1},
                "dataset_manifest_path": "dataset_manifest.yaml",
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "dataset_manifest.yaml").write_text(
        yaml.dump({"datasets": [{"id": "x", "train_data_file_key": "k.csv", "label_column": "y", "task_type": "binary"}]}),
        encoding="utf-8",
    )
    return path


def test_resolve_tabular_from_cli(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_benchmark(config_dir)
    cfg: dict = yaml.safe_load((config_dir / "benchmark.yaml").read_text(encoding="utf-8"))

    resolve_automl_pipeline_package_paths(
        cfg,
        config_dir,
        cli_tabular=str(tabular_pipeline_path),
        cli_timeseries=str(timeseries_pipeline_path),
        needs_tabular=True,
        needs_timeseries=True,
    )
    assert Path(cfg["pipeline"]["package_path"]) == tabular_pipeline_path.resolve()
    assert Path(cfg["pipeline"]["timeseries_package_path"]) == timeseries_pipeline_path.resolve()


def test_resolve_tabular_from_env(
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_benchmark(config_dir)
    cfg: dict = yaml.safe_load((config_dir / "benchmark.yaml").read_text(encoding="utf-8"))

    monkeypatch.setenv("BENCHMARK_TABULAR_PACKAGE_PATH", str(tabular_pipeline_path))
    monkeypatch.setenv("BENCHMARK_TIMESERIES_PACKAGE_PATH", str(timeseries_pipeline_path))

    resolve_automl_pipeline_package_paths(
        cfg,
        config_dir,
        cli_tabular=None,
        cli_timeseries=None,
        needs_tabular=True,
        needs_timeseries=True,
    )
    assert Path(cfg["pipeline"]["package_path"]) == tabular_pipeline_path.resolve()


def test_resolve_from_benchmark_yaml_paths(
    automl_fixture_dir: Path,
    tabular_pipeline_path: Path,
    timeseries_pipeline_path: Path,
) -> None:
    cfg = yaml.safe_load((automl_fixture_dir / "benchmark.yaml").read_text(encoding="utf-8"))
    resolve_automl_pipeline_package_paths(
        cfg,
        automl_fixture_dir,
        cli_tabular=None,
        cli_timeseries=None,
        needs_tabular=True,
        needs_timeseries=True,
    )
    assert Path(cfg["pipeline"]["package_path"]) == tabular_pipeline_path.resolve()
    assert Path(cfg["pipeline"]["timeseries_package_path"]) == timeseries_pipeline_path.resolve()


def test_cli_missing_file_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_benchmark(config_dir)
    cfg: dict = yaml.safe_load((config_dir / "benchmark.yaml").read_text(encoding="utf-8"))

    with pytest.raises(FileNotFoundError, match="package_path"):
        resolve_automl_pipeline_package_paths(
            cfg,
            config_dir,
            cli_tabular=str(tmp_path / "missing.yaml"),
            cli_timeseries=None,
            needs_tabular=True,
            needs_timeseries=False,
        )


def test_timeseries_only_reuses_tabular_path_when_unconfigured(
    tabular_pipeline_path: Path,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    rel = os.path.relpath(tabular_pipeline_path, config_dir)
    bench = config_dir / "benchmark.yaml"
    bench.write_text(
        yaml.dump(
            {
                "pipeline": {
                    "package_path": rel,
                    "timeseries_package_path": rel,
                },
                "dataset_manifest_path": "m.yaml",
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "m.yaml").write_text("datasets: []\n", encoding="utf-8")
    cfg = yaml.safe_load(bench.read_text(encoding="utf-8"))

    resolve_automl_pipeline_package_paths(
        cfg,
        config_dir,
        cli_tabular=None,
        cli_timeseries=None,
        needs_tabular=False,
        needs_timeseries=True,
    )
    # When only time series is needed, tabular path is copied to timeseries slot if missing
    assert Path(cfg["pipeline"]["timeseries_package_path"]).resolve() == tabular_pipeline_path.resolve()
