"""Tests for benchmark_common.managed_pipelines and pipeline_target_resolve."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark_common.managed_pipelines import (
    PipelineRunTarget,
    _find_pipeline_by_display_name,
    get_managed_kfp_pipeline_name,
    get_managed_pipeline_wait_timeout,
    resolve_benchmark_pipeline_mode,
    wait_for_managed_pipeline,
)
from benchmark_common.pipeline_run import submit_pipeline_run
from benchmark_common.pipeline_target_resolve import (
    resolve_automl_pipeline_targets,
    resolve_autorag_pipeline_target,
)


class TestResolveBenchmarkPipelineMode:
    def test_defaults_to_package(self):
        assert resolve_benchmark_pipeline_mode({}) == "package"

    def test_config_managed(self):
        cfg = {"pipeline": {"mode": "managed"}}
        assert resolve_benchmark_pipeline_mode(cfg) == "managed"

    def test_config_package_explicit(self):
        cfg = {"pipeline": {"mode": "package"}}
        assert resolve_benchmark_pipeline_mode(cfg) == "package"

    def test_env_var_true_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", "true")
        cfg = {"pipeline": {"mode": "package"}}
        assert resolve_benchmark_pipeline_mode(cfg) == "managed"

    def test_env_var_false_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", "false")
        cfg = {"pipeline": {"mode": "managed"}}
        assert resolve_benchmark_pipeline_mode(cfg) == "package"

    @pytest.mark.parametrize("val", ["1", "yes", "on", "True", "TRUE"])
    def test_env_var_true_values(self, monkeypatch, val):
        monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", val)
        assert resolve_benchmark_pipeline_mode({}) == "managed"

    @pytest.mark.parametrize("val", ["0", "no", "off", "False", "FALSE"])
    def test_env_var_false_values(self, monkeypatch, val):
        monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", val)
        assert resolve_benchmark_pipeline_mode({}) == "package"

    def test_env_var_empty_falls_through_to_config(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", "")
        cfg = {"pipeline": {"mode": "managed"}}
        assert resolve_benchmark_pipeline_mode(cfg) == "managed"


class TestGetManagedKfpPipelineName:
    def test_default_tabular(self):
        assert get_managed_kfp_pipeline_name("tabular", {}) == "autogluon-tabular-training-pipeline"

    def test_default_timeseries(self):
        assert get_managed_kfp_pipeline_name("timeseries", {}) == "autogluon-timeseries-training-pipeline"

    def test_default_autorag(self):
        assert get_managed_kfp_pipeline_name("autorag", {}) == "documents-rag-optimization-pipeline"

    def test_config_override(self):
        cfg = {"pipeline": {"kfp_pipeline_names": {"tabular": "my-custom-pipeline"}}}
        assert get_managed_kfp_pipeline_name("tabular", cfg) == "my-custom-pipeline"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MANAGED_PIPELINE_TABULAR", "env-pipeline")
        assert get_managed_kfp_pipeline_name("tabular", {}) == "env-pipeline"

    def test_env_var_takes_priority_over_config(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MANAGED_PIPELINE_TABULAR", "env-pipeline")
        cfg = {"pipeline": {"kfp_pipeline_names": {"tabular": "config-pipeline"}}}
        assert get_managed_kfp_pipeline_name("tabular", cfg) == "env-pipeline"


class TestGetManagedPipelineWaitTimeout:
    def test_default(self):
        assert get_managed_pipeline_wait_timeout({}) == 300

    def test_config_override(self):
        cfg = {"pipeline": {"managed_pipeline_wait_timeout": 120}}
        assert get_managed_pipeline_wait_timeout(cfg) == 120

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BENCHMARK_MANAGED_PIPELINE_WAIT_TIMEOUT", "200")
        assert get_managed_pipeline_wait_timeout({}) == 200

    def test_clamped_min(self):
        cfg = {"pipeline": {"managed_pipeline_wait_timeout": 5}}
        assert get_managed_pipeline_wait_timeout(cfg) == 30

    def test_clamped_max(self):
        cfg = {"pipeline": {"managed_pipeline_wait_timeout": 9999}}
        assert get_managed_pipeline_wait_timeout(cfg) == 600


class TestFindPipelineByDisplayName:
    def test_found_via_get_pipeline_id(self):
        client = MagicMock()
        client.get_pipeline_id.return_value = "pid-123"
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="vid-456")]
        )
        result, names = _find_pipeline_by_display_name(client, "my-pipeline")
        assert result == ("pid-123", "vid-456")
        client.get_pipeline_id.assert_called_once_with("my-pipeline")

    def test_not_found_via_get_pipeline_id(self):
        client = MagicMock()
        client.get_pipeline_id.return_value = None
        client.list_pipelines.return_value = SimpleNamespace(
            pipelines=[SimpleNamespace(display_name="other-pipeline")]
        )
        result, names = _find_pipeline_by_display_name(client, "my-pipeline")
        assert result is None
        assert "other-pipeline" in names

    def test_fallback_to_list_pipelines(self):
        client = MagicMock(spec=[])
        pipe = SimpleNamespace(
            display_name="target-pipeline",
            pipeline_id="pid-789",
            default_pipeline_version_id="vid-001",
        )
        client.list_pipelines = MagicMock(
            return_value=SimpleNamespace(pipelines=[pipe], next_page_token=""),
        )
        result, names = _find_pipeline_by_display_name(client, "target-pipeline")
        assert result == ("pid-789", "vid-001")


class TestWaitForManagedPipeline:
    def test_found_immediately(self):
        client = MagicMock()
        client.get_pipeline_id.return_value = "pid-abc"
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="vid-def")]
        )
        result = wait_for_managed_pipeline(client, "my-pipeline", timeout_seconds=10)
        assert result == ("pid-abc", "vid-def")

    def test_timeout_raises(self):
        client = MagicMock()
        client.get_pipeline_id.return_value = None
        client.list_pipelines.return_value = SimpleNamespace(
            pipelines=[SimpleNamespace(display_name="other")]
        )
        with pytest.raises(TimeoutError, match="not found in KFP"):
            wait_for_managed_pipeline(
                client, "missing-pipeline", timeout_seconds=0, poll_interval_seconds=0.01,
            )

    def test_empty_list_early_exit(self):
        client = MagicMock()
        client.get_pipeline_id.return_value = None
        client.list_pipelines.return_value = SimpleNamespace(pipelines=[])
        with pytest.raises(EnvironmentError, match="No managed pipelines registered"):
            wait_for_managed_pipeline(
                client, "missing", timeout_seconds=200, poll_interval_seconds=0.01,
            )


class TestPipelineRunTarget:
    def test_package_mode(self):
        t = PipelineRunTarget(
            mode="package",
            artifact_prefix="my-prefix",
            package_path="/tmp/pipeline.yaml",
        )
        assert t.mode == "package"
        assert t.package_path == "/tmp/pipeline.yaml"
        assert t.pipeline_id is None

    def test_managed_mode(self):
        t = PipelineRunTarget(
            mode="managed",
            artifact_prefix="my-prefix",
            pipeline_id="pid-123",
            pipeline_version_id="vid-456",
            kfp_pipeline_name="my-pipeline",
        )
        assert t.mode == "managed"
        assert t.package_path is None
        assert t.pipeline_id == "pid-123"


class TestSubmitPipelineRun:
    def test_package_mode_calls_create_run(self):
        client = MagicMock()
        client.create_run_from_pipeline_package.return_value = SimpleNamespace(run_id="r1")
        target = PipelineRunTarget(
            mode="package", artifact_prefix="pfx", package_path="/tmp/p.yaml",
        )
        result = submit_pipeline_run(
            client, target,
            arguments={"a": "b"},
            run_name="test-run",
            experiment_name="exp",
            enable_caching=False,
        )
        client.create_run_from_pipeline_package.assert_called_once()
        assert result.run_id == "r1"

    def test_managed_mode_calls_run_pipeline(self):
        client = MagicMock()
        exp = SimpleNamespace(experiment_id="exp-id")
        client.create_experiment.return_value = exp
        client.run_pipeline.return_value = SimpleNamespace(run_id="r2")
        target = PipelineRunTarget(
            mode="managed", artifact_prefix="pfx",
            pipeline_id="pid-1", pipeline_version_id="vid-1",
            kfp_pipeline_name="my-pipeline",
        )
        result = submit_pipeline_run(
            client, target,
            arguments={"x": "y"},
            run_name="test-run",
            experiment_name="my-exp",
            enable_caching=False,
        )
        client.run_pipeline.assert_called_once()
        assert result.run_id == "r2"
        call_kwargs = client.run_pipeline.call_args
        assert call_kwargs.kwargs["pipeline_id"] == "pid-1"
        assert call_kwargs.kwargs["version_id"] == "vid-1"

    def test_managed_mode_existing_experiment(self):
        client = MagicMock()
        client.create_experiment.side_effect = Exception("already exists")
        exp = SimpleNamespace(experiment_id="exp-existing")
        client.get_experiment.return_value = exp
        client.run_pipeline.return_value = SimpleNamespace(run_id="r3")
        target = PipelineRunTarget(
            mode="managed", artifact_prefix="pfx",
            pipeline_id="pid-1", pipeline_version_id="vid-1",
        )
        result = submit_pipeline_run(
            client, target,
            arguments={},
            run_name="test",
            experiment_name="exp",
            enable_caching=False,
        )
        client.get_experiment.assert_called_once_with(experiment_name="exp")
        assert result.run_id == "r3"

    def test_package_mode_without_path_raises(self):
        client = MagicMock()
        target = PipelineRunTarget(mode="package", artifact_prefix="pfx")
        with pytest.raises(ValueError, match="package_path"):
            submit_pipeline_run(
                client, target, arguments={}, run_name="r",
                experiment_name="e", enable_caching=False,
            )

    def test_managed_mode_without_pipeline_id_raises(self):
        client = MagicMock()
        target = PipelineRunTarget(mode="managed", artifact_prefix="pfx")
        with pytest.raises(ValueError, match="pipeline_id"):
            submit_pipeline_run(
                client, target, arguments={}, run_name="r",
                experiment_name="e", enable_caching=False,
            )

    def test_unknown_mode_raises(self):
        client = MagicMock()
        target = PipelineRunTarget(mode="unknown", artifact_prefix="pfx")
        with pytest.raises(ValueError, match="Unknown"):
            submit_pipeline_run(
                client, target, arguments={}, run_name="r",
                experiment_name="e", enable_caching=False,
            )


class TestResolveManagedTargetsWithoutClient:
    """dry_run passes client=None; managed mode must not call KFP."""

    def test_automl_stubs_without_client(self, tmp_path: Path):
        cfg = {"pipeline": {"mode": "managed"}}
        targets = resolve_automl_pipeline_targets(
            cfg, tmp_path, client=None, needs_tabular=True, needs_timeseries=True,
        )
        assert targets["tabular"].mode == "managed"
        assert targets["tabular"].pipeline_id is None
        assert targets["tabular"].kfp_pipeline_name == "autogluon-tabular-training-pipeline"
        assert targets["timeseries"].kfp_pipeline_name == "autogluon-timeseries-training-pipeline"

    def test_autorag_stub_without_client(self, tmp_path: Path):
        cfg = {"pipeline": {"mode": "managed"}}
        target = resolve_autorag_pipeline_target(cfg, tmp_path, client=None)
        assert target.mode == "managed"
        assert target.pipeline_id is None
        assert target.kfp_pipeline_name == "documents-rag-optimization-pipeline"

    def test_automl_with_client_waits(self, tmp_path: Path):
        client = MagicMock()
        client.get_pipeline_id.return_value = "pid-1"
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="vid-1")],
        )
        cfg = {"pipeline": {"mode": "managed"}}
        targets = resolve_automl_pipeline_targets(
            cfg, tmp_path, client=client, needs_tabular=True, needs_timeseries=False,
        )
        assert targets["tabular"].pipeline_id == "pid-1"
        assert targets["tabular"].pipeline_version_id == "vid-1"
