"""Load per-pipeline JSON test configurations for root RHOAI tests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from autox_tests.lib.env import tests_dir
from autox_tests.lib.settings import TEST_CONFIG_TAGS_ENV

_AUTOML_CONFIG_DIR = tests_dir() / "automl" / "config"
_AUTORAG_CONFIG_DIR = tests_dir() / "autorag" / "config"


def _filter_by_tags(configs: list[Any], get_tags: Any) -> list[Any]:
    raw = os.environ.get(TEST_CONFIG_TAGS_ENV)
    if not raw or not str(raw).strip():
        return configs
    allowed = {t.strip().lower() for t in str(raw).split(",") if t.strip()}
    if not allowed:
        return configs
    return [c for c in configs if any(t.lower() in allowed for t in get_tags(c))]


@dataclass
class AutomlTabularTestConfig:
    """One tabular AutoML scenario (AutoGluon tabular training pipeline)."""

    __test__ = False

    id: str
    dataset_path: str | None
    label_column: str
    problem_type: str
    task_type: str
    automl_settings: dict[str, Any]
    tags: list[str]
    data_mode: str = "upload"
    dataset_bucket: str | None = None
    dataset_key: str | None = None

    def get_pipeline_arguments(
        self,
        train_data_bucket_name: str,
        train_data_file_key: str,
        train_data_secret_name: str,
    ) -> dict[str, Any]:
        """Build keyword arguments for the tabular AutoML training pipeline."""
        return {
            "train_data_secret_name": train_data_secret_name,
            "train_data_bucket_name": train_data_bucket_name,
            "train_data_file_key": train_data_file_key,
            "label_column": self.label_column,
            "task_type": self.task_type,
            **self.automl_settings,
        }


def _load_tabular_raw() -> list[AutomlTabularTestConfig]:
    path = _AUTOML_CONFIG_DIR / "automl_tabular_test_configs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("automl_tabular_test_configs.json must be a JSON array")
    out: list[AutomlTabularTestConfig] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"tabular config[{i}] must be an object")
        tags_raw = item.get("tags") or []
        if not isinstance(tags_raw, list):
            raise ValueError(f"tabular config[{i}].tags must be a list")
        data_mode = (item.get("data_mode") or "upload").strip().lower()
        if data_mode not in ("upload", "existing_s3"):
            raise ValueError(f"tabular config[{i}].data_mode must be upload or existing_s3")
        ds_path = item.get("dataset_path")
        ds_bucket = item.get("dataset_bucket")
        ds_key = item.get("dataset_key")
        if data_mode == "upload":
            if not ds_path:
                raise ValueError(f"tabular config[{i}] data_mode=upload requires dataset_path")
        elif data_mode == "existing_s3":
            if not ds_key or not str(ds_key).strip():
                raise ValueError(f"tabular config[{i}] data_mode=existing_s3 requires dataset_key")
        out.append(
            AutomlTabularTestConfig(
                id=item["id"],
                dataset_path=str(ds_path) if ds_path is not None else None,
                label_column=item["label_column"],
                problem_type=item["problem_type"],
                task_type=item["task_type"],
                automl_settings=item.get("automl_settings") or {},
                tags=[str(t) for t in tags_raw],
                data_mode=data_mode,
                dataset_bucket=str(ds_bucket).strip() if ds_bucket else None,
                dataset_key=str(ds_key).strip() if ds_key else None,
            )
        )
    return out


def get_automl_tabular_configs_for_run() -> list[AutomlTabularTestConfig]:
    """Return tabular AutoML configs, optionally filtered by ``RHOAI_TEST_CONFIG_TAGS``."""
    configs = _load_tabular_raw()
    return _filter_by_tags(configs, lambda c: c.tags)


def get_automl_tabular_dataset_paths() -> list[str]:
    """Distinct repo-relative dataset files to upload for ``data_mode=upload``."""
    paths: list[str] = []
    seen: set[str] = set()
    for c in _load_tabular_raw():
        if c.data_mode != "upload" or not c.dataset_path:
            continue
        if c.dataset_path not in seen:
            seen.add(c.dataset_path)
            paths.append(c.dataset_path)
    return paths


@dataclass
class AutomlTimeseriesTestConfig:
    """One time series AutoML scenario (AutoGluon time series training pipeline)."""

    __test__ = False

    id: str
    dataset_path: str | None
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str]
    prediction_length: int
    top_n: int
    tags: list[str]
    data_mode: str = "upload"
    dataset_bucket: str | None = None
    dataset_key: str | None = None

    def get_pipeline_arguments(
        self,
        train_data_bucket_name: str,
        train_data_file_key: str,
        train_data_secret_name: str,
    ) -> dict[str, Any]:
        """Build keyword arguments for the time series AutoML training pipeline."""
        return {
            "train_data_secret_name": train_data_secret_name,
            "train_data_bucket_name": train_data_bucket_name,
            "train_data_file_key": train_data_file_key,
            "target": self.target,
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "known_covariates_names": self.known_covariates_names,
            "prediction_length": self.prediction_length,
            "top_n": self.top_n,
        }


def _load_timeseries_raw() -> list[AutomlTimeseriesTestConfig]:
    path = _AUTOML_CONFIG_DIR / "automl_timeseries_test_configs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("automl_timeseries_test_configs.json must be a JSON array")
    out: list[AutomlTimeseriesTestConfig] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"timeseries config[{i}] must be an object")
        tags_raw = item.get("tags") or []
        kc = item.get("known_covariates_names") or []
        if not isinstance(tags_raw, list) or not isinstance(kc, list):
            raise ValueError(f"timeseries config[{i}] tags/known_covariates_names must be lists")
        data_mode = (item.get("data_mode") or "upload").strip().lower()
        if data_mode not in ("upload", "existing_s3"):
            raise ValueError(f"timeseries config[{i}].data_mode must be upload or existing_s3")
        ds_path = item.get("dataset_path")
        ds_bucket = item.get("dataset_bucket")
        ds_key = item.get("dataset_key")
        if data_mode == "upload":
            if not ds_path:
                raise ValueError(f"timeseries config[{i}] data_mode=upload requires dataset_path")
        elif data_mode == "existing_s3":
            if not ds_key or not str(ds_key).strip():
                raise ValueError(f"timeseries config[{i}] data_mode=existing_s3 requires dataset_key")
        out.append(
            AutomlTimeseriesTestConfig(
                id=item["id"],
                dataset_path=str(ds_path) if ds_path is not None else None,
                target=item["target"],
                id_column=item["id_column"],
                timestamp_column=item["timestamp_column"],
                known_covariates_names=[str(x) for x in kc],
                prediction_length=int(item["prediction_length"]),
                top_n=int(item["top_n"]),
                tags=[str(t) for t in tags_raw],
                data_mode=data_mode,
                dataset_bucket=str(ds_bucket).strip() if ds_bucket else None,
                dataset_key=str(ds_key).strip() if ds_key else None,
            )
        )
    return out


def get_automl_timeseries_configs_for_run() -> list[AutomlTimeseriesTestConfig]:
    """Return time series AutoML configs, optionally filtered by ``RHOAI_TEST_CONFIG_TAGS``."""
    configs = _load_timeseries_raw()
    return _filter_by_tags(configs, lambda c: c.tags)


def get_automl_timeseries_dataset_paths() -> list[str]:
    """Distinct repo-relative dataset files to upload for ``data_mode=upload``."""
    paths: list[str] = []
    seen: set[str] = set()
    for c in _load_timeseries_raw():
        if c.data_mode != "upload" or not c.dataset_path:
            continue
        if c.dataset_path not in seen:
            seen.add(c.dataset_path)
            paths.append(c.dataset_path)
    return paths


@dataclass
class AutoragOptimizationTestConfig:
    """One AutoRAG optimization scenario (same ``data_mode`` vocabulary as AutoML).

    * ``data_mode=upload`` — upload ``documents_directory_path`` and ``benchmark_dataset_path`` from the repo.
    * ``data_mode=existing_s3`` — use ``test_data_*`` / ``input_data_*`` bucket and key fields (optional
      buckets: fall back to ``TEST_DATA_SOURCE_BUCKET`` / ``RHOAI_TEST_DATA_BUCKET``).
    """

    __test__ = False

    id: str
    tags: list[str]
    data_mode: str = "upload"
    documents_directory_path: str | None = None
    benchmark_dataset_path: str | None = None
    test_data_bucket: str | None = None
    test_data_key: str | None = None
    input_data_bucket: str | None = None
    input_data_key: str | None = None
    argument_overrides: dict[str, Any] = field(default_factory=dict)


def _load_autorag_raw() -> list[AutoragOptimizationTestConfig]:
    path = _AUTORAG_CONFIG_DIR / "autorag_test_configs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("autorag_test_configs.json must be a JSON array")
    out: list[AutoragOptimizationTestConfig] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"autorag config[{i}] must be an object")
        tags_raw = item.get("tags") or []
        ov = item.get("argument_overrides")
        if ov is not None and not isinstance(ov, dict):
            raise ValueError(f"autorag config[{i}].argument_overrides must be an object")
        data_mode = (item.get("data_mode") or "upload").strip().lower()
        if data_mode not in ("upload", "existing_s3"):
            raise ValueError(f"autorag config[{i}].data_mode must be upload or existing_s3")
        doc_dir = item.get("documents_directory_path")
        bench = item.get("benchmark_dataset_path")
        tb = item.get("test_data_bucket")
        tk = item.get("test_data_key")
        ib = item.get("input_data_bucket")
        ik = item.get("input_data_key")
        if data_mode == "upload":
            if not doc_dir or not bench:
                raise ValueError(
                    f"autorag config[{i}] data_mode=upload requires documents_directory_path and benchmark_dataset_path"
                )
        elif data_mode == "existing_s3":
            if not tk or not ik:
                raise ValueError(
                    f"autorag config[{i}] data_mode=existing_s3 requires test_data_key and input_data_key"
                )
        out.append(
            AutoragOptimizationTestConfig(
                id=item["id"],
                tags=[str(t) for t in tags_raw],
                data_mode=data_mode,
                documents_directory_path=str(doc_dir) if doc_dir else None,
                benchmark_dataset_path=str(bench) if bench else None,
                test_data_bucket=str(tb).strip() if tb else None,
                test_data_key=str(tk).strip() if tk else None,
                input_data_bucket=str(ib).strip() if ib else None,
                input_data_key=str(ik).strip() if ik else None,
                argument_overrides=dict(ov or {}),
            )
        )
    return out


def get_autorag_configs_for_run() -> list[AutoragOptimizationTestConfig]:
    """Return AutoRAG configs, optionally filtered by ``RHOAI_TEST_CONFIG_TAGS``."""
    configs = _load_autorag_raw()
    return _filter_by_tags(configs, lambda c: c.tags)


def get_automl_tabular_baseline_config() -> AutomlTabularTestConfig:
    """First ``data_mode=upload`` tabular scenario for negative tests (ignores tag filter)."""
    for c in _load_tabular_raw():
        if c.data_mode == "upload" and c.dataset_path:
            return c
    raise RuntimeError("automl_tabular_test_configs.json has no upload scenario")


def get_automl_timeseries_baseline_config() -> AutomlTimeseriesTestConfig:
    """First ``data_mode=upload`` time series scenario for negative tests (ignores tag filter)."""
    for c in _load_timeseries_raw():
        if c.data_mode == "upload" and c.dataset_path:
            return c
    raise RuntimeError("automl_timeseries_test_configs.json has no upload scenario")


def get_autorag_baseline_config_from_run() -> AutoragOptimizationTestConfig | None:
    """First selected AutoRAG scenario (respects ``RHOAI_TEST_CONFIG_TAGS``), or ``None`` if empty."""
    configs = get_autorag_configs_for_run()
    return configs[0] if configs else None
