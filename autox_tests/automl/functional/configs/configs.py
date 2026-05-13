"""Test configurations for parametrized AutoML functional tests.

Tabular configs are loaded from tabular_test_configs.json.
Timeseries configs are loaded from timeseries_test_configs.json.
Filter scenarios by tags with AUTOML_FUNCTIONAL_TESTS_TAGS (comma-separated).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).parent
_TABULAR_JSON = _CONFIGS_DIR / "tabular_test_configs.json"
_TIMESERIES_JSON = _CONFIGS_DIR / "timeseries_test_configs.json"


@dataclass
class AutoMLTabularFunctionalConfig:
    """Single tabular AutoML test scenario loaded from tabular_test_configs.json."""

    __test__ = False

    id: str
    dataset_path: str
    label_column: str
    task_type: str
    top_n: int
    train_data_file_key: str
    tags: list[str]
    inference_sample: list[dict] | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        return {
            "train_data_secret_name": base_config["train_data_secret_name"],
            "train_data_bucket_name": base_config["train_data_bucket_name"],
            "train_data_file_key": self.train_data_file_key,
            "label_column": self.label_column,
            "task_type": self.task_type,
            "top_n": self.top_n,
        }


@dataclass
class AutoMLTimeseriesFunctionalConfig:
    """Single timeseries AutoML test scenario loaded from timeseries_test_configs.json."""

    __test__ = False

    id: str
    dataset_path: str
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str]
    prediction_length: int
    top_n: int
    train_data_file_key: str
    tags: list[str]
    add_dummy_item_id: bool = False
    add_dummy_timestamp: bool = False
    inference_sample: list[dict] | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        return {
            "train_data_secret_name": base_config["train_data_secret_name"],
            "train_data_bucket_name": base_config["train_data_bucket_name"],
            "train_data_file_key": self.train_data_file_key,
            "target": self.target,
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "known_covariates_names": self.known_covariates_names,
            "prediction_length": self.prediction_length,
            "top_n": self.top_n,
        }


def _load_tabular_configs() -> list[AutoMLTabularFunctionalConfig]:
    """Parse tabular_test_configs.json into config dataclass instances."""
    data = json.loads(_TABULAR_JSON.read_text(encoding="utf-8"))
    configs = []
    for item in data:
        configs.append(AutoMLTabularFunctionalConfig(
            id=item["id"],
            dataset_path=item["dataset_path"],
            label_column=item["label_column"],
            task_type=item["task_type"],
            top_n=item["top_n"],
            train_data_file_key=item["train_data_file_key"],
            tags=item.get("tags", []),
            inference_sample=item.get("inference_sample"),
        ))
    return configs


def _load_timeseries_configs() -> list[AutoMLTimeseriesFunctionalConfig]:
    """Parse timeseries_test_configs.json into config dataclass instances."""
    data = json.loads(_TIMESERIES_JSON.read_text(encoding="utf-8"))
    configs = []
    for item in data:
        configs.append(AutoMLTimeseriesFunctionalConfig(
            id=item["id"],
            dataset_path=item["dataset_path"],
            target=item["target"],
            id_column=item["id_column"],
            timestamp_column=item["timestamp_column"],
            known_covariates_names=item.get("known_covariates_names", []),
            prediction_length=item["prediction_length"],
            top_n=item["top_n"],
            train_data_file_key=item["train_data_file_key"],
            tags=item.get("tags", []),
            add_dummy_item_id=item.get("add_dummy_item_id", False),
            add_dummy_timestamp=item.get("add_dummy_timestamp", False),
            inference_sample=item.get("inference_sample"),
        ))
    return configs


def _filter_by_tags(configs: list, tags_env: str = "AUTOML_FUNCTIONAL_TESTS_TAGS") -> list:
    """Return only configs whose tags overlap with the comma-separated env var, or all if unset."""
    raw = os.getenv(tags_env)
    if not raw or not raw.strip():
        return configs
    allowed = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if not allowed:
        return configs
    return [c for c in configs if any(t.lower() in allowed for t in c.tags)]


def get_tabular_configs_for_run() -> list[AutoMLTabularFunctionalConfig]:
    """Return tabular configs for this session, filtered by AUTOML_FUNCTIONAL_TESTS_TAGS."""
    return _filter_by_tags(_load_tabular_configs())


def get_timeseries_configs_for_run() -> list[AutoMLTimeseriesFunctionalConfig]:
    """Return timeseries configs for this session, filtered by AUTOML_FUNCTIONAL_TESTS_TAGS."""
    return _filter_by_tags(_load_timeseries_configs())
