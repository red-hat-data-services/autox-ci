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
    label_column: str
    task_type: str
    top_n: int
    train_data_file_key: str
    tags: list[str] = field(default_factory=list)
    inference_sample: list[dict] | None = None
    # Negative-path fields (None for positive scenarios)
    fault_category: str | None = None
    injected_fault: str | None = None
    expected_failing_stage: str | None = None
    expected_failing_task: list[str] | None = None
    train_data_secret_name_override: str | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        effective_secret = (
            self.train_data_secret_name_override
            or base_config["train_data_secret_name"]
        )
        return {
            "train_data_secret_name": effective_secret,
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
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str]
    prediction_length: int
    top_n: int
    train_data_file_key: str
    tags: list[str] = field(default_factory=list)
    inference_sample: list[dict] | None = None
    # Negative-path fields (None for positive scenarios)
    fault_category: str | None = None
    injected_fault: str | None = None
    expected_failing_stage: str | None = None
    expected_failing_task: list[str] | None = None
    train_data_secret_name_override: str | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        effective_secret = (
            self.train_data_secret_name_override
            or base_config["train_data_secret_name"]
        )
        return {
            "train_data_secret_name": effective_secret,
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
        raw_failing_task = item.get("expected_failing_task")
        configs.append(
            AutoMLTabularFunctionalConfig(
                id=item["id"],
                label_column=item["label_column"],
                task_type=item["task_type"],
                top_n=item["top_n"],
                train_data_file_key=item["train_data_file_key"],
                tags=item.get("tags", []),
                inference_sample=item.get("inference_sample"),
                fault_category=item.get("fault_category"),
                injected_fault=item.get("injected_fault"),
                expected_failing_stage=item.get("expected_failing_stage"),
                expected_failing_task=raw_failing_task
                if isinstance(raw_failing_task, list)
                else None,
                train_data_secret_name_override=item.get(
                    "train_data_secret_name_override"
                ),
            )
        )
    return configs


def _load_timeseries_configs() -> list[AutoMLTimeseriesFunctionalConfig]:
    """Parse timeseries_test_configs.json into config dataclass instances."""
    data = json.loads(_TIMESERIES_JSON.read_text(encoding="utf-8"))
    configs = []
    for item in data:
        raw_failing_task = item.get("expected_failing_task")
        configs.append(
            AutoMLTimeseriesFunctionalConfig(
                id=item["id"],
                target=item["target"],
                id_column=item["id_column"],
                timestamp_column=item["timestamp_column"],
                known_covariates_names=item.get("known_covariates_names", []),
                prediction_length=int(item.get("prediction_length", 1)),
                top_n=int(item.get("top_n", 1)),
                train_data_file_key=item["train_data_file_key"],
                tags=item.get("tags", []),
                inference_sample=item.get("inference_sample"),
                fault_category=item.get("fault_category"),
                injected_fault=item.get("injected_fault"),
                expected_failing_stage=item.get("expected_failing_stage"),
                expected_failing_task=raw_failing_task
                if isinstance(raw_failing_task, list)
                else None,
                train_data_secret_name_override=item.get(
                    "train_data_secret_name_override"
                ),
            )
        )
    return configs


def _filter_by_tags(
    configs: list, tags_env: str = "AUTOML_FUNCTIONAL_TESTS_TAGS"
) -> list:
    """Return only configs whose tags overlap with the comma-separated env var, or all if unset."""
    raw = os.getenv(tags_env)
    if not raw or not raw.strip():
        return configs
    allowed = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if not allowed:
        return configs
    return [c for c in configs if any(t.lower() in allowed for t in c.tags)]


def _split_by_pass_type(configs: list, pass_type: str | None) -> list:
    """Split configs into positive or negative subsets, or return all when pass_type is None."""
    if pass_type == "positive":
        return [c for c in configs if "negative" not in c.tags]
    if pass_type == "negative":
        return [c for c in configs if "negative" in c.tags]
    return configs


def get_tabular_configs_for_run(
    pass_type: str | None = None,
) -> list[AutoMLTabularFunctionalConfig]:
    """Return tabular configs filtered by AUTOML_FUNCTIONAL_TESTS_TAGS and optional pass_type."""
    return _filter_by_tags(_split_by_pass_type(_load_tabular_configs(), pass_type))


def get_timeseries_configs_for_run(
    pass_type: str | None = None,
) -> list[AutoMLTimeseriesFunctionalConfig]:
    """Return timeseries configs filtered by AUTOML_FUNCTIONAL_TESTS_TAGS and optional pass_type."""
    return _filter_by_tags(_split_by_pass_type(_load_timeseries_configs(), pass_type))
