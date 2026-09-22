"""Test configurations for parametrized AutoML functional tests.

Tabular configs are loaded from tabular_test_configs.json by default.
Timeseries configs are loaded from timeseries_test_configs.json by default.
Set AUTOML_TABULAR_TEST_CONFIGS_PATH or AUTOML_TIMESERIES_TEST_CONFIGS_PATH
to load from custom JSON files instead. Filter scenarios by tags with
AUTOML_FUNCTIONAL_TESTS_TAGS (comma-separated).
"""

import json
import os
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).parent
_TABULAR_JSON = Path(
    os.getenv("AUTOML_TABULAR_TEST_CONFIGS_PATH")
    or (_CONFIGS_DIR / "tabular_test_configs.json")
)
_TIMESERIES_JSON = Path(
    os.getenv("AUTOML_TIMESERIES_TEST_CONFIGS_PATH")
    or (_CONFIGS_DIR / "timeseries_test_configs.json")
)


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
    run_notebook: bool = False
    deploy: bool = False
    # Negative-path fields (None for positive scenarios)
    fault_category: str | None = None
    injected_fault: str | None = None
    expected_failing_stage: str | None = None
    expected_failing_task: list[str] | None = None
    # Regex matched against task errors + failed-pod logs. Pins the *reason* for the
    # failure so a scenario cannot pass on an unrelated fault in the expected task.
    expected_error_pattern: str | None = None
    # S3 keys this scenario needs to be absent (missing-object faults). Every other key
    # referenced by any scenario is verified present before the suite runs.
    missing_object_keys: list[str] = field(default_factory=list)
    train_data_secret_name_override: str | None = None
    # User-provided test dataset (pipelines-components AutoML test_data_* inputs).
    # Omitted from pipeline arguments when test_data_file_key is unset so older IR still runs.
    # Credentials come from train_data_secret_name (mapped to AWS_* and TEST_DATA_AWS_*).
    test_data_file_key: str | None = None
    test_data_bucket_name_override: str | None = None
    expected_test_dataset_rows: int | None = None
    expected_test_dataset_contains: str | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        effective_secret = (
            self.train_data_secret_name_override
            or base_config["train_data_secret_name"]
        )
        args: dict[str, Any] = {
            "train_data_secret_name": effective_secret,
            "train_data_bucket_name": base_config["train_data_bucket_name"],
            "train_data_file_key": self.train_data_file_key,
            "label_column": self.label_column,
            "task_type": self.task_type,
            "top_n": self.top_n,
        }
        _attach_user_test_data_args(
            args,
            test_data_file_key=self.test_data_file_key,
            test_data_bucket_name_override=self.test_data_bucket_name_override,
            base_config=base_config,
        )
        return args


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
    run_notebook: bool = False
    deploy: bool = False
    # Future known covariate rows for the forecast horizon (id + timestamp + covariate cols only).
    # Required when known_covariates_names is non-empty and this scenario enables deploy.
    known_covariates_sample: list[dict] | None = None
    # Negative-path fields (None for positive scenarios)
    fault_category: str | None = None
    injected_fault: str | None = None
    expected_failing_stage: str | None = None
    expected_failing_task: list[str] | None = None
    # Regex matched against task errors + failed-pod logs. Pins the *reason* for the
    # failure so a scenario cannot pass on an unrelated fault in the expected task.
    expected_error_pattern: str | None = None
    # S3 keys this scenario needs to be absent (missing-object faults). Every other key
    # referenced by any scenario is verified present before the suite runs.
    missing_object_keys: list[str] = field(default_factory=list)
    eval_metric: str | None = None
    train_data_secret_name_override: str | None = None
    test_data_file_key: str | None = None
    test_data_bucket_name_override: str | None = None
    expected_test_dataset_rows: int | None = None
    expected_test_dataset_contains: str | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Merge scenario-specific fields with shared S3 secret/bucket from base config."""
        effective_secret = (
            self.train_data_secret_name_override
            or base_config["train_data_secret_name"]
        )
        args: dict[str, Any] = {
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
        if self.eval_metric is not None:
            args["eval_metric"] = self.eval_metric
        _attach_user_test_data_args(
            args,
            test_data_file_key=self.test_data_file_key,
            test_data_bucket_name_override=self.test_data_bucket_name_override,
            base_config=base_config,
        )
        return args


def _attach_user_test_data_args(
    args: dict[str, Any],
    *,
    test_data_file_key: str | None,
    test_data_bucket_name_override: str | None,
    base_config: dict[str, Any],
) -> None:
    """Add test_data_* pipeline inputs only when a user test object key is configured.

    Latest AutoML IR (pipelines-components#227) accepts ``test_data_bucket_name`` and
    ``test_data_file_key`` only. S3 credentials for that object come from
    ``train_data_secret_name`` (mounted as both ``AWS_*`` and ``TEST_DATA_AWS_*``).
    """
    key = (test_data_file_key or "").strip()
    if not key:
        return
    train_bucket = base_config["train_data_bucket_name"]
    args["test_data_file_key"] = key
    args["test_data_bucket_name"] = (
        (test_data_bucket_name_override or "").strip() or train_bucket
    )


_TABULAR_FIELDS: set[str] | None = None
_TIMESERIES_FIELDS: set[str] | None = None


def _load_tabular_configs() -> list[AutoMLTabularFunctionalConfig]:
    """Parse tabular_test_configs.json into config dataclass instances."""
    global _TABULAR_FIELDS
    if _TABULAR_FIELDS is None:
        _TABULAR_FIELDS = {f.name for f in dc_fields(AutoMLTabularFunctionalConfig)}
    data = json.loads(_TABULAR_JSON.read_text(encoding="utf-8"))
    return [
        AutoMLTabularFunctionalConfig(
            **{k: v for k, v in item.items() if k in _TABULAR_FIELDS}
        )
        for item in data
    ]


def _load_timeseries_configs() -> list[AutoMLTimeseriesFunctionalConfig]:
    """Parse timeseries_test_configs.json into config dataclass instances."""
    global _TIMESERIES_FIELDS
    if _TIMESERIES_FIELDS is None:
        _TIMESERIES_FIELDS = {
            f.name for f in dc_fields(AutoMLTimeseriesFunctionalConfig)
        }
    data = json.loads(_TIMESERIES_JSON.read_text(encoding="utf-8"))
    return [
        AutoMLTimeseriesFunctionalConfig(
            **{k: v for k, v in item.items() if k in _TIMESERIES_FIELDS}
        )
        for item in data
    ]


def _filter_by_tags(
    configs: list, tags_env: str = "AUTOML_FUNCTIONAL_TESTS_TAGS"
) -> list:
    """Return only configs that have ALL requested tags (comma-separated env var), or all if unset."""
    raw = os.getenv(tags_env)
    if not raw or not raw.strip():
        return configs
    requested = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if not requested:
        return configs
    return [
        c
        for c in configs
        if all(t in {tag.lower() for tag in c.tags} for t in requested)
    ]


def _split_by_pass_type(configs: list, pass_type: str | None) -> list:
    """Split configs into positive or negative subsets, or return all when pass_type is None."""
    if pass_type == "positive":
        return [c for c in configs if "negative" not in c.tags]
    if pass_type == "negative":
        return [c for c in configs if "negative" in c.tags]
    return configs


def get_all_train_data_file_keys() -> list[str]:
    """Return unique train and user-test S3 keys across tabular and timeseries configs."""
    keys: list[str] = []
    for c in _load_tabular_configs():
        keys.append(c.train_data_file_key)
        if c.test_data_file_key:
            keys.append(c.test_data_file_key)
    for c in _load_timeseries_configs():
        keys.append(c.train_data_file_key)
        if c.test_data_file_key:
            keys.append(c.test_data_file_key)
    return keys


def get_dataset_key_expectations() -> tuple[list[str], list[str]]:
    """Return ``(keys_that_must_exist, keys_that_must_be_absent)`` in the train-data bucket.

    Scoped to the scenarios selected by ``AUTOML_FUNCTIONAL_TESTS_TAGS``, so a filtered run
    is never blocked by a dataset it does not touch. Absent keys come from
    ``missing_object_keys`` on the missing-object negative scenarios; every other key those
    scenarios reference must be present, so a fixture that never made it to S3 fails at
    session setup instead of surfacing as an opaque pipeline error.

    Raises:
        ValueError: If a key is declared missing by one scenario and used as real data by
            another — the two expectations cannot both hold.
    """
    configs = [*get_tabular_configs_for_run(), *get_timeseries_configs_for_run()]

    all_keys: set[str] = set()
    absent: set[str] = set()
    for c in configs:
        all_keys.add(c.train_data_file_key)
        if c.test_data_file_key:
            all_keys.add(c.test_data_file_key)
        absent.update(c.missing_object_keys)

    unknown = absent - all_keys
    if unknown:
        raise ValueError(
            f"missing_object_keys reference keys no selected scenario uses: {sorted(unknown)}. "
            "Fix the typo, or drop the entry."
        )

    conflicting = {
        key
        for key in absent
        for c in configs
        if key in (c.train_data_file_key, c.test_data_file_key)
        and key not in c.missing_object_keys
    }
    if conflicting:
        raise ValueError(
            f"Keys declared missing by one scenario but used as real data by another: "
            f"{sorted(conflicting)}. A missing-object fault needs its own dedicated key."
        )

    return sorted(all_keys - absent), sorted(absent)


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
