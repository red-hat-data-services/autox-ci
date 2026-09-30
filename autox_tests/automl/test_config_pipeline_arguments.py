"""Local tests for AutoML scenario config loading and artifact validation."""

from datetime import datetime, timedelta
from io import BytesIO

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from autox_tests.automl.configs.configs import (
    get_all_train_data_file_keys,
    get_tabular_configs_for_run,
    get_timeseries_configs_for_run,
)
from autox_tests.automl import utils
from autox_tests.automl.utils import (
    assert_sampled_test_dataset,
    find_test_dataset_artifact,
)

pytestmark = pytest.mark.config

_BASE = {
    "train_data_secret_name": "s3-connection",
    "train_data_bucket_name": "train-bucket",
}


@pytest.fixture(autouse=True)
def _clear_scenario_tag_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOML_FUNCTIONAL_TESTS_TAGS", raising=False)


def test_default_tabular_scenario_omits_test_data_args() -> None:
    cfg = next(c for c in get_tabular_configs_for_run("positive") if c.id == "TC-A-1_regression")
    args = cfg.get_pipeline_arguments(_BASE)
    assert "test_data_file_key" not in args
    assert "test_data_bucket_name" not in args
    assert "test_data_secret_name" not in args


def test_tabular_user_test_scenario_wires_test_data_args() -> None:
    cfg = next(
        c
        for c in get_tabular_configs_for_run("positive")
        if c.id == "TC-A-7_user_provided_test_data"
    )
    args = cfg.get_pipeline_arguments(_BASE)
    assert args["test_data_file_key"].endswith("german_credit_data_biased_test.csv")
    assert args["test_data_bucket_name"] == "train-bucket"
    assert "test_data_secret_name" not in args
    assert args["train_data_secret_name"] == "s3-connection"
    assert args["label_column"] == "Risk"
    assert args["task_type"] == "binary"
    assert cfg.expected_test_dataset_rows == 400


def test_timeseries_user_test_scenario_wires_test_data_args() -> None:
    cfg = next(
        c
        for c in get_timeseries_configs_for_run("positive")
        if c.id == "TC-B-3_user_provided_test_data"
    )
    args = cfg.get_pipeline_arguments(_BASE)
    assert args["test_data_file_key"].endswith("poland_daily_cases_03_04-28_2021.csv")
    assert args["train_data_file_key"].endswith("poland_daily_cases_03_03_2021.csv")
    assert args["test_data_bucket_name"] == "train-bucket"
    assert "test_data_secret_name" not in args
    assert args["train_data_secret_name"] == "s3-connection"
    assert args["target"] == "daily_cases"
    assert args["id_column"] == ""
    assert args["timestamp_column"] == "date"
    assert args["known_covariates_names"] == []
    assert args["prediction_length"] == 7
    assert cfg.expected_test_dataset_rows == 25
    assert cfg.expected_test_dataset_contains == "2021-03-28"


def test_user_test_keys_are_included_for_s3_upload() -> None:
    keys = get_all_train_data_file_keys()
    assert any(k.endswith("german_credit_data_biased_train.csv") for k in keys)
    assert any(k.endswith("german_credit_data_biased_test.csv") for k in keys)
    assert any(k.endswith("german_credit_data_biased_test_missing_features.csv") for k in keys)
    assert any(k.endswith("poland_daily_cases_03_03_2021.csv") for k in keys)
    assert any(k.endswith("poland_daily_cases_03_04-28_2021.csv") for k in keys)


def test_assert_sampled_test_dataset_accepts_matching_csv() -> None:
    body = 'price,area\n"42424242\nalt",9999\n1,2\n'

    class _FakeS3:
        def get_object(self, Bucket, Key):  # noqa: N803
            assert Bucket == "b"
            assert Key == "sampled_test_dataset.csv"
            return {"Body": BytesIO(body.encode("utf-8"))}

    assert_sampled_test_dataset(
        _FakeS3(),
        "b",
        "sampled_test_dataset.csv",
        scenario_id="TC-A-7",
        expected_rows=2,
        must_contain="42424242",
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["run/sampled_test_dataset.csv"], "run/sampled_test_dataset.csv"),
        (["run/sampled_test_dataset.parquet"], "run/sampled_test_dataset.parquet"),
        (
            ["run/sampled_test_dataset.csv", "run/sampled_test_dataset.parquet"],
            "run/sampled_test_dataset.parquet",
        ),
    ],
)
def test_find_test_dataset_artifact(
    monkeypatch, keys: list[str], expected: str
) -> None:
    monkeypatch.setattr(
        utils,
        "list_s3_objects",
        lambda client, bucket, prefix: [{"Key": key} for key in keys],
    )
    assert find_test_dataset_artifact(object(), "b", "run/") == expected


@pytest.mark.parametrize("scenario_id", ["TC-A-7", "TC-B-3"])
def test_assert_sampled_test_dataset_accepts_matching_parquet(scenario_id: str) -> None:
    if scenario_id == "TC-A-7":
        table = pa.table({"Risk": ["good"] * 400})
        expected_rows = 400
        must_contain = "good"
    else:
        table = pa.table(
            {"date": [datetime(2021, 3, 4) + timedelta(days=day) for day in range(25)]}
        )
        expected_rows = 25
        must_contain = "2021-03-28"

    body = BytesIO()
    pq.write_table(table, body)

    class _FakeS3:
        def get_object(self, Bucket, Key):  # noqa: N803
            assert Bucket == "b"
            assert Key == "sampled_test_dataset.parquet"
            return {"Body": BytesIO(body.getvalue())}

    assert_sampled_test_dataset(
        _FakeS3(),
        "b",
        "sampled_test_dataset.parquet",
        scenario_id=scenario_id,
        expected_rows=expected_rows,
        must_contain=must_contain,
    )
