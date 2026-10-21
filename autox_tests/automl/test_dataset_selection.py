"""Regression tests for dataset setup after pytest scenario selection, without S3 access."""

import json
from pathlib import Path

import pytest

from autox_tests.automl.configs.configs import get_dataset_key_expectations

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.config

_TABULAR = "functional-test/tabular/"
_TIMESERIES = "functional-test/timeseries/"
_TITANIC = _TABULAR + "classification_binary/titanic.csv"
_HOUSING = _TABULAR + "regression/housing_pricing.csv"
_CREDIT_TRAIN = _TABULAR + "classification_binary/german_credit_data_biased_train.csv"
_CREDIT_TEST = _TABULAR + "classification_binary/german_credit_data_biased_test.csv"
_CASES_TRAIN = _TIMESERIES + "poland_daily_cases_03_03_2021.csv"
_CASES_TEST = _TIMESERIES + "poland_daily_cases_03_04-28_2021.csv"


@pytest.mark.parametrize(
    ("args", "tags", "upload", "required", "absent", "passed"),
    [
        (["-k", "TC-A-2"], "", True, [_TITANIC], [], 1),
        (["-k", "TC-A-2"], "smoke", True, [_TITANIC], [], 1),
        (["-k", "TC-A-2"], "", False, [_TITANIC], [], 1),
        (
            ["-k", "TC-A-7 or TC-B-3"], "", True,
            [_CREDIT_TRAIN, _CREDIT_TEST, _CASES_TRAIN, _CASES_TEST], [], 2,
        ),
        (["-k", "TC-A-1 or TC-NA-1"], "", True, [_HOUSING], [], 2),
        (
            ["-k", "TC-NA-7 or TC-NB-6"], "", True,
            [_CREDIT_TRAIN, _CASES_TRAIN],
            [
                "functional-test/neg/tc-na-7/does-not-exist-user-test.csv",
                "functional-test/neg/tc-nb-6/does-not-exist-user-test.csv",
            ],
            2,
        ),
        (
            ["-m", "timeseries and positive"], "", True,
            [
                _TIMESERIES + "fruits_daily_price.csv",
                _TIMESERIES + "m4_hourly_subset_train_ultra_light.csv",
                _CASES_TRAIN, _CASES_TEST,
                _TIMESERIES + "demand_single_item.csv",
            ],
            [], 4,
        ),
        (["test_tabular.py", "-k", "TC-A-2"], "", True, [_TITANIC], [], 1),
        (["test_local_config.py"], "", True, [], [], 1),
        (["-k", "nonexistent_scenario"], "", True, [], [], 0),
    ],
    ids=[
        "keyword", "tags-and-keyword", "verify-without-upload", "user-test-data",
        "shared-dataset", "missing-objects", "markers", "test-path",
        "no-scenarios", "empty-selection",
    ],
)
def test_dataset_setup_follows_pytest_selection(
    pytester, monkeypatch, args, tags, upload, required, absent, passed,
):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    monkeypatch.setenv("AUTOML_FUNCTIONAL_TESTS_TAGS", tags)
    monkeypatch.setenv("AUTOML_UPLOAD_TEST_DATASETS", str(upload).lower())
    monkeypatch.delenv("AUTOML_TABULAR_TEST_CONFIGS_PATH", raising=False)
    monkeypatch.delenv("AUTOML_TIMESERIES_TEST_CONFIGS_PATH", raising=False)
    pytester.makeini("""
        [pytest]
        markers =
            positive
            negative
            tabular
            timeseries
    """)
    pytester.makeconftest("""
        import json
        from pathlib import Path

        import pytest
        from botocore.exceptions import ClientError
        from autox_tests.automl.conftest import upload_datasets_if_requested

        calls = {"uploads": [], "heads": [], "deletes": []}

        class FakeS3:
            def upload_file(self, filename, bucket, key):
                assert Path(filename).is_file()
                assert bucket == "test-bucket"
                calls["uploads"].append(key)

            def head_object(self, Bucket, Key):
                assert Bucket == "test-bucket"
                calls["heads"].append(Key)
                if Key.startswith("functional-test/neg/"):
                    raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
                return {}

            def delete_objects(self, Bucket, Delete):
                assert Bucket == "test-bucket"
                calls["deletes"].extend(obj["Key"] for obj in Delete["Objects"])
                return {}

        @pytest.fixture(scope="session")
        def automl_functional_config():
            return {"train_data_bucket_name": "test-bucket"}

        @pytest.fixture(scope="session")
        def s3_client_automl_functional():
            return FakeS3()

        def pytest_sessionfinish(session, exitstatus):
            Path("s3_calls.json").write_text(json.dumps(calls))
    """)
    for family in ("tabular", "timeseries"):
        pytester.makepyfile(**{f"test_{family}": f"""
            import pytest
            from autox_tests.automl.configs.configs import get_{family}_configs_for_run

            pytestmark = pytest.mark.{family}
            positive = get_{family}_configs_for_run("positive")
            negative = get_{family}_configs_for_run("negative")

            @pytest.mark.positive
            @pytest.mark.parametrize("test_config", positive, ids=[c.id for c in positive])
            def test_positive(test_config):
                pass

            @pytest.mark.negative
            @pytest.mark.parametrize("test_config", negative, ids=[c.id for c in negative])
            def test_negative(test_config):
                pass
        """})
    pytester.makepyfile(test_local_config="def test_config(): pass")

    result = pytester.runpytest_subprocess(*args, "-q")
    if passed:
        result.assert_outcomes(passed=passed)
    else:
        assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    calls = json.loads((pytester.path / "s3_calls.json").read_text())
    assert calls["uploads"] == (sorted(required) if upload else [])
    assert calls["heads"] == sorted(required) + sorted(absent)
    assert calls["deletes"] == calls["uploads"]


def test_empty_scenario_list_requires_no_datasets():
    assert get_dataset_key_expectations([]) == ([], [])
