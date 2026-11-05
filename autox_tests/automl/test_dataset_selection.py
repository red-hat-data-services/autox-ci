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
    ("args", "tags", "required", "absent", "passed"),
    [
        (
            [], "",
            [
                _TITANIC, _HOUSING, _CREDIT_TRAIN, _CREDIT_TEST,
                _TABULAR + "classification_binary/credit_default_binary_light.csv",
                _TABULAR + "classification_binary/german_credit_data_biased_test_missing_features.csv",
                _TABULAR + "classification_multiclass/car_rental_training_data_light.csv",
                _TABULAR + "classification_multiclass/wine_quality_multiclass_limited_classes_light.csv",
                _TABULAR + "regression/energy_efficiency_regression.csv",
                _TIMESERIES + "fruits_daily_price.csv",
                _TIMESERIES + "m4_hourly_subset_train_ultra_light.csv",
                _CASES_TRAIN, _CASES_TEST,
                _TIMESERIES + "demand_single_item.csv",
            ],
            [
                "functional-test/neg/tc-na-4/does-not-exist-00000000.csv",
                "functional-test/neg/tc-na-7/does-not-exist-user-test.csv",
                "functional-test/neg/tc-nb-3/does-not-exist-00000000.csv",
                "functional-test/neg/tc-nb-6/does-not-exist-user-test.csv",
            ],
            27,
        ),
        (
            [], "smoke",
            [
                _TITANIC, _HOUSING,
                _TIMESERIES + "fruits_daily_price.csv",
                _TIMESERIES + "demand_single_item.csv",
            ],
            [], 5,
        ),
        (["-k", "TC-A-2"], "", [_TITANIC], [], 1),
        (["-k", "TC-A-2"], "smoke", [_TITANIC], [], 1),
        (
            ["-k", "TC-A-7 or TC-B-3"], "",
            [_CREDIT_TRAIN, _CREDIT_TEST, _CASES_TRAIN, _CASES_TEST], [], 2,
        ),
        (["-k", "TC-A-1 or TC-NA-1"], "", [_HOUSING], [], 2),
        (
            ["-k", "TC-NA-7 or TC-NB-6"], "",
            [_CREDIT_TRAIN, _CASES_TRAIN],
            [
                "functional-test/neg/tc-na-7/does-not-exist-user-test.csv",
                "functional-test/neg/tc-nb-6/does-not-exist-user-test.csv",
            ],
            2,
        ),
        (
            ["-m", "timeseries and positive"], "",
            [
                _TIMESERIES + "fruits_daily_price.csv",
                _TIMESERIES + "m4_hourly_subset_train_ultra_light.csv",
                _CASES_TRAIN, _CASES_TEST,
                _TIMESERIES + "demand_single_item.csv",
            ],
            [], 4,
        ),
        (
            ["-k", "TC-A-2 or TC-B-3", "-m", "timeseries and positive"],
            "", [_CASES_TRAIN, _CASES_TEST], [], 1,
        ),
        (["-k", "TC-A-2", "-m", "timeseries"], "", [], [], 0),
        (
            ["-k", "TC-A-2 or TC-B-1", "-m", "timeseries"],
            "smoke", [_TIMESERIES + "fruits_daily_price.csv"], [], 1,
        ),
        (["test_tabular.py", "-k", "TC-A-2"], "", [_TITANIC], [], 1),
        (["test_local_config.py"], "", [], [], 1),
        (["-k", "nonexistent_scenario"], "", [], [], 0),
    ],
    ids=[
        "no-filters", "tags-without-keyword",
        "keyword", "tags-and-keyword", "user-test-data",
        "shared-dataset", "missing-objects", "markers", "keyword-and-markers",
        "keyword-and-markers-empty", "tags-keyword-and-markers", "test-path",
        "no-scenarios", "empty-selection",
    ],
)
def test_dataset_setup_follows_pytest_selection(
    pytester, monkeypatch, args, tags, required, absent, passed,
):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    monkeypatch.setenv("AUTOML_FUNCTIONAL_TESTS_TAGS", tags)
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
        from autox_tests.automl.conftest import upload_datasets

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
        # Smoke tags leave both negative parametrizations empty; pytest skips each one.
        skipped = 2 if tags == "smoke" and not args else 0
        result.assert_outcomes(passed=passed, skipped=skipped)
    else:
        assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    calls = json.loads((pytester.path / "s3_calls.json").read_text())
    assert calls["uploads"] == sorted(required)
    assert calls["heads"] == sorted(required) + sorted(absent)
    assert calls["deletes"] == calls["uploads"]


def test_empty_scenario_list_requires_no_datasets():
    assert get_dataset_key_expectations([]) == ([], [])


@pytest.mark.parametrize("failure", ["upload", "verification", "test"])
def test_datasets_are_cleaned_after_failure(pytester, monkeypatch, failure):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    pytester.makeconftest(f"""
        import json
        from pathlib import Path

        import pytest
        from botocore.exceptions import ClientError
        from autox_tests.automl.conftest import upload_datasets

        failure = {failure!r}
        calls = {{"uploads": [], "deletes": []}}

        class FakeS3:
            def upload_file(self, filename, bucket, key):
                assert Path(filename).is_file()
                if failure == "upload" and key.endswith("housing_pricing.csv"):
                    raise RuntimeError("upload failed")
                calls["uploads"].append(key)

            def head_object(self, Bucket, Key):
                if failure == "verification":
                    raise ClientError({{"Error": {{"Code": "404"}}}}, "HeadObject")
                return {{}}

            def delete_objects(self, Bucket, Delete):
                calls["deletes"].extend(obj["Key"] for obj in Delete["Objects"])
                return {{}}

        @pytest.fixture(scope="session")
        def automl_functional_config():
            return {{"train_data_bucket_name": "test-bucket"}}

        @pytest.fixture(scope="session")
        def s3_client_automl_functional():
            return FakeS3()

        def pytest_sessionfinish(session, exitstatus):
            Path("s3_calls.json").write_text(json.dumps(calls))
    """)
    pytester.makepyfile(f"""
        import pytest
        from autox_tests.automl.configs.configs import AutoMLTabularFunctionalConfig

        @pytest.mark.parametrize("test_config", [
            AutoMLTabularFunctionalConfig(
                id=str(index), label_column="target", task_type="regression",
                top_n=1, train_data_file_key=key,
            )
            for index, key in enumerate({[_TITANIC, _HOUSING]!r})
        ])
        def test_pipeline(test_config):
            assert {failure!r} != "test", "pipeline failed"
    """)

    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(**({"failed": 2} if failure == "test" else {"errors": 2}))
    calls = json.loads((pytester.path / "s3_calls.json").read_text())
    expected = [_TITANIC] if failure == "upload" else [_TITANIC, _HOUSING]
    assert calls["uploads"] == expected
    assert calls["deletes"] == expected
