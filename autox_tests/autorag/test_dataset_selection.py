"""Exercise AutoRAG dataset setup and teardown without a cluster or object store."""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.config

_PREFIX = "datasets/rag/neo4j_light"
_DOCUMENTS = [f"{_PREFIX}/documents/{name}.md" for name in ("aster_platform", "aster_runbook")]
_BENCHMARK = f"{_PREFIX}/benchmark_data.json"


@pytest.mark.parametrize(
    ("selection", "failure", "passed"),
    [
        ([], "", 3),
        (["-k", "optimization"], "", 2),
        (["-k", "indexing"], "", 1),
        (["test_local.py"], "", 1),
        (["-k", "optimization"], "upload", 0),
        (["-k", "optimization"], "test", 0),
    ],
    ids=["both-pipelines", "optimization", "indexing-only", "local-only", "upload-failure", "test-failure"],
)
def test_dataset_upload_and_cleanup(pytester, monkeypatch, selection, failure, passed):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    pytester.makeconftest(f"""
        import json
        from pathlib import Path

        import pytest
        from autox_tests.autorag.conftest import upload_datasets

        calls = {{"uploads": [], "deletes": []}}

        class FakeS3:
            def upload_file(self, filename, bucket, key, **kwargs):
                assert Path(filename).is_file()
                if {failure!r} == "upload" and key.endswith("aster_runbook.md"):
                    raise RuntimeError("upload failed")
                calls["uploads"].append([bucket, key])

            def delete_objects(self, Bucket, Delete):
                calls["deletes"].extend([Bucket, obj["Key"]] for obj in Delete["Objects"])
                return {{}}

        @pytest.fixture(scope="session")
        def functional_env_config():
            assert {selection!r} != ["-k", "indexing"], "indexing must not require optimization config"
            return {{"input_data_bucket_name": "inputs", "test_data_bucket_name": "benchmarks"}}

        @pytest.fixture(scope="session")
        def indexing_functional_env_config():
            return {{"input_data_bucket_name": "inputs"}}

        @pytest.fixture(scope="session")
        def s3_client_functional():
            return FakeS3()

        @pytest.fixture(scope="session")
        def s3_client_indexing_functional():
            return FakeS3()

        def pytest_sessionfinish(session, exitstatus):
            Path("s3_calls.json").write_text(json.dumps(calls))
    """)
    pytester.makepyfile(test_pipelines=f"""
        import pytest
        from autox_tests.autorag.configs.configs import AutoRAGTestConfig, IndexingTestConfig

        @pytest.mark.parametrize("test_scenario_config", [
            AutoRAGTestConfig(
                id=str(index), description="optimization", tags=[], expected_result="pass",
                input_data_keys=[{_PREFIX + '/documents'!r}], test_data_key={_BENCHMARK!r},
            )
            for index in range(2)
        ])
        def test_optimization(test_scenario_config):
            assert {failure!r} != "test", "pipeline failed"

        @pytest.mark.parametrize("test_config", [
            IndexingTestConfig(
                id="indexing", description="indexing", tags=[], expected_result="pass",
                embedding_model_id="model", input_data_keys=[{_PREFIX + '/documents'!r}],
            )
        ])
        def test_indexing(test_config):
            pass
    """)
    pytester.makepyfile(test_local="def test_config(): pass")
    args = selection or ["test_pipelines.py"]
    result = pytester.runpytest_subprocess(*args, "-q")
    if failure:
        result.assert_outcomes(**({"errors": 2} if failure == "upload" else {"failed": 2}))
    else:
        result.assert_outcomes(passed=passed)

    expected = [["inputs", key] for key in _DOCUMENTS]
    if selection == ["test_local.py"]:
        expected = []
    elif failure == "upload":
        expected = expected[:1]
    elif selection != ["-k", "indexing"]:
        expected.append(["benchmarks", _BENCHMARK])
    calls = json.loads((pytester.path / "s3_calls.json").read_text())
    assert sorted(calls["uploads"]) == sorted(expected)
    assert sorted(calls["deletes"]) == sorted(expected)
