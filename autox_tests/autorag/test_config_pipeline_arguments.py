"""Unit tests for AutoRAG scenario-to-pipeline argument mapping."""

import pytest

from autox_tests.autorag.configs.configs import get_test_configs_for_run


pytestmark = pytest.mark.config

_BASE = {
    "test_data_secret_name": "test-s3",
    "test_data_bucket_name": "test-bucket",
    "input_data_secret_name": "input-s3",
    "input_data_bucket_name": "input-bucket",
    "maas_secret_name": "maas",
    "vector_db_secret_name": "standard-vector-db",
}


@pytest.fixture(autouse=True)
def _clear_scenario_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTORAG_FUNCTIONAL_TESTS_TAGS", raising=False)
    monkeypatch.delenv("NEO4J_DB_SECRET_NAME", raising=False)
    monkeypatch.setenv("AUTORAG_EMBEDDING_MODELS", '["embedding-model"]')
    monkeypatch.setenv("AUTORAG_GENERATION_MODELS", '["generation-model"]')


def test_standard_scenario_uses_default_database_secret() -> None:
    scenario = next(c for c in get_test_configs_for_run("positive") if c.id == "TC-P-1")

    assert scenario.get_pipeline_arguments(_BASE)["db_secret_name"] == "standard-vector-db"


def test_neo4j_scenario_uses_its_dedicated_database_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEO4J_DB_SECRET_NAME", "neo4j-vector-db")
    scenario = next(c for c in get_test_configs_for_run("positive") if c.id == "TC-P-7")

    assert scenario.get_pipeline_arguments(_BASE)["db_secret_name"] == "neo4j-vector-db"


def test_dedicated_database_secret_env_is_required() -> None:
    scenario = next(c for c in get_test_configs_for_run("positive") if c.id == "TC-P-7")

    with pytest.raises(EnvironmentError, match="NEO4J_DB_SECRET_NAME"):
        scenario.get_pipeline_arguments(_BASE)
