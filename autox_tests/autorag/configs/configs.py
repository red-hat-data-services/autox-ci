"""Test configurations for parametrized functional tests of the Documents RAG Optimization pipeline.

Configurations are loaded from optimisation_test_configs.json in this directory by default.
Set AUTORAG_TEST_CONFIGS_PATH to load from a custom JSON file instead.
Each entry specifies pipeline parameter overrides, expected result (pass/fail),
and optional tags for filtering. Use TESTS_TAGS (comma-separated) to
run only configs that have all of the given tags.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_CONFIGS_JSON_PATH = Path(
    os.getenv("AUTORAG_TEST_CONFIGS_PATH")
    or (Path(__file__).parent / "optimisation_test_configs.json")
)
_INDEXING_CONFIGS_JSON_PATH = Path(
    os.getenv("AUTORAG_INDEXING_TEST_CONFIGS_PATH")
    or (Path(__file__).parent / "indexing_test_configs.json")
)


@dataclass
class AutoRAGTestConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "positive"]).
            Use TESTS_TAGS to run only configs that have all of the given tags.
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        pipeline_params_overrides: Keys matching pipeline parameter names. Values
            are resolved against the base config using these rules:
            - null/None: use base config value from env
            - "": pass empty string explicitly
            - "ENV": read from dedicated env var (for model lists)
            - "milvus-lite"/"milvus-remote": read provider ID from corresponding env var
            - any other value: use as-is
    """

    __test__ = False  # prevent pytest collection

    id: str
    description: str
    tags: list[str]
    expected_result: str
    vector_io_provider_type: str | None = None
    vector_io_provider_id: str | None = None
    embedding_models: str | list[str] | None = None
    generation_models: str | list[str] | None = None
    optimization_max_rag_patterns: int | None = None
    input_data_key: str | None = None
    test_data_key: str | None = None
    optimization_metric: str | None = None

    def __post_init__(self):
        if self.embedding_models == "env":
            self.embedding_models = os.getenv("AUTORAG_EMBEDDING_MODELS")
            if self.embedding_models is None:
                raise EnvironmentError("AUTORAG_EMBEDDING_MODELS env variable must be set.")

        if self.generation_models == "env":
            self.generation_models = os.getenv("AUTORAG_GENERATION_MODELS")
            if self.generation_models is None:
                raise EnvironmentError("AUTORAG_EMBEDDING_MODELS env variable must be set.")

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Build pipeline arguments dict by merging base config with overrides.

        Args:
            base_config: Functional config dict from get_functional_config().

        Returns:
            Pipeline arguments dict ready for KFP submission.
        """
        arguments = {
            "test_data_secret_name": base_config["test_data_secret_name"],
            "test_data_bucket_name": base_config["test_data_bucket_name"],
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "ogx_secret_name": base_config["ogx_secret_name"],
            "test_data_key": self.test_data_key or "",
            "input_data_key": self.input_data_key or "",
            "optimization_metric": self.optimization_metric or "",
        }

        if self.vector_io_provider_id:
            arguments["vector_io_provider_id"] = self.vector_io_provider_id
        if self.optimization_max_rag_patterns is not None:
            arguments["optimization_max_rag_patterns"] = self.optimization_max_rag_patterns
        if self.embedding_models:
            arguments["embedding_models"] = self.embedding_models
        if self.generation_models:
            arguments["generation_models"] = self.generation_models

        return arguments


_C = TypeVar("_C")


def _load_test_configs_from_json(path: Path, cls: type[_C], label: str, pass_type: str) -> list[_C]:
    """Load and validate test configs from a JSON file, filtered by pass_type."""
    with open(path) as f:
        all_items = json.load(f)

    expected = "pass" if pass_type == "positive" else "fail"
    data = [item for item in all_items if item.get("expected_result") == expected]

    configs = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"{label}[{i}] must be a dict; got {type(raw).__name__}")
        try:
            item = dict(raw)
            raw_tags = item.pop("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(f"{label}[{i}] 'tags' must be a list; got {type(raw_tags).__name__}")
            if item.get("expected_result") not in ("pass", "fail"):
                raise ValueError(
                    f"{label}[{i}] 'expected_result' must be 'pass' or 'fail'; "
                    f"got '{item.get('expected_result')}'"
                )
            configs.append(cls(tags=tags, **item))
        except (KeyError, TypeError) as e:
            raise ValueError(f"{label}[{i}] missing or invalid required field: {e}") from e
    return configs


def _filter_by_tags(configs: list, tags: list[str]) -> list:
    """Filter configs to those matching all given tags plus any tags from TESTS_TAGS env var."""
    env_tags_raw = os.getenv("TESTS_TAGS")
    env_tags = [t.strip().lower() for t in env_tags_raw.split(",") if t.strip()] if env_tags_raw else []
    all_tags = {t.lower() for t in (tags + env_tags)}
    if not all_tags:
        return configs
    return [c for c in configs if all(t in c.tags for t in all_tags)]


def _load_configs(pass_type: str) -> list[AutoRAGTestConfig]:
    return _load_test_configs_from_json(_CONFIGS_JSON_PATH, AutoRAGTestConfig, "optimisation_test_configs", pass_type)


def get_all_dataset_keys() -> tuple[list[str], list[str]]:
    """Return (input_data_keys, test_data_keys) deduplicated across all test configs."""
    with open(_CONFIGS_JSON_PATH) as f:
        all_items = json.load(f)
    input_keys = list({item["input_data_key"] for item in all_items if item.get("input_data_key")})
    test_keys = list({item["test_data_key"] for item in all_items if item.get("test_data_key")})
    return input_keys, test_keys


@dataclass
class IndexingTestConfig:
    """Single test configuration for one documents-indexing-pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "remote::milvus"]).
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        vector_io_provider_id: OGX vector IO provider ID (e.g. "milvus-remote").
        embedding_model_id: Embedding model ID for the vector store.
            Use "env" to read from the ``AUTORAG_INDEXING_EMBEDDING_MODEL_ID`` env var.
        input_data_key: Path to folder with input documents within the bucket.
        vector_store_id: OGX vector store / collection id to reuse. Omit to create new.
        chunking_method: Chunking method (default: "recursive").
        chunk_size: Maximum chunk size in tokens (default: 1024).
        chunk_overlap: Token overlap between consecutive chunks (default: 0).
        batch_size: Number of documents per batch (default: 20).
        expected_failing_task: For negative scenarios, KFP task display names expected to fail.
    """

    __test__ = False

    id: str
    description: str
    tags: list[str]
    expected_result: str
    vector_io_provider_id: str
    embedding_model_id: str
    input_data_key: str | None = None
    vector_store_id: str | None = None
    chunking_method: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    batch_size: int | None = None
    expected_failing_task: list[str] | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Build pipeline arguments dict by merging base config with per-scenario overrides.

        Args:
            base_config: Functional config dict from get_indexing_functional_config().

        Returns:
            Pipeline arguments dict ready for KFP submission.

        Raises:
            EnvironmentError: When embedding_model_id is "env" and
                AUTORAG_INDEXING_EMBEDDING_MODEL_ID is not set.
        """
        embedding_model_id = self.embedding_model_id
        if embedding_model_id == "env":
            embedding_model_id = os.getenv("AUTORAG_INDEXING_EMBEDDING_MODEL_ID")
            if embedding_model_id is None:
                raise EnvironmentError(
                    "AUTORAG_INDEXING_EMBEDDING_MODEL_ID env variable must be set "
                    "for indexing pipeline tests that use embedding_model_id: \"env\"."
                )

        arguments: dict[str, Any] = {
            "ogx_secret_name": base_config["ogx_secret_name"],
            "embedding_model_id": embedding_model_id,
            "vector_io_provider_id": self.vector_io_provider_id,
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "input_data_key": self.input_data_key or "",
        }
        if self.vector_store_id is not None:
            arguments["vector_store_id"] = self.vector_store_id
        if self.chunking_method is not None:
            arguments["chunking_method"] = self.chunking_method
        if self.chunk_size is not None:
            arguments["chunk_size"] = self.chunk_size
        if self.chunk_overlap is not None:
            arguments["chunk_overlap"] = self.chunk_overlap
        if self.batch_size is not None:
            arguments["batch_size"] = self.batch_size
        return arguments


def _load_indexing_configs(pass_type: str) -> list[IndexingTestConfig]:
    return _load_test_configs_from_json(
        _INDEXING_CONFIGS_JSON_PATH, IndexingTestConfig, "indexing_test_configs", pass_type
    )


def get_indexing_configs_for_run(
    pass_type: str, tags: None | list[str] = None
) -> list[IndexingTestConfig]:
    """Return indexing pipeline configs to run for this session, optionally filtered by tags.

    Args:
        pass_type (str): 'positive' or 'negative'.
        tags (None | list[str]): Only return configs that have all of these tags.

    Returns:
        list[IndexingTestConfig]: Filtered list of IndexingTestConfig instances.
    """
    return _filter_by_tags(_load_indexing_configs(pass_type), list(tags or []))


def get_test_configs_for_run(pass_type: str, tags: None | list[str] = None) -> list[AutoRAGTestConfig]:
    """Return configs to run for this session, optionally filtered by tags.

    If tags are passed, only configs that have all of those tags are returned.
    All configs are returned otherwise.

    Args:
        pass_type (str): Type of pass to run for this session. 'positive' or 'negative'.
        tags (None | list[str]): List of tags to run for this session.

    Returns:
        list[AutoRAGTestConfig]: List of TestConfig instances.
    """
    return _filter_by_tags(_load_configs(pass_type), list(tags or []))
