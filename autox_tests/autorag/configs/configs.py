"""Test configurations for parametrized functional tests of the Documents RAG Optimization pipeline.

Configurations are loaded from optimisation_test_configs.json in this directory by default.
Set AUTORAG_TEST_CONFIGS_PATH to load from a custom JSON file instead.
Each entry specifies pipeline parameter overrides, expected result (pass/fail),
and optional tags for filtering. Use AUTORAG_FUNCTIONAL_TESTS_TAGS (comma-separated) to
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


def _resolve_model_list(value: str | list[str] | None, env_name: str) -> list[str] | None:
    """Resolve a model-list field to a ``list[str]`` for KFP submission.

    The MaaS pipeline requires ``embedding_models`` / ``generation_models`` as
    lists of model IDs (they can no longer be inferred server-side). Accepted inputs:

    - ``list``: used as-is (e.g. explicit IDs, or deliberately-invalid IDs for
      negative scenarios).
    - ``None``: returns ``None`` (caller decides whether that is an error).
    - the sentinel ``"env"``: read ``env_name`` and parse it as a JSON array,
      falling back to a comma-separated list.

    Raises:
        EnvironmentError: When ``value`` is ``"env"`` but ``env_name`` is unset.
        ValueError: When the resolved value is not a list.
    """
    if value == "env":
        raw = os.getenv(env_name)
        if raw is None:
            raise EnvironmentError(f"{env_name} env variable must be set.")
        value = raw
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = [v.strip() for v in stripped.split(",") if v.strip()]
    if not isinstance(value, list):
        raise ValueError(f"{env_name} must resolve to a list; got {type(value).__name__}")
    return value


@dataclass
class AutoRAGTestConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "positive"]).
            Use AUTORAG_FUNCTIONAL_TESTS_TAGS to run only configs that have all of the given tags.
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        embedding_models: Embedding model IDs for the search space. Required by the
            MaaS pipeline. A JSON list, or the sentinel "env" to read a JSON array /
            comma-separated list from AUTORAG_EMBEDDING_MODELS.
        generation_models: Generation model IDs for the search space. Required by the
            MaaS pipeline. A JSON list, or "env" to read from AUTORAG_GENERATION_MODELS.
        optimization_max_rag_patterns: Cap on the number of RAG patterns explored.
        input_data_key: Path to the input documents folder within the bucket.
        test_data_key: Path to the benchmark JSON within the test-data bucket.
        optimization_metric: Metric to optimize (e.g. "faithfulness").

    The vector-store backend is no longer a pipeline parameter: the pipeline
    auto-detects it from the secret named by ``vector_db_secret_name`` (MILVUS_* vs
    PGVECTOR_* keys), which the harness wires from the VECTOR_DB_SECRET_NAME env var.
    """

    __test__ = False  # prevent pytest collection

    id: str
    description: str
    tags: list[str]
    expected_result: str
    embedding_models: str | list[str] | None = None
    generation_models: str | list[str] | None = None
    optimization_max_rag_patterns: int | None = None
    input_data_key: str | None = None
    test_data_key: str | None = None
    optimization_metric: str | None = None

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Build pipeline arguments dict by merging base config with overrides.

        Args:
            base_config: Functional config dict from get_functional_config().

        Returns:
            Pipeline arguments dict ready for KFP submission.

        Raises:
            EnvironmentError: When a model list uses the "env" sentinel but the
                corresponding env var (AUTORAG_EMBEDDING_MODELS /
                AUTORAG_GENERATION_MODELS) is not set.
        """
        arguments = {
            "test_data_secret_name": base_config["test_data_secret_name"],
            "test_data_bucket_name": base_config["test_data_bucket_name"],
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "maas_secret_name": base_config["maas_secret_name"],
            "vector_db_secret_name": base_config["vector_db_secret_name"],
            "test_data_key": self.test_data_key or "",
            "input_data_key": self.input_data_key or "",
            "optimization_metric": self.optimization_metric or "",
        }

        if self.optimization_max_rag_patterns is not None:
            arguments["optimization_max_rag_patterns"] = self.optimization_max_rag_patterns

        embedding_models = _resolve_model_list(self.embedding_models, "AUTORAG_EMBEDDING_MODELS")
        if embedding_models:
            arguments["embedding_models"] = embedding_models
        generation_models = _resolve_model_list(self.generation_models, "AUTORAG_GENERATION_MODELS")
        if generation_models:
            arguments["generation_models"] = generation_models

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
    """Filter configs to those matching all given tags plus any tags from AUTORAG_FUNCTIONAL_TESTS_TAGS env var."""
    env_tags_raw = os.getenv("AUTORAG_FUNCTIONAL_TESTS_TAGS")
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
        tags: Optional list of tags for filtering (e.g. ["smoke", "indexing"]).
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        embedding_model_id: Embedding model ID served by MaaS. Use "env" to read from
            the ``AUTORAG_INDEXING_EMBEDDING_MODEL_ID`` env var.
        input_data_key: Path to folder with input documents within the bucket.
        collection_name: Vector store collection to reuse. Omit to create a new one.
        chunking_method: Chunking method (default: "recursive").
        chunk_size: Maximum chunk size in tokens (default: 1024).
        chunk_overlap: Token overlap between consecutive chunks (default: 0).
        batch_size: Number of documents per batch (default: 20).
        expected_failing_task: For negative scenarios, KFP task display names expected to fail.

    The vector-store backend is auto-detected by the pipeline from the secret named by
    ``vector_db_secret_name`` (MILVUS_* vs PGVECTOR_* keys), wired from VECTOR_DB_SECRET_NAME.
    """

    __test__ = False

    id: str
    description: str
    tags: list[str]
    expected_result: str
    embedding_model_id: str
    input_data_key: str | None = None
    collection_name: str | None = None
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
            "maas_secret_name": base_config["maas_secret_name"],
            "vector_db_secret_name": base_config["vector_db_secret_name"],
            "embedding_model_id": embedding_model_id,
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "input_data_key": self.input_data_key or "",
        }
        if self.collection_name is not None:
            arguments["collection_name"] = self.collection_name
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
