"""Test configurations for parametrized functional tests of the Documents RAG Optimization pipeline.

Configurations are loaded from test_configs.json in this directory. Each entry
specifies pipeline parameter overrides, expected result (pass/fail), and optional
tags for filtering. Use FUNCTIONAL_TESTS_TAGS (comma-separated) to run only
configs that have all of the given tags.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIGS_JSON_PATH = Path(__file__).parent / "test_configs.json"


@dataclass
class AutoRAGTestConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "positive"]).
            Use FUNCTIONAL_TESTS_TAGS to run only configs that have all of the given tags.
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
    llama_stack_vector_io_provider_type: str | None = None
    llama_stack_vector_io_provider_id: str | None = None
    embeddings_models: str | list[str] | None = None
    generation_models: str | list[str] | None = None
    optimization_max_rag_patterns: int | None = None
    input_data_key: str | None = None
    test_data_key: str | None = None
    optimization_metric: str | None = None

    def __post_init__(self):
        if self.embeddings_models == "env":
            self.embeddings_models = os.getenv("AUTORAG_EMBEDDING_MODELS")
            if self.embeddings_models is None:
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
            "llama_stack_secret_name": base_config["llama_stack_secret_name"],
            "test_data_key": self.test_data_key or "",
            "input_data_key": self.input_data_key or "",
            "optimization_metric": self.optimization_metric or "",
        }

        if self.llama_stack_vector_io_provider_id:
            arguments["llama_stack_vector_io_provider_id"] = self.llama_stack_vector_io_provider_id
        if self.optimization_max_rag_patterns is not None:
            arguments["optimization_max_rag_patterns"] = self.optimization_max_rag_patterns
        if self.embeddings_models:
            arguments["embeddings_models"] = self.embeddings_models
        if self.generation_models:
            arguments["generation_models"] = self.generation_models

        return arguments


def _read_raw_configs() -> list[dict]:
    with open(_CONFIGS_JSON_PATH) as f:
        return json.load(f)


def _load_configs(pass_type: str) -> list[AutoRAGTestConfig]:
    """Load test configs from test_configs.json and return AutoRAGTestConfig instances."""
    all_items = _read_raw_configs()

    expected = "pass" if pass_type == "positive" else "fail"
    data = [item for item in all_items if item.get("expected_result") == expected]

    configs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"test_configs[{i}] must be a dict; got {type(item).__name__}")
        try:
            raw_tags = item.pop("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(f"test_configs[{i}] 'tags' must be a list; got {type(raw_tags).__name__}")

            expected_result = item["expected_result"]
            if expected_result not in ("pass", "fail"):
                raise ValueError(
                    f"test_configs[{i}] 'expected_result' must be 'pass' or 'fail'; got '{expected_result}'"
                )

            configs.append(AutoRAGTestConfig(tags=tags, **item))
        except KeyError as e:
            raise ValueError(f"test_configs[{i}] missing required key {e}") from e
    return configs


def get_test_configs_for_run(pass_type: str, tags: None | list[str] = None) -> list[AutoRAGTestConfig]:
    """Return configs to run for this session, optionally filtered by tags.

    If tags are passed, only configs that have all of those tags are returned.
    All configs are returned otherwise.

    Args:
        pass_type (str): Type of pass to run for this session. 'positive' or negative'
        tags (None | list[str]): List of tags to run for this session.

    Returns:
        list[AutoRAGTestConfig]: List of TestConfig instances.
    """
    test_configs: list[AutoRAGTestConfig] = _load_configs(pass_type)

    tags = tags or []

    env_tags_raw = os.getenv("TESTS_TAGS")
    env_tags = [t.strip().lower() for t in env_tags_raw.split(",") if t.strip()] if env_tags_raw else []

    all_tags = {t.lower() for t in (tags + env_tags)}

    if not all_tags:
        return test_configs
    return [c for c in test_configs if all(t.lower() in c.tags for t in all_tags)]


def get_all_dataset_keys() -> tuple[list[str], list[str]]:
    """Return (input_data_keys, test_data_keys) deduplicated across all test configs."""
    all_items = _read_raw_configs()
    input_keys = list({item["input_data_key"] for item in all_items if item.get("input_data_key")})
    test_keys = list({item["test_data_key"] for item in all_items if item.get("test_data_key")})
    return input_keys, test_keys
