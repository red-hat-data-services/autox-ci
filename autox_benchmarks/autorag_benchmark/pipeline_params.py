"""Map dataset manifest entries to RAG pipeline argument dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autorag_benchmark.settings import BenchmarkSettings


def pipeline_file_for_dataset(dataset: dict[str, Any], settings: BenchmarkSettings) -> Path:
    return settings.pipeline_yaml


def indexing_preset_for_dataset(dataset: dict[str, Any], settings: BenchmarkSettings) -> str | None:
    """Use the HPO quality tier for indexing unless an explicit override is set."""
    return settings.indexing_preset or dataset.get("preset") or settings.preset or None


def align_database_secret_argument(
    arguments: dict[str, Any], supported_parameters: set[str] | None,
) -> dict[str, Any]:
    """Use the database-secret input name declared by a compiled AutoRAG package."""
    if (
        supported_parameters is not None
        and "db_secret_name" in supported_parameters
        and "vector_db_secret_name" not in supported_parameters
        and "vector_db_secret_name" in arguments
    ):
        aligned = dict(arguments)
        aligned["db_secret_name"] = aligned.pop("vector_db_secret_name")
        return aligned
    return arguments


def _merge_manifest_pipeline_overrides(
    arguments: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    extra = dataset.get("pipeline_arguments") or dataset.get("pipeline_params")
    if not isinstance(extra, dict) or not extra:
        return arguments
    return {**arguments, **extra}


def build_pipeline_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "input_data_bucket_name": settings.input_data_bucket_name,
        "input_data_secret_name": settings.input_data_secret_name,
        "test_data_bucket_name": settings.test_data_bucket_name,
        "test_data_secret_name": settings.test_data_secret_name,
        "test_data_key": str(dataset["test_data_key"]),
        "maas_secret_name": settings.maas_secret_name,
        "vector_db_secret_name": settings.vector_db_secret_name,
    }

    # The optimization pipeline accepts a list; document discovery uses its first key.
    if dataset.get("input_data_key"):
        args["input_data_keys"] = [str(dataset["input_data_key"])]

    if "optimization_metric" in dataset:
        args["optimization_metric"] = str(dataset["optimization_metric"])
    else:
        args["optimization_metric"] = settings.optimization_metric

    if "optimization_max_rag_patterns" in dataset:
        args["optimization_max_rag_patterns"] = int(dataset["optimization_max_rag_patterns"])
    else:
        args["optimization_max_rag_patterns"] = settings.optimization_max_rag_patterns

    preset = str(dataset.get("preset", "")).strip() or settings.preset
    if preset:
        args["preset"] = preset
    args["gpu_acceleration"] = settings.gpu_acceleration

    # Model lists are required by the MaaS pipeline: prefer per-dataset overrides,
    # else fall back to the benchmark-wide defaults from settings.
    embedding_models = dataset.get("embedding_models") or settings.embedding_models
    if embedding_models:
        args["embedding_models"] = list(embedding_models)
    generation_models = dataset.get("generation_models") or settings.generation_models
    if generation_models:
        args["generation_models"] = list(generation_models)

    return _merge_manifest_pipeline_overrides(args, dataset)
