"""Map dataset manifest entries to RAG pipeline argument dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autorag_benchmark.settings import BenchmarkSettings


def pipeline_file_for_dataset(dataset: dict[str, Any], settings: BenchmarkSettings) -> Path:
    return settings.pipeline_yaml


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
        "llama_stack_secret_name": settings.llama_stack_secret_name,
        "llama_stack_vector_io_provider_id": settings.llama_stack_vector_io_provider_id,
    }

    if "input_data_key" in dataset and dataset["input_data_key"]:
        args["input_data_key"] = str(dataset["input_data_key"])

    if "optimization_metric" in dataset:
        args["optimization_metric"] = str(dataset["optimization_metric"])
    else:
        args["optimization_metric"] = settings.optimization_metric

    if "optimization_max_rag_patterns" in dataset:
        args["optimization_max_rag_patterns"] = int(dataset["optimization_max_rag_patterns"])
    else:
        args["optimization_max_rag_patterns"] = settings.optimization_max_rag_patterns

    for model_type in ("embedding_models", "retrieval_models", "generation_models"):
        if model_type in dataset and isinstance(dataset[model_type], list):
            args[model_type] = dataset[model_type]

    return args
