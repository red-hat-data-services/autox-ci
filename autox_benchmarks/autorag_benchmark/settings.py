"""Resolved AutoRAG benchmark settings from raw config dict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_common.paths import resolve_under


@dataclass(frozen=True)
class BenchmarkSettings:
    config_dir: Path
    pipeline_yaml: Path
    input_data_secret_name: str
    input_data_bucket_name: str
    test_data_secret_name: str
    test_data_bucket_name: str
    ogx_secret_name: str
    vector_io_provider_id: str
    optimization_metric: str
    optimization_max_rag_patterns: int
    system_prompt: str
    poll_interval_seconds: float
    timeout_seconds: float
    enable_caching: bool
    experiment_name: str
    run_name_prefix: str
    benchmark_s3_prefix: str
    upload_benchmark_results: bool


def benchmark_settings_from_config(cfg: dict[str, Any], config_dir: Path) -> BenchmarkSettings:
    pipeline_cfg = cfg.get("pipeline") or {}
    storage_cfg = cfg.get("storage") or {}
    run_cfg = cfg.get("run") or {}
    kfp_cfg = cfg.get("kfp") or {}

    pipeline_yaml = resolve_under(
        config_dir,
        str(pipeline_cfg.get("package_path", "../pipelines/documents-rag-optimization-pipeline.yaml")),
    )

    input_data_secret = pipeline_cfg.get("input_data_secret_name")
    input_data_bucket = storage_cfg.get("input_data_bucket_name")
    test_data_secret = pipeline_cfg.get("test_data_secret_name")
    test_data_bucket = storage_cfg.get("test_data_bucket_name")
    ogx_secret = pipeline_cfg.get("ogx_secret_name")
    vector_io_provider = pipeline_cfg.get("vector_io_provider_id")

    if not all(
        [
            input_data_secret,
            input_data_bucket,
            test_data_secret,
            test_data_bucket,
            ogx_secret,
            vector_io_provider,
        ]
    ):
        raise ValueError(
            "Required configuration missing. Please set in credentials.ini: "
            "pipeline.input_data_secret_name, pipeline.test_data_secret_name, "
            "pipeline.ogx_secret_name, pipeline.vector_io_provider_id, "
            "storage.input_data_bucket_name, storage.test_data_bucket_name"
        )

    bench_prefix = str(storage_cfg.get("benchmark_s3_prefix") or "benchmarks/rag").strip().strip("/")
    upload_raw = storage_cfg.get("upload_benchmark_results")
    if upload_raw is None:
        upload_benchmark_results = True
    elif isinstance(upload_raw, bool):
        upload_benchmark_results = upload_raw
    else:
        upload_benchmark_results = str(upload_raw).strip().lower() in ("1", "true", "yes", "on")

    return BenchmarkSettings(
        config_dir=config_dir,
        pipeline_yaml=pipeline_yaml,
        input_data_secret_name=str(input_data_secret),
        input_data_bucket_name=str(input_data_bucket),
        test_data_secret_name=str(test_data_secret),
        test_data_bucket_name=str(test_data_bucket),
        ogx_secret_name=str(ogx_secret),
        vector_io_provider_id=str(vector_io_provider),
        optimization_metric=str(run_cfg.get("optimization_metric", "faithfulness")),
        optimization_max_rag_patterns=int(run_cfg.get("optimization_max_rag_patterns", 8)),
        system_prompt=str(run_cfg.get("system_prompt", "")),
        poll_interval_seconds=float(run_cfg.get("poll_interval_seconds", 30)),
        timeout_seconds=float(run_cfg.get("timeout_seconds", 86400)),
        enable_caching=bool(run_cfg.get("enable_caching", False)),
        experiment_name=str(kfp_cfg.get("experiment_name", "rag-optimization-benchmark")),
        run_name_prefix=str(run_cfg.get("run_name_prefix", "rag-benchmark")),
        benchmark_s3_prefix=bench_prefix,
        upload_benchmark_results=upload_benchmark_results,
    )
