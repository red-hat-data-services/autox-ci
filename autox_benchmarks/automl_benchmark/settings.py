"""Resolved benchmark settings from raw config dict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_common.paths import resolve_under

# User-facing AutoML training quality tiers (pipelines-components preset param).
VALID_PRESETS: frozenset[str] = frozenset({"speed", "balanced"})
DEFAULT_PRESETS: tuple[str, ...] = ("speed",)


def _artifact_root_from_storage(storage_cfg: dict[str, Any], key: str, default: str) -> str:
    """Strip slashes; empty string allowed when the key is set explicitly in config."""
    if key not in storage_cfg:
        return default.strip().strip("/")
    return str(storage_cfg.get(key) or "").strip().strip("/")


def normalize_presets(value: Any) -> tuple[str, ...]:
    """Parse ``run.presets`` / CLI / env into an ordered, deduplicated tuple of valid presets.

    Accepts a comma-separated string, a single preset string, or a list/tuple.
    """
    if value is None:
        return DEFAULT_PRESETS
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
    else:
        raise ValueError(
            f"presets must be a string or list of strings; got {type(value).__name__}: {value!r}"
        )
    if not parts:
        raise ValueError(f"presets must be non-empty; valid values: {sorted(VALID_PRESETS)}")
    invalid = [p for p in parts if p not in VALID_PRESETS]
    if invalid:
        raise ValueError(
            f"Invalid preset(s) {invalid}; valid values: {sorted(VALID_PRESETS)}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def presets_from_run_config(run_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Resolve presets from ``run.presets`` (preferred) or singular ``run.preset``."""
    if "presets" in run_cfg and run_cfg.get("presets") is not None:
        return normalize_presets(run_cfg.get("presets"))
    if "preset" in run_cfg and run_cfg.get("preset") is not None:
        return normalize_presets(run_cfg.get("preset"))
    return DEFAULT_PRESETS


@dataclass(frozen=True)
class BenchmarkSettings:
    """artifact_s3_root_*: first path segment under the bucket for KFP artifacts (pipeline template name)."""

    config_dir: Path
    pipeline_yaml: Path | None
    timeseries_pipeline_yaml: Path | None
    train_data_secret_name: str
    train_data_bucket_name: str
    artifact_s3_root_tabular: str
    artifact_s3_root_timeseries: str
    top_n: int
    presets: tuple[str, ...]
    poll_interval_seconds: float
    timeout_seconds: float
    enable_caching: bool
    experiment_name: str
    run_name_prefix: str
    benchmark_s3_prefix: str
    upload_benchmark_results: bool
    pipeline_mode: str = "package"


def benchmark_settings_from_config(cfg: dict[str, Any], config_dir: Path) -> BenchmarkSettings:
    from benchmark_common.managed_pipelines import (
        get_managed_kfp_pipeline_name,
        resolve_benchmark_pipeline_mode,
    )

    pipeline_cfg = cfg.get("pipeline") or {}
    storage_cfg = cfg.get("storage") or {}
    run_cfg = cfg.get("run") or {}
    kfp_cfg = cfg.get("kfp") or {}

    mode = resolve_benchmark_pipeline_mode(cfg)

    if mode == "managed":
        pipeline_yaml = None
        timeseries_pipeline_yaml = None
    else:
        pipeline_yaml = resolve_under(config_dir, str(pipeline_cfg.get("package_path", "../pipelines/autogluon-tabular-training-pipeline.yaml")))
        timeseries_pipeline_yaml = resolve_under(
            config_dir,
            str(pipeline_cfg.get("timeseries_package_path", "../pipelines/autogluon-timeseries-training-pipeline.yaml")),
        )
    secret = pipeline_cfg.get("train_data_secret_name")
    bucket = storage_cfg.get("train_data_bucket_name")
    if not secret or not bucket:
        raise ValueError(
            "pipeline.train_data_secret_name and storage.train_data_bucket_name are required "
            "(set in .env only, not in benchmark.yaml)"
        )

    if mode == "managed":
        tab_default = get_managed_kfp_pipeline_name("tabular", cfg)
        ts_default = get_managed_kfp_pipeline_name("timeseries", cfg)
    else:
        tab_default = "autogluon-tabular-training-pipeline"
        ts_default = "autogluon-timeseries-training-pipeline"

    tab_root = _artifact_root_from_storage(storage_cfg, "artifact_s3_prefix", tab_default)
    ts_root = _artifact_root_from_storage(storage_cfg, "timeseries_artifact_s3_prefix", ts_default)

    bench_prefix = str(storage_cfg.get("benchmark_s3_prefix") or "benchmarks/ml").strip().strip("/")
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
        timeseries_pipeline_yaml=timeseries_pipeline_yaml,
        train_data_secret_name=str(secret),
        train_data_bucket_name=str(bucket),
        artifact_s3_root_tabular=tab_root,
        artifact_s3_root_timeseries=ts_root,
        top_n=int(run_cfg.get("top_n", 3)),
        presets=presets_from_run_config(run_cfg),
        poll_interval_seconds=float(run_cfg.get("poll_interval_seconds", 30)),
        timeout_seconds=float(run_cfg.get("timeout_seconds", 86400)),
        enable_caching=bool(run_cfg.get("enable_caching", False)),
        experiment_name=str(kfp_cfg.get("experiment_name", "autogluon-benchmark")),
        run_name_prefix=str(run_cfg.get("run_name_prefix", "benchmark")),
        benchmark_s3_prefix=bench_prefix,
        upload_benchmark_results=upload_benchmark_results,
        pipeline_mode=mode,
    )
