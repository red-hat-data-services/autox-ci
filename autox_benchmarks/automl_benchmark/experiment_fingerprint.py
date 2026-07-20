"""Canonical experiment identity for S3 dedupe (SHA-256 over sorted JSON)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from automl_benchmark.benchmark_run_metadata import sha256_file
from automl_benchmark.pipeline_params import is_timeseries_dataset
from automl_benchmark.settings import BenchmarkSettings


def _endpoint_hostname(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    s = str(endpoint).strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    return s.split("/")[0] or None


def _json_normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_normalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, list):
        return [_json_normalize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _dataset_manifest_entry(dataset: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(k for k in dataset.keys() if not str(k).startswith("_"))
    return {k: _json_normalize(dataset[k]) for k in keys}


def _environment_identity(
    cfg: dict[str, Any],
    settings: BenchmarkSettings,
    s3_cfg: dict[str, Any],
) -> dict[str, Any]:
    kfp = cfg.get("kfp") or {}
    return {
        "kfp_host": str(kfp.get("host", "")).strip() or None,
        "kfp_namespace": str(kfp.get("namespace", "")).strip() or None,
        "kfp_experiment_name": settings.experiment_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "artifact_s3_root_tabular": settings.artifact_s3_root_tabular,
        "artifact_s3_root_timeseries": settings.artifact_s3_root_timeseries,
        "benchmark_s3_prefix": settings.benchmark_s3_prefix,
        "s3_region": str(s3_cfg.get("aws_default_region") or "").strip() or None,
        "s3_endpoint_host": _endpoint_hostname(s3_cfg.get("endpoint")),
    }


def _orchestrator_options(settings: BenchmarkSettings, dataset_filter: str) -> dict[str, Any]:
    return {
        "dataset_filter": dataset_filter,
        "enable_caching": settings.enable_caching,
        "top_n": settings.top_n,
        "poll_interval_seconds": settings.poll_interval_seconds,
        "timeout_seconds": settings.timeout_seconds,
        "run_name_prefix": settings.run_name_prefix,
    }


def build_fingerprint_payload(
    *,
    pipeline_ir_path: Path | None,
    pipeline_arguments: dict[str, Any],
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    s3_cfg: dict[str, Any],
    dataset_filter: str,
    pipeline_id: str | None = None,
    pipeline_version_id: str | None = None,
) -> dict[str, Any]:
    kind = "timeseries" if is_timeseries_dataset(dataset) else "tabular"
    pipeline: dict[str, Any] = {"pipeline_kind": kind}
    if pipeline_ir_path is not None and pipeline_ir_path.is_file():
        pipeline["compiled_ir_sha256"] = sha256_file(pipeline_ir_path)
    elif pipeline_id:
        pipeline["pipeline_id"] = pipeline_id
        pipeline["pipeline_version_id"] = pipeline_version_id
    else:
        pipeline["compiled_ir_sha256"] = None
    return {
        "pipeline": pipeline,
        "pipeline_arguments": _json_normalize(pipeline_arguments),
        "dataset_manifest_entry": _dataset_manifest_entry(dataset),
        "environment_identity": _json_normalize(_environment_identity(cfg, settings, s3_cfg)),
        "orchestrator_options": _json_normalize(_orchestrator_options(settings, dataset_filter)),
    }


def compute_experiment_fingerprint(
    *,
    pipeline_ir_path: Path | None,
    pipeline_arguments: dict[str, Any],
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    s3_cfg: dict[str, Any],
    dataset_filter: str,
    pipeline_id: str | None = None,
    pipeline_version_id: str | None = None,
) -> str:
    payload = build_fingerprint_payload(
        pipeline_ir_path=pipeline_ir_path,
        pipeline_arguments=pipeline_arguments,
        dataset=dataset,
        settings=settings,
        cfg=cfg,
        s3_cfg=s3_cfg,
        dataset_filter=dataset_filter,
        pipeline_id=pipeline_id,
        pipeline_version_id=pipeline_version_id,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
