"""JSON metadata payloads for S3 benchmark result uploads (no secrets)."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from automl_benchmark.settings import BenchmarkSettings


SCHEMA_VERSION = "1.0"


def try_git_commit_hash(cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pipeline_template_name_from_ir(path: Path) -> str | None:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return _pipeline_name_from_comment(path)
    if isinstance(data, dict):
        pi = data.get("pipelineInfo")
        if isinstance(pi, dict) and pi.get("name"):
            return str(pi["name"])
        if data.get("name"):
            return str(data["name"])
    return _pipeline_name_from_comment(path)


def _pipeline_name_from_comment(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:15]:
            m = re.match(r"^#\s*Name:\s*(\S+)", line.strip())
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def pipeline_definition_block(ir_path: Path) -> dict[str, Any]:
    name = pipeline_template_name_from_ir(ir_path)
    out: dict[str, Any] = {
        "compiled_ir_path": str(ir_path),
        "pipeline_template_name": name,
    }
    if ir_path.is_file():
        out["compiled_ir_sha256"] = sha256_file(ir_path)
    return out


def task_components_from_metrics_blob(metrics_blob: str) -> list[str]:
    if not metrics_blob or not metrics_blob.strip():
        return []
    try:
        d = json.loads(metrics_blob)
    except json.JSONDecodeError:
        return []
    tasks = d.get("task_details") or []
    if not isinstance(tasks, list):
        return []
    names: list[str] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        dn = t.get("display_name")
        if isinstance(dn, str) and dn.strip():
            names.append(dn.strip())
    return names


def _endpoint_hostname(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    s = str(endpoint).strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    return s.split("/")[0] or None


def environment_context(
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
        "s3_region": str(s3_cfg.get("aws_default_region") or "").strip() or None,
        "s3_endpoint_host": _endpoint_hostname(s3_cfg.get("endpoint")),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def build_input_params(
    *,
    settings: BenchmarkSettings,
    dataset: dict[str, Any],
    arguments: dict[str, Any] | None,
    pipeline_file: Path,
    dataset_filter: str,
    fail_fast: bool,
    artifact_s3_root: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "top_n": settings.top_n,
        "run_name_prefix": settings.run_name_prefix,
        "enable_caching": settings.enable_caching,
        "poll_interval_seconds": settings.poll_interval_seconds,
        "timeout_seconds": settings.timeout_seconds,
        "dataset_filter": dataset_filter,
        "fail_fast": fail_fast,
        "artifact_s3_prefix": artifact_s3_root,
        "pipeline_package_name": pipeline_file.name,
        "manifest_dataset": {k: dataset.get(k) for k in dataset if k is not None},
    }
    if arguments is not None:
        out["pipeline_arguments"] = arguments
    return out


def build_run_metadata(
    *,
    row: dict[str, Any],
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    s3_cfg: dict[str, Any],
    pipeline_ir_path: Path,
    s3_benchmark_key_prefix: str,
    arguments: dict[str, Any] | None,
    dataset_filter: str,
    fail_fast: bool,
    artifact_s3_root: str,
    repo_root: Path | None,
    experiment_fingerprint: str | None = None,
) -> dict[str, Any]:
    metrics_blob = str(row.get("metrics_blob") or "")
    commit = try_git_commit_hash(repo_root)
    ts = datetime.now(timezone.utc).isoformat()
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": ts,
        "commit_hash": commit,
        "pipeline_definition": pipeline_definition_block(pipeline_ir_path),
        "pipeline_components": task_components_from_metrics_blob(metrics_blob),
        "input_params": build_input_params(
            settings=settings,
            dataset=dataset,
            arguments=arguments,
            pipeline_file=pipeline_ir_path,
            dataset_filter=dataset_filter,
            fail_fast=fail_fast,
            artifact_s3_root=artifact_s3_root,
        ),
        "downstream_dependencies": [
            {
                "name": "aggregated_merged_leaderboards",
                "relative_s3_key": "aggregated/merged_leaderboards.csv",
                "description": "Batch-level long-form CSV merged from leaderboard HTML tables",
            }
        ],
        "environment_context": environment_context(cfg, settings, s3_cfg),
        "run_id": str(row.get("run_id") or ""),
        "dataset_id": str(row.get("dataset_id") or dataset.get("id") or ""),
        "leaderboard_html_s3_uri": str(row.get("leaderboard_html_s3_uri") or ""),
        "s3_benchmark_prefix": s3_benchmark_key_prefix,
    }
    if experiment_fingerprint:
        out["experiment_fingerprint"] = experiment_fingerprint
    if str(row.get("dedupe_cache_hit") or "").lower() in ("1", "true", "yes"):
        out["dedupe_cache_hit"] = True
    return out


def build_batch_metadata(
    *,
    batch_id: str,
    started_at: str,
    finished_at: str,
    manifest_relative: str,
    settings: BenchmarkSettings,
    cfg: dict[str, Any],
    dataset_ids: list[str],
    row_count: int,
    output_csv_relative: str,
    repo_root: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "commit_hash": try_git_commit_hash(repo_root),
        "kfp_experiment_name": settings.experiment_name,
        "dataset_manifest_path": manifest_relative,
        "benchmark_s3_prefix": settings.benchmark_s3_prefix,
        "dataset_ids": dataset_ids,
        "benchmark_row_count": row_count,
        "local_output_csv": output_csv_relative,
    }


def sanitize_path_segment(s: str) -> str:
    out: list[str] = []
    for c in s:
        if c.isalnum() or c in "-_.":
            out.append(c)
        else:
            out.append("_")
    t = "".join(out).strip("._")
    return t if t else "unknown"


def dataset_results_subpath(dataset: dict[str, Any]) -> str:
    raw = (dataset.get("train_data_file_key") or "").strip().lstrip("/")
    ds_id = sanitize_path_segment(str(dataset.get("id", dataset.get("name", "unknown"))))
    if not raw:
        return f"unknown/{ds_id}"
    pp = PurePosixPath(raw)
    parts = pp.parts
    if parts and parts[0] == "datasets":
        inner = PurePosixPath(*parts[1:]) if len(parts) > 1 else PurePosixPath("")
    else:
        inner = pp
    if str(inner) in ("", ".", "/"):
        return f"unknown/{ds_id}"
    if inner.suffix:
        inner = inner.parent / inner.stem
    segs = [sanitize_path_segment(p) for p in inner.parts if p not in (".", "..", "")]
    if not segs:
        return f"unknown/{ds_id}"
    return "/".join(segs)
