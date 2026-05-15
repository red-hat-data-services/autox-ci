"""Load optional credentials and cluster settings from a .ini file (configparser)."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_credentials_ini(path: Path) -> dict[str, Any]:
    """
    Parse INI into a dict shaped like benchmark.yaml (nested sections).

    Recognized sections (case-insensitive): kfp, storage, pipeline, s3, run.
    Supports both AutoML (train_data_*, artifact prefixes) and AutoRAG keys.
    """
    cp = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        raise FileNotFoundError(path)
    read = cp.read(path)
    if not read:
        raise OSError(f"Could not read INI: {path}")

    by_section: dict[str, dict[str, str]] = {}
    for sec in cp.sections():
        by_section[sec.lower()] = {k: v.strip() for k, v in cp.items(sec)}

    out: dict[str, Any] = {}

    if "kfp" in by_section:
        kfp: dict[str, Any] = {}
        raw = by_section["kfp"]
        for key in ("host", "namespace", "token", "token_file", "token_env", "experiment_name"):
            if key in raw and raw[key] != "":
                kfp[key] = raw[key]
        if kfp:
            out["kfp"] = kfp

    if "storage" in by_section:
        st: dict[str, Any] = {}
        raw = by_section["storage"]
        if raw.get("train_data_bucket_name"):
            st["train_data_bucket_name"] = raw["train_data_bucket_name"]
        for key in ("input_data_bucket_name", "test_data_bucket_name"):
            if raw.get(key):
                st[key] = raw[key]
        for key in ("artifact_s3_prefix", "timeseries_artifact_s3_prefix", "benchmark_s3_prefix"):
            if key in raw:
                st[key] = raw[key]
        if "upload_benchmark_results" in raw:
            st["upload_benchmark_results"] = _truthy(raw["upload_benchmark_results"])
        if st:
            out["storage"] = st

    if "pipeline" in by_section:
        pl: dict[str, Any] = {}
        raw = by_section["pipeline"]
        for key in (
            "train_data_secret_name",
            "package_path",
            "timeseries_package_path",
            "input_data_secret_name",
            "test_data_secret_name",
            "ogx_secret_name",
            "vector_io_provider_id",
            # Legacy names for backwards compatibility
            "llama_stack_secret_name",
            "llama_stack_vector_io_provider_id",
        ):
            if raw.get(key):
                pl[key] = raw[key]

        # Map legacy names to new names if new names not present
        if "llama_stack_secret_name" in pl and "ogx_secret_name" not in pl:
            pl["ogx_secret_name"] = pl["llama_stack_secret_name"]
        if "llama_stack_vector_io_provider_id" in pl and "vector_io_provider_id" not in pl:
            pl["vector_io_provider_id"] = pl["llama_stack_vector_io_provider_id"]

        if pl:
            out["pipeline"] = pl

    if "s3" in by_section:
        s3: dict[str, Any] = {}
        raw = by_section["s3"]
        endpoint = raw.get("endpoint") or raw.get("aws_s3_endpoint")
        if endpoint:
            s3["endpoint"] = endpoint
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_default_region"):
            if raw.get(key):
                s3[key] = raw[key]
        if s3:
            out["s3"] = s3

    if "run" in by_section:
        rn: dict[str, Any] = {}
        raw = by_section["run"]
        if raw.get("top_n"):
            rn["top_n"] = int(raw["top_n"])
        if raw.get("optimization_metric"):
            rn["optimization_metric"] = raw["optimization_metric"]
        if raw.get("optimization_max_rag_patterns"):
            rn["optimization_max_rag_patterns"] = int(raw["optimization_max_rag_patterns"])
        if raw.get("poll_interval_seconds"):
            rn["poll_interval_seconds"] = float(raw["poll_interval_seconds"])
        if raw.get("timeout_seconds"):
            rn["timeout_seconds"] = float(raw["timeout_seconds"])
        if "enable_caching" in raw:
            rn["enable_caching"] = _truthy(raw["enable_caching"])
        if raw.get("run_name_prefix"):
            rn["run_name_prefix"] = raw["run_name_prefix"]
        if rn:
            out["run"] = rn

    return out
