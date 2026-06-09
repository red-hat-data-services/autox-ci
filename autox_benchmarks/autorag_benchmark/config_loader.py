"""Load benchmark YAML and merge credentials from .env (or legacy INI) for AutoRAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark_common.credentials import _CREDENTIALS_HELP, load_credentials_overlay
from benchmark_common.kubernetes_config import load_benchmark_config_file
from benchmark_common.merge import deep_merge

logger = logging.getLogger(__name__)


def validate_merged_benchmark_config(cfg: dict[str, Any]) -> None:
    kfp = cfg.get("kfp") or {}
    if not str(kfp.get("host", "")).strip():
        raise ValueError(f"Missing kfp.host. {_CREDENTIALS_HELP}")
    if not str(kfp.get("namespace", "")).strip():
        raise ValueError(f"Missing kfp.namespace. {_CREDENTIALS_HELP}")

    storage = cfg.get("storage") or {}
    if not str(storage.get("input_data_bucket_name", "")).strip():
        raise ValueError(f"Missing storage.input_data_bucket_name. {_CREDENTIALS_HELP}")
    if not str(storage.get("test_data_bucket_name", "")).strip():
        raise ValueError(f"Missing storage.test_data_bucket_name. {_CREDENTIALS_HELP}")

    pipeline = cfg.get("pipeline") or {}
    if not str(pipeline.get("input_data_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.input_data_secret_name. {_CREDENTIALS_HELP}")
    if not str(pipeline.get("test_data_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.test_data_secret_name. {_CREDENTIALS_HELP}")
    if not str(pipeline.get("ogx_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.ogx_secret_name. {_CREDENTIALS_HELP}")
    if not str(pipeline.get("vector_io_provider_id", "")).strip():
        raise ValueError(f"Missing pipeline.vector_io_provider_id. {_CREDENTIALS_HELP}")


def load_merged_benchmark_config(
    config_path: Path,
    credentials_ini_path: Path | None = None,
    env_file: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg, config_dir = load_benchmark_config_file(config_path)
    overlay, source = load_credentials_overlay(
        env_file=env_file,
        credentials_path=credentials_ini_path,
    )
    merged = deep_merge(cfg, overlay)
    logger.info("Merged credentials from %s", source)
    validate_merged_benchmark_config(merged)
    return merged, config_dir
