"""Load benchmark YAML and merge credentials from .env for AutoRAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark_common.credentials import CREDENTIALS_HELP, load_credentials_overlay
from benchmark_common.kubernetes_config import load_benchmark_config_file
from benchmark_common.merge import deep_merge

logger = logging.getLogger(__name__)


def validate_merged_benchmark_config(cfg: dict[str, Any]) -> None:
    kfp = cfg.get("kfp") or {}
    if not str(kfp.get("host", "")).strip():
        raise ValueError(f"Missing kfp.host. {CREDENTIALS_HELP}")
    if not str(kfp.get("namespace", "")).strip():
        raise ValueError(f"Missing kfp.namespace. {CREDENTIALS_HELP}")

    storage = cfg.get("storage") or {}
    if not str(storage.get("input_data_bucket_name", "")).strip():
        raise ValueError(f"Missing storage.input_data_bucket_name. {CREDENTIALS_HELP}")
    if not str(storage.get("test_data_bucket_name", "")).strip():
        raise ValueError(f"Missing storage.test_data_bucket_name. {CREDENTIALS_HELP}")

    pipeline = cfg.get("pipeline") or {}
    if not str(pipeline.get("input_data_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.input_data_secret_name. {CREDENTIALS_HELP}")
    if not str(pipeline.get("test_data_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.test_data_secret_name. {CREDENTIALS_HELP}")
    if not str(pipeline.get("ogx_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.ogx_secret_name. {CREDENTIALS_HELP}")
    if not str(pipeline.get("vector_io_provider_id", "")).strip():
        raise ValueError(f"Missing pipeline.vector_io_provider_id. {CREDENTIALS_HELP}")


def load_merged_benchmark_config(
    config_path: Path,
    env_file: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg, config_dir = load_benchmark_config_file(config_path)
    overlay, source = load_credentials_overlay(env_file=env_file)
    merged = deep_merge(cfg, overlay)
    logger.info("Merged credentials from %s", source)
    validate_merged_benchmark_config(merged)
    return merged, config_dir
