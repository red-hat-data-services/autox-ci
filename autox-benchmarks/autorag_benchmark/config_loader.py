"""Load benchmark YAML and merge credentials INI for AutoRAG benchmarks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from benchmark_common.ini_credentials import load_credentials_ini
from benchmark_common.kubernetes_config import load_benchmark_config_file
from benchmark_common.merge import deep_merge

logger = logging.getLogger(__name__)

_CREDENTIALS_HELP = (
    "Copy config/credentials.example.ini to config/credentials.ini, set [kfp], [storage], [pipeline], "
    "or pass --credentials / set $BENCHMARK_CREDENTIALS_PATH."
)


def resolve_credentials_ini_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Credentials INI not found: {p}")
        return p
    env_p = os.environ.get("BENCHMARK_CREDENTIALS_PATH")
    if env_p:
        p = Path(env_p).resolve()
        return p if p.is_file() else None
    default = Path("config/credentials.ini").resolve()
    return default if default.is_file() else None


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
    if not str(pipeline.get("llama_stack_secret_name", "")).strip():
        raise ValueError(f"Missing pipeline.llama_stack_secret_name. {_CREDENTIALS_HELP}")
    if not str(pipeline.get("llama_stack_vector_io_provider_id", "")).strip():
        raise ValueError(f"Missing pipeline.llama_stack_vector_io_provider_id. {_CREDENTIALS_HELP}")


def load_merged_benchmark_config(
    config_path: Path,
    credentials_ini_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg, config_dir = load_benchmark_config_file(config_path)
    try:
        ini_path = resolve_credentials_ini_path(credentials_ini_path)
    except FileNotFoundError:
        raise
    if ini_path is None:
        raise FileNotFoundError(f"No credentials INI found. {_CREDENTIALS_HELP}")
    overlay = load_credentials_ini(ini_path)
    merged = deep_merge(cfg, overlay)
    logger.info("Merged credentials from %s", ini_path)
    validate_merged_benchmark_config(merged)
    return merged, config_dir
