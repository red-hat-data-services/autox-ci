"""Dataset manifest loading (inline config or external YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_common.paths import resolve_under
from benchmark_common.yaml_io import load_yaml_dict


def load_dataset_entries(cfg: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    embedded = cfg.get("datasets")
    if embedded:
        return list(embedded)
    path_key = cfg.get("dataset_manifest_path")
    if not path_key:
        raise ValueError("Config must contain 'datasets' or 'dataset_manifest_path'")
    manifest_path = resolve_under(config_dir, str(path_key))
    m = load_yaml_dict(manifest_path)
    ds = m.get("datasets")
    if not ds:
        raise ValueError(f"No datasets: in manifest {manifest_path}")
    return list(ds)
