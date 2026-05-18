"""YAML file loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_dict(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
