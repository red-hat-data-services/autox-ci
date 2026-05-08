"""Kubeflow Pipelines / OpenShift Data Science Pipelines API client factory."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_kfp_token(kfp_cfg: dict[str, Any]) -> str:
    t = kfp_cfg.get("token")
    if t is not None and str(t).strip():
        return str(t).strip()
    tf = kfp_cfg.get("token_file")
    if tf:
        path = Path(str(tf)).expanduser()
        return path.read_text(encoding="utf-8").strip()
    token_env = kfp_cfg.get("token_env", "KFP_API_TOKEN")
    return os.environ.get(str(token_env), "")


def create_kfp_client(cfg: dict[str, Any]):
    from kfp.client import Client

    kfp_cfg = cfg.get("kfp") or {}
    host = kfp_cfg.get("host")
    if not host:
        raise ValueError("kfp.host is required in config")
    ns = kfp_cfg.get("namespace")
    token = resolve_kfp_token(kfp_cfg)
    if not token:
        token_env = kfp_cfg.get("token_env", "KFP_API_TOKEN")
        logger.warning(
            "No KFP token (set [kfp] token / token_file in credentials.ini or %s)",
            token_env,
        )
    kwargs: dict[str, Any] = {"host": host, "namespace": ns}
    if token:
        kwargs["existing_token"] = token
    return Client(**kwargs)
