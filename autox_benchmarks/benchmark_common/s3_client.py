"""Shared boto3 S3 client construction from credentials.ini [s3] section."""

from __future__ import annotations

from typing import Any


def s3_cfg_usable(s3_cfg: dict[str, Any] | None) -> bool:
    if not s3_cfg or not isinstance(s3_cfg, dict):
        return False
    return bool(str(s3_cfg.get("aws_access_key_id", "")).strip()) and bool(
        str(s3_cfg.get("aws_secret_access_key", "")).strip()
    )


def make_s3_client(s3_cfg: dict[str, Any]) -> Any:
    import boto3

    kwargs: dict[str, Any] = {}
    ep = s3_cfg.get("endpoint")
    if ep:
        kwargs["endpoint_url"] = str(ep).strip()
    return boto3.client(
        "s3",
        aws_access_key_id=str(s3_cfg.get("aws_access_key_id", "")).strip() or None,
        aws_secret_access_key=str(s3_cfg.get("aws_secret_access_key", "")).strip() or None,
        region_name=str(s3_cfg.get("aws_default_region") or "us-east-1").strip(),
        **kwargs,
    )
