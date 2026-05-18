"""Shared S3 upload utilities for benchmark result uploads."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any

from benchmark_common.s3_client import make_s3_client


def build_batch_id() -> str:
    """Generate timestamp-based batch ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def join_s3_key(*parts: str) -> str:
    """Join S3 key parts with slashes, stripping leading/trailing slashes."""
    return "/".join(p.strip().strip("/") for p in parts if p and str(p).strip())


def row_to_csv_bytes(row: dict[str, Any]) -> bytes:
    """Convert a single result row dict to CSV bytes (header + row)."""
    keys = sorted(row.keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerow({k: "" if row.get(k) is None else row.get(k) for k in keys})
    return buf.getvalue().encode("utf-8")


def put_s3_bytes(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
) -> None:
    """Upload bytes to S3 with specified content type."""
    client = make_s3_client(s3_cfg)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def _s3_object_not_found(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    err = exc.response.get("Error") or {}
    return err.get("Code", "") in ("404", "NoSuchKey", "NotFound")


def try_get_s3_object_bytes(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    key: str,
) -> bytes | None:
    """Return object body or ``None`` if the key is missing."""
    from benchmark_common.s3_client import s3_cfg_usable

    if not s3_cfg_usable(s3_cfg):
        return None
    client = make_s3_client(s3_cfg)
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:
        if _s3_object_not_found(e):
            return None
        raise
