"""S3 helpers shared across RHOAI test suites."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _s3_error_code(exc: BaseException) -> str:
    """Extract the S3 error code string from a botocore ClientError."""
    if hasattr(exc, "response"):
        err = getattr(exc, "response", {}) or {}
        if isinstance(err, dict):
            return str((err.get("Error") or {}).get("Code") or "")
    return ""


def _s3_http_status(exc: BaseException) -> int | None:
    """Extract the HTTP status code from a botocore ClientError."""
    if hasattr(exc, "response"):
        err = getattr(exc, "response", {}) or {}
        if isinstance(err, dict):
            meta = err.get("ResponseMetadata") or {}
            code = meta.get("HTTPStatusCode")
            return int(code) if code is not None else None
    return None


def _bucket_not_found(exc: BaseException) -> bool:
    """Return True if the exception indicates the S3 bucket does not exist."""
    code = _s3_error_code(exc)
    status = _s3_http_status(exc)
    if status == 404:
        return True
    return code in ("404", "NoSuchBucket", "NotFound")


def ensure_s3_bucket_exists(
    s3_client: Any,
    bucket: str,
    *,
    region: str | None = None,
) -> None:
    """Create ``bucket`` when it is missing, if the API allows (no-op when it already exists).

    Uses :func:`head_bucket` first. For creation: MinIO and most S3-compatible endpoints use a
    plain ``create_bucket``; AWS S3 outside ``us-east-1`` uses ``LocationConstraint``.

    Raises the underlying client error if the bucket is missing and cannot be created.
    """
    b = (bucket or "").strip()
    if not b:
        return

    from botocore.exceptions import ClientError

    try:
        s3_client.head_bucket(Bucket=b)
        return
    except ClientError as e:
        if not _bucket_not_found(e):
            raise

    endpoint = (getattr(s3_client.meta, "endpoint_url", None) or "").lower()
    is_aws = "amazonaws.com" in endpoint
    reg = (region or "us-east-1").strip() or "us-east-1"

    try:
        if is_aws and reg != "us-east-1":
            s3_client.create_bucket(
                Bucket=b,
                CreateBucketConfiguration={"LocationConstraint": reg},
            )
        else:
            s3_client.create_bucket(Bucket=b)
    except ClientError as e:
        code = _s3_error_code(e)
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return
        raise
    logger.info("Created S3 bucket %r (region=%s)", b, reg)


def upload_file_to_s3(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    local_path: Path,
) -> None:
    """Upload a single file to S3 with a best-effort Content-Type."""
    content_type, _ = mimetypes.guess_type(str(local_path))
    extra_args: dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type
    s3_client.upload_file(str(local_path), bucket, key, ExtraArgs=extra_args or None)


def upload_tree_to_s3_prefix(
    s3_client: Any,
    *,
    bucket: str,
    key_prefix: str,
    local_root: Path,
) -> None:
    """Upload every file under ``local_root`` preserving relative paths under ``key_prefix``."""
    if not local_root.is_dir():
        raise ValueError(f"Not a directory: {local_root}")
    prefix = key_prefix.strip("/")
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root)
        key = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
        upload_file_to_s3(s3_client, bucket=bucket, key=key, local_path=path)


def list_s3_objects(s3_client: Any, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns list of {Key, Size, ...} dicts."""
    paginator = s3_client.get_paginator("list_objects_v2")
    return [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents") or []
    ]


def delete_s3_objects(s3_client: Any, bucket: str, keys: list[str]) -> int:
    """Delete S3 objects in batches of 1000. Returns the count of confirmed deleted objects."""
    deleted = 0
    batch_size = 1000
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        delete_req = {"Objects": [{"Key": k} for k in batch], "Quiet": True}
        try:
            response = s3_client.delete_objects(Bucket=bucket, Delete=delete_req)
            errors = response.get("Errors", [])
            for err in errors:
                logger.warning(
                    "Failed to delete s3://%s/%s: %s", bucket, err["Key"], err.get("Message")
                )
            deleted += len(batch) - len(errors)
        except Exception as e:
            logger.warning(
                "Failed to delete %d objects from s3://%s: %s", len(batch), bucket, e
            )
    return deleted


class S3CleanupTracker:
    """Accumulates S3 artifact prefixes to delete during session teardown."""

    def __init__(self) -> None:
        self.artifact_prefixes: dict[str, list[str]] = {}

    def track_artifact_prefix(self, bucket: str, prefix: str) -> None:
        """Record a pipeline artifact prefix for teardown cleanup."""
        self.artifact_prefixes.setdefault(bucket, []).append(prefix)
