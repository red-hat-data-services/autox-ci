"""S3 helpers for benchmark compare UI: list batches, fetch CSVs, local cache."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from benchmark_common.s3_client import make_s3_client, s3_cfg_usable
from benchmark_common.s3_upload import try_get_s3_object_bytes

_BATCH_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autox_benchmarks" / "compare"


def storage_from_credentials(ini_cfg: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return (bucket, benchmark_s3_prefix, s3_cfg) from merged credentials dict."""
    storage = ini_cfg.get("storage") or {}
    bucket = str(storage.get("train_data_bucket_name") or "").strip()
    if not bucket:
        raise ValueError(".env BENCHMARK_TRAIN_DATA_BUCKET_NAME is required")
    prefix = str(storage.get("benchmark_s3_prefix") or "benchmarks").strip().strip("/")
    s3_cfg = ini_cfg.get("s3") or {}
    if not s3_cfg_usable(s3_cfg):
        raise ValueError(".env AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required")
    return bucket, prefix, s3_cfg


def joined_results_key(benchmark_prefix: str) -> str:
    return f"{benchmark_prefix.strip('/')}/joined_results.csv"


def merged_leaderboards_key(benchmark_prefix: str, batch_id: str) -> str:
    return f"{benchmark_prefix.strip('/')}/{batch_id}/aggregated/merged_leaderboards.csv"


def list_batch_ids(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_prefix: str,
) -> list[str]:
    """List batch_id folders that have aggregated/merged_leaderboards.csv."""
    import botocore

    client = make_s3_client(s3_cfg)
    prefix = benchmark_prefix.strip("/") + "/"
    merged_suffix = "/aggregated/merged_leaderboards.csv"
    found: set[str] = set()

    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = str(obj.get("Key") or "")
                if not key.endswith(merged_suffix):
                    continue
                rest = key[len(prefix) : -len(merged_suffix)]
                batch_id = rest.strip("/")
                if _BATCH_ID_RE.match(batch_id):
                    found.add(batch_id)
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(f"Failed to list s3://{bucket}/{prefix}: {e}") from e

    return sorted(found, reverse=True)


def _head_object_meta(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    key: str,
) -> dict[str, str] | None:
    client = make_s3_client(s3_cfg)
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    etag = str(resp.get("ETag") or "").strip('"')
    lm = str(resp.get("LastModified") or "")
    return {"etag": etag, "last_modified": lm}


def _cache_path(cache_dir: Path, bucket: str, key: str, meta: dict[str, str]) -> Path:
    digest = hashlib.sha256(f"{bucket}/{key}/{meta.get('etag')}/{meta.get('last_modified')}".encode()).hexdigest()[:16]
    safe = key.replace("/", "__")
    return cache_dir / f"{safe}.{digest}.csv"


def _cache_meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def fetch_csv_cached(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    key: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> bytes:
    """Download S3 object bytes, using a local cache keyed by ETag / LastModified."""
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    meta = _head_object_meta(s3_cfg=s3_cfg, bucket=bucket, key=key)
    if meta is None:
        raw = try_get_s3_object_bytes(s3_cfg=s3_cfg, bucket=bucket, key=key)
        if raw is None:
            raise FileNotFoundError(f"s3://{bucket}/{key}")
        return raw

    path = _cache_path(cache_root, bucket, key, meta)
    meta_path = _cache_meta_path(path)
    if not force_refresh and path.is_file() and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored.get("etag") == meta.get("etag") and stored.get("last_modified") == meta.get("last_modified"):
                return path.read_bytes()
        except (json.JSONDecodeError, OSError):
            pass

    raw = try_get_s3_object_bytes(s3_cfg=s3_cfg, bucket=bucket, key=key)
    if raw is None:
        raise FileNotFoundError(f"s3://{bucket}/{key}")
    path.write_bytes(raw)
    meta_path.write_text(json.dumps({"bucket": bucket, "key": key, **meta}), encoding="utf-8")
    return raw


def fetch_joined_results(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_prefix: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> bytes:
    key = joined_results_key(benchmark_prefix)
    return fetch_csv_cached(
        s3_cfg=s3_cfg,
        bucket=bucket,
        key=key,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )


def fetch_merged_leaderboards(
    *,
    s3_cfg: dict[str, Any],
    bucket: str,
    benchmark_prefix: str,
    batch_id: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> bytes:
    key = merged_leaderboards_key(benchmark_prefix, batch_id)
    return fetch_csv_cached(
        s3_cfg=s3_cfg,
        bucket=bucket,
        key=key,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
