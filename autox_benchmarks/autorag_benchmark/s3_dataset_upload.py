"""S3 upload utilities for dataset generation."""

import os
import sys
from pathlib import Path
from typing import Any

# Import existing S3 utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "e2e-tests" / "lib"))
from s3_data import upload_file_to_s3, upload_tree_to_s3_prefix, ensure_s3_bucket_exists  # noqa: E402

from benchmark_common.credentials import load_credentials_overlay  # noqa: E402


def _s3_section_to_boto_config(s3_section: dict[str, Any]) -> dict[str, Any] | None:
    if not s3_section.get("aws_access_key_id") or not s3_section.get("aws_secret_access_key"):
        return None
    config: dict[str, Any] = {
        "aws_access_key_id": s3_section["aws_access_key_id"],
        "aws_secret_access_key": s3_section["aws_secret_access_key"],
    }
    endpoint = s3_section.get("endpoint")
    if endpoint:
        config["endpoint_url"] = endpoint
    region = s3_section.get("aws_default_region", "us-east-1")
    if region:
        config["region_name"] = region
    return config


def get_s3_boto_config(env_file: Path | None = None) -> dict[str, Any] | None:
    """Read S3 configuration from .env (or shell environment)."""
    try:
        overlay, _ = load_credentials_overlay(env_file=env_file)
        return _s3_section_to_boto_config(overlay.get("s3") or {})
    except (FileNotFoundError, ValueError):
        return get_s3_boto_config_from_env()


def get_s3_boto_config_from_env() -> dict[str, Any] | None:
    """Read S3 configuration from environment variables."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        return None

    config: dict[str, Any] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }

    endpoint = os.getenv("AWS_S3_ENDPOINT")
    if endpoint:
        config["endpoint_url"] = endpoint

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    if region:
        config["region_name"] = region

    return config


def upload_dataset_to_s3(
    s3_client: Any,
    *,
    local_kb_dir: Path,
    local_bench_path: Path,
    bucket: str,
    prefix: str,
) -> tuple[str, str]:
    """Upload knowledge base directory and benchmark JSON to S3."""
    prefix = prefix.strip("/")

    kb_prefix = f"{prefix}/knowledge_base" if prefix else "knowledge_base"
    print(f"Uploading knowledge base to s3://{bucket}/{kb_prefix}...")
    upload_tree_to_s3_prefix(
        s3_client,
        bucket=bucket,
        key_prefix=kb_prefix,
        local_root=local_kb_dir,
    )

    bench_key = f"{prefix}/benchmark_data.json" if prefix else "benchmark_data.json"
    print(f"Uploading benchmark data to s3://{bucket}/{bench_key}...")
    upload_file_to_s3(
        s3_client,
        bucket=bucket,
        key=bench_key,
        local_path=local_bench_path,
    )

    print("\nUpload complete!")
    print(f"  Knowledge base: s3://{bucket}/{kb_prefix}")
    print(f"  Benchmark data: s3://{bucket}/{bench_key}")

    return (kb_prefix, bench_key)


__all__ = [
    "upload_dataset_to_s3",
    "ensure_s3_bucket_exists",
    "get_s3_boto_config",
    "get_s3_boto_config_from_env",
]
