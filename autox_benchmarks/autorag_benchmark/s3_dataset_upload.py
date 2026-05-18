"""S3 upload utilities for dataset generation."""

import os
import sys
from pathlib import Path
from typing import Any

# Import existing S3 utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "e2e-tests" / "lib"))
from s3_data import upload_file_to_s3, upload_tree_to_s3_prefix, ensure_s3_bucket_exists  # noqa: E402

# Import credentials loader
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark_common"))
from ini_credentials import load_credentials_ini  # noqa: E402


def get_s3_boto_config(credentials_path: Path | None = None) -> dict[str, Any] | None:
    """Read S3 configuration from credentials.ini file or environment variables.

    Args:
        credentials_path: Path to credentials.ini file (optional).
                         If None, tries: config/credentials.ini, then environment variables.

    Returns:
        boto3.client('s3', **config) compatible dict or None if credentials missing.

    Credentials file format ([s3] section):
        endpoint = https://s3.amazonaws.com
        aws_access_key_id = YOUR_KEY
        aws_secret_access_key = YOUR_SECRET
        aws_default_region = us-east-1

    Environment variables (fallback):
        AWS_S3_ENDPOINT: S3 endpoint URL (optional, defaults to AWS)
        AWS_ACCESS_KEY_ID: Access key (required)
        AWS_SECRET_ACCESS_KEY: Secret key (required)
        AWS_DEFAULT_REGION: Region (optional, defaults to us-east-1)
    """
    # Try credentials.ini first
    if credentials_path is None:
        # Try default location
        default_path = Path(__file__).parent.parent / "config" / "credentials.ini"
        if default_path.is_file():
            credentials_path = default_path

    if credentials_path is not None and credentials_path.is_file():
        try:
            creds = load_credentials_ini(credentials_path)
            s3_section = creds.get("s3", {})

            if s3_section.get("aws_access_key_id") and s3_section.get("aws_secret_access_key"):
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
        except Exception:
            # Fall through to environment variables
            pass

    # Fallback to environment variables
    return get_s3_boto_config_from_env()


def get_s3_boto_config_from_env() -> dict[str, Any] | None:
    """Read S3 configuration from environment variables.

    Returns boto3.client('s3', **config) compatible dict or None if credentials missing.

    Environment variables:
        AWS_S3_ENDPOINT: S3 endpoint URL (optional, defaults to AWS)
        AWS_ACCESS_KEY_ID: Access key (required)
        AWS_SECRET_ACCESS_KEY: Secret key (required)
        AWS_DEFAULT_REGION: Region (optional, defaults to us-east-1)
    """
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
    """Upload knowledge base directory and benchmark JSON to S3.

    Args:
        s3_client: Boto3 S3 client
        local_kb_dir: Path to local knowledge_base directory
        local_bench_path: Path to local benchmark_data.json file
        bucket: S3 bucket name
        prefix: S3 key prefix (e.g., "datasets/beir_scifact_50")

    Returns:
        (input_data_key, test_data_key) for dataset_manifest.yaml
    """
    prefix = prefix.strip("/")

    # Upload knowledge base directory
    kb_prefix = f"{prefix}/knowledge_base" if prefix else "knowledge_base"
    print(f"Uploading knowledge base to s3://{bucket}/{kb_prefix}...")
    upload_tree_to_s3_prefix(
        s3_client,
        bucket=bucket,
        key_prefix=kb_prefix,
        local_root=local_kb_dir,
    )

    # Upload benchmark JSON
    bench_key = f"{prefix}/benchmark_data.json" if prefix else "benchmark_data.json"
    print(f"Uploading benchmark data to s3://{bucket}/{bench_key}...")
    upload_file_to_s3(
        s3_client,
        bucket=bucket,
        key=bench_key,
        local_path=local_bench_path,
    )

    print(f"\nUpload complete!")
    print(f"  Knowledge base: s3://{bucket}/{kb_prefix}")
    print(f"  Benchmark data: s3://{bucket}/{bench_key}")

    return (kb_prefix, bench_key)


__all__ = [
    "upload_dataset_to_s3",
    "ensure_s3_bucket_exists",
    "get_s3_boto_config",
    "get_s3_boto_config_from_env",
]
