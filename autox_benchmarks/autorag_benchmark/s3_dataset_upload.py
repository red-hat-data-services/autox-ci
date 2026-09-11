"""S3 upload utilities for dataset generation."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Import existing S3 utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "autox_tests" / "lib"))
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


def _resolve_document_keys(local_kb_dir: Path, kb_prefix: str, values: list[str]) -> list[str]:
    """Turn benchmark document references into full S3 object keys.

    Generators emit bare file names because the S3 prefix does not exist yet at
    generation time.  A document's key downstream is its full object key -- the
    value ai4rag stores as ``DoclingDocument.name`` and matches benchmark data
    against -- so the names are resolved here, where ``kb_prefix`` is known.

    Names are resolved against the knowledge base on disk so that a file nested
    inside ``local_kb_dir`` keeps its relative path, mirroring how
    :func:`upload_tree_to_s3_prefix` lays the tree out in S3.
    """
    by_relpath: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}
    for path in sorted(local_kb_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_kb_dir).as_posix()
        by_relpath[rel] = rel
        by_name.setdefault(path.name, []).append(rel)

    resolved: list[str] = []
    for value in values:
        # Tolerate an already-prefixed value so re-uploading is not destructive.
        candidate = value[len(kb_prefix) + 1 :] if value.startswith(f"{kb_prefix}/") else value

        if candidate in by_relpath:
            rel = candidate
        else:
            matches = by_name.get(Path(candidate).name, [])
            if not matches:
                raise ValueError(
                    f"Benchmark references {value!r} but no such file exists under {local_kb_dir}. "
                    "Every correct_answer_document_keys entry must name a knowledge base file."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Benchmark reference {value!r} is ambiguous -- it matches {matches}. "
                    "Use the path relative to the knowledge base directory instead."
                )
            rel = matches[0]

        resolved.append(f"{kb_prefix}/{rel}" if kb_prefix else rel)
    return resolved


def _rewrite_benchmark_for_upload(local_bench_path: Path, local_kb_dir: Path, kb_prefix: str) -> str:
    """Return benchmark JSON text with document keys expanded to full S3 keys."""
    records = json.loads(local_bench_path.read_text(encoding="utf-8"))
    for record in records:
        keys = record.get("correct_answer_document_keys")
        if keys:
            record["correct_answer_document_keys"] = _resolve_document_keys(local_kb_dir, kb_prefix, keys)
    return json.dumps(records, indent=2, ensure_ascii=False)


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
    rewritten = _rewrite_benchmark_for_upload(local_bench_path, local_kb_dir, kb_prefix)
    with tempfile.TemporaryDirectory() as tmp_dir:
        staged = Path(tmp_dir) / "benchmark_data.json"
        staged.write_text(rewritten, encoding="utf-8")
        upload_file_to_s3(
            s3_client,
            bucket=bucket,
            key=bench_key,
            local_path=staged,
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
