"""Extract pattern scores from S3 rag_patterns artifacts."""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def create_s3_client(config: dict[str, Any]):
    """Create boto3 S3 client from config."""
    s3_config = config.get("s3") or {}

    endpoint = s3_config.get("endpoint", "https://s3.amazonaws.com")
    access_key = s3_config.get("aws_access_key_id", "")
    secret_key = s3_config.get("aws_secret_access_key", "")
    region = s3_config.get("aws_default_region", "us-east-1")

    if not access_key or not secret_key:
        raise ValueError("S3 credentials (aws_access_key_id, aws_secret_access_key) required in [s3] section")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def extract_pattern_scores(
    run_id: str,
    config: dict[str, Any],
    bucket: str = "ai-eng-cracow",
    pipeline_name: str = "documents-rag-optimization-pipeline",
) -> dict[str, Any]:
    """
    Extract pattern scores from S3 rag_patterns artifacts.

    Args:
        run_id: KFP run ID
        config: Config dict with [s3] credentials
        bucket: S3 bucket name
        pipeline_name: Pipeline name prefix in S3

    Returns:
        Dict with pattern scores and metadata
    """
    try:
        s3_client = create_s3_client(config)
    except Exception as e:
        logger.warning("Could not create S3 client: %s", e)
        return {"error": f"S3 client creation failed: {e}"}

    # Find rag_patterns directory under the run
    prefix = f"{pipeline_name}/{run_id}/"
    rag_patterns_prefix = None

    try:
        # List to find rag-templates-optimization task
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

        for page in pages:
            for common_prefix in page.get("CommonPrefixes", []):
                prefix_path = common_prefix["Prefix"]
                if "rag-templates-optimization" in prefix_path:
                    # Found the task directory, now find the execution ID
                    sub_pages = paginator.paginate(Bucket=bucket, Prefix=prefix_path, Delimiter="/")
                    for sub_page in sub_pages:
                        for sub_prefix in sub_page.get("CommonPrefixes", []):
                            execution_path = sub_prefix["Prefix"]
                            # Check if rag_patterns exists
                            test_prefix = f"{execution_path}rag_patterns/"
                            test_response = s3_client.list_objects_v2(
                                Bucket=bucket,
                                Prefix=test_prefix,
                                MaxKeys=1,
                            )
                            if test_response.get("KeyCount", 0) > 0:
                                rag_patterns_prefix = test_prefix
                                break
                    if rag_patterns_prefix:
                        break

        if not rag_patterns_prefix:
            logger.warning("rag_patterns directory not found for run %s", run_id)
            return {"error": "rag_patterns directory not found"}

        logger.info("Found rag_patterns at: s3://%s/%s", bucket, rag_patterns_prefix)

        # List all pattern.json files
        pattern_scores = []
        pages = paginator.paginate(Bucket=bucket, Prefix=rag_patterns_prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/pattern.json"):
                    # Extract pattern name from path
                    # Format: .../rag_patterns/Pattern1/pattern.json
                    rel_path = key[len(rag_patterns_prefix):]
                    pattern_name = rel_path.split("/")[0]

                    try:
                        # Download and parse pattern.json
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        pattern_data = json.loads(response["Body"].read())

                        # Extract relevant fields
                        pattern_info = {
                            "pattern_name": pattern_name,
                            "final_score": pattern_data.get("final_score"),
                            "scores": pattern_data.get("scores", {}),
                            "execution_time": pattern_data.get("execution_time"),
                            "pattern_id": pattern_data.get("pattern_name"),
                        }

                        # Include indexing and rag params if available
                        if "indexing_params" in pattern_data:
                            pattern_info["indexing_params"] = pattern_data["indexing_params"]
                        if "rag_params" in pattern_data:
                            pattern_info["rag_params"] = pattern_data["rag_params"]

                        pattern_scores.append(pattern_info)
                        logger.info(
                            "Extracted scores for %s: final_score=%s",
                            pattern_name,
                            pattern_info["final_score"],
                        )

                    except Exception as e:
                        logger.error("Error reading pattern %s: %s", key, e)
                        pattern_scores.append({
                            "pattern_name": pattern_name,
                            "error": str(e),
                        })

        if not pattern_scores:
            logger.warning("No pattern.json files found in %s", rag_patterns_prefix)
            return {"error": "No pattern.json files found"}

        # Sort by final_score descending
        pattern_scores.sort(
            key=lambda x: x.get("final_score", -1) if x.get("final_score") is not None else -1,
            reverse=True,
        )

        result = {
            "rag_patterns_s3_uri": f"s3://{bucket}/{rag_patterns_prefix}",
            "num_patterns": len(pattern_scores),
            "patterns": pattern_scores,
        }

        # Add best pattern summary
        if pattern_scores and pattern_scores[0].get("final_score") is not None:
            best = pattern_scores[0]
            result["best_pattern"] = {
                "name": best.get("pattern_name"),
                "final_score": best.get("final_score"),
                "scores": best.get("scores", {}),
            }

        return result

    except ClientError as e:
        logger.error("S3 error extracting patterns for run %s: %s", run_id, e)
        return {"error": f"S3 error: {e}"}
    except Exception as e:
        logger.exception("Unexpected error extracting patterns for run %s", run_id)
        return {"error": f"Unexpected error: {e}"}
