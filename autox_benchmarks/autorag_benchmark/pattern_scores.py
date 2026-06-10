"""Extract pattern scores from S3 rag_patterns artifacts."""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autorag_benchmark.storage_buckets import resolve_pattern_artifacts_bucket

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
    bucket: str | None = None,
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
    bucket = resolve_pattern_artifacts_bucket(config, explicit=bucket)

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

                        # Include settings if available
                        if "settings" in pattern_data:
                            pattern_info["settings"] = pattern_data["settings"]

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


def extract_pattern_scores_tabular(
    run_id: str,
    config: dict[str, Any],
    bucket: str | None = None,
    pipeline_name: str = "documents-rag-optimization-pipeline",
    optimization_metric: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract pattern scores in tabular format suitable for CSV rows.

    Args:
        run_id: KFP run ID
        config: Config dict with [s3] credentials
        bucket: S3 bucket name
        pipeline_name: Pipeline name prefix in S3
        optimization_metric: The metric used for optimization (for reference)

    Returns:
        List of flat dicts, one per pattern, with columns:
        - pattern_name: Pattern identifier
        - final_score: Overall score for the pattern
        - Individual metric scores (faithfulness, answer_relevance, etc.)
        - execution_time: Time taken to execute pattern
        - error: Error message if pattern failed

    Returns empty list if extraction fails.
    """
    result = extract_pattern_scores(
        run_id=run_id,
        config=config,
        bucket=bucket,
        pipeline_name=pipeline_name,
    )

    if "error" in result:
        error_msg = result["error"]
        logger.warning("Pattern extraction failed for run %s: %s", run_id, error_msg)
        if "not found" in error_msg.lower() or "not exist" in error_msg.lower():
            logger.info("Pattern files may not have been created yet. The pipeline needs to complete successfully and create pattern.json files in S3.")
        return []

    patterns = result.get("patterns", [])
    if not patterns:
        logger.warning("No patterns found for run %s", run_id)
        logger.info("Expected S3 path: %s", result.get("rag_patterns_s3_uri", "unknown"))
        return []

    logger.info("Successfully found %d patterns for run %s", len(patterns), run_id)

    # Convert nested pattern data to flat rows
    rows = []
    for pattern in patterns:
        row = {
            "pattern_name": pattern.get("pattern_name", ""),
            "final_score": pattern.get("final_score"),
            "execution_time": pattern.get("execution_time"),
        }

        # Add individual metric scores (mean values if dict with mean/ci_low/ci_high)
        scores = pattern.get("scores", {})
        if isinstance(scores, dict):
            # Common RAG metrics
            for metric in ["faithfulness", "answer_relevance", "answer_correctness", "context_precision", "context_recall", "context_correctness"]:
                if metric in scores:
                    score_val = scores[metric]
                    # Handle dict format with mean/ci_low/ci_high
                    if isinstance(score_val, dict) and "mean" in score_val:
                        row[f"mean_{metric}"] = score_val["mean"]
                    elif isinstance(score_val, (int, float)):
                        row[f"mean_{metric}"] = score_val

            # Include optimization metric explicitly if specified
            if optimization_metric and optimization_metric in scores:
                score_val = scores[optimization_metric]
                if isinstance(score_val, dict) and "mean" in score_val:
                    row[f"mean_{optimization_metric}"] = score_val["mean"]
                elif isinstance(score_val, (int, float)):
                    row[f"mean_{optimization_metric}"] = score_val

        # Extract parameters from settings
        settings = pattern.get("settings", {})
        if isinstance(settings, dict):
            # Chunking parameters
            chunking = settings.get("chunking", {})
            if isinstance(chunking, dict):
                row["chunking_method"] = chunking.get("method", "")
                row["chunking_chunk_size"] = chunking.get("chunk_size")
                row["chunking_chunk_overlap"] = chunking.get("chunk_overlap")

            # Embedding parameters (note: singular 'embedding', not 'embeddings')
            embedding = settings.get("embedding", {})
            if isinstance(embedding, dict):
                row["embeddings_model_id"] = embedding.get("model_id", "")

            # Retrieval parameters
            retrieval = settings.get("retrieval", {})
            if isinstance(retrieval, dict):
                row["retrieval_method"] = retrieval.get("method", "")
                row["retrieval_number_of_chunks"] = retrieval.get("number_of_chunks")
                row["retrieval_search_mode"] = retrieval.get("search_mode", "")
                # Derive ranker_strategy from ranker_k and ranker_alpha if needed
                ranker_k = retrieval.get("ranker_k")
                ranker_alpha = retrieval.get("ranker_alpha")
                if ranker_k is not None and ranker_alpha is not None:
                    row["retrieval_ranker_strategy"] = f"k={ranker_k},alpha={ranker_alpha}"

            # Generation parameters
            generation = settings.get("generation", {})
            if isinstance(generation, dict):
                row["generation_model_id"] = generation.get("model_id", "")
        else:
            logger.warning("Pattern %s has no settings field or settings is not a dict", pattern.get("pattern_name"))

        # Include error if present
        if "error" in pattern:
            row["error"] = pattern["error"]
            row["final_score"] = None  # No score if errored

        rows.append(row)

    logger.info("Extracted %d pattern rows for run %s", len(rows), run_id)
    return rows
