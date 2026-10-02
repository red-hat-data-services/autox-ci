"""End-to-end evaluation: run the standalone evaluator against an indexed collection."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME

logger = logging.getLogger(__name__)


def run_full_corpus_evaluation(
    *,
    indexing_run_id: str,
    hpo_run_id: str,
    config_path: Path,
    env_file: Path | None,
    config: dict[str, Any],
    bucket: str,
    test_data_key: str,
    pattern_name: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the standalone evaluator after indexing and return its S3-collected metrics.

    This intentionally invokes the evaluator as a separate process: it needs the
    optional RAGAS dependencies, while a normal benchmark run does not.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_indexed_rag.py"
    command = [
        sys.executable, str(script), "--indexing-run-id", indexing_run_id,
        "--hpo-run-id", hpo_run_id, "--config", str(config_path),
        "--test-data-key", test_data_key,
    ]
    if env_file:
        command.extend(["--env-file", str(env_file)])
    if pattern_name:
        command.extend(["--pattern-name", pattern_name])
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=True, timeout=timeout_seconds,
        )
        if completed.stdout.strip():
            logger.info("Full-corpus evaluator: %s", completed.stdout.strip())
    except subprocess.TimeoutExpired:
        logger.error("Full-corpus evaluator timed out after %ss", timeout_seconds)
        return {"e2e_error": f"evaluator timed out after {timeout_seconds}s"}
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        logger.error("Full-corpus evaluator failed: %s", detail)
        return {"e2e_error": detail}
    return collect_full_corpus_evaluation_metrics(
        config=config, bucket=bucket, indexing_run_id=indexing_run_id, pattern_name=pattern_name,
    )


def collect_full_corpus_evaluation_metrics(
    config: dict[str, Any],
    bucket: str,
    indexing_run_id: str,
    *,
    pattern_name: str | None = None,
) -> dict[str, Any]:
    """Read a dedicated full-corpus QA evaluator artifact, if one was produced.

    A valid full-corpus evaluator must query the indexing run's collection with
    benchmark questions and write the resulting QA rows under its indexing-run
    artifact prefix. This deliberately never falls back to HPO results.
    """
    from autorag_benchmark.metrics.quality_metrics import aggregate_evaluation_results
    from autorag_benchmark.pattern_scores import create_s3_client

    flat: dict[str, Any] = {}
    try:
        s3_client = create_s3_client(config)
        prefix = f"{INDEXING_PIPELINE_NAME}/{indexing_run_id}/"
        paginator = s3_client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith("full_corpus_evaluation_results.json"))
        if pattern_name:
            matching = [key for key in keys if f"/rag_patterns/{pattern_name}/" in key]
            keys = matching or keys
        if keys:
            response = s3_client.get_object(Bucket=bucket, Key=sorted(keys)[0])
            flat = aggregate_evaluation_results(json.loads(response["Body"].read()), prefix="e2e_")
    except Exception as e:
        logger.warning("Could not collect E2E metrics: %s", e)
        flat["e2e_error"] = str(e)

    if not flat:
        flat["e2e_note"] = "full_corpus_evaluation_results.json not found"

    return flat
