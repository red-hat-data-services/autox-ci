"""End-to-end evaluation: run the standalone evaluator against an indexed collection."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME

logger = logging.getLogger(__name__)


def rebase_test_data_document_keys(
    records: list[dict[str, Any]], *, test_data_key: str, indexed_data_key: str,
) -> list[dict[str, Any]]:
    """Point gold document IDs at the corpus indexed for this E2E evaluation."""
    source_prefix = f"{test_data_key.rsplit('/', 1)[0]}/knowledge_base/"
    target_prefix = f"{indexed_data_key.rstrip('/')}/"

    def rebase(key: str) -> str:
        if key.startswith(target_prefix):
            return key
        if key.startswith(source_prefix):
            return target_prefix + key[len(source_prefix):]
        if "/" not in key:
            return target_prefix + key
        raise ValueError(f"Gold document key {key!r} is outside {source_prefix!r}")

    rewritten = []
    for record in records:
        item = dict(record)
        for field in ("correct_answer_document_keys", "correct_answer_document_ids"):
            if field in item:
                item[field] = [rebase(key) for key in item[field]]
        rewritten.append(item)
    return rewritten


def _evaluator_python() -> str:
    """Select a Python environment with the generated notebook runtime installed."""
    override = os.environ.get("BENCHMARK_EVALUATOR_PYTHON")
    if override:
        return override
    if importlib.util.find_spec("ai4rag") is not None:
        return sys.executable

    repository_python = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    if repository_python.is_file():
        probe = subprocess.run(
            [str(repository_python), "-c", "import importlib.util; assert importlib.util.find_spec('ai4rag')"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if probe.returncode == 0:
            return str(repository_python)
    raise RuntimeError(
        "ai4rag is unavailable in the benchmark and repository Python environments; "
        "install ai4rag~=0.15.0 or set BENCHMARK_EVALUATOR_PYTHON"
    )


def run_full_corpus_evaluation(
    *,
    indexing_run_id: str,
    hpo_run_id: str,
    config_path: Path,
    env_file: Path | None,
    config: dict[str, Any],
    bucket: str,
    test_data_key: str,
    indexed_data_key: str,
    pattern_name: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the standalone evaluator after indexing and return its S3-collected metrics.

    This intentionally invokes the evaluator as a separate process: it needs the
    optional RAGAS dependencies, while a normal benchmark run does not.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_indexed_rag.py"
    try:
        evaluator_python = _evaluator_python()
    except RuntimeError as exc:
        logger.error("Full-corpus evaluator unavailable: %s", exc)
        return {"e2e_error": str(exc)}
    logger.info("Using evaluator Python: %s", evaluator_python)
    command = [
        evaluator_python, str(script), "--indexing-run-id", indexing_run_id,
        "--hpo-run-id", hpo_run_id, "--config", str(config_path),
        "--test-data-key", test_data_key,
        "--indexed-data-key", indexed_data_key,
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
