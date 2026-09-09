"""Shared utilities for AutoRAG functional tests."""

import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from autox_tests.lib.kfp_run_state import _get_failed_task_names, _normalize_state  # noqa: F401
from autox_tests.lib.notebooks import run_notebooks_as_k8s_job
from autox_tests.lib.s3_data import upload_file_to_s3

logger = logging.getLogger(__name__)




def _make_docrag_run_name():
    """Return a run name: docrag-func-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"autorag-func-test-{hex_part}-{time_part}"


def _run_pipeline_and_wait(client, pipeline_target, arguments, timeout):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    from autox_tests.lib.managed_pipelines import submit_pipeline_run_and_wait

    run_name = _make_docrag_run_name()
    return submit_pipeline_run_and_wait(
        client,
        pipeline_target,
        arguments,
        run_name=run_name,
        timeout=timeout,
    )



def _collect_failure_details(client, run_id, config=None):
    """Collect failure details from a failed pipeline run via the Kubernetes API.

    Uses the Kubernetes client to find pods for failed pipeline tasks (via Tekton
    labels such as ``tekton.dev/pipelineTask``) and fetches their logs. Task-level
    metadata is still pulled from the KFP v2 API for context.

    Args:
        client: KFP client instance (used for run-level / task-level metadata).
        run_id: The pipeline run ID.
        config: Functional config dict with ``rhoai_token``, ``rhoai_kfp_url``,
            and ``rhoai_project`` keys used for Kubernetes authentication.

    Returns:
        Formatted string with failure details and pod logs.
    """
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]
    failed_task_names: list[str] = []

    # --- Run-level and task-level details from KFP v2 API ---
    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)

        run_error = getattr(run_obj, "error", None)
        if run_error:
            error_msg = getattr(run_error, "message", str(run_error))
            lines.append(f"\nRUN ERROR: {error_msg}")

        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None

        if task_list:
            _INTERNAL_SUFFIXES = ("-driver",)
            _INTERNAL_NAMES = ("root", "executor")

            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                state = getattr(task, "state", None)
                state_str = _normalize_state(state) or "NOT_STARTED"

                if name in _INTERNAL_NAMES or any(name.endswith(s) for s in _INTERNAL_SUFFIXES):
                    continue

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")

                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")

                    failed_task_names.append(name)

                    start = getattr(task, "start_time", None)
                    end = getattr(task, "end_time", None)
                    if start and end:
                        lines.append(f"  Duration: {start} -> {end}")
                else:
                    lines.append(f"  TASK: {name} — {state_str}")
        else:
            lines.append("\n[No task_details in run response]")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details from KFP API: {e}]")

    # Fetch logs from failed pods only (Tekton-backed managed pipelines)
    if config:
        from autox_tests.lib.k8s_utils import append_failed_task_pod_logs_safe

        append_failed_task_pod_logs_safe(
            lines,
            run_id,
            config,
            failed_task_names,
            logger=logger,
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def _validate_artifacts_in_s3(s3_client, bucket, prefix):
    """List and categorize S3 artifacts under prefix.

    Returns:
        Dict with keys: "pattern_keys", "indexing_notebook_keys", "inference_notebook_keys",
        "evaluation_results_keys", "leaderboard_keys", "responses_body_keys", "all_keys".

    Raises:
        AssertionError: If S3 listing fails.
    """
    result = {
        "pattern_keys": [],
        "indexing_notebook_keys": [],
        "inference_notebook_keys": [],
        "evaluation_results_keys": [],
        "leaderboard_keys": [],
        "responses_body_keys": [],
        "all_keys": [],
    }
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                result["all_keys"].append(key)
                lower_key = key.lower()
                if key.endswith("pattern.json") or "rag_patterns" in lower_key:
                    result["pattern_keys"].append(key)
                if key.endswith(".ipynb") and "indexing" in lower_key:
                    result["indexing_notebook_keys"].append(key)
                if key.endswith(".ipynb") and "inference" in lower_key:
                    result["inference_notebook_keys"].append(key)
                if "evaluation_results.json" in key:
                    result["evaluation_results_keys"].append(key)
                if "leaderboard" in lower_key or key.endswith(".html") or key.endswith("/leaderboard_html"):
                    result["leaderboard_keys"].append(key)
                if "v1_responses_body.json" in key:
                    result["responses_body_keys"].append(key)
    except Exception as e:
        raise AssertionError(f"Failed to list S3 artifacts under s3://{bucket}/{prefix}: {e}") from e
    return result


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def upload_test_datasets(
    s3_client,
    bucket: str,
    s3_keys: list[str],
    local_data_dir: Path,
) -> list[str]:
    """Upload local files or directories to S3 for each key in s3_keys.

    Keys with a file extension are matched by filename (basename). Keys without an
    extension are treated as S3 directory prefixes: the matching local directory is found
    by its name, and all files within it are uploaded under that prefix.

    When multiple local files or directories share the same name, the one whose parent
    directory name shares the longest common prefix with the S3 key's parent component
    is chosen. Keys with no local match are skipped (covers intentional negative-test keys).

    Returns the list of S3 keys that were actually uploaded.
    """
    if not local_data_dir.is_dir():
        raise FileNotFoundError(
            f"AUTORAG_UPLOAD_TEST_DATASETS is set but local data directory does not exist: {local_data_dir}"
        )

    local_files: dict[str, list[Path]] = {}
    local_dirs: dict[str, list[Path]] = {}
    for entry in local_data_dir.rglob("*"):
        if entry.is_file():
            local_files.setdefault(entry.name, []).append(entry)
        elif entry.is_dir():
            local_dirs.setdefault(entry.name, []).append(entry)

    uploaded_keys: list[str] = []
    failed_uploads: list[str] = []

    for s3_key in sorted(set(s3_keys)):
        key_path = Path(s3_key)
        parent_hint = key_path.parent.name
        leaf = key_path.name

        if key_path.suffix:
            candidates = sorted(local_files.get(leaf, []))
            if not candidates:
                logger.debug("No local file for key %r — skipping upload", s3_key)
                continue
            scores = [_common_prefix_len(p.parent.name, parent_hint) for p in candidates]
            best_score = max(scores)
            if len(candidates) > 1:
                if best_score == 0:
                    logger.warning(
                        "Cannot discriminate among %d candidates for key %r — picking %s",
                        len(candidates), s3_key, candidates[0],
                    )
                else:
                    logger.warning(
                        "Multiple local files named %r; picking best match for parent %r",
                        leaf, parent_hint,
                    )
            local_path = candidates[scores.index(best_score)]
            try:
                logger.info("Uploading %s → s3://%s/%s", local_path, bucket, s3_key)
                upload_file_to_s3(s3_client, bucket=bucket, key=s3_key, local_path=local_path)
                uploaded_keys.append(s3_key)
            except Exception as exc:
                logger.error(
                    "Failed to upload %s → s3://%s/%s: %s", local_path, bucket, s3_key, exc
                )
                failed_uploads.append(s3_key)
        else:
            candidates = sorted(local_dirs.get(leaf, []))
            if not candidates:
                logger.debug("No local directory for key %r — skipping upload", s3_key)
                continue
            scores = [_common_prefix_len(d.parent.name, parent_hint) for d in candidates]
            best_score = max(scores)
            if len(candidates) > 1:
                if best_score == 0:
                    logger.warning(
                        "Cannot discriminate among %d candidates for key %r — picking %s",
                        len(candidates), s3_key, candidates[0],
                    )
                else:
                    logger.warning(
                        "Multiple local directories named %r; picking best match for parent %r",
                        leaf, parent_hint,
                    )
            local_dir = candidates[scores.index(best_score)]
            for f in sorted(local_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(local_dir)
                file_s3_key = f"{s3_key}/{rel}"
                try:
                    logger.info("Uploading %s → s3://%s/%s", f, bucket, file_s3_key)
                    upload_file_to_s3(s3_client, bucket=bucket, key=file_s3_key, local_path=f)
                    uploaded_keys.append(file_s3_key)
                except Exception as exc:
                    logger.error(
                        "Failed to upload %s → s3://%s/%s: %s", f, bucket, file_s3_key, exc
                    )
                    failed_uploads.append(file_s3_key)

    logger.info(
        "Dataset upload complete: %d file(s) uploaded to s3://%s",
        len(uploaded_keys),
        bucket,
    )
    if failed_uploads:
        raise RuntimeError(
            f"Failed to upload {len(failed_uploads)} dataset file(s) to s3://{bucket}: {failed_uploads}"
        )
    return uploaded_keys


def _pick_best_pattern_notebooks(s3_client, bucket, artifacts):
    """Return (indexing_key, inference_key) for the best pattern per the leaderboard.

    Downloads the leaderboard HTML artifact and parses the best pattern name from
    the ``<div class="best-model-footer">Best pattern: <strong>NAME</strong></div>``
    footer written by ai4rag's ``build_leaderboard_html()``. Notebooks are always
    named ``indexing.ipynb`` / ``inference.ipynb`` under ``rag_patterns/{pattern_name}/``,
    so the best-pattern name is matched against the S3 key suffix.

    Falls back to the first available indexing/inference key pair when the leaderboard
    is absent or unparseable.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        artifacts: Dict returned by ``_validate_artifacts_in_s3``.

    Returns:
        Tuple of (indexing_notebook_key, inference_notebook_key).

    Raises:
        ValueError: If no indexing or inference notebooks are present.
    """
    import re

    indexing_keys = artifacts["indexing_notebook_keys"]
    inference_keys = artifacts["inference_notebook_keys"]
    leaderboard_keys = artifacts["leaderboard_keys"]

    if not indexing_keys:
        raise ValueError("No indexing notebooks found in artifacts")
    if not inference_keys:
        raise ValueError("No inference notebooks found in artifacts")

    best_pattern_name = None
    if leaderboard_keys:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=leaderboard_keys[0])
            html = response["Body"].read().decode("utf-8")
            match = re.search(r'Best pattern: <strong>([^<]+)</strong>', html)
            if match:
                best_pattern_name = match.group(1)
                logger.info("Best pattern from leaderboard: %s", best_pattern_name)
        except Exception as e:
            logger.warning("Could not parse leaderboard HTML: %s", e)

    if best_pattern_name:
        # Notebooks live at rag_patterns/{pattern_name}/indexing.ipynb (inference.ipynb)
        pattern_prefix = f"{best_pattern_name}/"
        indexing_match = next((k for k in indexing_keys if f"/{pattern_prefix}" in k), None)
        inference_match = next((k for k in inference_keys if f"/{pattern_prefix}" in k), None)
        if indexing_match and inference_match:
            return indexing_match, inference_match
        logger.warning(
            "Best pattern %r not matched in notebook keys — falling back to first available",
            best_pattern_name,
        )

    return indexing_keys[0], inference_keys[0]


def _download_and_execute_notebooks(s3_client, bucket, notebook_keys, *, config):
    """Execute generated notebooks sequentially in one Kubernetes Job pod.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        notebook_keys: List of S3 keys pointing to .ipynb files.

    Raises:
        AssertionError: If any notebook fails execution.
    """
    del s3_client  # The Job downloads notebooks with its injected S3 secret.
    run_notebooks_as_k8s_job(
        bucket=bucket,
        notebook_keys=notebook_keys,
        config=config,
        secret_names=[
            str(config.get("s3_secret_name") or config.get("input_data_secret_name") or ""),
            str(config.get("maas_secret_name") or ""),
            str(config.get("vector_db_secret_name") or ""),
        ],
        inject_mock_input=True,
    )
