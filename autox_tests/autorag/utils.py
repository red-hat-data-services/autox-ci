"""Shared utilities for AutoRAG functional tests."""

import logging
import os
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from autox_tests.lib.kfp_run_state import _normalize_state
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
                if "leaderboard" in lower_key or key.endswith(".html"):
                    result["leaderboard_keys"].append(key)
                if "v1_responses_body.json" in key:
                    result["responses_body_keys"].append(key)
    except Exception as e:
        raise AssertionError(f"Failed to list S3 artifacts under s3://{bucket}/{prefix}: {e}") from e
    return result


_NOTEBOOK_ENV_PREFIXES = ("OGX_CLIENT_", "AWS_")
_SYSTEM_ENV_KEYS = frozenset({"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL"})


def _inject_and_run(notebook_path: Path, output_path: Path) -> None:
    """Inject mocked input() function into the notebook and execute it."""
    import nbformat
    import papermill as pm

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mock_code = 'def input(prompt=""):\n    return "Sample query?"'
    nb.cells.insert(0, nbformat.v4.new_code_cell(mock_code))

    injected_path = notebook_path.with_name(f"injected_{notebook_path.name}")
    with open(injected_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    original_cwd = os.getcwd()
    original_environ = os.environ.copy()
    try:
        safe_cwd = output_path.parent
        safe_cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(safe_cwd)

        filtered_env = {
            k: v
            for k, v in original_environ.items()
            if k in _SYSTEM_ENV_KEYS or any(k.startswith(p) for p in _NOTEBOOK_ENV_PREFIXES)
        }
        os.environ.clear()
        os.environ.update(filtered_env)

        pm.execute_notebook(str(injected_path), str(output_path), kernel_name="python3")
    finally:
        os.environ.clear()
        os.environ.update(original_environ)
        os.chdir(original_cwd)
        injected_path.unlink(missing_ok=True)


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


def _download_s3_json(s3_client, bucket: str, key: str):
    """Download and parse a JSON object from S3."""
    import json

    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def _load_pattern_json_objects(s3_client, bucket: str, pattern_keys: list[str]) -> list[dict]:
    patterns: list[dict] = []
    errors: list[str] = []
    for key in pattern_keys:
        try:
            data = _download_s3_json(s3_client, bucket, key)
            if isinstance(data, dict):
                patterns.append(data)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise AssertionError("Failed to load pattern.json files:\n" + "\n".join(errors))
    return patterns


def _maybe_run_llm_judge_validation(
    patterns: list[dict],
    *,
    scenario_id: str,
    enabled: bool,
) -> None:
    """Optionally score a sample of answers with LLM-as-a-Judge when enabled."""
    if not enabled:
        return

    base_url = (os.environ.get("OGX_CLIENT_BASE_URL") or "").strip()
    api_key = (os.environ.get("OGX_CLIENT_API_KEY") or "").strip()
    if not base_url or not api_key:
        logger.warning(
            "[%s] llm_judge tag set but OGX_CLIENT_BASE_URL/API_KEY missing — skipping",
            scenario_id,
        )
        return

    from .response_validation import collect_answers_from_patterns

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[%s] openai package not installed — skipping llm_judge", scenario_id)
        return

    answers = collect_answers_from_patterns(patterns)
    if not answers:
        logger.warning("[%s] no answers available for llm_judge sampling", scenario_id)
        return

    model_id = (os.environ.get("AUTORAG_LLM_JUDGE_MODEL") or "").strip()
    if not model_id:
        raise AssertionError(
            f"[{scenario_id}] llm_judge enabled but AUTORAG_LLM_JUDGE_MODEL is not set — "
            "configure a foundation model id available on your cluster"
        )

    sample_size = min(3, len(answers))
    sample = answers[:sample_size]
    client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key=api_key)

    scores: list[float] = []
    for row in sample:
        question = row.get("question") or ""
        predicted = row.get("answer") or ""
        if not predicted or str(predicted).startswith("Error:"):
            scores.append(0.0)
            continue
        prompt = (
            "Rate the predicted answer from 1-5 versus the question. "
            "Reply with ONLY a digit 1-5.\n\n"
            f"Question: {question}\nPredicted: {predicted}"
        )
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8,
            )
            text = (response.choices[0].message.content or "").strip()
            match = re.search(r"[1-5]", text)
            scores.append(int(match.group()) / 5.0 if match else 0.5)
        except Exception as exc:
            logger.warning("[%s] llm_judge call failed: %s", scenario_id, exc)
            scores.append(0.5)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    logger.info("[%s] llm_judge sample mean=%.2f (n=%d)", scenario_id, mean_score, len(scores))
    assert mean_score > 0.0, f"[{scenario_id}] llm_judge sample mean must be > 0"


def _validate_response_quality_artifacts(
    s3_client,
    bucket: str,
    artifacts: dict,
    *,
    scenario_id: str,
    optimization_metric: str,
    vector_io_provider_id: str | None = None,
    min_patterns: int = 1,
    min_evaluation_questions: int | None = None,
    require_leaderboard: bool = False,
    require_responses_export: bool = False,
    run_llm_judge: bool = False,
) -> None:
    """Validate evaluation scores, prompts, and optional Responses API export in S3 artifacts."""
    from .response_validation import (
        collect_answers_from_patterns,
        compute_answer_quality_stats,
        select_best_pattern,
        validate_evaluation_results_payload,
        validate_generation_prompt_template,
        validate_pattern_scores,
        validate_responses_export,
    )

    if require_leaderboard:
        assert len(artifacts["leaderboard_keys"]) >= 1, (
            f"[{scenario_id}] Expected leaderboard artifact under run prefix; "
            f"found {artifacts['leaderboard_keys']}"
        )

    if artifacts["evaluation_results_keys"]:
        eval_key = artifacts["evaluation_results_keys"][0]
        eval_data = _download_s3_json(s3_client, bucket, eval_key)
        validate_evaluation_results_payload(
            eval_data,
            min_patterns=min_patterns,
            min_questions=min_evaluation_questions,
            scenario_id=scenario_id,
        )

    assert artifacts["pattern_keys"], (
        f"[{scenario_id}] response_quality validation requires pattern.json artifacts"
    )
    patterns = _load_pattern_json_objects(s3_client, bucket, artifacts["pattern_keys"])
    assert len(patterns) >= min_patterns, (
        f"[{scenario_id}] expected >={min_patterns} pattern.json files, found {len(patterns)}"
    )

    best_pattern = select_best_pattern(patterns)
    validate_pattern_scores(
        best_pattern,
        optimization_metric=optimization_metric,
        scenario_id=scenario_id,
    )
    validate_generation_prompt_template(best_pattern, scenario_id=scenario_id)

    if require_responses_export:
        assert vector_io_provider_id, (
            f"[{scenario_id}] responses_api validation requires vector_io_provider_id"
        )
        validate_responses_export(
            best_pattern,
            provider_id=vector_io_provider_id,
            scenario_id=scenario_id,
        )

    answers = collect_answers_from_patterns(patterns)
    if min_evaluation_questions and answers:
        assert len(answers) >= min_evaluation_questions, (
            f"[{scenario_id}] expected >={min_evaluation_questions} answers in pattern.json, "
            f"found {len(answers)}"
        )

    if answers:
        stats = compute_answer_quality_stats(answers)
        logger.info(
            "[%s] answer quality: citation_rate=%.1f%% multilingual_rate=%.1f%% (n=%d)",
            scenario_id,
            stats["citation_rate"] * 100,
            stats["multilingual_rate"] * 100,
            len(answers),
        )

    _maybe_run_llm_judge_validation(
        patterns,
        scenario_id=scenario_id,
        enabled=run_llm_judge,
    )


def _download_and_execute_notebooks(s3_client, bucket, notebook_keys):
    """Download notebooks from S3 and execute them via papermill.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        notebook_keys: List of S3 keys pointing to .ipynb files.

    Raises:
        AssertionError: If any notebook fails execution.
    """
    import papermill as pm

    errors = []
    with tempfile.TemporaryDirectory(prefix="autorag-pipeline-notebook-") as tmpdir:
        for key in notebook_keys:
            filename = Path(key).name
            input_path = Path(tmpdir) / f"input_{filename}"
            output_path = Path(tmpdir) / f"output_{filename}"

            s3_client.download_file(bucket, key, str(input_path))

            try:
                _inject_and_run(input_path, output_path)
            except pm.PapermillExecutionError as e:
                errors.append(f"Notebook {filename} (key={key}) failed: {e}")
            except Exception as e:
                errors.append(f"Notebook {filename} (key={key}) execution error: {e}")

    if errors:
        raise AssertionError("Notebook execution failures:\n" + "\n".join(errors))
