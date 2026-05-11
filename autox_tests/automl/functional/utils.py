"""Shared utilities for AutoML functional tests."""

import logging
import os
import secrets
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TASK_PRIMARY_METRICS = {
    "binary": "roc_auc",
    "multiclass": "balanced_accuracy",
    "regression": "r2",
}


def make_kfp_client(config):
    """Create a KFP client from a config dict; returns None if config is None."""
    if config is None:
        return None
    import kfp

    host = config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return kfp.Client(
        host=host,
        namespace=config["rhoai_project"],
        existing_token=config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )


def make_s3_client(config):
    """Create a boto3 S3 client from a config dict; returns None if not configured."""
    if config is None or not config.get("s3_endpoint"):
        return None
    try:
        import boto3
    except ImportError:
        return None
    return boto3.client(
        "s3",
        endpoint_url=config["s3_endpoint"],
        aws_access_key_id=config["s3_access_key"],
        aws_secret_access_key=config["s3_secret_key"],
        region_name=config["s3_region"],
    )


def make_run_name(prefix: str) -> str:
    """Return a unique run name: ``<prefix>-<6 hex chars>-<YYYYMMDD-HHMMSS>``."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{hex_part}-{time_part}"


def run_pipeline_and_wait(client, compiled_path, arguments, timeout):
    """Submit a pipeline run and block until completion; return ``(run_id, detail)``."""
    run_name = make_run_name("automl-func")
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
        enable_caching=False,
    )
    run_id = run.run_id
    detail = client.wait_for_run_completion(run_id, timeout=timeout)
    return run_id, detail


def normalize_state(state):
    """Normalize a state value (str or enum) to an uppercase string."""
    if state is None:
        return None
    return str(getattr(state, "name", state)).upper()


def get_run_state(detail):
    """Extract the run state string from a KFP run detail object."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    return normalize_state(state)


def run_succeeded(detail):
    """Return True if the run finished with SUCCEEDED state."""
    return get_run_state(detail) == "SUCCEEDED"


def run_failed(detail):
    """Return True if the run finished with FAILED state."""
    return get_run_state(detail) == "FAILED"


def validate_artifacts_in_s3(s3_client, bucket, prefix):
    """List and categorize S3 artifacts under prefix.

    Returns dict with keys: model_keys, leaderboard_keys, notebook_keys, all_keys.
    """
    result = {
        "model_keys": [],
        "leaderboard_keys": [],
        "notebook_keys": [],
        "all_keys": [],
    }
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                result["all_keys"].append(key)
                lower_key = key.lower()
                if key.endswith(".pkl") or "model" in lower_key:
                    result["model_keys"].append(key)
                if "leaderboard" in lower_key or key.endswith(".html"):
                    result["leaderboard_keys"].append(key)
                if key.endswith(".ipynb"):
                    result["notebook_keys"].append(key)
    except Exception as e:
        raise AssertionError(f"Failed to list S3 artifacts under s3://{bucket}/{prefix}: {e}") from e
    return result


def collect_failure_details(client, run_id, config=None):
    """Collect failure details from a failed pipeline run."""
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]

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
            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                state = getattr(task, "state", None)
                state_str = normalize_state(state) or "NOT_STARTED"

                if name in ("root", "executor") or name.endswith("-driver"):
                    continue

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")
                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")
                else:
                    lines.append(f"  TASK: {name} -- {state_str}")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details: {e}]")

    lines.append("=" * 80)
    return "\n".join(lines)
