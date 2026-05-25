"""Submit a compiled pipeline package and wait until the run reaches a terminal state."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_common.run_state import is_terminal_state, read_run_state, unwrap_run_from_get_run
from benchmark_common.yaml_io import load_yaml_dict

logger = logging.getLogger(__name__)


def get_pipeline_supported_params(pipeline_file: str | Path) -> set[str] | None:
    """Return the set of root input parameter names declared in a compiled KFP pipeline YAML.

    Returns *None* if the structure cannot be parsed (e.g. the file is not a
    compiled KFP YAML), so callers can fall back to passing all arguments.
    """
    import yaml

    try:
        with open(pipeline_file, encoding="utf-8") as f:
            for doc in yaml.safe_load_all(f):
                if not isinstance(doc, dict):
                    continue
                params = doc.get("root", {}).get("inputDefinitions", {}).get("parameters", {})
                if isinstance(params, dict) and params:
                    return set(params.keys())
    except Exception:
        pass
    return None


def filter_pipeline_arguments(
    arguments: dict[str, Any],
    pipeline_file: str | Path,
) -> dict[str, Any]:
    """Drop arguments that the pipeline does not declare as root inputs."""
    supported = get_pipeline_supported_params(pipeline_file)
    if supported is None:
        return arguments
    dropped = set(arguments) - supported
    if dropped:
        logger.info("Dropping unsupported pipeline parameters: %s", ", ".join(sorted(dropped)))
    return {k: v for k, v in arguments.items() if k in supported}


def submit_pipeline_package(
    client: Any,
    *,
    pipeline_file: str,
    arguments: dict[str, Any],
    run_name: str,
    experiment_name: str,
    enable_caching: bool,
) -> Any:
    try:
        return client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
            enable_caching=enable_caching,
        )
    except TypeError:
        return client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
        )


def extract_run_id(run_result: Any) -> str:
    rid = getattr(run_result, "run_id", None)
    if rid is None and isinstance(run_result, dict):
        rid = run_result.get("run_id")
    return str(rid) if rid is not None else ""


def wait_for_terminal_run(
    client: Any,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[Any | None, bool]:
    deadline = time.monotonic() + timeout_seconds
    detail = None
    last_token_refresh = time.monotonic()
    token_refresh_interval = 1800.0

    while time.monotonic() < deadline:
        if time.monotonic() - last_token_refresh > token_refresh_interval:
            try:
                result = subprocess.run(
                    ["oc", "whoami", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    fresh_token = result.stdout.strip()
                    if fresh_token and hasattr(client, "_config") and hasattr(client._config, "api_key"):
                        client._config.api_key["authorization"] = f"Bearer {fresh_token}"
                        last_token_refresh = time.monotonic()
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        detail = client.get_run(run_id)
        run_obj = unwrap_run_from_get_run(detail)
        st = read_run_state(run_obj).upper()
        if is_terminal_state(st):
            return detail, False
        time.sleep(poll_interval_seconds)
    return detail, True
