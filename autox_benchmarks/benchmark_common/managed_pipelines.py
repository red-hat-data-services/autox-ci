"""Managed KFP pipeline support: resolve and submit runs against DSPA-registered pipelines."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

BENCHMARK_USE_MANAGED_PIPELINES_ENV = "BENCHMARK_USE_MANAGED_PIPELINES"
BENCHMARK_MANAGED_PIPELINE_WAIT_TIMEOUT_ENV = "BENCHMARK_MANAGED_PIPELINE_WAIT_TIMEOUT"

BENCHMARK_MANAGED_PIPELINE_TABULAR_ENV = "BENCHMARK_MANAGED_PIPELINE_TABULAR"
BENCHMARK_MANAGED_PIPELINE_TIMESERIES_ENV = "BENCHMARK_MANAGED_PIPELINE_TIMESERIES"
BENCHMARK_MANAGED_PIPELINE_AUTORAG_ENV = "BENCHMARK_MANAGED_PIPELINE_AUTORAG"

PIPELINE_DEFAULT_NAMES: dict[str, str] = {
    "tabular": "autogluon-tabular-training-pipeline",
    "timeseries": "autogluon-timeseries-training-pipeline",
    "autorag": "documents-rag-optimization-pipeline",
}

_KIND_ENV_VARS: dict[str, str] = {
    "tabular": BENCHMARK_MANAGED_PIPELINE_TABULAR_ENV,
    "timeseries": BENCHMARK_MANAGED_PIPELINE_TIMESERIES_ENV,
    "autorag": BENCHMARK_MANAGED_PIPELINE_AUTORAG_ENV,
}

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class PipelineRunTarget:
    """How to start a pipeline run: package upload or managed KFP pipeline."""

    mode: Literal["package", "managed"]
    artifact_prefix: str
    package_path: str | None = None
    pipeline_id: str | None = None
    pipeline_version_id: str | None = None
    kfp_pipeline_name: str | None = None


def resolve_benchmark_pipeline_mode(cfg: dict[str, Any]) -> Literal["package", "managed"]:
    """Determine pipeline mode from env var (highest priority) or config."""
    raw = (os.environ.get(BENCHMARK_USE_MANAGED_PIPELINES_ENV) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return "managed"
    if raw in _FALSE_VALUES:
        return "package"
    pipeline_cfg = cfg.get("pipeline") or {}
    mode = str(pipeline_cfg.get("mode", "package")).strip().lower()
    if mode == "managed":
        return "managed"
    return "package"


def get_managed_kfp_pipeline_name(
    kind: Literal["tabular", "timeseries", "autorag"],
    cfg: dict[str, Any],
) -> str:
    """Return the KFP display name for a managed pipeline."""
    env_var = _KIND_ENV_VARS[kind]
    env_val = (os.environ.get(env_var) or "").strip()
    if env_val:
        return env_val
    pipeline_cfg = cfg.get("pipeline") or {}
    names = pipeline_cfg.get("kfp_pipeline_names")
    if isinstance(names, dict) and kind in names:
        val = str(names[kind]).strip()
        if val:
            return val
    return PIPELINE_DEFAULT_NAMES[kind]


def _clamp_timeout(parsed: int) -> int:
    clamped = max(30, min(parsed, 600))
    if parsed != clamped:
        logger.warning(
            "Managed pipeline wait timeout %ds out of range [30, 600]; clamping to %ds",
            parsed, clamped,
        )
    return clamped


def get_managed_pipeline_wait_timeout(cfg: dict[str, Any]) -> int:
    """Return the wait timeout for managed pipeline discovery."""
    env_val = (os.environ.get(BENCHMARK_MANAGED_PIPELINE_WAIT_TIMEOUT_ENV) or "").strip()
    if env_val:
        try:
            return _clamp_timeout(int(env_val))
        except ValueError:
            pass
    pipeline_cfg = cfg.get("pipeline") or {}
    raw = pipeline_cfg.get("managed_pipeline_wait_timeout")
    if raw is not None:
        try:
            return _clamp_timeout(int(raw))
        except (ValueError, TypeError):
            pass
    return 300


def _resolve_latest_pipeline_version_id(client: Any, pipeline_id: str) -> str | None:
    """Return the newest pipeline version id, or None on error."""
    try:
        versions = client.list_pipeline_versions(
            pipeline_id=pipeline_id,
            page_size=10,
            sort_by="created_at desc",
        )
        version_list = getattr(versions, "pipeline_versions", None) or []
        if version_list:
            vid = getattr(version_list[0], "pipeline_version_id", None)
            if vid:
                return vid
    except Exception as exc:
        logger.debug("list_pipeline_versions failed for %s: %s", pipeline_id, exc)
    return None


def _list_pipeline_display_names(client: Any, *, page_size: int = 50) -> list[str]:
    """Return display names from the first page of list_pipelines (for timeout hints)."""
    try:
        resp = client.list_pipelines(page_size=page_size)
    except Exception as exc:
        logger.debug("list_pipelines failed (hint unavailable): %s", exc)
        return []
    return [
        (getattr(p, "display_name", None) or "").strip()
        for p in (getattr(resp, "pipelines", None) or [])
    ]


def _find_pipeline_by_display_name(
    client: Any,
    display_name: str,
) -> tuple[tuple[str, str | None] | None, list[str]]:
    """Return (result_or_None, list_of_known_names).

    Uses kfp.Client.get_pipeline_id (KFP v2 JSON filter) when available,
    falls back to paginated list_pipelines with client-side name matching.
    """
    get_pipeline_id = getattr(client, "get_pipeline_id", None)
    if callable(get_pipeline_id):
        try:
            pipeline_id = get_pipeline_id(display_name)
        except ValueError:
            logger.warning("Multiple KFP pipelines named %r", display_name)
            return None, _list_pipeline_display_names(client)
        if not pipeline_id:
            return None, _list_pipeline_display_names(client)
        return (
            pipeline_id,
            _resolve_latest_pipeline_version_id(client, pipeline_id),
        ), []

    known_names: list[str] = []
    page_token = ""
    while True:
        resp = client.list_pipelines(page_token=page_token, page_size=50)
        for pipeline in getattr(resp, "pipelines", None) or []:
            name = (getattr(pipeline, "display_name", None) or "").strip()
            if name:
                known_names.append(name)
            if name == display_name:
                pipeline_id = getattr(pipeline, "pipeline_id", None)
                if not pipeline_id:
                    continue
                version_id = getattr(pipeline, "default_pipeline_version_id", None)
                if version_id:
                    return (pipeline_id, version_id), known_names
                return (
                    pipeline_id,
                    _resolve_latest_pipeline_version_id(client, pipeline_id),
                ), known_names
        page_token = getattr(resp, "next_page_token", None) or ""
        if not page_token:
            break
    return None, known_names


def wait_for_managed_pipeline(
    client: Any,
    kfp_pipeline_name: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: float = 10.0,
) -> tuple[str, str | None]:
    """Poll KFP until a managed pipeline with the given display name is registered.

    Returns (pipeline_id, version_id_or_None).

    Raises EnvironmentError if the pipeline list stays empty for >90s (DSPA misconfiguration).
    Raises TimeoutError if not found within timeout_seconds.
    """
    _EARLY_EXIT_EMPTY_THRESHOLD_SECONDS = 90

    start = time.monotonic()
    deadline = start + timeout_seconds
    last_names: list[str] = []
    consecutive_empty = 0

    while time.monotonic() < deadline:
        found, last_names = _find_pipeline_by_display_name(client, kfp_pipeline_name)
        if found:
            pipeline_id, version_id = found
            logger.info(
                "Managed pipeline %r available (pipeline_id=%s, version_id=%s)",
                kfp_pipeline_name,
                pipeline_id,
                version_id or "(default)",
            )
            return found

        if not last_names:
            consecutive_empty += 1
            elapsed = time.monotonic() - start
            if elapsed > _EARLY_EXIT_EMPTY_THRESHOLD_SECONDS:
                raise EnvironmentError(
                    f"No managed pipelines registered in KFP after {elapsed:.0f}s "
                    f"({consecutive_empty} consecutive empty checks). "
                    "Verify DSPA spec.apiServer.managedPipelines configuration and check "
                    "DSPA pod logs for reconciliation errors."
                )
        else:
            consecutive_empty = 0

        time.sleep(poll_interval_seconds)

    hint = f" Known pipelines: {last_names!r}" if last_names else ""
    raise TimeoutError(
        f"Managed pipeline {kfp_pipeline_name!r} not found in KFP within {timeout_seconds}s.{hint}"
    )
