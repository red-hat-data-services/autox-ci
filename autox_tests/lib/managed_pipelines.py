"""Resolve pipeline run targets: precompiled YAML package or KFP managed pipeline."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from autox_tests.lib.env import load_tests_env
from autox_tests.lib.settings import parse_timeout_seconds_from_env
from autox_tests.lib.pipeline_yaml_sources import (
    PIPELINE_YAML_AUTORAG_ENV,
    PIPELINE_YAML_TABULAR_ENV,
    PIPELINE_YAML_TIMESERIES_ENV,
    resolve_precompiled_pipeline_yaml,
)

logger = logging.getLogger(__name__)

RHOAI_USE_MANAGED_PIPELINES_ENV = "RHOAI_USE_MANAGED_PIPELINES"
RHOAI_MANAGED_PIPELINE_WAIT_TIMEOUT_ENV = "RHOAI_MANAGED_PIPELINE_WAIT_TIMEOUT"

KFP_NAME_TABULAR_ENV = "RHOAI_MANAGED_PIPELINE_TABULAR"
KFP_NAME_TIMESERIES_ENV = "RHOAI_MANAGED_PIPELINE_TIMESERIES"
KFP_NAME_AUTORAG_ENV = "RHOAI_MANAGED_PIPELINE_AUTORAG"

ARTIFACT_PREFIX_TABULAR_ENV = "RHOAI_TABULAR_PIPELINE_ARTIFACT_PREFIX"
ARTIFACT_PREFIX_TIMESERIES_ENV = "RHOAI_TIMESERIES_PIPELINE_ARTIFACT_PREFIX"
ARTIFACT_PREFIX_AUTORAG_ENV = "RHOAI_AUTORAG_PIPELINE_ARTIFACT_PREFIX"

_PIPELINE_DEFAULT_NAMES: dict[str, str] = {
    "tabular": "autogluon-tabular-training-pipeline",
    "timeseries": "autogluon-timeseries-training-pipeline",
    "autorag": "documents-rag-optimization-pipeline",
}

_LEGACY_PACKAGE_PATH_ENVS = (
    PIPELINE_YAML_TABULAR_ENV,
    PIPELINE_YAML_TIMESERIES_ENV,
    PIPELINE_YAML_AUTORAG_ENV,
)

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _legacy_pipeline_package_configured() -> bool:
    """Return True when any legacy ``*_PIPELINE_PATH`` env var is set."""
    return any(
        (os.environ.get(name) or "").strip() for name in _LEGACY_PACKAGE_PATH_ENVS
    )


def use_managed_pipelines_from_env() -> bool:
    """Return whether tests submit runs from KFP-managed pipelines.

    Default is **managed** (no ``pipeline.yaml`` required). Set
    ``RHOAI_USE_MANAGED_PIPELINES=false`` or provide ``AUTOML_*`` / ``AUTORAG_PIPELINE_PATH``
    for legacy package upload mode.
    """
    load_tests_env()
    raw = (os.environ.get(RHOAI_USE_MANAGED_PIPELINES_ENV) or "").strip().lower()
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    if _legacy_pipeline_package_configured():
        return False
    return True


def _env_or_default(env_var: str, default: str) -> str:
    raw = (os.environ.get(env_var) or "").strip()
    return raw or default


def get_managed_kfp_pipeline_name(
    kind: Literal["tabular", "timeseries", "autorag"],
) -> str:
    """Return the KFP display name for a managed pipeline (from env)."""
    env_by_kind = {
        "tabular": KFP_NAME_TABULAR_ENV,
        "timeseries": KFP_NAME_TIMESERIES_ENV,
        "autorag": KFP_NAME_AUTORAG_ENV,
    }
    return _env_or_default(env_by_kind[kind], _PIPELINE_DEFAULT_NAMES[kind])


def get_pipeline_artifact_prefix(
    kind: Literal["tabular", "timeseries", "autorag"],
) -> str:
    """Return the S3 artifact path prefix segment for a pipeline run.

    Uses ``RHOAI_*_PIPELINE_ARTIFACT_PREFIX`` when set; otherwise the managed
    pipeline display name (``RHOAI_MANAGED_PIPELINE_*`` or the built-in default).
    """
    env_by_kind = {
        "tabular": ARTIFACT_PREFIX_TABULAR_ENV,
        "timeseries": ARTIFACT_PREFIX_TIMESERIES_ENV,
        "autorag": ARTIFACT_PREFIX_AUTORAG_ENV,
    }
    override = (os.environ.get(env_by_kind[kind]) or "").strip()
    if override:
        return override
    return get_managed_kfp_pipeline_name(kind)


@dataclass(frozen=True)
class PipelineRunTarget:
    """How to start a pipeline run and which name to use for artifact paths."""

    mode: Literal["package", "managed"]
    artifact_prefix: str
    package_path: str | None = None
    pipeline_id: str | None = None
    pipeline_version_id: str | None = None
    kfp_pipeline_name: str | None = None


def _resolve_latest_pipeline_version_id(client: Any, pipeline_id: str) -> str | None:
    """Return the newest pipeline version id, or ``None`` if none are listed or on error."""
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
    """Return display names from the first page of ``list_pipelines`` (for timeout hints)."""
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
    """Return ``(result_or_None, list_of_known_names)``.

    Uses :meth:`kfp.Client.get_pipeline_id` (KFP v2 JSON filter), not legacy
    ``display_name="..."`` syntax which the API rejects with HTTP 400.

    Known names come from the same ``list_pipelines`` traversal on the fallback
    client path, or from a single first-page list when using ``get_pipeline_id``.
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

    # Fallback for older clients: list and match client-side (no server filter).
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
    early_exit_checks: int = 3,
) -> tuple[str, str | None]:
    """Poll KFP until a managed pipeline is registered.

    Args:
        client: KFP client instance
        kfp_pipeline_name: Display name of the pipeline to wait for
        timeout_seconds: Maximum time to wait
        poll_interval_seconds: Seconds between poll attempts
        early_exit_checks: Number of consecutive empty-list checks before raising EnvironmentError

    Raises:
        EnvironmentError: If pipeline list remains empty after early_exit_checks attempts
        TimeoutError: If pipeline not found within timeout_seconds
    """
    deadline = time.monotonic() + timeout_seconds
    last_names: list[str] = []
    checks = 0
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

        checks += 1

        # Early exit: if pipeline list is consistently empty, DSPA may not be creating pipelines
        if not last_names:
            consecutive_empty += 1
            if consecutive_empty >= early_exit_checks:
                elapsed = checks * poll_interval_seconds
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


def resolve_managed_pipeline_target(
    client: Any,
    *,
    kind: Literal["tabular", "timeseries", "autorag"],
    path_env_var: str,
    cache_dir: Any,
    cache_file_name: str,
) -> PipelineRunTarget:
    """Build a :class:`PipelineRunTarget` for managed KFP or legacy package upload."""
    artifact_prefix = get_pipeline_artifact_prefix(kind)

    if use_managed_pipelines_from_env():
        kfp_name = get_managed_kfp_pipeline_name(kind)
        wait_timeout = parse_timeout_seconds_from_env(
            RHOAI_MANAGED_PIPELINE_WAIT_TIMEOUT_ENV, 300, max_seconds=600
        )
        pipeline_id, version_id = wait_for_managed_pipeline(
            client,
            kfp_name,
            timeout_seconds=wait_timeout,
        )
        return PipelineRunTarget(
            mode="managed",
            artifact_prefix=artifact_prefix,
            pipeline_id=pipeline_id,
            pipeline_version_id=version_id,
            kfp_pipeline_name=kfp_name,
        )

    package_path = resolve_precompiled_pipeline_yaml(
        path_env_var=path_env_var,
        cache_dir=cache_dir,
        cache_file_name=cache_file_name,
    )
    return PipelineRunTarget(
        mode="package",
        artifact_prefix=artifact_prefix,
        package_path=package_path,
    )


_KF_DEFAULT_EXPERIMENT = "KF_PIPELINES_DEFAULT_EXPERIMENT_NAME"
_KF_OVERRIDE_EXPERIMENT = "KF_PIPELINES_OVERRIDE_EXPERIMENT_NAME"


def submit_pipeline_run_and_wait(
    client: Any,
    target: PipelineRunTarget,
    arguments: dict[str, Any],
    *,
    run_name: str,
    timeout: int,
    experiment_name: str | None = None,
    namespace: str | None = None,
) -> tuple[str, Any]:
    """Start a run and block until completion; return ``(run_id, run_detail)``."""
    if target.mode == "package":
        if not target.package_path:
            raise ValueError("package mode requires package_path on PipelineRunTarget")
        run = client.create_run_from_pipeline_package(
            target.package_path,
            arguments=arguments,
            run_name=run_name,
            enable_caching=False,
            experiment_name=experiment_name,
            namespace=namespace,
        )
        run_id = run.run_id
    elif target.mode == "managed":
        if not target.pipeline_id:
            raise ValueError("managed mode requires pipeline_id on PipelineRunTarget")
        exp_name = experiment_name
        if exp_name is None:
            exp_name = os.environ.get(_KF_DEFAULT_EXPERIMENT)
            overridden = os.environ.get(_KF_OVERRIDE_EXPERIMENT, exp_name)
            if overridden != exp_name:
                logger.warning(
                    "Changing experiment name from %r to %r.",
                    exp_name or "(default)",
                    overridden,
                )
            exp_name = overridden or "Default"
        try:
            experiment = client.create_experiment(name=exp_name, namespace=namespace)
        except Exception as e:
            # Only catch conflict (experiment already exists) - let other errors propagate
            # KFP SDK doesn't expose specific exception types, so we check the message
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                experiment = client.get_experiment(
                    experiment_name=exp_name, namespace=namespace
                )
            else:
                logger.error("Failed to create experiment %r: %s", exp_name, e)
                raise
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=run_name,
            pipeline_id=target.pipeline_id,
            version_id=target.pipeline_version_id,
            params=arguments,
            enable_caching=False,
        )
        run_id = run.run_id
    else:
        raise ValueError(f"Unknown pipeline run mode: {target.mode!r}")

    detail = client.wait_for_run_completion(run_id, timeout=timeout)
    return run_id, detail
