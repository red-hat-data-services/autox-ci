"""Resolve pipeline run targets: package-mode paths or managed KFP pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark_common.managed_pipelines import (
    PipelineRunTarget,
    get_managed_kfp_pipeline_name,
    get_managed_pipeline_wait_timeout,
    resolve_benchmark_pipeline_mode,
    wait_for_managed_pipeline,
)
from benchmark_common.pipeline_package_resolve import (
    resolve_automl_pipeline_package_paths,
    resolve_autorag_pipeline_package_path,
)

logger = logging.getLogger(__name__)


def resolve_automl_pipeline_targets(
    cfg: dict[str, Any],
    config_dir: Path,
    client: Any,
    *,
    cli_tabular: str | None = None,
    cli_timeseries: str | None = None,
    needs_tabular: bool = True,
    needs_timeseries: bool = False,
    compile_cache_root: Path | None = None,
) -> dict[str, PipelineRunTarget]:
    """Return ``{"tabular": target, "timeseries": target}`` for the configured mode."""
    mode = resolve_benchmark_pipeline_mode(cfg)

    if mode == "managed":
        targets: dict[str, PipelineRunTarget] = {}
        # dry_run passes client=None — skip KFP discovery and return name-only stubs
        if client is None:
            for kind, needed in [("tabular", needs_tabular), ("timeseries", needs_timeseries)]:
                if not needed:
                    continue
                kfp_name = get_managed_kfp_pipeline_name(kind, cfg)
                targets[kind] = PipelineRunTarget(
                    mode="managed",
                    artifact_prefix=kfp_name,
                    kfp_pipeline_name=kfp_name,
                )
        else:
            timeout = get_managed_pipeline_wait_timeout(cfg)
            for kind, needed in [("tabular", needs_tabular), ("timeseries", needs_timeseries)]:
                if not needed:
                    continue
                kfp_name = get_managed_kfp_pipeline_name(kind, cfg)
                pipeline_id, version_id = wait_for_managed_pipeline(
                    client, kfp_name, timeout_seconds=timeout,
                )
                targets[kind] = PipelineRunTarget(
                    mode="managed",
                    artifact_prefix=kfp_name,
                    pipeline_id=pipeline_id,
                    pipeline_version_id=version_id,
                    kfp_pipeline_name=kfp_name,
                )

        if needs_tabular and not needs_timeseries and "timeseries" not in targets:
            targets["timeseries"] = targets["tabular"]
        if needs_timeseries and not needs_tabular and "tabular" not in targets:
            targets["tabular"] = targets["timeseries"]

        return targets

    resolve_automl_pipeline_package_paths(
        cfg,
        config_dir,
        cli_tabular=cli_tabular,
        cli_timeseries=cli_timeseries,
        needs_tabular=needs_tabular,
        needs_timeseries=needs_timeseries,
        compile_cache_root=compile_cache_root,
    )

    pipeline_cfg = cfg.get("pipeline") or {}
    tab_path = pipeline_cfg.get("package_path")
    ts_path = pipeline_cfg.get("timeseries_package_path")

    from benchmark_common.managed_pipelines import _PIPELINE_DEFAULT_NAMES

    targets = {}
    if tab_path:
        targets["tabular"] = PipelineRunTarget(
            mode="package",
            artifact_prefix=_PIPELINE_DEFAULT_NAMES["tabular"],
            package_path=tab_path,
        )
    if ts_path:
        targets["timeseries"] = PipelineRunTarget(
            mode="package",
            artifact_prefix=_PIPELINE_DEFAULT_NAMES["timeseries"],
            package_path=ts_path,
        )

    if "tabular" in targets and "timeseries" not in targets:
        targets["timeseries"] = targets["tabular"]
    if "timeseries" in targets and "tabular" not in targets:
        targets["tabular"] = targets["timeseries"]

    return targets


def resolve_autorag_pipeline_target(
    cfg: dict[str, Any],
    config_dir: Path,
    client: Any,
    *,
    cli_package: str | None = None,
    compile_cache_root: Path | None = None,
) -> PipelineRunTarget:
    """Return a single PipelineRunTarget for the AutoRAG pipeline."""
    mode = resolve_benchmark_pipeline_mode(cfg)

    if mode == "managed":
        kfp_name = get_managed_kfp_pipeline_name("autorag", cfg)
        # dry_run passes client=None — skip KFP discovery and return a name-only stub
        if client is None:
            return PipelineRunTarget(
                mode="managed",
                artifact_prefix=kfp_name,
                kfp_pipeline_name=kfp_name,
            )
        timeout = get_managed_pipeline_wait_timeout(cfg)
        pipeline_id, version_id = wait_for_managed_pipeline(
            client, kfp_name, timeout_seconds=timeout,
        )
        return PipelineRunTarget(
            mode="managed",
            artifact_prefix=kfp_name,
            pipeline_id=pipeline_id,
            pipeline_version_id=version_id,
            kfp_pipeline_name=kfp_name,
        )

    resolve_autorag_pipeline_package_path(
        cfg, config_dir, cli_package=cli_package, compile_cache_root=compile_cache_root,
    )

    pipeline_cfg = cfg.get("pipeline") or {}
    package_path = pipeline_cfg.get("package_path")
    from benchmark_common.managed_pipelines import _PIPELINE_DEFAULT_NAMES

    return PipelineRunTarget(
        mode="package",
        artifact_prefix=_PIPELINE_DEFAULT_NAMES["autorag"],
        package_path=package_path,
    )
