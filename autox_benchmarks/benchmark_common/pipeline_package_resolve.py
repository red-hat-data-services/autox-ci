"""Resolve pipeline IR paths: static package_path or compile-from-Git (pipelines-components)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from benchmark_common.paths import resolve_under
from benchmark_common.pipeline_compile import compile_kfp_pipeline_yaml, default_compile_cache_root

logger = logging.getLogger(__name__)

DEFAULT_COMPILE_GIT_URL = "https://github.com/opendatahub-io/pipelines-components.git"
DEFAULT_COMPILE_GIT_REF = "main"
DEFAULT_TABULAR_ENTRYPOINT = "pipelines/training/automl/autogluon_tabular_training_pipeline/pipeline.py"
DEFAULT_TIMESERIES_ENTRYPOINT = "pipelines/training/automl/autogluon_timeseries_training_pipeline/pipeline.py"
DEFAULT_AUTORAG_ENTRYPOINT = "pipelines/training/autorag/documents_rag_optimization_pipeline/pipeline.py"


def _compile_section(pipeline_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = pipeline_cfg.get("compile")
    return raw if isinstance(raw, dict) else {}


def _compile_git_url(c: dict[str, Any]) -> str:
    return str(c.get("git_url") or DEFAULT_COMPILE_GIT_URL).strip()


def _compile_git_ref(c: dict[str, Any]) -> str:
    return str(c.get("git_ref") or DEFAULT_COMPILE_GIT_REF).strip()


def _first_env_path(*names: str) -> str | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return None


def resolve_automl_pipeline_package_paths(
    cfg: dict[str, Any],
    config_dir: Path,
    *,
    cli_tabular: str | None,
    cli_timeseries: str | None,
    needs_tabular: bool,
    needs_timeseries: bool,
    compile_cache_root: Path | None = None,
) -> None:
    """
    Mutate ``cfg['pipeline']`` with absolute ``package_path`` / ``timeseries_package_path`` strings.

    When a slot is not needed for this run and no path is configured, the other slot's
    resolved path is reused so :class:`BenchmarkSettings` always points at existing files.
    """
    pl = cfg.setdefault("pipeline", {})
    c = _compile_section(pl)
    cache = compile_cache_root if compile_cache_root is not None else default_compile_cache_root()

    if cli_tabular is None:
        cli_tabular = _first_env_path("BENCHMARK_TABULAR_PACKAGE_PATH", "TABULAR_PACKAGE_PATH")
    if cli_timeseries is None:
        cli_timeseries = _first_env_path(
            "BENCHMARK_TIMESERIES_PACKAGE_PATH",
            "TIMESERIES_PACKAGE_PATH",
        )

    def _resolve_slot(
        *,
        cli_val: str | None,
        yaml_key: str,
        entry_key: str,
        default_entry: str,
        needed: bool,
    ) -> None:
        if not needed:
            return
        if cli_val:
            p = Path(cli_val).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"{yaml_key}: path is not a file: {p}")
            pl[yaml_key] = str(p)
            logger.info("Using %s from CLI: %s", yaml_key, p)
            return
        raw = pl.get(yaml_key)
        if isinstance(raw, str) and raw.strip():
            p = resolve_under(config_dir, raw.strip())
            if p.is_file():
                pl[yaml_key] = str(p.resolve())
                logger.info("Using %s from config: %s", yaml_key, p)
                return
            raise FileNotFoundError(
                f"pipeline.{yaml_key} is set but file not found: {p}. "
                "Fix the path, remove the key to compile from Git, or pass a CLI override."
            )
        entry = str(c.get(entry_key) or default_entry).strip() or default_entry
        out = compile_kfp_pipeline_yaml(
            git_url=_compile_git_url(c),
            git_ref=_compile_git_ref(c),
            entrypoint_rel=entry,
            cache_root=cache,
        )
        pl[yaml_key] = str(out)
        logger.info("Compiled %s from Git -> %s", yaml_key, out)

    _resolve_slot(
        cli_val=cli_tabular,
        yaml_key="package_path",
        entry_key="tabular_entrypoint",
        default_entry=DEFAULT_TABULAR_ENTRYPOINT,
        needed=needs_tabular,
    )
    _resolve_slot(
        cli_val=cli_timeseries,
        yaml_key="timeseries_package_path",
        entry_key="timeseries_entrypoint",
        default_entry=DEFAULT_TIMESERIES_ENTRYPOINT,
        needed=needs_timeseries,
    )

    if needs_tabular and not needs_timeseries and not pl.get("timeseries_package_path"):
        pl["timeseries_package_path"] = pl["package_path"]
    if needs_timeseries and not needs_tabular and not pl.get("package_path"):
        pl["package_path"] = pl["timeseries_package_path"]


def resolve_autorag_pipeline_package_path(
    cfg: dict[str, Any],
    config_dir: Path,
    *,
    cli_package: str | None,
    compile_cache_root: Path | None = None,
) -> None:
    """Mutate ``cfg['pipeline']['package_path']`` to an absolute path (static or compiled)."""
    pl = cfg.setdefault("pipeline", {})
    c = _compile_section(pl)
    cache = compile_cache_root if compile_cache_root is not None else default_compile_cache_root()

    if cli_package is None:
        cli_package = _first_env_path("BENCHMARK_PACKAGE_PATH", "RAG_PACKAGE_PATH")

    if cli_package:
        p = Path(cli_package).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"package_path: CLI path is not a file: {p}")
        pl["package_path"] = str(p)
        logger.info("Using package_path from CLI: %s", p)
        return
    raw = pl.get("package_path")
    if isinstance(raw, str) and raw.strip():
        p = resolve_under(config_dir, raw.strip())
        if p.is_file():
            pl["package_path"] = str(p.resolve())
            logger.info("Using package_path from config: %s", p)
            return
        raise FileNotFoundError(
            f"pipeline.package_path is set but file not found: {p}. "
            "Fix the path, remove the key to compile from Git, or pass --package-path."
        )
    entry = str(c.get("autorag_entrypoint") or DEFAULT_AUTORAG_ENTRYPOINT).strip() or DEFAULT_AUTORAG_ENTRYPOINT
    out = compile_kfp_pipeline_yaml(
        git_url=_compile_git_url(c),
        git_ref=_compile_git_ref(c),
        entrypoint_rel=entry,
        cache_root=cache,
    )
    pl["package_path"] = str(out)
    logger.info("Compiled RAG pipeline from Git -> %s", out)
