"""Clone pipelines-components, compile KFP IR in an isolated venv (kfp 2.16+ / Python 3.11+)."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "autox_benchmarks" / "pipeline-compile"

_CLONE_SUBDIR = "repos"
_VENV_SUBDIR = "venvs"
_ARTIFACT_SUBDIR = "artifacts"


def default_compile_cache_root() -> Path:
    return _DEFAULT_CACHE_ROOT


def _sha16(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _find_python_for_compile_venv() -> str:
    """Return a Python >=3.11 executable path (upstream kfp-components requires 3.11+)."""
    if sys.version_info >= (3, 11):
        return sys.executable
    for cand in ("python3.12", "python3.11", "python3"):
        exe = shutil.which(cand)
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe, "-c", "import sys; assert sys.version_info >= (3, 11); print('ok')"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out.stdout.strip() == "ok":
                return exe
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            continue
    raise RuntimeError(
        "Python 3.11+ is required to compile pipeline YAML from Git (upstream kfp-components). "
        "Install python3.11 or run the orchestrator with Python >=3.11."
    )


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    logger.debug("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _ensure_shallow_repo(*, clone_root: Path, git_url: str, git_ref: str) -> Path:
    clone_root.parent.mkdir(parents=True, exist_ok=True)
    if (clone_root / ".git").is_dir():
        try:
            _run(["git", "-C", str(clone_root), "fetch", "--depth", "1", "origin", git_ref])
            _run(["git", "-C", str(clone_root), "checkout", "-B", "_benchmark_compile", "FETCH_HEAD"])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"git fetch/checkout failed for {git_url!r} ref={git_ref!r}: {e.stderr or e.stdout}"
            ) from e
        return clone_root

    try:
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                git_ref,
                git_url,
                str(clone_root),
            ]
        )
    except subprocess.CalledProcessError:
        if clone_root.exists():
            shutil.rmtree(clone_root, ignore_errors=True)
        try:
            _run(["git", "clone", "--depth", "1", git_url, str(clone_root)])
            _run(["git", "-C", str(clone_root), "fetch", "--depth", "1", "origin", git_ref])
            _run(["git", "-C", str(clone_root), "checkout", "FETCH_HEAD"])
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(
                f"git clone failed for {git_url!r} (ref {git_ref!r}): {e2.stderr or e2.stdout}"
            ) from e2
    return clone_root


def _ensure_venv_with_editable(*, venv_dir: Path, py_exe: str, repo_root: Path) -> Path:
    pip = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("pip.exe" if os.name == "nt" else "pip")
    python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    marker = venv_dir / ".autox_benchmark_venv_ok"
    if marker.is_file() and python.is_file():
        return python

    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    _run([py_exe, "-m", "venv", str(venv_dir)])
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([str(python), "-m", "pip", "install", "-e", str(repo_root)])
    marker.write_text("ok", encoding="utf-8")
    return python


def compile_kfp_pipeline_yaml(
    *,
    git_url: str,
    git_ref: str,
    entrypoint_rel: str,
    cache_root: Path | None = None,
) -> Path:
    """
    Return path to a compiled ``pipeline.yaml`` for the given repo entrypoint.

    Caches clone, venv (per resolved commit), and the emitted YAML artifact.
    """
    root = (cache_root or default_compile_cache_root()).resolve()
    url = git_url.strip()
    ref = git_ref.strip()
    entry = entrypoint_rel.strip().strip("/")
    if not url or not ref or not entry:
        raise ValueError("git_url, git_ref, and entrypoint_rel must be non-empty.")

    if not shutil.which("git"):
        raise RuntimeError("git executable not found on PATH (required to compile pipeline YAML from Git).")

    clone_key = _sha16(url, ref)
    clone_dir = root / _CLONE_SUBDIR / clone_key
    _ensure_shallow_repo(clone_root=clone_dir, git_url=url, git_ref=ref)

    entry_path = (clone_dir / entry).resolve()
    if not entry_path.is_file():
        raise FileNotFoundError(f"Pipeline entrypoint not found in clone: {entry_path}")

    commit = _git_head(clone_dir)
    artifact_key = _sha16(commit, entry)
    artifact_yaml = root / _ARTIFACT_SUBDIR / artifact_key / "pipeline.yaml"
    if artifact_yaml.is_file() and artifact_yaml.stat().st_size > 0:
        logger.info("Using cached compiled pipeline %s (commit=%s)", artifact_yaml, commit[:8])
        return artifact_yaml

    py_for_venv = _find_python_for_compile_venv()
    venv_key = _sha16(commit, url)
    venv_dir = root / _VENV_SUBDIR / venv_key
    python = _ensure_venv_with_editable(venv_dir=venv_dir, py_exe=py_for_venv, repo_root=clone_dir)

    built_yaml = entry_path.with_suffix(".yaml")
    if built_yaml.is_file():
        built_yaml.unlink()

    try:
        _run([str(python), str(entry_path)], cwd=clone_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Compiling {entry} failed (commit {commit[:8]}): {e.stderr or e.stdout or e}"
        ) from e

    if not built_yaml.is_file() or built_yaml.stat().st_size == 0:
        raise RuntimeError(f"Compile step did not produce {built_yaml}")

    artifact_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_yaml, artifact_yaml)
    logger.info("Compiled pipeline IR -> %s (commit=%s)", artifact_yaml, commit[:8])
    return artifact_yaml
