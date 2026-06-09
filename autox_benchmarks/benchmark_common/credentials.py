"""Load benchmark cluster/storage credentials from .env (preferred) or legacy INI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from benchmark_common.ini_credentials import load_credentials_ini

logger = logging.getLogger(__name__)

_CREDENTIALS_HELP = (
    "Copy .env.example to .env and set KFP / storage / pipeline / S3 variables, "
    "or pass --env-file PATH. Legacy INI: --credentials PATH or config/credentials.ini."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _truthy_env(*names: str, default: bool | None = None) -> bool | None:
    raw = _get_env(*names)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def resolve_env_file_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_file() else None
    env_p = _get_env("BENCHMARK_ENV_FILE")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        return p if p.is_file() else None
    for candidate in (_repo_root() / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_benchmark_dotenv(explicit: Path | None = None) -> Path | None:
    """Load .env into os.environ (existing shell/CI vars keep precedence)."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError("python-dotenv is required; pip install -e .") from exc

    path = resolve_env_file_path(explicit)
    if path is None:
        return None
    load_dotenv(path, override=False)
    logger.debug("Loaded benchmark env from %s", path)
    return path


def credentials_dict_from_env() -> dict[str, Any]:
    """Build nested credentials dict from environment variables."""
    out: dict[str, Any] = {}

    kfp_host = _get_env("BENCHMARK_KFP_HOST", "RHOAI_KFP_URL", "KFP_HOST")
    kfp_namespace = _get_env("BENCHMARK_KFP_NAMESPACE", "RHOAI_PROJECT_NAME", "KFP_NAMESPACE")
    kfp_token = _get_env("BENCHMARK_KFP_TOKEN", "RHOAI_TOKEN", "KFP_API_TOKEN", "KFP_TOKEN")
    kfp_token_file = _get_env("BENCHMARK_KFP_TOKEN_FILE", "KFP_TOKEN_FILE")
    kfp_token_env = _get_env("BENCHMARK_KFP_TOKEN_ENV", "KFP_TOKEN_ENV")
    kfp_experiment = _get_env("BENCHMARK_KFP_EXPERIMENT_NAME", "KFP_EXPERIMENT_NAME")
    if any((kfp_host, kfp_namespace, kfp_token, kfp_token_file, kfp_experiment)):
        kfp: dict[str, Any] = {}
        if kfp_host:
            kfp["host"] = kfp_host
        if kfp_namespace:
            kfp["namespace"] = kfp_namespace
        if kfp_token:
            kfp["token"] = kfp_token
        if kfp_token_file:
            kfp["token_file"] = kfp_token_file
        if kfp_token_env:
            kfp["token_env"] = kfp_token_env
        if kfp_experiment:
            kfp["experiment_name"] = kfp_experiment
        out["kfp"] = kfp

    storage: dict[str, Any] = {}
    train_bucket = _get_env(
        "BENCHMARK_TRAIN_DATA_BUCKET_NAME",
        "AUTOML_TRAIN_DATA_BUCKET_NAME",
        "RHOAI_TEST_DATA_BUCKET",
    )
    if train_bucket:
        storage["train_data_bucket_name"] = train_bucket
    input_bucket = _get_env("BENCHMARK_INPUT_DATA_BUCKET_NAME", "INPUT_DATA_BUCKET_NAME")
    if input_bucket:
        storage["input_data_bucket_name"] = input_bucket
    test_bucket = _get_env("BENCHMARK_TEST_DATA_BUCKET_NAME", "TEST_DATA_BUCKET_NAME")
    if test_bucket:
        storage["test_data_bucket_name"] = test_bucket
    artifact_prefix = _get_env("BENCHMARK_ARTIFACT_S3_PREFIX", "ARTIFACT_S3_PREFIX")
    if artifact_prefix:
        storage["artifact_s3_prefix"] = artifact_prefix
    ts_prefix = _get_env("BENCHMARK_TIMESERIES_ARTIFACT_S3_PREFIX", "TIMESERIES_ARTIFACT_S3_PREFIX")
    if ts_prefix:
        storage["timeseries_artifact_s3_prefix"] = ts_prefix
    bench_prefix = _get_env("BENCHMARK_S3_PREFIX", "BENCHMARK_BENCHMARK_S3_PREFIX")
    if bench_prefix:
        storage["benchmark_s3_prefix"] = bench_prefix
    upload = _truthy_env("BENCHMARK_UPLOAD_RESULTS", "UPLOAD_BENCHMARK_RESULTS")
    if upload is not None:
        storage["upload_benchmark_results"] = upload
    if storage:
        out["storage"] = storage

    pipeline: dict[str, Any] = {}
    train_secret = _get_env(
        "BENCHMARK_TRAIN_DATA_SECRET_NAME",
        "RHOAI_TEST_S3_SECRET_NAME",
        "TRAIN_DATA_SECRET_NAME",
    )
    if train_secret:
        pipeline["train_data_secret_name"] = train_secret
    input_secret = _get_env("BENCHMARK_INPUT_DATA_SECRET_NAME", "INPUT_DATA_SECRET_NAME")
    if input_secret:
        pipeline["input_data_secret_name"] = input_secret
    test_secret = _get_env("BENCHMARK_TEST_DATA_SECRET_NAME", "TEST_DATA_SECRET_NAME")
    if test_secret:
        pipeline["test_data_secret_name"] = test_secret
    ogx_secret = _get_env("BENCHMARK_OGX_SECRET_NAME", "OGX_SECRET_NAME", "LLAMA_STACK_SECRET_NAME")
    if ogx_secret:
        pipeline["ogx_secret_name"] = ogx_secret
    vector_io = _get_env(
        "BENCHMARK_VECTOR_IO_PROVIDER_ID",
        "VECTOR_IO_PROVIDER_ID",
        "LLAMA_STACK_VECTOR_IO_PROVIDER_ID",
    )
    if vector_io:
        pipeline["vector_io_provider_id"] = vector_io
    package_path = _get_env(
        "BENCHMARK_TABULAR_PACKAGE_PATH",
        "TABULAR_PACKAGE_PATH",
        "BENCHMARK_PACKAGE_PATH",
        "RAG_PACKAGE_PATH",
        "AUTORAG_PIPELINE_PATH",
    )
    if package_path:
        pipeline["package_path"] = package_path
    ts_package = _get_env("BENCHMARK_TIMESERIES_PACKAGE_PATH", "TIMESERIES_PACKAGE_PATH")
    if ts_package:
        pipeline["timeseries_package_path"] = ts_package
    if pipeline:
        out["pipeline"] = pipeline

    s3: dict[str, Any] = {}
    endpoint = _get_env("AWS_S3_ENDPOINT", "S3_ENDPOINT", "ARTIFACTS_AWS_S3_ENDPOINT")
    access_key = _get_env("AWS_ACCESS_KEY_ID", "ARTIFACTS_AWS_ACCESS_KEY_ID")
    secret_key = _get_env("AWS_SECRET_ACCESS_KEY", "ARTIFACTS_AWS_SECRET_ACCESS_KEY")
    region = _get_env("AWS_DEFAULT_REGION", "ARTIFACTS_AWS_DEFAULT_REGION", "AWS_S3_REGION")
    if endpoint:
        s3["endpoint"] = endpoint
    if access_key:
        s3["aws_access_key_id"] = access_key
    if secret_key:
        s3["aws_secret_access_key"] = secret_key
    if region:
        s3["aws_default_region"] = region
    if s3:
        out["s3"] = s3

    run: dict[str, Any] = {}
    top_n = _get_env("BENCHMARK_TOP_N")
    if top_n:
        run["top_n"] = int(top_n)
    opt_metric = _get_env("BENCHMARK_OPTIMIZATION_METRIC")
    if opt_metric:
        run["optimization_metric"] = opt_metric
    max_patterns = _get_env("BENCHMARK_OPTIMIZATION_MAX_RAG_PATTERNS")
    if max_patterns:
        run["optimization_max_rag_patterns"] = int(max_patterns)
    poll = _get_env("BENCHMARK_POLL_INTERVAL_SECONDS", "RHOAI_KFP_POLL_INTERVAL_SECONDS")
    if poll:
        run["poll_interval_seconds"] = float(poll)
    timeout = _get_env("BENCHMARK_TIMEOUT_SECONDS", "RHOAI_PIPELINE_RUN_TIMEOUT")
    if timeout:
        run["timeout_seconds"] = float(timeout)
    caching = _truthy_env("BENCHMARK_ENABLE_CACHING")
    if caching is not None:
        run["enable_caching"] = caching
    prefix = _get_env("BENCHMARK_RUN_NAME_PREFIX")
    if prefix:
        run["run_name_prefix"] = prefix
    if run:
        out["run"] = run

    return out


def _env_has_kfp_storage_pipeline(overlay: dict[str, Any]) -> bool:
    kfp = overlay.get("kfp") or {}
    storage = overlay.get("storage") or {}
    pipeline = overlay.get("pipeline") or {}
    return bool(
        str(kfp.get("host", "")).strip()
        and str(kfp.get("namespace", "")).strip()
        and (
            str(storage.get("train_data_bucket_name", "")).strip()
            or str(storage.get("input_data_bucket_name", "")).strip()
        )
        and (
            str(pipeline.get("train_data_secret_name", "")).strip()
            or str(pipeline.get("input_data_secret_name", "")).strip()
        )
    )


def resolve_legacy_ini_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Credentials file not found: {p}")
        return p
    env_p = _get_env("BENCHMARK_CREDENTIALS_PATH")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        return p if p.is_file() else None
    default = (_repo_root() / "config" / "credentials.ini").resolve()
    return default if default.is_file() else None


def load_credentials_overlay(
    *,
    env_file: Path | None = None,
    credentials_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Load credentials overlay for merging into benchmark.yaml.

    Priority:
    1. Explicit ``credentials_path`` (.ini legacy)
    2. Explicit ``env_file`` (.env)
    3. Auto-discovered ``.env`` (project root or cwd)
    4. Legacy ``config/credentials.ini`` or ``$BENCHMARK_CREDENTIALS_PATH``
    """
    if credentials_path is not None:
        ini_path = credentials_path.expanduser().resolve()
        if not ini_path.is_file():
            raise FileNotFoundError(f"Credentials file not found: {ini_path}")
        if ini_path.suffix.lower() == ".ini" or ini_path.name.startswith("credentials"):
            logger.warning(
                "Loading legacy credentials.ini from %s — prefer .env (see .env.example)",
                ini_path,
            )
            return load_credentials_ini(ini_path), str(ini_path)
        load_benchmark_dotenv(ini_path)
        overlay = credentials_dict_from_env()
        if not _env_has_kfp_storage_pipeline(overlay):
            raise ValueError(f".env at {ini_path} is missing required benchmark variables. {_CREDENTIALS_HELP}")
        return overlay, str(ini_path)

    load_benchmark_dotenv(env_file)
    overlay = credentials_dict_from_env()
    if _env_has_kfp_storage_pipeline(overlay):
        source = str(resolve_env_file_path(env_file) or ".env (environment)")
        return overlay, source

    ini_path = resolve_legacy_ini_path(None)
    if ini_path is not None:
        logger.warning(
            "Using legacy credentials.ini at %s — copy .env.example to .env and migrate",
            ini_path,
        )
        return load_credentials_ini(ini_path), str(ini_path)

    raise FileNotFoundError(f"No credentials found. {_CREDENTIALS_HELP}")
