"""Session setup for online AutoML integration tests (credentials.ini or env only)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmark_common.kfp_client import create_kfp_client, resolve_kfp_token
from benchmark_common.s3_client import make_s3_client, s3_cfg_usable
from benchmark_common.s3_upload import _s3_object_not_found
from tests.conftest import REPO_ROOT

INTEGRATION_CONFIG_DIR = REPO_ROOT / "tests" / "fixtures" / "automl" / "integration"
INTEGRATION_BENCHMARK_YAML = INTEGRATION_CONFIG_DIR / "benchmark.yaml"
LOCAL_SMOKE_TRAIN_CSV = INTEGRATION_CONFIG_DIR / "breast-w_n200.csv"
SMOKE_TRAIN_DATA_KEY = "benchmark/smoke/breast-w_n200.csv"
TABULAR_PIPELINE = REPO_ROOT / "pipelines" / "autogluon-tabular-training-pipeline.yaml"


@dataclass(frozen=True)
class IntegrationContext:
    credentials_path: Path
    merged_config: dict[str, Any]
    config_dir: Path
    kfp_client: Any


def _resolve_credentials_path() -> Path | None:
    env_p = os.environ.get("BENCHMARK_CREDENTIALS_PATH", "").strip()
    if env_p:
        p = Path(env_p).expanduser().resolve()
        return p if p.is_file() else None
    default = (REPO_ROOT / "config" / "credentials.ini").resolve()
    return default if default.is_file() else None


def _collect_prerequisite_errors() -> list[str]:
    errors: list[str] = []

    creds = _resolve_credentials_path()
    if creds is None:
        env_hint = (
            f"Set BENCHMARK_CREDENTIALS_PATH or create {REPO_ROOT / 'config' / 'credentials.ini'}"
        )
        return [f"Missing credentials.ini ({env_hint})"]

    if not INTEGRATION_BENCHMARK_YAML.is_file():
        errors.append(f"Missing integration benchmark config: {INTEGRATION_BENCHMARK_YAML}")
    if not LOCAL_SMOKE_TRAIN_CSV.is_file():
        errors.append(f"Missing bundled smoke CSV fixture: {LOCAL_SMOKE_TRAIN_CSV}")
    if not TABULAR_PIPELINE.is_file():
        errors.append(f"Missing tabular pipeline YAML: {TABULAR_PIPELINE}")

    try:
        from automl_benchmark.config_loader import load_merged_benchmark_config

        cfg, _config_dir = load_merged_benchmark_config(INTEGRATION_BENCHMARK_YAML, creds)
    except Exception as exc:
        errors.append(f"Failed to load benchmark config with credentials: {exc}")
        return errors

    timeout_env = os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "").strip()
    if timeout_env:
        cfg.setdefault("run", {})["timeout_seconds"] = float(timeout_env)

    kfp = cfg.get("kfp") or {}
    if not str(kfp.get("host", "")).startswith("http"):
        errors.append("[kfp] host must be an http(s) URL in credentials.ini")
    if not str(kfp.get("namespace", "")).strip():
        errors.append("[kfp] namespace is required in credentials.ini")
    if not str(kfp.get("experiment_name", "")).strip():
        errors.append("[kfp] experiment_name is required in credentials.ini")
    if not resolve_kfp_token(kfp):
        token_env = str(kfp.get("token_env", "KFP_API_TOKEN"))
        errors.append(
            f"KFP token required ([kfp] token/token_file in credentials.ini or {token_env} env)"
        )

    storage = cfg.get("storage") or {}
    if not str(storage.get("train_data_bucket_name", "")).strip():
        errors.append("[storage] train_data_bucket_name is required in credentials.ini")

    s3_cfg = cfg.get("s3")
    if not s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None):
        errors.append("[s3] aws_access_key_id and aws_secret_access_key are required in credentials.ini")

    pipeline_cfg = cfg.get("pipeline") or {}
    if not str(pipeline_cfg.get("package_path", "")).strip() and not TABULAR_PIPELINE.is_file():
        errors.append("[pipeline] package_path or bundled tabular pipeline YAML is required")

    return errors


def _ensure_smoke_training_csv(cfg: dict[str, Any]) -> None:
    storage = cfg.get("storage") or {}
    bucket = str(storage.get("train_data_bucket_name", "")).strip()
    s3_cfg = cfg.get("s3")
    assert s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None)

    client = make_s3_client(s3_cfg)
    try:
        client.head_object(Bucket=bucket, Key=SMOKE_TRAIN_DATA_KEY)
        return
    except Exception as exc:
        if not _s3_object_not_found(exc):
            raise

    client.upload_file(
        str(LOCAL_SMOKE_TRAIN_CSV),
        bucket,
        SMOKE_TRAIN_DATA_KEY,
        ExtraArgs={"ContentType": "text/csv"},
    )


def _build_integration_context() -> IntegrationContext:
    errors = _collect_prerequisite_errors()
    if errors:
        pytest.fail(
            "Integration prerequisites not met:\n  - " + "\n  - ".join(errors),
            pytrace=False,
        )

    creds = _resolve_credentials_path()
    assert creds is not None

    from automl_benchmark.config_loader import load_merged_benchmark_config

    cfg, config_dir = load_merged_benchmark_config(INTEGRATION_BENCHMARK_YAML, creds)
    timeout_env = os.environ.get("BENCHMARK_INTEGRATION_TIMEOUT_SECONDS", "").strip()
    if timeout_env:
        cfg.setdefault("run", {})["timeout_seconds"] = float(timeout_env)

    try:
        _ensure_smoke_training_csv(cfg)
    except Exception as exc:
        pytest.fail(f"Could not ensure smoke training CSV on S3: {exc}", pytrace=False)

    try:
        kfp_client = create_kfp_client(cfg)
        kfp_client.list_experiments(page_size=1)
    except Exception as exc:
        pytest.fail(f"KFP connectivity check failed: {exc}", pytrace=False)

    return IntegrationContext(
        credentials_path=creds,
        merged_config=cfg,
        config_dir=config_dir,
        kfp_client=kfp_client,
    )


@pytest.fixture(scope="session")
def integration_context() -> IntegrationContext:
    return _build_integration_context()


@pytest.fixture(scope="session", autouse=True)
def _integration_session_gate(integration_context: IntegrationContext) -> None:
    """Validate credentials/KFP/S3 once; abort the rest of the suite on failure."""


@pytest.fixture(scope="session")
def integration_credentials_path(integration_context: IntegrationContext) -> Path:
    return integration_context.credentials_path


@pytest.fixture(scope="session")
def integration_benchmark_yaml() -> Path:
    assert INTEGRATION_BENCHMARK_YAML.is_file()
    return INTEGRATION_BENCHMARK_YAML


@pytest.fixture(scope="session")
def integration_merged_config(
    integration_context: IntegrationContext,
) -> tuple[dict[str, Any], Path]:
    return integration_context.merged_config, integration_context.config_dir


@pytest.fixture(scope="session")
def integration_kfp_client(integration_context: IntegrationContext):
    return integration_context.kfp_client


@pytest.fixture
def integration_output_csv(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("integration_out") / "smoke_benchmark_runs.csv"
