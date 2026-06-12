"""MLflow tracking settings from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from benchmark_common.credentials import _get_env, _truthy_env


@dataclass(frozen=True)
class MlflowSettings:
    enabled: bool
    tracking_uri: str
    token: str
    workspace: str
    experiment_name: str
    benchmark_kind: str
    task_type_normalize: bool
    filter_parse_ok: bool


def mlflow_settings_from_env() -> MlflowSettings | None:
    """Return settings when MLflow upload is enabled; otherwise None."""
    enabled = _truthy_env("BENCHMARK_UPLOAD_MLFLOW", "MLFLOW_UPLOAD_BENCHMARKS")
    if enabled is not True:
        return None

    tracking_uri = _get_env("MLFLOW_TRACKING_URI", "BENCHMARK_MLFLOW_TRACKING_URI")
    token = _get_env("MLFLOW_TRACKING_TOKEN", "BENCHMARK_MLFLOW_TOKEN", "RHOAI_TOKEN")
    workspace = _get_env(
        "MLFLOW_TRACKING_WORKSPACE",
        "MLFLOW_WORKSPACE",
        "BENCHMARK_KFP_NAMESPACE",
        "RHOAI_PROJECT_NAME",
    )
    experiment = _get_env(
        "BENCHMARK_MLFLOW_EXPERIMENT",
        "MLFLOW_EXPERIMENT_NAME",
        "BENCHMARK_KFP_EXPERIMENT_NAME",
    )
    kind = _get_env("BENCHMARK_MLFLOW_KIND", "BENCHMARK_KIND").lower() or "automl"
    normalize = _truthy_env("BENCHMARK_MLFLOW_NORMALIZE_TASK_TYPE")
    if normalize is None:
        normalize = True
    filter_ok = _truthy_env("BENCHMARK_MLFLOW_FILTER_PARSE_OK")
    if filter_ok is None:
        filter_ok = True

    missing = [
        name
        for name, val in (
            ("MLFLOW_TRACKING_URI", tracking_uri),
            ("MLFLOW_TRACKING_TOKEN", token),
            ("MLFLOW_WORKSPACE", workspace),
            ("MLFLOW_EXPERIMENT_NAME", experiment),
        )
        if not val
    ]
    if missing:
        raise ValueError(
            "MLflow upload enabled but missing: "
            + ", ".join(missing)
            + ". Set them in .env or disable BENCHMARK_UPLOAD_MLFLOW."
        )

    return MlflowSettings(
        enabled=True,
        tracking_uri=tracking_uri,
        token=token,
        workspace=workspace,
        experiment_name=experiment,
        benchmark_kind=kind if kind in ("automl", "autorag") else "automl",
        task_type_normalize=bool(normalize),
        filter_parse_ok=bool(filter_ok),
    )


def apply_mlflow_env(settings: MlflowSettings) -> None:
    """Configure process env for OpenShift AI / MLflow tracking API."""
    os.environ["MLFLOW_TRACKING_URI"] = settings.tracking_uri
    os.environ["MLFLOW_TRACKING_TOKEN"] = settings.token
    os.environ["MLFLOW_TRACKING_WORKSPACE"] = settings.workspace
