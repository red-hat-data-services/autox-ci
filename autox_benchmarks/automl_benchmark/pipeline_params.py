"""Map dataset manifest entries to pipeline argument dicts."""

from __future__ import annotations

from typing import Any

from automl_benchmark.settings import BenchmarkSettings, normalize_presets
from benchmark_common.managed_pipelines import PipelineRunTarget

# Keys that select the preset sweep; applied by the orchestrator, not as raw overrides.
_PRESET_OVERRIDE_KEYS = frozenset({"preset", "presets"})


def is_timeseries_dataset(dataset: dict[str, Any]) -> bool:
    t = dataset.get("task_type")
    if t is None:
        return False
    return str(t).strip().lower() == "timeseries"


def target_for_dataset(
    dataset: dict[str, Any],
    targets: dict[str, PipelineRunTarget],
) -> PipelineRunTarget:
    kind = "timeseries" if is_timeseries_dataset(dataset) else "tabular"
    try:
        return targets[kind]
    except KeyError:
        raise ValueError(
            f"No pipeline target configured for {kind!r} datasets. "
            f"Available targets: {sorted(targets)}"
        ) from None


def _manifest_pipeline_overrides(dataset: dict[str, Any]) -> dict[str, Any]:
    extra = dataset.get("pipeline_arguments") or dataset.get("pipeline_params")
    if not isinstance(extra, dict) or not extra:
        return {}
    return dict(extra)


def presets_for_dataset(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
) -> tuple[str, ...]:
    """Presets to run for one dataset: manifest override, else ``settings.presets``."""
    extra = _manifest_pipeline_overrides(dataset)
    if "presets" in extra and extra.get("presets") is not None:
        return normalize_presets(extra["presets"])
    if "preset" in extra and extra.get("preset") is not None:
        return normalize_presets(extra["preset"])
    return settings.presets


def _merge_manifest_pipeline_overrides(
    arguments: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Apply optional per-dataset overrides from the manifest (excluding preset keys)."""
    extra = _manifest_pipeline_overrides(dataset)
    if not extra:
        return arguments
    overrides = {k: v for k, v in extra.items() if k not in _PRESET_OVERRIDE_KEYS}
    if not overrides:
        return arguments
    return {**arguments, **overrides}


def build_pipeline_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    *,
    preset: str,
) -> dict[str, Any]:
    if is_timeseries_dataset(dataset):
        base = _build_timeseries_arguments(dataset, settings, preset=preset)
    else:
        base = _build_tabular_arguments(dataset, settings, preset=preset)
    return _merge_manifest_pipeline_overrides(base, dataset)


def _build_tabular_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    *,
    preset: str,
) -> dict[str, Any]:
    return {
        "train_data_secret_name": settings.train_data_secret_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "train_data_file_key": str(dataset["train_data_file_key"]),
        "label_column": str(dataset["label_column"]),
        "task_type": str(dataset["task_type"]),
        "top_n": settings.top_n,
        "preset": preset,
    }


def _build_timeseries_arguments(
    dataset: dict[str, Any],
    settings: BenchmarkSettings,
    *,
    preset: str,
) -> dict[str, Any]:
    target = dataset.get("target") or dataset.get("label_column")
    if not target:
        raise ValueError("timeseries datasets require 'target' or 'label_column'")
    args: dict[str, Any] = {
        "train_data_secret_name": settings.train_data_secret_name,
        "train_data_bucket_name": settings.train_data_bucket_name,
        "train_data_file_key": str(dataset["train_data_file_key"]),
        "target": str(target),
        "id_column": str(dataset["id_column"]),
        "timestamp_column": str(dataset["timestamp_column"]),
        "top_n": settings.top_n,
        "preset": preset,
    }
    kc = dataset.get("known_covariates_names")
    if isinstance(kc, list) and kc:
        args["known_covariates_names"] = [str(x) for x in kc]
    pl = dataset.get("prediction_length")
    if pl is not None and str(pl).strip() != "":
        args["prediction_length"] = int(pl)
    return args
