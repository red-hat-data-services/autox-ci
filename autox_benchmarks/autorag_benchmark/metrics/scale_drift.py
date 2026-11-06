"""Compare HPO-subsample quality with full-corpus evaluation quality."""

from __future__ import annotations

from typing import Any


def calculate_scale_drift(hpo_metrics: dict[str, Any], full_metrics: dict[str, Any]) -> dict[str, Any]:
    """Return full-minus-HPO deltas for mean quality metrics sharing a suffix."""
    result: dict[str, Any] = {}
    for hpo_key, baseline in hpo_metrics.items():
        if not hpo_key.startswith("hpo_") or not hpo_key.endswith("_mean") or not isinstance(baseline, (int, float)):
            continue
        suffix = hpo_key.removeprefix("hpo_")
        observed = full_metrics.get(f"e2e_{suffix}")
        if not isinstance(observed, (int, float)):
            continue
        metric = suffix.removesuffix("_mean")
        delta = observed - baseline
        result[f"scale_delta_{metric}"] = round(delta, 6)
        # Percentage degradation is undefined against a zero baseline; the raw
        # delta above still captures the change in that case.
        if baseline != 0:
            result[f"scale_degradation_pct_{metric}"] = round((baseline - observed) / baseline * 100, 3)
    return result
