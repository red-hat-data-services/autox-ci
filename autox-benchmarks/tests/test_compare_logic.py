"""Tests for automl_benchmark.compare_logic."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from automl_benchmark.compare_logic import (
    align_score_matrix,
    baseline_only_keys,
    collapse_baseline_latest_per_dataset,
    compare_only_keys,
    compare_to_baseline,
    coverage_stats,
    detect_model_column,
    detect_score_column,
    score_matrix_for_heatmap,
)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def baseline_long() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-01T10:00:00Z",
                "run_id": "run-old",
                "model": "m1",
                "score_val": 0.80,
            },
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-01T10:00:00Z",
                "run_id": "run-old",
                "model": "m2",
                "score_val": 0.70,
            },
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-02T12:00:00Z",
                "run_id": "run-new",
                "model": "m1",
                "score_val": 0.85,
            },
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-02T12:00:00Z",
                "run_id": "run-new",
                "model": "m2",
                "score_val": 0.75,
            },
            {
                "dataset_name": "ds_b",
                "finished_at": "2026-01-01T08:00:00Z",
                "run_id": "run-b",
                "model": "m1",
                "score_val": 0.90,
            },
        ]
    )


@pytest.fixture
def compare_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-03T09:00:00Z",
                "model": "m1",
                "score_val": 0.88,
            },
            {
                "dataset_name": "ds_a",
                "finished_at": "2026-01-03T09:00:00Z",
                "model": "m2",
                "score_val": 0.80,
            },
            {
                "dataset_name": "ds_c",
                "finished_at": "2026-01-03T09:00:00Z",
                "model": "m1",
                "score_val": 0.50,
            },
        ]
    )


def test_collapse_baseline_latest_per_dataset(baseline_long: pd.DataFrame) -> None:
    out = collapse_baseline_latest_per_dataset(baseline_long)
    ds_a = out[out["dataset_name"] == "ds_a"]
    assert len(ds_a) == 2
    assert set(ds_a["run_id"]) == {"run-new"}
    assert set(ds_a["score_val"].tolist()) == {0.85, 0.75}
    ds_b = out[out["dataset_name"] == "ds_b"]
    assert len(ds_b) == 1
    assert ds_b.iloc[0]["score_val"] == 0.90


def test_compare_to_baseline_join_and_delta(
    baseline_long: pd.DataFrame, compare_batch: pd.DataFrame
) -> None:
    matched, warnings = compare_to_baseline(
        baseline_long,
        compare_batch,
        compare_batch_id="20260103T090000Z",
    )
    assert not warnings or isinstance(warnings, list)
    assert len(matched) == 2
    row_m1 = matched[matched["model"] == "m1"].iloc[0]
    assert row_m1["baseline_score"] == pytest.approx(0.85)
    assert row_m1["compare_score"] == pytest.approx(0.88)
    assert row_m1["delta"] == pytest.approx(0.03)
    row_m2 = matched[matched["model"] == "m2"].iloc[0]
    assert row_m2["delta"] == pytest.approx(0.05)


def test_compare_only_and_baseline_only_keys(
    baseline_long: pd.DataFrame, compare_batch: pd.DataFrame
) -> None:
    b_only = baseline_only_keys(baseline_long, compare_batch)
    assert "ds_b" in b_only["dataset_name"].values
    c_only = compare_only_keys(baseline_long, compare_batch)
    assert "ds_c" in c_only["dataset_name"].values


def test_detect_columns(baseline_long: pd.DataFrame) -> None:
    assert detect_model_column(baseline_long) == "model"
    assert detect_score_column(baseline_long) == "score_val"


def test_coverage_stats(baseline_long: pd.DataFrame, compare_batch: pd.DataFrame) -> None:
    matched, _ = compare_to_baseline(baseline_long, compare_batch)
    collapsed = collapse_baseline_latest_per_dataset(baseline_long)
    stats = coverage_stats(matched, collapsed, compare_batch, batch_id="test")
    assert stats["matched_pairs"] == 2
    assert stats["matched_datasets"] == 1
    assert stats["baseline_datasets"] == 2


def test_score_matrix_for_heatmap(baseline_long: pd.DataFrame) -> None:
    collapsed = collapse_baseline_latest_per_dataset(baseline_long)
    matrix = score_matrix_for_heatmap(collapsed, score_column="score_val")
    assert matrix.loc["ds_a", "m1"] == pytest.approx(0.85)
    assert matrix.shape == (2, 2)  # ds_a, ds_b × m1 (ds_b only has m1)


def test_align_score_matrix(baseline_long: pd.DataFrame, compare_batch: pd.DataFrame) -> None:
    base_m = score_matrix_for_heatmap(
        collapse_baseline_latest_per_dataset(baseline_long), score_column="score_val"
    )
    cmp_m = score_matrix_for_heatmap(compare_batch, score_column="score_val")
    aligned = align_score_matrix(base_m, cmp_m)
    assert list(aligned.index) == list(base_m.index)
    assert list(aligned.columns) == list(base_m.columns)


def test_load_merged_csv_bytes() -> None:
    from automl_benchmark.compare_logic import load_merged_csv

    df = pd.DataFrame([{"dataset_name": "x", "model": "m", "score_val": 1.0}])
    loaded = load_merged_csv(_csv_bytes(df))
    assert len(loaded) == 1
