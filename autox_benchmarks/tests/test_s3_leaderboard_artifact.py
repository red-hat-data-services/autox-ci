"""Unit tests for leaderboard S3 path matching and scores JSON assembly."""

from __future__ import annotations

from automl_benchmark.s3_leaderboard_artifact import (
    TABULAR_ARTIFACT_FOLDERS,
    _key_is_html_artifact,
    build_leaderboard_rows_from_metrics_jsons,
)


def test_html_key_matches_training_component_layout() -> None:
    key = (
        "autogluon-tabular-training-pipeline/"
        "rid-1/autogluon-models-training/exec-abc/html_artifact"
    )
    assert _key_is_html_artifact(
        key, "rid-1", "autogluon-models-training", "autogluon-tabular-training-pipeline"
    )


def test_html_key_matches_training_component_2_branch() -> None:
    key = (
        "autogluon-tabular-training-pipeline/"
        "rid-1/autogluon-models-training-2/exec-abc/html_artifact"
    )
    assert _key_is_html_artifact(
        key, "rid-1", "autogluon-models-training-2", "autogluon-tabular-training-pipeline"
    )


def test_html_key_still_matches_legacy_leaderboard_evaluation() -> None:
    key = (
        "autogluon-tabular-training-pipeline/"
        "rid-1/leaderboard-evaluation/exec-abc/html_artifact"
    )
    assert _key_is_html_artifact(
        key, "rid-1", "leaderboard-evaluation", "autogluon-tabular-training-pipeline"
    )


def test_tabular_folders_prefer_training_over_legacy() -> None:
    assert TABULAR_ARTIFACT_FOLDERS[0] == "autogluon-models-training"
    assert "leaderboard-evaluation" in TABULAR_ARTIFACT_FOLDERS


def test_build_leaderboard_rows_from_metrics_jsons_ranks_by_accuracy() -> None:
    rows = build_leaderboard_rows_from_metrics_jsons(
        {
            "ModelB_FULL": {"accuracy": 0.80, "f1": 0.7},
            "ModelA_FULL": {"accuracy": 0.95, "f1": 0.9},
        }
    )
    assert [r["model"] for r in rows] == ["ModelA_FULL", "ModelB_FULL"]
    assert rows[0]["rank"] == 1
    assert rows[0]["accuracy"] == 0.95
