"""The HPO upload remains available before the full-index result is ready."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autorag_benchmark import orchestrator as module
from autorag_benchmark import s3_benchmark_upload
from autorag_benchmark.result_rows import indexing_row


def test_stage_uploads_use_separate_s3_keys_and_metadata(monkeypatch) -> None:
    written: dict[str, bytes] = {}
    monkeypatch.setattr(s3_benchmark_upload, "s3_cfg_usable", lambda config: True)
    monkeypatch.setattr(
        s3_benchmark_upload,
        "put_s3_bytes",
        lambda *, key, body, **kwargs: written.__setitem__(key, body),
    )
    settings = SimpleNamespace(
        upload_benchmark_results=True, benchmark_s3_prefix="benchmarks/rag", suite="base", rhoai_version="v1",
    )
    base = {"run_id": "hpo-1", "state": "SUCCEEDED", "dataset_id": "dataset-1"}
    full = indexing_row(
        base, "hpo-1", "idx-1", "SUCCEEDED", 12,
        indexing_run_name="index-dataset-1", indexing_started_at="2026-10-19T10:00:00+00:00",
        indexing_finished_at="2026-10-19T10:00:12+00:00",
        e2e_metrics={"e2e_question_count": 10},
    )
    common = dict(
        s3_cfg={}, bucket="bucket", settings=settings, cfg={}, batch_id="batch-1",
        dataset={"id": "dataset-1"}, pipeline_yaml_path=None, arguments={},
        dataset_filter="all", fail_fast=False, repo_root=None,
    )

    s3_benchmark_upload.upload_single_dataset_results(**common, row=base)
    s3_benchmark_upload.upload_single_dataset_results(**common, row=full, stage="full_index")

    root = "benchmarks/rag/batch-1/datasets/dataset-1"
    assert f"{root}/results.csv" in written
    assert f"{root}/full_index/results.csv" in written
    assert b"e2e_question_count" in written[f"{root}/full_index/results.csv"]
    hpo_meta = json.loads(written[f"{root}/metadata.json"])
    full_meta = json.loads(written[f"{root}/full_index/metadata.json"])
    assert (hpo_meta["stage"], hpo_meta["run_id"]) == ("hpo", "hpo-1")
    assert (full_meta["stage"], full_meta["run_id"], full_meta["state"]) == (
        "full_index", "idx-1", "SUCCEEDED",
    )
    assert full_meta["run_name"] == "index-dataset-1"
    assert full_meta["started_at"] == "2026-10-19T10:00:00+00:00"


@pytest.mark.parametrize(
    ("full_key", "e2e_metrics", "expected_stages", "expected_code"),
    [
        ("full/knowledge_base", {"e2e_question_count": 10}, ["hpo", "full_index"], 0),
        ("full/knowledge_base", {"e2e_error": "evaluator failed"}, ["hpo", "full_index"], 1),
        (None, {}, ["hpo"], 1),
    ],
)
def test_orchestrator_uploads_hpo_then_index_and_reports_stage_failure(
    tmp_path: Path, monkeypatch, full_key, e2e_metrics, expected_stages, expected_code,
) -> None:
    settings = SimpleNamespace(
        pipeline_mode="managed", test_data_bucket_name="bucket", upload_benchmark_results=True,
        benchmark_s3_prefix="benchmarks/rag", run_name_prefix="bench", suite="base",
        rhoai_version="v1", experiment_name="exp", enable_caching=False,
        timeout_seconds=60, poll_interval_seconds=1, run_indexing=True,
        run_e2e_evaluation=True, indexing_pipeline_yaml=None,
    )
    dataset = {"id": "dataset-1", "input_data_key": "small/knowledge_base", "test_data_key": "test.json"}
    if full_key:
        dataset["full_input_data_key"] = full_key
    target = SimpleNamespace(package_path=None, kfp_pipeline_name="hpo-pipeline")
    monkeypatch.setattr(
        module.BenchmarkOrchestrator, "load_config_and_datasets",
        lambda self, **kwargs: (
            {"s3": {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}},
            settings, [dataset], target, object(),
        ),
    )
    monkeypatch.setattr(module, "build_pipeline_arguments", lambda dataset, settings: {})
    monkeypatch.setattr(module, "submit_pipeline_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "extract_run_id", lambda result: "hpo-1")
    monkeypatch.setattr(module, "wait_for_terminal_run", lambda *args, **kwargs: (object(), False))
    monkeypatch.setattr(module, "completed_row", lambda base, rid, detail: {**base, "run_id": rid, "state": "SUCCEEDED"})
    monkeypatch.setattr(module, "extract_pattern_scores_tabular", lambda **kwargs: [])
    monkeypatch.setattr(module, "upload_batch_aggregated", lambda **kwargs: None)

    def fake_index(self, **kwargs):
        return [indexing_row(kwargs["base_row"], "hpo-1", "idx-1", "SUCCEEDED", 12, e2e_metrics=e2e_metrics)], {
            "input_data_key": full_key,
        }

    monkeypatch.setattr(module.BenchmarkOrchestrator, "_run_indexing_from_hpo", fake_index)
    uploads: list[str] = []
    monkeypatch.setattr(
        module, "upload_single_dataset_results",
        lambda **kwargs: uploads.append(kwargs.get("stage", "hpo")),
    )

    output = tmp_path / "results.csv"
    code = module.BenchmarkOrchestrator(tmp_path / "config.yaml").execute(output_csv=output)

    assert code == expected_code
    assert uploads == expected_stages
    assert output.is_file()
