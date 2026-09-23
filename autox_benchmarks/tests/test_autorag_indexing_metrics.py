"""Unit tests for the AutoRAG indexing/e2e metric helpers and argument builders."""

from __future__ import annotations

from types import SimpleNamespace

from autorag_benchmark.indexing_from_hpo import build_indexing_arguments
from autorag_benchmark.indexing_report import flatten_report
from autorag_benchmark.metrics.quality_metrics import aggregate_evaluation_results
from autorag_benchmark.metrics.scale_drift import calculate_scale_drift
from autorag_benchmark.pipeline_params import build_pipeline_arguments


def _settings(**overrides: object) -> SimpleNamespace:
    base = {
        "input_data_bucket_name": "in-bucket",
        "input_data_secret_name": "in-secret",
        "test_data_bucket_name": "test-bucket",
        "test_data_secret_name": "test-secret",
        "maas_secret_name": "maas",
        "vector_db_secret_name": "milvus",
        "optimization_metric": "overall_score",
        "optimization_max_rag_patterns": 8,
        "preset": "",
        "embedding_models": [],
        "generation_models": [],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Pipeline argument builders: guard the singular input_data_key contract ──

def test_build_pipeline_arguments_passes_input_data_key_as_string() -> None:
    dataset = {"test_data_key": "t.json", "input_data_key": "datasets/rag/x/knowledge_base"}
    args = build_pipeline_arguments(dataset, _settings())
    assert args["input_data_key"] == "datasets/rag/x/knowledge_base"
    assert "input_data_keys" not in args


def test_build_indexing_arguments_passes_full_key_as_string() -> None:
    pattern = {"pattern_name": "P1", "embedding": {"model_id": "bge-m3"}, "chunking": {}}
    args = build_indexing_arguments(pattern, _settings(), full_input_data_key="datasets/full/kb")
    assert args["input_data_key"] == "datasets/full/kb"
    assert "input_data_keys" not in args


def test_build_indexing_arguments_maps_chunking() -> None:
    pattern = {
        "pattern_name": "P1",
        "embedding": {"model_id": "bge-m3"},
        "chunking": {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 0},
    }
    args = build_indexing_arguments(pattern, _settings(), full_input_data_key="k")
    assert args["embedding_model_id"] == "bge-m3"
    assert args["chunking_method"] == "recursive"
    assert args["chunk_size"] == 1024
    assert args["chunk_overlap"] == 0


# ── scale drift ──

def test_calculate_scale_drift_delta_and_degradation() -> None:
    hpo = {"hpo_faithfulness_mean": 0.8}
    full = {"e2e_faithfulness_mean": 0.6}
    drift = calculate_scale_drift(hpo, full)
    assert drift["scale_delta_faithfulness"] == -0.2
    assert drift["scale_degradation_pct_faithfulness"] == 25.0


def test_calculate_scale_drift_zero_baseline_skips_percentage() -> None:
    drift = calculate_scale_drift({"hpo_x_mean": 0.0}, {"e2e_x_mean": 0.5})
    assert drift["scale_delta_x"] == 0.5
    assert "scale_degradation_pct_x" not in drift


def test_calculate_scale_drift_ignores_unmatched_metrics() -> None:
    assert calculate_scale_drift({"hpo_a_mean": 0.5}, {"e2e_b_mean": 0.5}) == {}


# ── quality aggregation ──

def test_aggregate_evaluation_results_exact_match_and_token_f1() -> None:
    rows = [
        {"answer": "Paris", "correct_answers": ["Paris"], "metrics": []},
        {"answer": "London", "correct_answers": ["Paris"], "metrics": []},
    ]
    out = aggregate_evaluation_results(rows, prefix="e2e_")
    assert out["e2e_question_count"] == 2
    assert out["e2e_answer_exact_match_mean"] == 0.5


def test_aggregate_evaluation_results_groups_judge_metrics() -> None:
    rows = [{"answer": "", "metrics": [{"name": "faithfulness", "evaluator": "unitxt", "score": 1.0}]}]
    out = aggregate_evaluation_results(rows, prefix="e2e_")
    assert out["e2e_unitxt_faithfulness_mean"] == 1.0


def test_aggregate_evaluation_results_handles_unsupported_schema() -> None:
    out = aggregate_evaluation_results({"results": 123}, prefix="e2e_")
    assert "e2e_quality_note" in out


# ── indexing report flattening ──

def test_flatten_report_core_fields_and_storage_estimate() -> None:
    report = {
        "total_documents": 10,
        "completed": 10,
        "failed": 0,
        "total_chunks": 100,
        "settings": {
            "vector_store_binding": {"collection_name": "c1", "provider_type": "milvus"},
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 0},
            "embedding": {"model_id": "bge-m3", "embedding_params": {"embedding_dimension": 1024}},
        },
    }
    flat = flatten_report(report)
    assert flat["indexing_total_documents"] == 10
    assert flat["indexing_collection_name"] == "c1"
    assert flat["indexing_embedding_dimension"] == 1024
    # 100 chunks * 1024 dims * 4 bytes / 1024**2
    assert flat["indexing_estimated_vector_storage_mb"] == round(100 * 1024 * 4 / 1024**2, 3)


def test_flatten_report_chunk_distribution() -> None:
    report = {"documents": [{"chunk_count": 1}, {"chunk_count": 3}, {"chunk_count": 5}]}
    flat = flatten_report(report)
    assert flat["indexing_chunks_per_document_min"] == 1
    assert flat["indexing_chunks_per_document_max"] == 5
    assert flat["indexing_chunks_per_document_median"] == 3
