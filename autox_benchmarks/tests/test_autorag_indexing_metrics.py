"""Unit tests for the AutoRAG indexing/e2e metric helpers and argument builders."""

from __future__ import annotations

from types import SimpleNamespace

from autorag_benchmark.indexing_from_hpo import build_indexing_arguments
from autorag_benchmark.e2e_evaluation import rebase_test_data_document_keys
from autorag_benchmark.indexing_report import flatten_report, store_binding_from_report
from autorag_benchmark.metrics.quality_metrics import aggregate_evaluation_results, extract_quality_metrics
from autorag_benchmark.metrics.scale_drift import calculate_scale_drift
from autorag_benchmark.pipeline_params import align_database_secret_argument, build_pipeline_arguments, indexing_preset_for_dataset


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
        "gpu_acceleration": False,
        "indexing_gpu_acceleration": False,
        "embedding_models": [],
        "generation_models": [],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Pipeline argument builders ──

def test_build_pipeline_arguments_passes_input_data_keys_as_list() -> None:
    dataset = {"test_data_key": "t.json", "input_data_key": "datasets/rag/x/knowledge_base"}
    args = build_pipeline_arguments(dataset, _settings())
    assert args["input_data_keys"] == ["datasets/rag/x/knowledge_base"]
    assert "input_data_key" not in args


def test_build_indexing_arguments_passes_full_key_as_string() -> None:
    pattern = {"pattern_name": "P1", "embedding": {"model_id": "bge-m3"}, "chunking": {}}
    args = build_indexing_arguments(pattern, _settings(), full_input_data_key="datasets/full/kb")
    assert args["input_data_key"] == "datasets/full/kb"
    assert "input_data_keys" not in args
    assert args["ocr_lang"] == "en"


def test_build_indexing_arguments_uses_pattern_language_for_ocr() -> None:
    pattern = {
        "pattern_name": "P1",
        "embedding": {"model_id": "bge-m3"},
        "generation": {"language": {"code": "zh"}},
    }
    args = build_indexing_arguments(pattern, _settings(), full_input_data_key="k")
    assert args["ocr_lang"] == "zh"


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


def test_build_indexing_arguments_keeps_indexing_preset_separate() -> None:
    pattern = {"pattern_name": "P1", "embedding": {"model_id": "bge-m3"}, "chunking": {}}
    args = build_indexing_arguments(
        pattern,
        _settings(),
        full_input_data_key="k",
        preset="balanced",
    )
    assert args["preset"] == "balanced"


def test_indexing_uses_dataset_quality_tier_unless_overridden() -> None:
    dataset = {"preset": "balanced", "test_data_key": "t.json"}
    settings = _settings(indexing_preset="", preset="")
    assert build_pipeline_arguments(dataset, settings)["preset"] == "balanced"
    assert indexing_preset_for_dataset(dataset, settings) == "balanced"
    assert indexing_preset_for_dataset(dataset, _settings(indexing_preset="speed")) == "speed"


def test_gpu_acceleration_is_passed_to_both_pipelines() -> None:
    settings = _settings(gpu_acceleration=True, indexing_gpu_acceleration=True)
    hpo_args = build_pipeline_arguments({"test_data_key": "t.json"}, settings)
    pattern = {"pattern_name": "P1", "embedding": {"model_id": "bge-m3"}}
    indexing_args = build_indexing_arguments(pattern, settings, full_input_data_key="k")
    assert hpo_args["gpu_acceleration"] is True
    assert indexing_args["gpu_acceleration"] is True


def test_compiled_package_uses_declared_database_secret_name() -> None:
    legacy_args = {"vector_db_secret_name": "milvus-secret", "gpu_acceleration": False}
    current = align_database_secret_argument(legacy_args, {"db_secret_name", "gpu_acceleration"})
    assert current == {"db_secret_name": "milvus-secret", "gpu_acceleration": False}
    assert legacy_args["vector_db_secret_name"] == "milvus-secret"
    assert align_database_secret_argument(legacy_args, {"vector_db_secret_name"}) == legacy_args


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


def test_retrieval_metrics_keep_document_paths_distinct() -> None:
    rows = [{
        "correct_answer_document_keys": ["corpus/set_a/readme.md"],
        "answer_contexts": [{"document_id": "corpus/set_b/readme.md"}],
    }]
    out = aggregate_evaluation_results(rows)
    assert out["e2e_retrieval_hit_mean"] == 0.0
    assert out["e2e_retrieval_document_recall_mean"] == 0.0


def test_retrieval_metrics_accept_filename_only_context_ids() -> None:
    rows = [{
        "correct_answer_document_keys": ["corpus/set_a/readme.md"],
        "answer_contexts": [{"document_id": "readme.md"}],
    }]
    assert aggregate_evaluation_results(rows)["e2e_retrieval_hit_mean"] == 1.0


def test_missing_selected_hpo_pattern_does_not_use_another_pattern(monkeypatch) -> None:
    from autorag_benchmark import pattern_scores

    class FakeS3:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, **kwargs):
            return [{"Contents": [{"Key": f"{kwargs['Prefix']}rag_patterns/Pattern2/evaluation_results.json"}]}]

        def get_object(self, **kwargs):
            raise AssertionError("another pattern's artifact must not be read")

    monkeypatch.setattr(pattern_scores, "create_s3_client", lambda config: FakeS3())
    out = extract_quality_metrics("hpo-1", {}, "bucket", pattern_name="Pattern1")
    assert out == {"hpo_quality_note": "evaluation_results.json not found for pattern Pattern1"}


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
            "store_binding": {"collection_name": "c1", "provider_type": "milvus"},
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 0},
            "embedding": {"model_id": "bge-m3", "embedding_params": {"embedding_dimension": 1024}},
        },
    }
    flat = flatten_report(report)
    assert flat["indexing_total_documents"] == 10
    assert flat["indexing_collection_name"] == "c1"
    assert flat["indexing_provider_type"] == "milvus"
    assert flat["indexing_embedding_dimension"] == 1024
    # 100 chunks * 1024 dims * 4 bytes / 1024**2
    assert flat["indexing_estimated_vector_storage_mb"] == round(100 * 1024 * 4 / 1024**2, 3)


def test_store_binding_accepts_legacy_report_name() -> None:
    report = {"settings": {"vector_store_binding": {"collection_name": "old", "provider_type": "milvus"}}}
    assert store_binding_from_report(report)["collection_name"] == "old"


def test_e2e_gold_document_keys_follow_indexed_corpus() -> None:
    source = "datasets/rag/spoken_squad/test/wav/10"
    target = "datasets/rag/spoken_squad/test/wav/capped-500docs-900mib/knowledge_base"
    records = [{"correct_answer_document_keys": [f"{source}/knowledge_base/a.wav"]}]
    rebased = rebase_test_data_document_keys(
        records, test_data_key=f"{source}/benchmark_data.json", indexed_data_key=target,
    )
    assert rebased[0]["correct_answer_document_keys"] == [f"{target}/a.wav"]
    assert records[0]["correct_answer_document_keys"] == [f"{source}/knowledge_base/a.wav"]


def test_flatten_report_chunk_distribution() -> None:
    report = {"documents": [{"chunk_count": 1}, {"chunk_count": 3}, {"chunk_count": 5}]}
    flat = flatten_report(report)
    assert flat["indexing_chunks_per_document_min"] == 1
    assert flat["indexing_chunks_per_document_max"] == 5
    assert flat["indexing_chunks_per_document_median"] == 3
