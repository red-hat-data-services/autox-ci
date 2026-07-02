"""Unit tests for response-quality artifact validation helpers."""

from autox_tests.autorag.response_validation import (
    collect_answers_from_patterns,
    compute_answer_quality_stats,
    count_citations,
    extract_metric_mean,
    validate_evaluation_results_payload,
    validate_generation_prompt_template,
    validate_pattern_scores,
    validate_responses_export,
)


def _sample_pattern(**overrides):
    base = {
        "pattern_name": "Pattern1",
        "final_score": 0.85,
        "scores": {
            "faithfulness": {"mean": 0.9},
            "answer_correctness": {"mean": 0.85},
        },
        "settings": {
            "vector_store_binding": {
                "provider_id": "pgvector",
                "vector_store_id": "vs-123",
            },
            "responses_template": {
                "model": "test-model",
                "tools": [{"type": "file_search", "vector_store_ids": ["vs-123"]}],
            },
            "generation": {
                "user_message_text": "Context:\n{reference_documents}\nQuestion: {question}",
            },
        },
        "evaluation_results": [
            {"question": "Q1?", "answer": "A1 with citation [1]"},
            {"question": "Q2?", "answer": "Réponse"},
        ],
    }
    base.update(overrides)
    return base


def test_extract_metric_mean_from_dict_and_scalar():
    assert extract_metric_mean({"faithfulness": {"mean": 0.42}}, "faithfulness") == 0.42
    assert extract_metric_mean({"answer_correctness": 0.7}, "answer_correctness") == 0.7
    assert extract_metric_mean({}, "faithfulness") is None


def test_validate_pattern_scores_accepts_optimization_metric():
    validate_pattern_scores(
        _sample_pattern(),
        optimization_metric="faithfulness",
        scenario_id="TC-P-4",
    )


def test_validate_responses_export_checks_binding_parity():
    validate_responses_export(
        _sample_pattern(),
        provider_id="pgvector",
        scenario_id="TC-P-3",
    )


def test_validate_generation_prompt_template_requires_question_placeholder():
    validate_generation_prompt_template(_sample_pattern(), scenario_id="TC-P-1")


def test_validate_evaluation_results_payload_nested_patterns():
    payload = {
        "patterns": [
            {
                "pattern_name": "Pattern1",
                "evaluation_results": [
                    {"question": "Q1", "answer": "A1", "scores": {"faithfulness": 0.9}},
                    {"question": "Q2", "answer": "A2", "scores": {"faithfulness": 0.8}},
                ],
            }
        ]
    }
    validate_evaluation_results_payload(
        payload,
        min_patterns=1,
        min_questions=2,
        scenario_id="TC-P-6",
    )


def test_collect_answers_and_quality_stats():
    patterns = [_sample_pattern()]
    answers = collect_answers_from_patterns(patterns)
    assert len(answers) == 2
    stats = compute_answer_quality_stats(answers)
    assert stats["citation_rate"] == 0.5
    assert stats["multilingual_rate"] == 0.5


def test_count_citations_supports_bracket_and_file_id_formats():
    assert count_citations("See [1] and <|doc-1|>") == 2
    assert count_citations("no cites") == 0
