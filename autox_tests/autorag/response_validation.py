"""Validate AutoRAG pipeline response-quality artifacts (scores, prompts, Responses API export)."""

from __future__ import annotations

import json
import re
from typing import Any


_UNITXT_METRICS = frozenset(
    {"faithfulness", "answer_correctness", "answer_relevance", "context_precision", "context_recall"}
)


def extract_metric_mean(scores: Any, metric_name: str) -> float | None:
    """Return a 0–1 mean for a metric from pattern or evaluation scores."""
    if not isinstance(scores, dict):
        return None
    data = scores.get("scores", scores) if "scores" in scores and metric_name not in scores else scores
    if not isinstance(data, dict) or metric_name not in data:
        return None
    metric = data[metric_name]
    if isinstance(metric, dict) and "mean" in metric:
        return float(metric["mean"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def validate_pattern_scores(
    pattern_data: dict[str, Any],
    *,
    optimization_metric: str,
    scenario_id: str,
) -> None:
    """Assert a pattern.json payload contains usable Unitxt-style scores."""
    scores = pattern_data.get("scores")
    assert isinstance(scores, dict) and scores, (
        f"[{scenario_id}] pattern.json missing non-empty 'scores'"
    )

    has_unitxt = any(m in scores for m in _UNITXT_METRICS)
    assert has_unitxt, (
        f"[{scenario_id}] pattern.json scores missing Unitxt metrics; keys={list(scores)}"
    )

    opt_mean = extract_metric_mean(scores, optimization_metric)
    assert opt_mean is not None, (
        f"[{scenario_id}] optimization metric {optimization_metric!r} not present in scores"
    )
    assert 0.0 <= opt_mean <= 1.0, (
        f"[{scenario_id}] {optimization_metric} mean out of range: {opt_mean}"
    )

    final_score = pattern_data.get("final_score")
    assert final_score is not None, f"[{scenario_id}] pattern.json missing final_score"
    assert isinstance(final_score, (int, float)), (
        f"[{scenario_id}] final_score must be numeric, got {type(final_score).__name__}"
    )


def validate_generation_prompt_template(
    pattern_data: dict[str, Any],
    *,
    scenario_id: str,
) -> None:
    """Assert generation prompts are exported and include a question placeholder."""
    settings = pattern_data.get("settings") or {}
    generation = settings.get("generation") or pattern_data.get("generation") or {}
    assert isinstance(generation, dict), f"[{scenario_id}] pattern.json missing generation settings"

    user_msg = generation.get("user_message_text") or ""
    assert isinstance(user_msg, str) and user_msg.strip(), (
        f"[{scenario_id}] generation.user_message_text is empty"
    )
    assert "{question}" in user_msg, (
        f"[{scenario_id}] generation.user_message_text missing {{question}} placeholder"
    )


def get_responses_template_vector_store_ids(template: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool in template.get("tools") or []:
        if tool.get("type") == "file_search":
            ids.extend(str(vs_id) for vs_id in (tool.get("vector_store_ids") or []) if vs_id)
    return ids


def validate_responses_export(
    pattern_data: dict[str, Any],
    *,
    provider_id: str,
    scenario_id: str,
) -> None:
    """Assert Responses API export fields exist and vector store IDs are consistent."""
    settings = pattern_data.get("settings") or {}
    binding = settings.get("vector_store_binding") or {}
    template = settings.get("responses_template") or {}

    assert binding, f"[{scenario_id}] pattern.json missing settings.vector_store_binding"
    assert template, f"[{scenario_id}] pattern.json missing settings.responses_template"

    binding_provider = binding.get("provider_id")
    assert binding_provider == provider_id, (
        f"[{scenario_id}] vector_store_binding.provider_id={binding_provider!r}, "
        f"expected {provider_id!r}"
    )

    binding_vs_id = binding.get("vector_store_id")
    assert binding_vs_id, f"[{scenario_id}] vector_store_binding missing vector_store_id"

    template_ids = get_responses_template_vector_store_ids(template)
    assert template_ids, (
        f"[{scenario_id}] responses_template has no file_search.vector_store_ids"
    )
    assert binding_vs_id in template_ids, (
        f"[{scenario_id}] binding vector_store_id {binding_vs_id!r} not in template ids {template_ids}"
    )


def _evaluation_entries_from_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if not isinstance(data, dict):
        return []

    for key in ("evaluation_results", "evaluations", "results", "patterns"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def validate_evaluation_results_payload(
    data: Any,
    *,
    min_patterns: int,
    min_questions: int | None,
    scenario_id: str,
) -> None:
    """Assert evaluation_results.json (or equivalent) contains evaluated patterns/answers."""
    assert data is not None, f"[{scenario_id}] evaluation_results.json is empty"

    entries = _evaluation_entries_from_payload(data)
    assert entries, (
        f"[{scenario_id}] evaluation_results.json has no recognizable evaluation entries"
    )

    if min_patterns > 1:
        pattern_ids = {
            row.get("pattern_id") or row.get("pattern_name") or row.get("name")
            for row in entries
        }
        pattern_ids.discard(None)
        if pattern_ids:
            assert len(pattern_ids) >= min_patterns, (
                f"[{scenario_id}] expected >={min_patterns} patterns in evaluation_results, "
                f"found {len(pattern_ids)}"
            )

    if min_questions is not None and min_questions > 0:
        flat_rows = [
            row
            for row in entries
            if row.get("question") and (row.get("answer") is not None or row.get("scores"))
        ]
        nested_rows = [
            inner
            for row in entries
            for inner in (row.get("evaluation_results") or [])
            if isinstance(inner, dict)
            and inner.get("question")
            and (inner.get("answer") is not None or inner.get("scores"))
        ]
        total_answers = len(flat_rows) + len(nested_rows)
        assert total_answers >= min_questions, (
            f"[{scenario_id}] expected >={min_questions} evaluated questions, found {total_answers}"
        )


def count_citations(answer: str) -> int:
    bracket = re.findall(r"\[\d+\]", answer or "")
    file_id = re.findall(r"<\|[^|]+\|>", answer or "")
    return len(bracket) + len(file_id)


def is_multilingual(answer: str) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", answer or ""))


def compute_answer_quality_stats(answers: list[dict[str, Any]]) -> dict[str, float]:
    if not answers:
        return {"citation_rate": 0.0, "multilingual_rate": 0.0}
    cited = sum(1 for row in answers if count_citations(str(row.get("answer", ""))))
    multi = sum(1 for row in answers if is_multilingual(str(row.get("answer", ""))))
    total = len(answers)
    return {
        "citation_rate": cited / total,
        "multilingual_rate": multi / total,
    }


def collect_answers_from_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-question answers from pattern.json evaluation_results lists."""
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        pattern_id = pattern.get("pattern_name") or pattern.get("pattern_id") or pattern.get("name")
        for entry in pattern.get("evaluation_results") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("question") and entry.get("answer") is not None:
                rows.append(
                    {
                        "pattern_id": pattern_id,
                        "question": entry["question"],
                        "answer": entry["answer"],
                    }
                )
    return rows


def select_best_pattern(patterns: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the pattern with the highest final_score."""
    if not patterns:
        raise ValueError("select_best_pattern requires a non-empty pattern list")
    scored = [p for p in patterns if p.get("final_score") is not None]
    if not scored:
        return patterns[0]
    return max(scored, key=lambda p: float(p["final_score"]))


def get_pattern_responses_template(pattern: dict[str, Any]) -> dict[str, Any] | None:
    template = (pattern.get("settings") or {}).get("responses_template")
    return template if isinstance(template, dict) else None


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def inject_question_into_responses_template(template: dict[str, Any], question: str) -> dict[str, Any]:
    """Return a copy of template with the user message set to the benchmark question."""
    payload = _json_safe(template)
    user_msg = next(
        (m for m in reversed(payload.get("input") or []) if m.get("role") == "user"),
        None,
    )
    if not user_msg:
        raise ValueError("responses_template has no user-role message")
    content = user_msg.get("content") or []
    if not content or not isinstance(content[0], dict):
        raise ValueError("responses_template user message must have text content")
    content[0]["text"] = question
    return payload


def extract_responses_answer(result: dict[str, Any]) -> str:
    """Extract assistant text from an OGX /v1/responses JSON body."""
    if not isinstance(result, dict):
        return ""

    top = result.get("output_text")
    if isinstance(top, str) and top.strip():
        return top.strip()

    output = result.get("output")
    if isinstance(output, list):
        for item in output:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    text = part.get("text") or part.get("value") or ""
                    if text:
                        return str(text).strip()
    return ""


def extract_file_search_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize file_search_call.results from a Responses API payload."""
    hits: list[dict[str, Any]] = []
    for item in result.get("output") or []:
        if item.get("type") != "file_search_call":
            continue
        for row in item.get("results") or []:
            hits.append(
                {
                    "file_id": row.get("file_id"),
                    "score": row.get("score"),
                    "text_preview": (row.get("text") or "")[:120],
                }
            )
    return hits


def post_ogx_responses(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    """POST to /v1/responses; raise with response body snippet on HTTP errors."""
    import requests

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
    return resp.json()


def pick_probe_question(pattern: dict[str, Any], *, fallback: str | None = None) -> str:
    """Select a benchmark question for a single Responses API probe call."""
    for entry in pattern.get("evaluation_results") or []:
        if isinstance(entry, dict):
            question = (entry.get("question") or "").strip()
            if question:
                return question
    return fallback or "Summarize the main topic covered in the indexed documents."


def validate_responses_api_probe_result(
    result: dict[str, Any],
    *,
    scenario_id: str,
    pattern_id: str,
) -> dict[str, Any]:
    """Assert a live /v1/responses probe returned retrieval hits and an answer."""
    hits = extract_file_search_hits(result)
    answer = extract_responses_answer(result)

    assert hits, (
        f"[{scenario_id}] Responses API probe for {pattern_id}: "
        "file_search returned no results (empty file_search_call.results)"
    )
    assert answer, (
        f"[{scenario_id}] Responses API probe for {pattern_id}: "
        "response had file_search hits but no extractable answer text"
    )

    return {
        "pattern_id": pattern_id,
        "file_search_hits": len(hits),
        "answer_chars": len(answer),
        "answer_preview": answer[:200],
    }


def run_responses_api_probe(
    pattern: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    scenario_id: str,
    question: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute one live Responses API call for the best pattern export."""
    template = get_pattern_responses_template(pattern)
    if not template:
        raise AssertionError(
            f"[{scenario_id}] cannot run Responses API probe: missing responses_template"
        )

    pattern_id = (
        pattern.get("pattern_name") or pattern.get("pattern_id") or pattern.get("name") or "unknown"
    )
    probe_question = question or pick_probe_question(pattern)
    payload = inject_question_into_responses_template(template, probe_question)
    result = post_ogx_responses(base_url, api_key, payload, timeout=timeout)
    summary = validate_responses_api_probe_result(
        result,
        scenario_id=scenario_id,
        pattern_id=pattern_id,
    )
    summary["question"] = probe_question
    return summary
