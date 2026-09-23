"""Aggregate per-question RAG evaluation results, including deterministic accuracy."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from typing import Any

from autorag_benchmark.pipeline_names import OPTIMIZATION_PIPELINE_NAME


def extract_quality_metrics(hpo_run_id: str, config: dict[str, Any], bucket: str, *, pattern_name: str | None = None, prefix: str = "hpo_") -> dict[str, Any]:
    """Download one pattern's evaluation results and return per-metric distributions."""
    from autorag_benchmark.pattern_scores import create_s3_client

    client = create_s3_client(config)
    root = f"{OPTIMIZATION_PIPELINE_NAME}/{hpo_run_id}/"
    keys: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root):
        keys.extend(obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith("evaluation_results.json"))
    if pattern_name:
        selected = [key for key in keys if f"/rag_patterns/{pattern_name}/" in key]
        keys = selected or keys
    if not keys:
        return {f"{prefix}quality_note": "evaluation_results.json not found"}
    response = client.get_object(Bucket=bucket, Key=sorted(keys)[0])
    return aggregate_evaluation_results(json.loads(response["Body"].read()), prefix=prefix)


def aggregate_evaluation_results(data: Any, *, prefix: str = "e2e_") -> dict[str, Any]:
    """Aggregate ai4rag's QA array without requiring numpy or an LLM.

    Alongside judge metrics, this calculates exact-match accuracy and SQuAD-style
    token F1 from ``answer`` and ``correct_answers`` whenever those fields exist.
    """
    entries = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(entries, list):
        return {f"{prefix}quality_note": "evaluation results have an unsupported schema"}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    exact_matches: list[float] = []
    token_f1s: list[float] = []
    retrieval_hits: list[float] = []
    retrieval_recalls: list[float] = []
    retrieval_precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for metric in entry.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            score = metric.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                name = _safe_name(str(metric.get("name") or "unknown"))
                evaluator = _safe_name(str(metric.get("evaluator") or "unknown"))
                grouped[(evaluator, name)].append(float(score))
        answer = entry.get("answer")
        references = entry.get("correct_answers") or entry.get("reference_answers") or entry.get("ground_truth")
        if isinstance(answer, str) and isinstance(references, list) and any(isinstance(ref, str) for ref in references):
            references = [ref for ref in references if isinstance(ref, str)]
            exact_matches.append(float(any(_normalise(answer) == _normalise(ref) for ref in references)))
            token_f1s.append(max((_token_f1(answer, ref) for ref in references), default=0.0))
        gold_documents = entry.get("correct_answer_document_keys") or entry.get("gold_document_keys")
        contexts = entry.get("answer_contexts") or entry.get("contexts")
        if isinstance(gold_documents, list) and isinstance(contexts, list):
            expected = {_document_key(value) for value in gold_documents if isinstance(value, str)}
            retrieved = [_document_key(context.get("document_id") or context.get("document_key") or context.get("source")) for context in contexts if isinstance(context, dict)]
            retrieved = [value for value in retrieved if value]
            if expected:
                matches = [index for index, value in enumerate(retrieved, start=1) if value in expected]
                retrieval_hits.append(float(bool(matches)))
                retrieval_recalls.append(len(set(retrieved) & expected) / len(expected))
                retrieval_precisions.append(len(matches) / len(retrieved) if retrieved else 0.0)
                reciprocal_ranks.append(1 / matches[0] if matches else 0.0)
    flat: dict[str, Any] = {f"{prefix}question_count": len(entries)}
    for (evaluator, name), values in sorted(grouped.items()):
        flat.update(_stats(values, f"{prefix}{evaluator}_{name}"))
    if exact_matches:
        flat.update(_stats(exact_matches, f"{prefix}answer_exact_match"))
        flat.update(_stats(token_f1s, f"{prefix}answer_token_f1"))
    if retrieval_hits:
        flat.update(_stats(retrieval_hits, f"{prefix}retrieval_hit"))
        flat.update(_stats(retrieval_recalls, f"{prefix}retrieval_document_recall"))
        flat.update(_stats(retrieval_precisions, f"{prefix}retrieval_document_precision"))
        flat.update(_stats(reciprocal_ranks, f"{prefix}retrieval_mrr"))
    return flat


def _stats(values: list[float], key: str) -> dict[str, float]:
    ordered = sorted(values)
    def percentile(p: float) -> float:
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {
        f"{key}_mean": round(statistics.fmean(values), 6), f"{key}_median": round(statistics.median(values), 6),
        f"{key}_std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        f"{key}_min": round(ordered[0], 6), f"{key}_max": round(ordered[-1], 6),
        f"{key}_p5": round(percentile(.05), 6), f"{key}_p95": round(percentile(.95), 6),
    }


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower()).split())


def _token_f1(answer: str, reference: str) -> float:
    answer_tokens, reference_tokens = _normalise(answer).split(), _normalise(reference).split()
    if not answer_tokens or not reference_tokens:
        return float(answer_tokens == reference_tokens)
    remaining = list(reference_tokens)
    common = 0
    for token in answer_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    return 2 * common / (len(answer_tokens) + len(reference_tokens))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _document_key(value: Any) -> str:
    """Compare document IDs robustly across S3 key and filename representations."""
    if not isinstance(value, str):
        return ""
    return value.rstrip("/").rsplit("/", 1)[-1]
