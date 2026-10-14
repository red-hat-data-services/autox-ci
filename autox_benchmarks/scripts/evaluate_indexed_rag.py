#!/usr/bin/env python3
"""Evaluate a completed index with the exact ai4rag pattern from its HPO notebook."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autorag_benchmark.config_loader import load_merged_benchmark_config
from autorag_benchmark.indexing_from_hpo import extract_best_pattern_settings
from autorag_benchmark.indexing_report import find_indexing_report_key
from autorag_benchmark.metrics.quality_metrics import aggregate_evaluation_results
from autorag_benchmark.pattern_scores import create_s3_client
from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME
from autorag_benchmark.settings import benchmark_settings_from_config

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full QA evaluation using the generated ai4rag RAG pattern.")
    p.add_argument("--indexing-run-id", required=True)
    p.add_argument("--hpo-run-id", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--test-data-key", required=True)
    p.add_argument("--pattern-name")
    p.add_argument("--max-questions", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-ragas", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _make_core_api(cfg: dict[str, Any]) -> Any:
    from autox_tests.lib.k8s_utils import make_k8s_core_api

    kfp = cfg.get("kfp") or {}
    return make_k8s_core_api(str(kfp.get("token") or ""), str(kfp.get("host") or ""))


def load_secret(cfg: dict[str, Any], name: str) -> None:
    """Read a namespaced k8s secret and export its keys as environment variables."""
    kfp = cfg.get("kfp") or {}
    api = _make_core_api(cfg)
    secret = api.read_namespaced_secret(name, str(kfp.get("namespace") or ""))
    for key, value in (secret.data or {}).items():
        os.environ.setdefault(key, base64.b64decode(value).decode())


def _resolve_external_milvus_uri(cfg: dict[str, Any], internal_uri: str) -> str | None:
    """Return the external LoadBalancer URI for an in-cluster Milvus, or None.

    Looks for a LoadBalancer service in the URI's namespace that exposes the same
    port and returns ``http://<external-host>:<port>``.
    """
    parsed = urlparse(internal_uri)
    host = parsed.hostname or ""
    port = parsed.port or 19530
    # host is like "milvus-service.milvus.svc.cluster.local"; the namespace is the 2nd label.
    labels = host.split(".")
    if len(labels) < 2:
        return None
    namespace = labels[1]
    api = _make_core_api(cfg)
    for svc in api.list_namespaced_service(namespace).items:
        if svc.spec.type != "LoadBalancer" or not any(p.port == port for p in (svc.spec.ports or [])):
            continue
        lb = svc.status.load_balancer
        for ingress in (lb.ingress or []) if lb else []:
            external = ingress.hostname or ingress.ip
            if external:
                return f"http://{external}:{port}"
    return None


_LOOPBACK_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def _reconcile_milvus_uri_from_secret(cfg: dict[str, Any], secret_name: str) -> None:
    """Overwrite a blank/loopback ``MILVUS_URI`` with the cluster secret's value.

    ``load_secret`` uses ``setdefault``, so a ``MILVUS_URI`` inherited from the
    caller's shell (commonly empty or ``localhost`` left over from local Milvus
    experiments) shadows the cluster secret. A loopback address is never a valid
    remote target for an off-cluster benchmark run, so the secret wins in that
    case; a genuine non-loopback override is left untouched.
    """
    host = urlparse(os.environ.get("MILVUS_URI", "")).hostname or ""
    if host not in _LOOPBACK_HOSTS:
        return
    kfp = cfg.get("kfp") or {}
    api = _make_core_api(cfg)
    secret = api.read_namespaced_secret(secret_name, str(kfp.get("namespace") or ""))
    encoded = (secret.data or {}).get("MILVUS_URI")
    if not encoded:
        return
    secret_uri = base64.b64decode(encoded).decode()
    if secret_uri:
        LOG.info("MILVUS_URI was %r; using cluster secret value %s", host, secret_uri)
        os.environ["MILVUS_URI"] = secret_uri


def apply_milvus_networking(cfg: dict[str, Any]) -> None:
    """Swap an unreachable in-cluster ``MILVUS_URI`` for its external LoadBalancer.

    Only acts when the URI is a ``*.svc.cluster.local`` address that this machine
    cannot resolve (i.e. the benchmark runs outside the cluster). In-cluster the
    name resolves, so the internal address is left untouched.
    """
    uri = os.environ.get("MILVUS_URI", "")
    host = urlparse(uri).hostname or ""
    if not host.endswith(".svc.cluster.local"):
        return
    try:
        socket.getaddrinfo(host, None)
        return  # resolvable -> running in-cluster, keep the internal address
    except socket.gaierror:
        pass
    external = _resolve_external_milvus_uri(cfg, uri)
    if external:
        LOG.info("MILVUS_URI %s is not resolvable here; using external LoadBalancer %s", host, external)
        os.environ["MILVUS_URI"] = external
    else:
        LOG.warning(
            "MILVUS_URI %s is not resolvable and no external LoadBalancer was found; "
            "set MILVUS_URI manually if running outside the cluster",
            host,
        )


def add_scores(rows: list[dict[str, Any]], result: Any, evaluator: str) -> None:
    """Preserve ai4rag's per-question scores in the established artifact format."""
    if not isinstance(result, dict):
        return
    question_scores = result.get("question_scores") or []

    # ai4rag >= 0.15 returns a question-centric list, matching the evaluator
    # contract used by the generated notebook.
    if isinstance(question_scores, list):
        for index, question_score in enumerate(question_scores):
            if not isinstance(question_score, dict) or index >= len(rows):
                continue
            for metric in question_score.get("metrics") or []:
                try:
                    rows[index]["metrics"].append({
                        "name": metric["name"],
                        "evaluator": metric.get("evaluator", evaluator),
                        "score": float(metric["value"]),
                    })
                except (KeyError, TypeError, ValueError):
                    pass
        return

    # Compatibility with the older ai4rag dict-of-question-scores result.
    if not isinstance(question_scores, dict):
        return
    for metric, values in question_scores.items():
        if not isinstance(values, dict):
            continue
        for index, value in values.items():
            try:
                rows[int(index)]["metrics"].append({"name": metric, "evaluator": evaluator, "score": float(value)})
            except (ValueError, TypeError, IndexError):
                pass


def generate_answers(rag: Any, test_data: list[dict[str, Any]], workers: int) -> list[Any]:
    """Answer every question in parallel; a single failure does not sink the run."""
    def answer(item: dict[str, Any]) -> Any:
        try:
            return rag.generate(question=item["question"])
        except Exception as exc:
            LOG.warning("rag.generate failed for a question: %s", exc)
            return {"answer": "", "reference_documents": []}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(answer, test_data))


def build_rows(test_data: list[dict[str, Any]], responses: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item, response in zip(test_data, responses):
        contexts = [
            {
                "text": str(getattr(d, "page_content", "")),
                "document_id": str((getattr(d, "metadata", {}) or {}).get("document_id", "")),
            }
            for d in response.get("reference_documents", [])
        ]
        rows.append({
            "question": item.get("question", ""),
            "correct_answers": item.get("correct_answers", []),
            "correct_answer_document_keys": item.get("correct_answer_document_keys", []),
            "answer": response.get("answer", ""),
            "answer_contexts": contexts,
            "metrics": [],
        })
    return rows


def main() -> int:
    a = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg, config_dir = load_merged_benchmark_config(a.config.resolve(), a.env_file)
    settings = benchmark_settings_from_config(cfg, config_dir)
    load_secret(cfg, settings.maas_secret_name)
    load_secret(cfg, settings.vector_db_secret_name)
    _reconcile_milvus_uri_from_secret(cfg, settings.vector_db_secret_name)
    apply_milvus_networking(cfg)
    if not os.getenv("MAAS_BASE_URL") or not os.getenv("MAAS_API_KEY"):
        raise RuntimeError("MaaS secret requires MAAS_BASE_URL and MAAS_API_KEY")

    s3 = create_s3_client(cfg)
    bucket = settings.test_data_bucket_name
    pattern = extract_best_pattern_settings(a.hpo_run_id, cfg, bucket, pattern_name_override=a.pattern_name)

    prefix = f"{INDEXING_PIPELINE_NAME}/{a.indexing_run_id}/"
    report_key = find_indexing_report_key(s3, bucket, prefix)
    if not report_key:
        raise RuntimeError(f"No indexing report under s3://{bucket}/{prefix}")
    report = json.loads(s3.get_object(Bucket=bucket, Key=report_key)["Body"].read())
    report_settings = report.get("settings") or {}
    binding = report_settings.get("vector_store_binding") or {}
    collection = binding.get("collection_name")
    provider = binding.get("provider_type")
    if not collection or not provider:
        raise RuntimeError("Indexing report lacks collection name/provider type")

    test_data = json.loads(s3.get_object(Bucket=bucket, Key=a.test_data_key)["Body"].read())
    if not isinstance(test_data, list):
        raise RuntimeError("Benchmark test data must be a JSON array")
    if a.max_questions:
        test_data = test_data[:a.max_questions]

    try:
        from ai4rag.core.experiment.benchmark_data import BenchmarkData
        from ai4rag.core.experiment.utils import build_evaluation_data
        from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel, OpenAIEmbeddingParams
        from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
        from ai4rag.rag.retrieval.retriever import Retriever
        from ai4rag.rag.template.simple_rag_template import SimpleRAG
        from ai4rag.rag.vector_store import get_vector_store, get_vector_store_config
        from ai4rag.utils.clients.maas_client import create_maas_client
        from pandas import DataFrame
    except ImportError as e:
        raise RuntimeError(
            "Install the generated notebook runtime: python3 -m pip install 'ai4rag~=0.15.0'"
        ) from e

    client = create_maas_client(base_url=os.environ["MAAS_BASE_URL"], api_key=os.environ["MAAS_API_KEY"])

    embedding = report_settings.get("embedding") or pattern.get("embedding") or {}
    em = OpenAIEmbeddingModel(
        client=client,
        model_id=embedding.get("model_id"),
        params=OpenAIEmbeddingParams(**(embedding.get("embedding_params") or {})),
    )
    gen = pattern.get("generation") or {}
    fm = OpenAIFoundationModel(
        client=client,
        model_id=gen.get("model_id"),
        system_message_text=gen.get("system_message_text"),
        user_message_text=gen.get("user_message_text"),
        context_template_text=gen.get("context_template_text"),
    )
    vs = get_vector_store(embedding_model=em, config=get_vector_store_config(provider), collection_name=collection)
    ret = pattern.get("retrieval") or {}
    retriever = Retriever(
        vector_store=vs,
        method=ret.get("method", "simple"),
        number_of_chunks=int(ret.get("number_of_chunks", 5)),
        search_mode=ret.get("search_mode", "vector"),
        ranker_strategy=ret.get("ranker_strategy"),
        ranker_k=ret.get("ranker_k"),
        ranker_alpha=ret.get("ranker_alpha"),
    )
    rag = SimpleRAG(foundation_model=fm, retriever=retriever)

    LOG.info("Evaluating %d questions against collection %s", len(test_data), collection)
    responses = generate_answers(rag, test_data, a.workers)
    rows = build_rows(test_data, responses)

    data = build_evaluation_data(BenchmarkData(DataFrame(data=test_data)), responses)
    from ai4rag.evaluator.metric import Metrics
    from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator

    unitxt_metrics = (Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS, Metrics.CONTEXT_CORRECTNESS)
    add_scores(rows, UnitxtEvaluator().evaluate_metrics(data, unitxt_metrics), "unitxt")
    if not a.skip_ragas:
        from ai4rag.evaluator.ragas_evaluator import RagasEvaluator

        ragas_metrics = (
            Metrics.RAGAS_FAITHFULNESS,
            Metrics.RAGAS_ANSWER_RELEVANCY,
            Metrics.RAGAS_CONTEXT_PRECISION,
            Metrics.RAGAS_CONTEXT_RECALL,
        )
        add_scores(rows, RagasEvaluator(model=fm, embedding_model=em).evaluate_metrics(data, ragas_metrics), "ragas")

    summary = aggregate_evaluation_results(rows, prefix="e2e_")
    key = f"{INDEXING_PIPELINE_NAME}/{a.indexing_run_id}/full_corpus_evaluation_results.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(rows, ensure_ascii=False).encode(),
        ContentType="application/json",
    )
    LOG.info("Uploaded s3://%s/%s", bucket, key)
    LOG.info("Summary: %s", json.dumps(summary, sort_keys=True))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
