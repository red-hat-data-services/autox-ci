"""MultiHop-RAG articles and questions for the AutoRAG benchmark.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG
The corpus is kept whole even when only a few questions are selected, so retrieval
still has to distinguish the evidence articles from the other articles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document

REPO_ID = "yixuantt/MultiHopRAG"
QUERY_FILE = "MultiHopRAG.json"
CORPUS_FILE = "corpus.json"


def _source_file(filename: str, local_dir: Path | None) -> Path:
    if local_dir is not None:
        path = Path(local_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"MultiHop-RAG source file not found: {path}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))


def _filename(url: str, output_format: str) -> str:
    return f"multihop_rag_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:20]}.{output_format}"


def _article_text(article: dict) -> str:
    fields = (
        ("Title", article.get("title")),
        ("Source", article.get("source")),
        ("Author", article.get("author")),
        ("Published", article.get("published_at")),
        ("Category", article.get("category")),
        ("URL", article.get("url")),
    )
    header = "\n".join(f"{label}: {value}" for label, value in fields if value)
    return f"{header}\n\n{article['body'].strip()}"


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    output_format: str = "txt",
    local_dir: Path | None = None,
    **_: object,
) -> tuple[int, int]:
    """Write all articles and the first ``num_samples`` answerable QA rows.

    ``local_dir`` can contain the two official JSON files for offline generation.
    Null questions have no evidence keys, which ai4rag's BenchmarkData rejects,
    so they are excluded from AutoRAG benchmark data.
    """
    if output_format not in {"txt", "md"}:
        raise ValueError("MultiHop-RAG supports only txt and md output formats")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    corpus = json.loads(_source_file(CORPUS_FILE, local_dir).read_text(encoding="utf-8"))
    questions = json.loads(_source_file(QUERY_FILE, local_dir).read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not isinstance(questions, list):
        raise ValueError("MultiHop-RAG corpus and questions must be JSON arrays")
    answerable: list[tuple[int, dict]] = []
    for index, row in enumerate(questions):
        evidence = row.get("evidence_list")
        if not isinstance(evidence, list):
            raise ValueError(f"MultiHop-RAG question {index} has no evidence list")
        if evidence:
            answerable.append((index, row))
    if num_samples > len(answerable):
        raise ValueError(
            f"MultiHop-RAG has {len(answerable)} questions with evidence; requested {num_samples}"
        )

    articles: dict[str, dict] = {}
    filenames: dict[str, str] = {}
    for article in corpus:
        url = article.get("url")
        if not url or not isinstance(article.get("body"), str) or not article["body"].strip():
            raise ValueError("MultiHop-RAG corpus article is missing a URL or body")
        if url in articles:
            raise ValueError(f"Duplicate MultiHop-RAG article URL: {url}")
        articles[url] = article
        filenames[url] = _filename(url, output_format)

    benchmark: list[dict] = []
    for index, row in answerable[:num_samples]:
        question = row.get("query")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"MultiHop-RAG question {index} has no query or answer")
        evidence = row.get("evidence_list")
        gold: list[str] = []
        for item in evidence:
            url = item.get("url")
            if url not in filenames:
                raise ValueError(f"MultiHop-RAG question {index} references missing article: {url}")
            filename = filenames[url]
            if filename not in gold:
                gold.append(filename)
        benchmark.append({
            "question": question.strip(),
            "correct_answers": [answer.strip()],
            "correct_answer_document_keys": gold,
        })

    kb_dir.mkdir(parents=True, exist_ok=True)
    for url, article in articles.items():
        filename = filenames[url]
        save_document(
            content=_article_text(article),
            output_path=kb_dir / filename,
            format=output_format,
            metadata={"title": article.get("title") or filename, "source": article.get("source") or ""},
        )

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(articles), len(benchmark)


register("multihop_rag", prepare, {"num_samples": 50})
