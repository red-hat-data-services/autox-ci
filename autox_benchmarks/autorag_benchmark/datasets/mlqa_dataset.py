"""MLQA dataset provider for multilingual RAG benchmarking.

Uses the facebook/mlqa dataset — a parallel extractive QA benchmark with
Wikipedia paragraph contexts in 7 languages. The same question+context
pairs exist across languages, enabling cross-lingual comparison.

Downloads the MLQA_V1.zip directly from Facebook's servers and parses
the SQuAD-format JSON files.

Supported languages: ar, de, en, es, hi, vi, zh.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document, get_file_extension

MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"

SUPPORTED_LANGUAGES = ["ar", "de", "en", "es", "hi", "vi", "zh"]


def _context_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _download_and_extract(language: str, split: str) -> list[dict]:
    """Download MLQA zip, extract the relevant JSON, return flat QA rows."""
    import requests

    cache_dir = Path(tempfile.gettempdir()) / "mlqa_cache"
    zip_path = cache_dir / "MLQA_V1.zip"

    if not zip_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading MLQA_V1.zip...")
        resp = requests.get(MLQA_URL, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        print(f"  Downloaded {len(resp.content) / 1024 / 1024:.1f} MB")

    json_name = f"MLQA_V1/{split}/{split}-context-{language}-question-{language}.json"
    print(f"  Extracting {json_name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(json_name) as f:
            data = json.load(f)

    rows: list[dict] = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                rows.append({
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": [a.get("text", "") for a in qa.get("answers", [])],
                    "id": qa.get("id", ""),
                })
    return rows


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    language: str = "en",
    split: str = "test",
    output_format: str = "txt",
    **_: object,
) -> tuple[int, int]:
    """Download MLQA and write kb docs + benchmark JSON.

    Returns (num_docs, num_entries).
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Available: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    print(f"Loading MLQA ({language}, split={split}) for up to {num_samples} samples...")
    rows = _download_and_extract(language, split)
    print(f"  Parsed {len(rows)} QA pairs")

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    written_docs: dict[str, str] = {}  # context_hash -> local_filename
    processed = 0
    skipped = 0

    for row in rows:
        if processed >= num_samples:
            break

        context = row["context"].strip()
        question = row["question"].strip()
        answer_texts = [a for a in row["answers"] if a.strip()]

        if not context or not question or not answer_texts:
            skipped += 1
            continue

        ctx_hash = _context_hash(context)

        if ctx_hash not in written_docs:
            safe_id = f"mlqa_{language}_{ctx_hash}"
            file_ext = get_file_extension(output_format)
            local_filename = f"{safe_id}{file_ext}"

            save_document(
                content=context,
                output_path=kb_dir / safe_id,
                format=output_format,
                metadata={
                    "source": "mlqa",
                    "language": language,
                    "context_hash": ctx_hash,
                },
            )
            written_docs[ctx_hash] = local_filename

        benchmark_data.append({
            "question": question,
            "correct_answers": answer_texts,
            "correct_answer_document_ids": [written_docs[ctx_hash]],
        })
        processed += 1

        if processed % 10 == 0:
            print(
                f"  Processed {processed}/{num_samples} "
                f"(skipped {skipped}, unique docs: {len(written_docs)})"
            )

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4, ensure_ascii=False)

    print(f"\nMLQA ({language}) generation complete:")
    print(f"  Benchmark entries: {len(benchmark_data)}")
    print(f"  Unique documents: {len(written_docs)}")
    print(f"  Skipped entries: {skipped}")
    print(f"  Format: {output_format}")

    return len(written_docs), len(benchmark_data)


register("mlqa", prepare, {
    "num_samples": 50,
    "language": "en",
    "split": "test",
})
