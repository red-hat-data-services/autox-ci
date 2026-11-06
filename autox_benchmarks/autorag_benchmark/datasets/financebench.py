"""Convert the public FinanceBench release into an AutoRAG PDF dataset.

The source is a local checkout of https://github.com/patronus-ai/financebench.
Every PDF listed in its document index is retained, including documents that
are not referenced by the selected questions. This keeps retrieval meaningful
when benchmarking a subset of the 150 public questions.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

from autorag_benchmark.datasets import register


QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCUMENTS_FILE = "financebench_document_information.jsonl"
SAMPLE_SEED = 42


def _source_dir(local_dir: str | Path | None) -> Path:
    configured = local_dir or os.environ.get("FINANCEBENCH_DIR")
    if not configured:
        raise ValueError(
            "FinanceBench requires --financebench-dir or FINANCEBENCH_DIR "
            "pointing to a checkout with data/ and pdfs/."
        )
    root = Path(configured).expanduser().resolve()
    required = (root / "data" / QUESTIONS_FILE, root / "data" / DOCUMENTS_FILE, root / "pdfs")
    if not all(path.exists() for path in required):
        raise ValueError(f"FinanceBench checkout at {root} must contain data/ and pdfs/.")
    return root


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _pdf_name(doc_name: str) -> str:
    if not doc_name or Path(doc_name).name != doc_name or doc_name in {".", ".."}:
        raise ValueError(f"Invalid FinanceBench document name: {doc_name!r}")
    return f"{doc_name}.pdf"


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as document:
        return document.read(4) == b"%PDF"


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    output_format: str = "pdf",
    local_dir: str | Path | None = None,
    **_: object,
) -> tuple[int, int]:
    """Write the full PDF corpus and a deterministic subset of public QA rows."""
    if output_format != "pdf":
        raise ValueError("FinanceBench requires --output-format pdf to test PDF ingestion.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    root = _source_dir(local_dir)
    questions = _read_jsonl(root / "data" / QUESTIONS_FILE)
    documents = _read_jsonl(root / "data" / DOCUMENTS_FILE)
    if not questions or not documents:
        raise ValueError("FinanceBench question and document indexes must not be empty")
    if num_samples > len(questions):
        raise ValueError(f"Requested {num_samples} questions, but only {len(questions)} are available")

    document_names = {str(row["doc_name"]) for row in documents}
    pdfs = {_pdf_name(name): root / "pdfs" / _pdf_name(name) for name in document_names}
    missing = sorted(name for name, path in pdfs.items() if not path.is_file())
    if missing:
        raise ValueError(f"FinanceBench checkout is missing {len(missing)} indexed PDFs, starting with {missing[:5]}")
    invalid = sorted(name for name, path in pdfs.items() if not _is_pdf(path))
    if invalid:
        raise ValueError(f"FinanceBench PDFs are invalid or Git LFS pointers, starting with {invalid[:5]}")

    chosen = random.Random(SAMPLE_SEED).sample(
        sorted(questions, key=lambda row: str(row["financebench_id"])), num_samples
    )
    benchmark: list[dict] = []
    for row in chosen:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        gold_names = list(dict.fromkeys(
            str(item["doc_name"]) for item in row.get("evidence") or []
        ))
        if not question or not answer or not gold_names:
            raise ValueError(f"FinanceBench question {row.get('financebench_id')} has incomplete QA or evidence")
        unknown = set(gold_names) - document_names
        if unknown:
            raise ValueError(f"FinanceBench question {row['financebench_id']} references unknown documents: {sorted(unknown)}")
        benchmark.append({
            "question": question,
            "correct_answers": [answer],
            "correct_answer_document_keys": [_pdf_name(name) for name in gold_names],
        })

    kb_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in kb_dir.iterdir() if path.name not in pdfs)
    if unexpected:
        raise ValueError(f"Knowledge base contains files outside the FinanceBench corpus: {unexpected[:5]}")
    for name, source in sorted(pdfs.items()):
        destination = kb_dir / name
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(pdfs), len(benchmark)


register("financebench", prepare)
