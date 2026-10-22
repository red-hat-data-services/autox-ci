"""FinanceBench conversion preserves its full retrieval corpus and gold IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autorag_benchmark.datasets import financebench


def _checkout(root: Path) -> Path:
    source = root / "source"
    (source / "data").mkdir(parents=True)
    (source / "pdfs").mkdir()
    documents = [{"doc_name": name} for name in ("ACME_2022_10K", "ACME_2023_10K", "BETA_2022_10K")]
    questions = [
        {"financebench_id": "q2", "question": "What was ACME's 2022 revenue?", "answer": "$12 million",
         "evidence": [{"doc_name": "ACME_2022_10K", "evidence_page_num": 3}]},
        {"financebench_id": "q1", "question": "How did ACME's 2023 revenue change?", "answer": "It grew",
         "evidence": [{"doc_name": "ACME_2023_10K", "evidence_page_num": 4},
                      {"doc_name": "ACME_2023_10K", "evidence_page_num": 5}]},
    ]
    for filename, rows in ((financebench.DOCUMENTS_FILE, documents), (financebench.QUESTIONS_FILE, questions)):
        (source / "data" / filename).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    for row in documents:
        (source / "pdfs" / f"{row['doc_name']}.pdf").write_bytes(b"%PDF-1.4\nfixture")
    return source


def test_prepare_keeps_distractors_and_matches_evidence_files(tmp_path: Path) -> None:
    source = _checkout(tmp_path)
    kb_dir = tmp_path / "output" / "knowledge_base"
    bench_path = tmp_path / "output" / "benchmark_data.json"

    assert financebench.prepare(kb_dir, bench_path, local_dir=source, num_samples=2) == (3, 2)
    records = json.loads(bench_path.read_text(encoding="utf-8"))
    assert {item["question"] for item in records} == {
        "What was ACME's 2022 revenue?", "How did ACME's 2023 revenue change?"
    }
    assert sorted(path.name for path in kb_dir.iterdir()) == [
        "ACME_2022_10K.pdf", "ACME_2023_10K.pdf", "BETA_2022_10K.pdf"
    ]
    assert all(len(item["correct_answer_document_keys"]) == 1 for item in records)
    assert all((kb_dir / key).is_file() for item in records for key in item["correct_answer_document_keys"])
    first = bench_path.read_bytes()
    financebench.prepare(kb_dir, bench_path, local_dir=source, num_samples=2)
    assert bench_path.read_bytes() == first


def test_prepare_rejects_incomplete_pdf_corpus(tmp_path: Path) -> None:
    source = _checkout(tmp_path)
    (source / "pdfs" / "BETA_2022_10K.pdf").unlink()
    with pytest.raises(ValueError, match="missing 1 indexed PDFs"):
        financebench.prepare(tmp_path / "kb", tmp_path / "benchmark.json", local_dir=source, num_samples=1)


def test_prepare_rejects_lfs_pointer(tmp_path: Path) -> None:
    source = _checkout(tmp_path)
    (source / "pdfs" / "BETA_2022_10K.pdf").write_text("version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError, match="Git LFS pointers"):
        financebench.prepare(tmp_path / "kb", tmp_path / "benchmark.json", local_dir=source, num_samples=1)
