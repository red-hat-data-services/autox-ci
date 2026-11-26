"""MultiHop-RAG conversion preserves the article corpus and gold evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autorag_benchmark.datasets import get, multihop_rag
from scripts import generate_rag_datasets


def _fixture(local_dir: Path) -> None:
    local_dir.mkdir()
    articles = [
        {"url": f"https://example.org/{i}", "title": f"Article {i}", "source": "Example",
         "published_at": "2023-10-01", "author": "Author", "category": "news", "body": f"Body {i}"}
        for i in range(3)
    ]
    questions = [
        {"query": "Combine these?", "answer": "Both", "question_type": "inference_query",
         "evidence_list": [{"url": articles[0]["url"]}, {"url": articles[1]["url"]}]},
        {"query": "Unknown?", "answer": "Insufficient information.", "question_type": "null_query",
         "evidence_list": []},
        {"query": "Third?", "answer": "Three", "question_type": "inference_query",
         "evidence_list": [{"url": articles[2]["url"]}]},
    ]
    (local_dir / "corpus.json").write_text(json.dumps(articles), encoding="utf-8")
    (local_dir / "MultiHopRAG.json").write_text(json.dumps(questions), encoding="utf-8")


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_prepare_preserves_full_corpus_and_multihop_evidence(tmp_path: Path, output_format: str) -> None:
    source = tmp_path / "source"
    _fixture(source)
    kb = tmp_path / "knowledge_base"
    bench = tmp_path / "benchmark_data.json"

    prepare, _ = get("multihop_rag")
    assert prepare(kb, bench, local_dir=source, num_samples=2, output_format=output_format) == (3, 2)

    records = json.loads(bench.read_text(encoding="utf-8"))
    assert records[0]["correct_answers"] == ["Both"]
    assert records[0]["correct_answer_document_keys"] == [
        multihop_rag._filename("https://example.org/0", output_format),
        multihop_rag._filename("https://example.org/1", output_format),
    ]
    assert records[1]["question"] == "Third?"
    assert records[1]["correct_answer_document_keys"] == [
        multihop_rag._filename("https://example.org/2", output_format)
    ]
    assert all(record["correct_answer_document_keys"] for record in records)
    assert {p.name for p in kb.iterdir()} == {
        multihop_rag._filename(f"https://example.org/{i}", output_format) for i in range(3)
    }
    assert "Body 0" in (kb / records[0]["correct_answer_document_keys"][0]).read_text()


def test_prepare_rejects_unresolvable_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fixture(source)
    questions_path = source / "MultiHopRAG.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions[0]["evidence_list"][0]["url"] = "https://example.org/missing"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    with pytest.raises(ValueError, match="references missing article"):
        multihop_rag.prepare(tmp_path / "kb", tmp_path / "bench.json", local_dir=source, num_samples=1)
    assert not (tmp_path / "bench.json").exists()


def test_prepare_rejects_more_questions_than_have_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fixture(source)
    with pytest.raises(ValueError, match="2 questions with evidence"):
        multihop_rag.prepare(tmp_path / "kb", tmp_path / "bench.json", local_dir=source, num_samples=3)


def test_cli_generates_local_dataset_and_manifest_hint(tmp_path: Path, monkeypatch, capsys) -> None:
    source = tmp_path / "source"
    _fixture(source)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "generate_rag_datasets.py", "--dataset", "multihop_rag",
        "--multihop-rag-dir", str(source), "--num-samples", "2",
        "--output-dir", str(output),
    ])

    generate_rag_datasets.main()

    assert len(list((output / "knowledge_base").iterdir())) == 3
    assert len(json.loads((output / "benchmark_data.json").read_text())) == 2
    assert "multihop-rag-2-txt" in capsys.readouterr().out
