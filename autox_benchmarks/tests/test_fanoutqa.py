"""FanOutQA selection and corpus conversion."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from autorag_benchmark.datasets import fanoutqa
from scripts import generate_rag_datasets


def test_prepare_keeps_full_corpus_and_only_answer_supported_questions(tmp_path: Path, monkeypatch, capsys) -> None:
    pages = [{"pageid": i, "revid": 100 + i} for i in range(1, 6)]
    rows = [
        {"id": "excluded", "question": "What are the current values?", "answer": {"a": 1, "b": 2, "c": 3, "d": 4},
         "decomposition": [{"evidence": p} for p in pages]},
        {"id": "included", "question": "What are all five values?", "answer": {f"Page {i}": i for i in range(1, 6)},
         "decomposition": [{"evidence": p} for p in pages]},
    ]
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    arrow = tmp_path / "corpus.arrow"
    corpus = pa.Table.from_pylist([
        {"doc_id": str(i), "title": f"Page {i}", "url": f"https://example.com/{i}",
         "text": f"Value for page {i}: {i}. " * 30} for i in range(1, 7)
    ])
    with arrow.open("wb") as output:
        with pa.ipc.new_stream(output, corpus.schema) as writer:
            writer.write_table(corpus)
    kb_dir = tmp_path / "knowledge_base"
    bench_path = tmp_path / "benchmark_data.json"
    assert fanoutqa.prepare(kb_dir, bench_path, source_json=source,
                            corpus_arrow=arrow, num_samples=1) == (6, 1)
    benchmark = json.loads(bench_path.read_text())
    assert benchmark[0]["question"] == "What are all five values?"
    assert benchmark[0]["correct_answers"] == ["Page 1: 1; Page 2: 2; Page 3: 3; Page 4: 4; Page 5: 5"]
    assert len(benchmark[0]["correct_answer_document_keys"]) == 5
    assert all((kb_dir / name).is_file() for name in benchmark[0]["correct_answer_document_keys"])
    assert (kb_dir / "fanoutqa_6.txt").is_file()  # Distractor page.

    output = tmp_path / "cli_output"
    monkeypatch.setattr(sys, "argv", [
        "generate_rag_datasets.py", "--dataset", "fanoutqa",
        "--fanoutqa-dev-json", str(source), "--fanoutqa-corpus-arrow", str(arrow),
        "--num-samples", "1", "--output-dir", str(output),
    ])
    generate_rag_datasets.main()
    assert len(list((output / "knowledge_base").iterdir())) == 6
    assert len(json.loads((output / "benchmark_data.json").read_text())) == 1
    assert "fanoutqa-1-txt" in capsys.readouterr().out


def test_select_rejects_empty_gold_page() -> None:
    row = {"id": "q", "question": "Combine these five pages?",
           "answer": {f"Page {i}": i for i in range(1, 6)},
           "decomposition": [{"evidence": {"pageid": i, "revid": i + 100}} for i in range(1, 6)]}
    corpus = {str(i): {"text": (f"Page {i}: {i}. " * 30) if i != 3 else ""}
              for i in range(1, 6)}
    with pytest.raises(ValueError, match="Only 0 FanOutQA questions"):
        fanoutqa._select([row], corpus, 1)
