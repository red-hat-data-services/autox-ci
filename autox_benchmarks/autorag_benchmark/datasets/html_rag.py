"""HtmlRag dataset provider using Google Natural Questions.

Downloads native Wikipedia HTML pages paired with questions and short answers
from the Natural Questions dataset (google-research-datasets/natural_questions)
via HuggingFace streaming to avoid the full 45GB download.

Each document is a complete Wikipedia HTML page. Questions come with
human-annotated short answers extracted from the page.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from autorag_benchmark.datasets import register

NQ_DATASET_ID = "google-research-datasets/natural_questions"


def _safe_filename(title: str) -> str:
    """Convert a Wikipedia title to a filesystem-safe filename."""
    safe = re.sub(r"[^\w\s\-]", "", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    safe = safe[:120]
    if not safe:
        safe = "untitled"
    return safe


def _extract_short_answers(row: dict) -> list[str]:
    """Extract unique short answer texts from all annotations.

    HuggingFace returns annotations as a columnar dict:
      annotations = {"id": [...], "short_answers": [sa0, sa1, ...], ...}
    Each sa is also columnar: {"text": ["ans1", "ans2"], ...}
    """
    answers: list[str] = []
    seen: set[str] = set()
    annotations = row.get("annotations", {})
    short_answers_list = annotations.get("short_answers", [])
    for sa in short_answers_list:
        texts = sa.get("text", [])
        for t in texts:
            t = t.strip()
            if t and t not in seen:
                answers.append(t)
                seen.add(t)
    return answers


def _doc_fingerprint(doc_id: str) -> str:
    """Short hash for dedup when title-based filename collides."""
    return hashlib.md5(doc_id.encode()).hexdigest()[:8]


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    split: str = "validation",
    **_: object,
) -> tuple[int, int]:
    """Stream Natural Questions and write native HTML docs + benchmark JSON.

    Returns (num_docs, num_entries).
    """
    from datasets import load_dataset

    print(f"Streaming Natural Questions ({split} split) for up to {num_samples} samples...")

    ds = load_dataset(NQ_DATASET_ID, split=split, streaming=True)

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    written_docs: dict[str, str] = {}  # doc_url -> filename
    processed = 0
    skipped = 0

    for row in ds:
        if processed >= num_samples:
            break

        doc = row.get("document", {})
        html_content = doc.get("html", "")
        doc_title = doc.get("title", "")
        doc_url = doc.get("url", "")
        question_text = row.get("question", {}).get("text", "").strip()

        if not html_content or not question_text:
            skipped += 1
            continue

        answers = _extract_short_answers(row)
        if not answers:
            skipped += 1
            continue

        # Deduplicate documents by URL
        if doc_url in written_docs:
            doc_filename = written_docs[doc_url]
        else:
            base = _safe_filename(doc_title) if doc_title else "nq_doc"
            doc_filename = f"nq_{base}_{_doc_fingerprint(doc_url)}.html"

            # Write native HTML
            output_path = kb_dir / doc_filename
            if not output_path.exists():
                output_path.write_text(html_content, encoding="utf-8")

            written_docs[doc_url] = doc_filename

        benchmark_data.append({
            "question": question_text,
            "correct_answers": answers,
            "correct_answer_document_ids": [doc_filename],
        })
        processed += 1

        if processed % 10 == 0:
            print(f"  Processed {processed}/{num_samples} (skipped {skipped}, unique docs: {len(written_docs)})")

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nHtmlRag (Natural Questions) generation complete:")
    print(f"  Benchmark entries: {len(benchmark_data)}")
    print(f"  Unique HTML documents: {len(written_docs)}")
    print(f"  Skipped entries: {skipped}")

    return len(written_docs), len(benchmark_data)


register("html_rag", prepare, {"num_samples": 50, "split": "validation"})
