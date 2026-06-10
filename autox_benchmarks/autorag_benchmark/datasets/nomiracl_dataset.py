"""NoMIRACL dataset provider for multilingual RAG benchmarking.

Downloads raw corpus, topics, and qrels files from the NoMIRACL repo
(miracl/nomiracl) via huggingface_hub and assembles them locally.
Uses the 'relevant' subset where each query has at least one judged
relevant passage.

Supported languages: ar, bn, de, en, es, fa, fi, fr, hi, id, ja, ko,
ru, sw, te, th, yo, zh.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document, get_file_extension

NOMIRACL_REPO_ID = "miracl/nomiracl"

LANG_CODE_TO_NAME = {
    "ar": "arabic",
    "bn": "bengali",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "ja": "japanese",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",
    "th": "thai",
    "yo": "yoruba",
    "zh": "chinese",
}


def _safe_docid(docid: str) -> str:
    """Sanitise a document ID for use as a filename."""
    safe = re.sub(r"[^\w\-.]", "_", docid)
    return safe[:120] or "doc"


def _load_corpus(filepath: str) -> dict[str, dict[str, str]]:
    """Load corpus.jsonl.gz into {docid: {text, title}}."""
    corpus: dict[str, dict[str, str]] = {}
    open_fn = gzip.open if filepath.endswith(".gz") else open
    with open_fn(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["docid"]] = {
                "text": row.get("text", "").strip(),
                "title": row.get("title", "").strip(),
            }
    return corpus


def _load_topics(filepath: str) -> dict[str, str]:
    """Load topics TSV into {qid: query_text}."""
    topics: dict[str, str] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= 2:
                topics[row[0]] = row[1]
    return topics


def _load_qrels(filepath: str) -> dict[str, dict[str, int]]:
    """Load qrels TSV into {qid: {docid: relevance}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
                qrels[qid][docid] = int(rel)
    return dict(qrels)


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    language: str = "en",
    split: str = "dev",
    output_format: str = "txt",
    **_: object,
) -> tuple[int, int]:
    """Download NoMIRACL raw files and write kb docs + benchmark JSON.

    Returns (num_docs, num_entries).
    """
    from huggingface_hub import hf_hub_download

    if language not in LANG_CODE_TO_NAME:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Available: {', '.join(sorted(LANG_CODE_TO_NAME))}"
        )

    lang_name = LANG_CODE_TO_NAME[language]
    subset = "relevant"
    data_prefix = f"data/{lang_name}"

    print(f"Downloading NoMIRACL files ({lang_name}, {split}.{subset})...")

    corpus_file = hf_hub_download(
        repo_id=NOMIRACL_REPO_ID,
        repo_type="dataset",
        filename=f"{data_prefix}/corpus.jsonl.gz",
    )
    topics_file = hf_hub_download(
        repo_id=NOMIRACL_REPO_ID,
        repo_type="dataset",
        filename=f"{data_prefix}/topics/{split}.{subset}.tsv",
    )
    qrels_file = hf_hub_download(
        repo_id=NOMIRACL_REPO_ID,
        repo_type="dataset",
        filename=f"{data_prefix}/qrels/{split}.{subset}.tsv",
    )

    print("Parsing corpus...")
    corpus = _load_corpus(corpus_file)
    topics = _load_topics(topics_file)
    qrels = _load_qrels(qrels_file)

    print(f"Corpus: {len(corpus)} docs, Topics: {len(topics)} queries, "
          f"Qrels: {len(qrels)} queries")

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    written_docs: dict[str, str] = {}
    processed = 0
    skipped = 0

    for qid, query_text in topics.items():
        if processed >= num_samples:
            break

        query_text = query_text.strip()
        if not query_text:
            skipped += 1
            continue

        query_qrels = qrels.get(qid, {})
        pos_docids = [did for did, rel in query_qrels.items() if rel == 1]
        if not pos_docids:
            skipped += 1
            continue

        doc_ids_for_query: list[str] = []
        answers_for_query: list[str] = []

        for docid in pos_docids:
            if docid not in corpus:
                continue

            doc = corpus[docid]
            text = doc["text"]
            title = doc["title"]

            if not text:
                continue

            if docid not in written_docs:
                safe_id = _safe_docid(docid)
                file_ext = get_file_extension(output_format)
                local_filename = f"nomiracl_{safe_id}{file_ext}"

                content = f"{title}\n\n{text}" if title else text

                save_document(
                    content=content,
                    output_path=kb_dir / f"nomiracl_{safe_id}",
                    format=output_format,
                    metadata={
                        "source": "nomiracl",
                        "language": language,
                        "doc_id": docid,
                        "title": title,
                    },
                )
                written_docs[docid] = local_filename

            doc_ids_for_query.append(written_docs[docid])
            answers_for_query.append(text)

        if not answers_for_query:
            skipped += 1
            continue

        benchmark_data.append({
            "question": query_text,
            "correct_answers": answers_for_query,
            "correct_answer_document_ids": doc_ids_for_query,
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

    print(f"\nNoMIRACL ({language}) generation complete:")
    print(f"  Benchmark entries: {len(benchmark_data)}")
    print(f"  Unique documents: {len(written_docs)}")
    print(f"  Skipped entries: {skipped}")
    print(f"  Format: {output_format}")

    return len(written_docs), len(benchmark_data)


register("nomiracl", prepare, {
    "num_samples": 50,
    "language": "en",
    "split": "dev",
})
