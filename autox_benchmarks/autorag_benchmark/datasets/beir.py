"""BEIR benchmark → knowledge base + project benchmark JSON.

Uses the official BEIR zip layout and ``GenericDataLoader``. Install with
``pip install beir`` or ``pip install -r requirements.txt``.

See https://github.com/beir-cellar/beir for datasets and splits.

**SciFact scores:** Claims are yes/no–style scientific statements. If you set
``correct_answers`` to the raw abstract, LLM judges usually give very low
**answer_correctness** (the model answers in prose, not by pasting the abstract)
even when **context_correctness** is high. When ``queries.jsonl`` includes a
``SUPPORT`` / ``CONTRADICT`` label for the gold document, we encode that as
short reference answers so metrics align better with claim verification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document, get_file_extension


def _safe_filename_part(s: str) -> str:
    """Filesystem-safe fragment (BEIR doc ids may contain odd characters)."""
    return re.sub(r"[^\w.\-]+", "_", str(s))[:200]


def _kb_basename(dataset_name: str, doc_id: str, file_ext: str = ".txt") -> str:
    return f"beir_{_safe_filename_part(dataset_name)}_{_safe_filename_part(doc_id)}{file_ext}"


def _corpus_passage(corpus_entry: dict) -> str:
    title = (corpus_entry.get("title") or "").strip()
    text = (corpus_entry.get("text") or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def _is_relevant(score: object) -> bool:
    try:
        return float(score) > 0
    except (TypeError, ValueError):
        return False


def _load_queries_jsonl_metadata(data_folder: Path) -> dict[str, dict]:
    """Map query id -> ``metadata`` dict from BEIR ``queries.jsonl`` (SciFact has evidence labels)."""
    path = data_folder / "queries.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row.get("_id", ""))
        meta = row.get("metadata")
        if qid and isinstance(meta, dict):
            out[qid] = meta
    return out


def _stance_for_gold_doc(query_metadata: dict, corpus_doc_id: str) -> str | None:
    """Return SciFact evidence label (e.g. SUPPORT, CONTRADICT) for ``corpus_doc_id`` if present."""
    if not query_metadata:
        return None
    raw = query_metadata.get(corpus_doc_id)
    if raw is None and corpus_doc_id.isdigit():
        raw = query_metadata.get(str(int(corpus_doc_id)))
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    if not isinstance(first, dict):
        return None
    label = first.get("label")
    return str(label).strip().upper() if label else None


def _correct_answers_for_scifact_stance(stance: str | None, passage_fallback: str) -> list[str]:
    """Reference answers suited to claim-style questions (short phrases + passage fallback)."""
    if stance == "SUPPORT":
        return [
            "The evidence supports the claim.",
            "Yes; the cited abstract supports this claim.",
        ]
    if stance in ("CONTRADICT", "REFUTE", "CONTRADICTION"):
        return [
            "The evidence contradicts the claim.",
            "No; the cited abstract contradicts this claim.",
        ]
    return [passage_fallback]


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    beir_dataset: str = "scifact",
    split: str = "test",
    download_root: Path | None = None,
    output_format: str = "txt",
    **_: object,
) -> tuple[int, int]:
    """Download a BEIR dataset, write passages under ``kb_dir``, benchmark JSON.

    Args:
        kb_dir: Directory to write knowledge base documents
        bench_path: Path to write benchmark JSON file
        num_samples: Number of samples to generate
        beir_dataset: BEIR dataset name (e.g., "scifact", "nfcorpus")
        split: Dataset split (e.g., "train", "test", "dev")
        download_root: Directory to cache BEIR downloads
        output_format: Output format for documents - "txt" or "md" (default: "txt")
                      Note: BEIR provides text data only. PDF format is not supported.

    Returns:
        (number of unique documents written this run, benchmark row count).
    """
    # Validate format for BEIR
    if output_format == "pdf":
        raise ValueError(
            "BEIR dataset does not support PDF format (text-only dataset). "
            "Use --output-format txt or --output-format md. "
            "For native PDFs, use the OpenRAGBench dataset."
        )
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    dataset = beir_dataset.strip().lower()
    if not dataset:
        raise ValueError("beir_dataset must be non-empty")

    if download_root is None:
        download_root = kb_dir.resolve().parent / "beir_datasets"
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    print(f"BEIR: fetching {dataset} (cached under {download_root})…")
    data_path_str = util.download_and_unzip(url, str(download_root))
    data_folder = Path(data_path_str)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(split=split)
    query_meta = _load_queries_jsonl_metadata(data_folder)

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    docs_written: set[str] = set()
    file_ext = get_file_extension(output_format)

    for qid in sorted(qrels.keys(), key=str):
        if len(benchmark_data) >= num_samples:
            break
        qkey = str(qid)
        rel = qrels[qid] or {}
        gold_ids = [did for did, score in rel.items() if _is_relevant(score)]
        if not gold_ids:
            continue
        question = queries.get(qid) or queries.get(qkey)
        if not question or not str(question).strip():
            continue

        basenames: list[str] = []
        ok = True
        bodies: list[str] = []
        for did in gold_ids:
            if did not in corpus:
                ok = False
                break
            body = _corpus_passage(corpus[did])
            if not body.strip():
                ok = False
                break
            bodies.append(body)
            basename = _kb_basename(dataset, did, file_ext)
            basenames.append(basename)

            # Save document in requested format
            doc_base = f"beir_{_safe_filename_part(dataset)}_{_safe_filename_part(did)}"
            out_path = kb_dir / doc_base
            if not (kb_dir / basename).exists():
                corpus_entry = corpus[did]
                title = (corpus_entry.get("title") or "").strip()
                save_document(
                    content=body,
                    output_path=out_path,
                    format=output_format,
                    metadata={
                        "source": "beir",
                        "dataset": dataset,
                        "doc_id": str(did),
                        "title": title or f"BEIR {dataset} Document {did}",
                    }
                )
            docs_written.add(basename)

        if not ok:
            continue

        preview = bodies[0]
        if len(preview) > 2000:
            preview = preview[:2000]

        meta_row = query_meta.get(qkey, {})
        primary_doc_id = str(gold_ids[0])
        stance = None
        if dataset == "scifact":
            stance = _stance_for_gold_doc(meta_row, primary_doc_id)
        answers = _correct_answers_for_scifact_stance(stance, preview)

        benchmark_data.append({
            "question": str(question).strip(),
            "correct_answers": answers,
            "correct_answer_document_ids": basenames,
        })

    if not benchmark_data:
        raise RuntimeError(
            f"No benchmark rows built for BEIR dataset {dataset!r} split={split!r}. "
            "Check the dataset name and split (e.g. scifact: train|test; nfcorpus: train|dev|test)."
        )

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    print(
        f"BEIR {dataset} ({split}): {len(benchmark_data)} questions, "
        f"{len(docs_written)} document files in kb."
    )
    return len(list(kb_dir.iterdir())), len(benchmark_data)


register(
    "beir",
    prepare,
    {"num_samples": 50, "beir_dataset": "scifact", "split": "test"},
)