"""EnterpriseRAG-Bench (onyx-dot-app) dataset provider.

Enterprise-scale RAG benchmark with ~500K documents across 9 source types
(Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira, Confluence)
and 500 questions spanning 10 categories.

Source: https://github.com/onyx-dot-app/EnterpriseRAG-Bench
HuggingFace: onyx-dot-app/EnterpriseRAG-Bench

Identity-preserving design (required for the official onyx leaderboard)
----------------------------------------------------------------------
The official onyx evaluator scores an answers file whose ``document_ids`` must be
the original ``dsid_<hex>`` document UUIDs. To keep that identity intact all the
way through indexing and retrieval, this provider:

  * names every knowledge-base file ``{dsid}.{ext}`` (the ``dsid_`` UUID is the
    filename stem), and
  * sets ``correct_answer_document_ids`` to those KB filenames (``{dsid}.{ext}``).

ai4rag's in-pipeline ``context_correctness`` metric compares the retrieved chunk's
``metadata["document_id"]`` — which is the full source filename *including* the
extension (e.g. ``dsid_<hex>.txt``) — verbatim against
``correct_answer_document_ids``. Storing bare ``dsid_`` values here makes that
overlap empty and pins ``context_correctness`` to 0.0, so the gold IDs must carry
the extension too. The raw ``dsid_`` UUIDs are still preserved in the
``selected_questions.jsonl`` sidecar (``expected_doc_ids``) for Phase 2b, which
recovers them by stripping the extension from each retrieved filename.

Data source
-----------
Builds directly from a local checkout of the onyx corpus (fast, no download):
a ``generated_data`` directory containing ``uuid_index.json`` (dsid -> relative
path), ``sources/`` (per-document JSON), and ``questions.jsonl`` (500 questions).
Resolution order for that directory:

  1. explicit ``local_dir`` option
  2. ``ENTERPRISE_RAGBENCH_DIR`` env var
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from autorag_benchmark.datasets import register

SOURCE_TYPES = [
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
]

QUESTION_CATEGORIES = [
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "miscellaneous",
    "high_level",
    "info_not_found",
]

# Deterministic distractor sampling so slices are reproducible run-to-run.
_RANDOM_SEED = 42


class EnterpriseRAGBenchError(RuntimeError):
    """Raised when the local corpus cannot be located or is malformed."""


# --------------------------------------------------------------------------- #
# Corpus location + document content extraction
# --------------------------------------------------------------------------- #
def _resolve_corpus_dir(local_dir: str | Path | None) -> Path:
    """Find the onyx ``generated_data`` directory (with uuid_index.json)."""
    candidates: list[Path] = []
    if local_dir:
        candidates.append(Path(local_dir))
    env_dir = os.environ.get("ENTERPRISE_RAGBENCH_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    for cand in candidates:
        if (cand / "uuid_index.json").is_file():
            return cand
        # Allow pointing at the repo root instead of generated_data.
        if (cand / "generated_data" / "uuid_index.json").is_file():
            return cand / "generated_data"

    if not candidates:
        raise EnterpriseRAGBenchError(
            "Could not locate the EnterpriseRAG-Bench corpus. No search paths "
            "configured — set the ENTERPRISE_RAGBENCH_DIR env var or pass "
            "--erb-local-dir pointing to the onyx 'generated_data' directory."
        )
    raise EnterpriseRAGBenchError(
        "Could not locate the EnterpriseRAG-Bench corpus. Expected a directory "
        "containing uuid_index.json (the onyx 'generated_data' dir). Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _extract_document_content(doc_data: dict) -> tuple[str, str]:
    """Extract (title, content) using the onyx field-label convention.

    Mirrors onyx ``src/utils/document_content.py`` so this provider does not
    depend on the onyx package being importable.
    """
    title_field = doc_data.get("title_field_name")
    if not title_field or title_field not in doc_data:
        raise EnterpriseRAGBenchError(
            f"Document missing valid title_field_name ({title_field!r})"
        )
    title = str(doc_data[title_field])

    content_fields = doc_data.get("content_field_names")
    if not isinstance(content_fields, list) or not content_fields:
        raise EnterpriseRAGBenchError("Document missing non-empty content_field_names")
    for field in content_fields:
        if field not in doc_data:
            raise EnterpriseRAGBenchError(f"content_field_name '{field}' not in document")

    if len(content_fields) == 1:
        content = str(doc_data[content_fields[0]])
    else:
        parts = []
        for field in content_fields:
            value = doc_data[field]
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            parts.append(f"{field}:\n{value}")
        content = "\n\n".join(parts)

    return title, content


def _write_kb_doc(
    kb_dir: Path, dsid: str, title: str, content: str, output_format: str
) -> str:
    """Write one KB file named ``{dsid}.{ext}``; return the filename."""
    ext = ".md" if output_format == "md" else ".txt"
    filename = f"{dsid}{ext}"
    out_path = kb_dir / filename
    if output_format == "md":
        escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
        body = (
            f"---\n"
            f"dataset_doc_uuid: {dsid}\n"
            f'title: "{escaped_title}"\n'
            f"---\n\n{content}"
        )
    else:
        body = f"{title}\n\n{content}" if title else content
    out_path.write_text(body, encoding="utf-8")
    return filename


# --------------------------------------------------------------------------- #
# Question selection
# --------------------------------------------------------------------------- #
def _load_questions(corpus_dir: Path) -> list[dict]:
    """Load questions.jsonl (from generated_data dir or its repo root)."""
    for path in (corpus_dir / "questions.jsonl", corpus_dir.parent / "questions.jsonl"):
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise EnterpriseRAGBenchError(
        f"questions.jsonl not found near {corpus_dir}"
    )


def _select_questions(
    questions: list[dict],
    *,
    source_type: str,
    question_category: str | None,
    num_samples: int | None,
    include_all: bool,
) -> list[dict]:
    if include_all:
        return list(questions)

    matched = []
    for q in questions:
        if question_category and q.get("question_type", "").lower() != question_category.lower():
            continue
        if source_type and source_type != "all":
            q_sources = q.get("source_types", [])
            if isinstance(q_sources, str):
                q_sources = [q_sources]
            if source_type not in q_sources:
                continue
        matched.append(q)

    rng = random.Random(_RANDOM_SEED)
    rng.shuffle(matched)
    if num_samples and num_samples > 0:
        matched = matched[:num_samples]
    return matched


# --------------------------------------------------------------------------- #
# Provider entry point
# --------------------------------------------------------------------------- #
def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 25,
    source_type: str = "all",
    question_category: str | None = None,
    include_all: bool = False,
    output_format: str = "txt",
    distractor_docs: int = 500,
    local_dir: str | Path | None = None,
    **_: object,
) -> tuple[int, int]:
    """Build a dsid-preserving KB + benchmark JSON from the local onyx corpus.

    Args:
        kb_dir: Directory to write knowledge-base documents (``{dsid}.{ext}``).
        bench_path: Path to write ``benchmark_data.json`` (ai4rag schema).
        num_samples: Max questions to select (ignored when ``include_all``).
        source_type: Filter selected questions to this onyx source type, or
            "all" for no filter.
        question_category: Optional filter by onyx ``question_type``.
        include_all: Include all 500 questions and the FULL corpus (all docs).
            Overrides ``num_samples``, ``source_type`` and ``distractor_docs``.
        output_format: "txt" (default) or "md".
        distractor_docs: Number of non-gold documents to add to the KB so
            retrieval is a real task during HPO (seeded sample; drawn from the
            same source types as the selected questions when a filter is set).
            Ignored when ``include_all``.
        local_dir: Explicit path to the onyx ``generated_data`` directory.

    Returns:
        (number of documents written, number of benchmark entries)
    """
    corpus_dir = _resolve_corpus_dir(local_dir)
    sources_dir = corpus_dir / "sources"
    uuid_index: dict[str, str] = json.loads(
        (corpus_dir / "uuid_index.json").read_text(encoding="utf-8")
    )

    print(f"Building EnterpriseRAG-Bench from local corpus: {corpus_dir}")
    print(f"  uuid_index entries: {len(uuid_index)}")

    questions = _load_questions(corpus_dir)
    print(f"  questions.jsonl rows: {len(questions)}")

    selected = _select_questions(
        questions,
        source_type=source_type,
        question_category=question_category,
        num_samples=num_samples,
        include_all=include_all,
    )
    print(f"  selected questions: {len(selected)}")
    if not selected:
        raise EnterpriseRAGBenchError("No questions matched the given filters")

    # Gold documents referenced by the selected questions.
    gold_dsids: set[str] = set()
    for q in selected:
        for did in q.get("expected_doc_ids") or []:
            gold_dsids.add(did)
    print(f"  gold documents referenced: {len(gold_dsids)}")

    # Determine the full KB document set.
    if include_all:
        kb_dsids = list(uuid_index.keys())
    else:
        kb_set = set(gold_dsids)
        if distractor_docs > 0:
            # Candidate distractors: prefer same source types as the selection
            # (path prefix in uuid_index is "<source_type>/..."), else global.
            wanted_prefixes = None
            if source_type and source_type != "all":
                wanted_prefixes = (f"{source_type}/",)
            candidates = [
                did
                for did, rel in uuid_index.items()
                if did not in gold_dsids
                and (wanted_prefixes is None or rel.startswith(wanted_prefixes))
            ]
            rng = random.Random(_RANDOM_SEED)
            rng.shuffle(candidates)
            kb_set.update(candidates[:distractor_docs])
        kb_dsids = list(kb_set)

    print(f"  total KB documents to write: {len(kb_dsids)}")

    # Write KB files, mapping each written dsid -> its ``{dsid}.{ext}`` filename.
    kb_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    skipped_docs = 0
    for i, dsid in enumerate(kb_dsids, 1):
        rel = uuid_index.get(dsid)
        if not rel:
            skipped_docs += 1
            continue
        doc_path = sources_dir / rel
        try:
            doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
            title, content = _extract_document_content(doc_data)
        except (OSError, json.JSONDecodeError, EnterpriseRAGBenchError) as exc:
            skipped_docs += 1
            if skipped_docs <= 5:
                print(f"    skip {dsid}: {exc}")
            continue
        if not content.strip():
            skipped_docs += 1
            continue
        written[dsid] = _write_kb_doc(kb_dir, dsid, title, content, output_format)
        if i % 5000 == 0:
            print(f"    written {len(written)}/{i} docs...")

    print(f"  documents written: {len(written)} (skipped {skipped_docs})")

    # Build ai4rag benchmark_data.json (frozen 3-field schema).
    benchmark_data: list[dict] = []
    selected_meta: list[dict] = []
    skipped_questions = 0
    for q in selected:
        # Map gold dsids to their written KB filenames (``{dsid}.{ext}``) so the
        # IDs match what the retriever reports for context_correctness scoring.
        expected = q.get("expected_doc_ids") or []
        gold = [written[d] for d in expected if d in written]
        if expected and not gold:
            skipped_questions += 1
            continue
        gold_answer = q.get("gold_answer") or ""
        benchmark_data.append(
            {
                "question": q.get("question", ""),
                "correct_answers": [gold_answer] if gold_answer else [],
                "correct_answer_document_ids": gold,
            }
        )
        selected_meta.append(q)

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    # Sidecar: full onyx question rows for the selected slice, preserving
    # question_id / answer_facts / expected_doc_ids for Phase 2b/3.
    sidecar = bench_path.parent / "selected_questions.jsonl"
    with open(sidecar, "w", encoding="utf-8") as f:
        for q in selected_meta:
            f.write(json.dumps(q) + "\n")

    print("\nEnterpriseRAG-Bench generation complete:")
    print(f"  Documents written: {len(written)}")
    print(f"  Benchmark entries: {len(benchmark_data)}")
    print(f"  Skipped questions (gold docs missing): {skipped_questions}")
    print(f"  Sidecar: {sidecar}")
    print(f"  Format: {output_format}")

    return len(written), len(benchmark_data)


register(
    "enterprise_ragbench",
    prepare,
    {"num_samples": 25, "source_type": "all", "distractor_docs": 500},
)
