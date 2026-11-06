"""FanOutQA dev questions and a published full-page Wikipedia retrieval corpus.

Questions: https://github.com/zhudotexe/fanoutqa
Corpus: https://huggingface.co/datasets/JinChao1022/fanoutqa-retrieval
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import requests

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document

DEV_URL = "https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"
CORPUS_URL = "https://huggingface.co/datasets/JinChao1022/fanoutqa-retrieval/resolve/main/corpus/data-00000-of-00001.arrow?download=true"
TIME_WORDS = re.compile(r"\b(current|currently|latest|most recent|today|now)\b", re.I)


def _evidence(row: dict) -> list[str]:
    """Return distinct source page IDs in decomposition order."""
    pages: dict[str, None] = {}
    for part in row.get("decomposition", []):
        evidence = part.get("evidence") or {}
        pageid = str(evidence.get("pageid") or "")
        if pageid.isdigit() and evidence.get("revid"):
            pages[pageid] = None
    return list(pages)


def _answer_text(answer: dict | list) -> str:
    if isinstance(answer, dict):
        return "; ".join(f"{key}: {value}" for key, value in answer.items())
    return "; ".join(str(value) for value in answer)


def _select(rows: list[dict], corpus: dict[str, dict], num_samples: int) -> list[tuple[dict, list[str]]]:
    selected = []
    for row in rows:
        ids = _evidence(row)
        answer = row.get("answer")
        question = row.get("question")
        if not (isinstance(question, str) and question.strip()
                and not TIME_WORDS.search(question)
                and isinstance(answer, (dict, list)) and 4 <= len(answer) <= 10
                and 5 <= len(ids) <= 10
                and all(pageid in corpus and len(corpus[pageid]["text"].strip()) >= 200
                        for pageid in ids)):
            continue
        # The published corpus can differ from the dated source revisions. Keep
        # only questions whose entire reference answer remains in its evidence.
        evidence_text = " ".join(corpus[pageid]["text"].casefold() for pageid in ids)
        values = answer.values() if isinstance(answer, dict) else answer
        if not all(str(value).casefold() in evidence_text for value in values):
            continue
        selected.append((row, ids))
        if len(selected) == num_samples:
            break
    if len(selected) != num_samples:
        raise ValueError(f"Only {len(selected)} FanOutQA questions have 5–10 available evidence pages with all answer values; requested {num_samples}")
    return selected


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=300, stream=True, headers={"User-Agent": "AutoRAGBenchmark/1.0"})
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            output.write(chunk)
    temporary.replace(path)


def _corpus(path: Path) -> dict[str, dict]:
    import pyarrow as pa

    table = pa.ipc.open_stream(str(path)).read_all()
    if not {"doc_id", "title", "url", "text"} <= set(table.column_names):
        raise ValueError("FanOutQA corpus Arrow file has unexpected columns")
    records = table.to_pylist()
    corpus = {str(row["doc_id"]): row for row in records}
    if len(corpus) != len(records):
        raise ValueError("Duplicate FanOutQA corpus document ID")
    return corpus


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 100,
    output_format: str = "txt",
    source_json: Path | None = None,
    corpus_arrow: Path | None = None,
    **_: object,
) -> tuple[int, int]:
    """Generate deterministic questions and the full 1,594-page corpus."""
    if output_format not in {"txt", "md"}:
        raise ValueError("FanOutQA supports only txt and md output formats")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    cache_dir = bench_path.parent / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_cache = cache_dir / "fanout-final-dev.json"
    corpus_cache = cache_dir / "fanoutqa_corpus.arrow"
    if source_json is not None:
        shutil.copyfile(source_json, source_cache)
    elif not source_cache.is_file():
        _download(DEV_URL, source_cache)
    if corpus_arrow is not None:
        shutil.copyfile(corpus_arrow, corpus_cache)
    elif not corpus_cache.is_file():
        _download(CORPUS_URL, corpus_cache)
    rows = json.loads(source_cache.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("FanOutQA dev source must be a JSON array")
    corpus = _corpus(corpus_cache)
    selected = _select(rows, corpus, num_samples)

    kb_dir.mkdir(parents=True, exist_ok=True)
    for count, (pageid, page) in enumerate(corpus.items(), 1):
        filename = f"fanoutqa_{pageid}.{output_format}"
        content = f"Title: {page['title']}\nSource: {page['url']}\n\n{page['text']}\n"
        save_document(content, kb_dir / filename, format=output_format,
                      metadata={"title": page["title"]})
        if count % 250 == 0 or count == len(corpus):
            print(f"Wrote {count}/{len(corpus)} FanOutQA pages", flush=True)

    benchmark = []
    for row, ids in selected:
        keys = [f"fanoutqa_{pageid}.{output_format}" for pageid in ids]
        benchmark.append({"question": row["question"].strip(),
                          "correct_answers": [_answer_text(row["answer"])],
                          "correct_answer_document_keys": keys})
    bench_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(corpus), len(benchmark)


register("fanoutqa", prepare, {"num_samples": 100})
