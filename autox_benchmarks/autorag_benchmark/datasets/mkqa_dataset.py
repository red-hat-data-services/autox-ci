"""MKQA dataset provider for multilingual RAG benchmarking.

Uses the apple/mkqa dataset (10K parallel questions in 26 languages) combined
with Wikipedia article fetching via Wikidata entity links. For each question
whose answer is a Wikidata entity, the corresponding full Wikipedia article is
fetched in the target language — giving truly parallel long documents across
all 26 languages.

Supported languages: ar, da, de, en, es, fi, fr, he, hu, it, ja, km, ko,
ms, nl, no, pl, pt, ru, sv, th, tr, vi, zh_cn, zh_hk, zh_tw.
"""

from __future__ import annotations

import gzip
import json
import re
import tempfile
import time
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document, get_file_extension

MKQA_DATA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"

SUPPORTED_LANGUAGES = [
    "ar", "da", "de", "en", "es", "fi", "fr", "he", "hu", "it", "ja",
    "km", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr",
    "vi", "zh_cn", "zh_hk", "zh_tw",
]

LANG_TO_WIKI = {
    "ar": "ar", "da": "da", "de": "de", "en": "en", "es": "es",
    "fi": "fi", "fr": "fr", "he": "he", "hu": "hu", "it": "it",
    "ja": "ja", "km": "km", "ko": "ko", "ms": "ms", "nl": "nl",
    "no": "no", "pl": "pl", "pt": "pt", "ru": "ru", "sv": "sv",
    "th": "th", "tr": "tr", "vi": "vi",
    "zh_cn": "zh", "zh_hk": "zh", "zh_tw": "zh",
}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKI_API_TMPL = "https://{lang}.wikipedia.org/w/api.php"
REQUEST_DELAY = 0.5


def _safe_docid(entity_id: str, lang: str) -> str:
    safe = re.sub(r"[^\w\-.]", "_", f"mkqa_{lang}_{entity_id}")
    return safe[:120] or "doc"


def _build_session():
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "AutoRAG-Benchmark/0.1 (dataset generation)"
    })
    return session


def _get_wiki_title(session, entity_id: str, wiki_lang: str) -> str | None:
    """Get Wikipedia article title for a Wikidata entity in a given language."""
    try:
        resp = session.get(WIKIDATA_API, params={
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "sitelinks",
            "format": "json",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        sitelinks = data.get("entities", {}).get(entity_id, {}).get("sitelinks", {})
        wiki_key = f"{wiki_lang}wiki"
        if wiki_key in sitelinks:
            return sitelinks[wiki_key].get("title")
    except Exception:
        pass
    return None


def _get_wiki_article(session, title: str, wiki_lang: str) -> str | None:
    """Fetch full plain-text Wikipedia article."""
    try:
        resp = session.get(WIKI_API_TMPL.format(lang=wiki_lang), params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": "true",
            "format": "json",
        }, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 100:
                return extract
    except Exception:
        pass
    return None


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    language: str = "en",
    output_format: str = "txt",
    **_: object,
) -> tuple[int, int]:
    """Download MKQA, fetch Wikipedia articles, write kb docs + benchmark JSON.

    Returns (num_docs, num_entries).
    """
    import requests

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Available: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )

    wiki_lang = LANG_TO_WIKI[language]

    cache_dir = Path(tempfile.gettempdir()) / "mkqa_cache"
    cache_path = cache_dir / "mkqa.jsonl.gz"

    if not cache_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading MKQA data...")
        resp = requests.get(MKQA_DATA_URL, timeout=120)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        print(f"  Downloaded {len(resp.content) / 1024 / 1024:.1f} MB")
    else:
        print("Using cached MKQA data...")

    print("Parsing MKQA and filtering entity-type answers...")
    rows: list[dict] = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            answers = row.get("answers", {}).get(language, [])
            if not answers:
                continue
            ans = answers[0]
            if (
                ans.get("type") == "entity"
                and re.match(r"^Q\d+$", ans.get("entity", ""))
                and ans.get("text", "").strip()
            ):
                rows.append(row)

    print(f"  Found {len(rows)} entity-type questions for '{language}'")

    # Collect unique entities needed
    entity_ids: set[str] = set()
    for row in rows:
        entity_ids.add(row["answers"][language][0]["entity"])

    print(f"  Unique Wikidata entities: {len(entity_ids)}")

    # Phase 1: Fetch Wikipedia articles for unique entities
    session = _build_session()
    entity_docs: dict[str, tuple[str, str]] = {}  # entity_id -> (title, text)
    fetched = 0
    failed = 0

    print(f"Fetching Wikipedia articles in '{wiki_lang}'...")
    for eid in entity_ids:
        if len(entity_docs) >= num_samples:
            break

        title = _get_wiki_title(session, eid, wiki_lang)
        if not title:
            failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        time.sleep(REQUEST_DELAY)

        text = _get_wiki_article(session, title, wiki_lang)
        if not text:
            failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        entity_docs[eid] = (title, text)
        fetched += 1

        if fetched % 10 == 0:
            print(f"  Fetched {fetched} articles (failed: {failed})")

        time.sleep(REQUEST_DELAY)

    print(f"  Total fetched: {len(entity_docs)}, failed: {failed}")

    # Phase 2: Write documents and build benchmark entries
    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    written_docs: dict[str, str] = {}  # entity_id -> local_filename
    processed = 0

    for row in rows:
        if processed >= num_samples:
            break

        ans = row["answers"][language][0]
        eid = ans["entity"]

        if eid not in entity_docs:
            continue

        # Get translated question
        queries = row.get("queries", {})
        question = queries.get(language, row.get("query", "")).strip()
        if not question:
            continue

        # Write document if not yet written
        if eid not in written_docs:
            title, article_text = entity_docs[eid]
            safe_id = _safe_docid(eid, language)
            file_ext = get_file_extension(output_format)
            local_filename = f"{safe_id}{file_ext}"

            content = f"{title}\n\n{article_text}"
            save_document(
                content=content,
                output_path=kb_dir / safe_id,
                format=output_format,
                metadata={
                    "source": "mkqa",
                    "language": language,
                    "entity_id": eid,
                    "title": title,
                },
            )
            written_docs[eid] = local_filename

        # Build answer list: primary text + aliases
        correct_answers = [ans["text"].strip()]
        for alias in ans.get("aliases", []):
            alias = alias.strip()
            if alias and alias not in correct_answers:
                correct_answers.append(alias)

        benchmark_data.append({
            "question": question,
            "correct_answers": correct_answers,
            "correct_answer_document_ids": [written_docs[eid]],
        })
        processed += 1

        if processed % 10 == 0:
            print(f"  Built {processed}/{num_samples} benchmark entries")

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4, ensure_ascii=False)

    print(f"\nMKQA ({language}) generation complete:")
    print(f"  Benchmark entries: {len(benchmark_data)}")
    print(f"  Unique documents: {len(written_docs)}")
    print(f"  Format: {output_format}")

    return len(written_docs), len(benchmark_data)


register("mkqa", prepare, {
    "num_samples": 50,
    "language": "en",
})
