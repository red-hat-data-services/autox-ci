"""Open RAGBench (Vectara / Hugging Face) dataset provider."""

import json
import re
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document, get_file_extension

# Constants
OPEN_RAGBENCH_REPO_ID = "vectara/open_ragbench"
OPEN_RAGBENCH_SUBSET = "pdf/arxiv"


def extract_text_only(corpus_data: dict) -> str:
    """Extract only text content from corpus, excluding images and encodings."""
    text_parts = []

    def extract_from_value(value):
        """Recursively extract text from nested structures."""
        if isinstance(value, str):
            # Skip if it looks like base64 or data URI
            if value.startswith('data:image') or value.startswith('data:application'):
                return
            if len(value) > 100 and re.match(r'^[A-Za-z0-9+/=]{100,}$', value):
                # Likely base64 encoded data
                return
            # Skip URLs to images
            if re.match(r'^https?://.*\.(jpg|jpeg|png|gif|svg|webp|bmp)', value, re.I):
                return
            # Valid text content
            text_parts.append(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                # Skip keys that typically contain images/encodings
                if k.lower() in ('image', 'img', 'figure', 'diagram', 'chart', 'base64', 'data_uri', 'binary'):
                    continue
                extract_from_value(v)
        elif isinstance(value, list):
            for item in value:
                extract_from_value(item)

    extract_from_value(corpus_data)

    # Join text parts with newlines, clean up whitespace
    full_text = '\n\n'.join(text_parts)
    # Remove excessive whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)

    return full_text.strip()


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    repo_id: str | None = None,
    output_format: str = "txt",
    **_: object,
) -> tuple[int, int]:
    """Download Open RAGBench and write kb_dir + bench_path. Returns (num_docs, num_entries).

    Args:
        kb_dir: Directory to write knowledge base documents
        bench_path: Path to write benchmark JSON file
        num_samples: Number of samples to generate
        repo_id: HuggingFace repository ID
        output_format: Output format for documents - "txt", "md", or "pdf" (default: "txt")
                      Note: "pdf" downloads native ArXiv PDFs, "txt"/"md" extract text from JSON

    Returns:
        (number of documents written, number of benchmark entries)
    """
    from huggingface_hub import hf_hub_download

    if repo_id is None:
        repo_id = OPEN_RAGBENCH_REPO_ID
    prefix = f"{OPEN_RAGBENCH_SUBSET}/"
    queries_path = f"{prefix}queries.json"
    answers_path = f"{prefix}answers.json"
    qrels_path = f"{prefix}qrels.json"
    pdf_urls_path = f"{prefix}pdf_urls.json"

    queries_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=queries_path)
    answers_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=answers_path)
    qrels_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=qrels_path)

    # Download PDF URLs mapping (needed for native PDF download)
    pdf_urls_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=pdf_urls_path)

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(answers_file, "r", encoding="utf-8") as f:
        answers = json.load(f)
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    with open(pdf_urls_file, "r", encoding="utf-8") as f:
        pdf_urls = json.load(f)

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    processed = 0
    skipped_no_text = 0
    use_native_pdf = (output_format == "pdf")

    if use_native_pdf:
        print(f"Processing Open RAGBench dataset (downloading native PDFs from ArXiv)...")
    else:
        print(f"Processing Open RAGBench dataset (extracting text, excluding images/encodings)...")

    for query_uuid, query_info in queries.items():
        if processed >= num_samples:
            break
        question_text = query_info.get("query")
        answer_text = answers.get(query_uuid)
        if not answer_text or query_uuid not in qrels:
            continue
        qrel_info = qrels[query_uuid]
        doc_id = ""
        if isinstance(qrel_info, dict):
            doc_id = qrel_info.get("doc_id", "")
        elif isinstance(qrel_info, list) and qrel_info:
            doc_id = qrel_info[0].get("doc_id", "")
        if not doc_id:
            continue

        # Determine file extension and name
        local_doc_base = f"open_ragbench_{doc_id}"
        file_ext = get_file_extension(output_format)
        local_doc_id = f"{local_doc_base}{file_ext}"
        output_path = kb_dir / local_doc_id

        # Handle native PDF download vs text extraction
        if use_native_pdf:
            # Download native PDF from ArXiv
            if doc_id not in pdf_urls:
                skipped_no_text += 1
                print(f"  Skipped {doc_id}: PDF URL not found")
                continue

            pdf_url = pdf_urls[doc_id]
            try:
                print(f"  Downloading PDF: {doc_id} from {pdf_url}")

                # Try using requests library first (better SSL handling)
                try:
                    import requests
                    response = requests.get(pdf_url, timeout=30)
                    response.raise_for_status()
                    output_path.write_bytes(response.content)
                except ImportError:
                    # Fallback to urllib with SSL context
                    import ssl
                    import urllib.request

                    # Create SSL context that uses system certificates
                    ssl_context = ssl.create_default_context()
                    # Note: If you're still having SSL issues, you can disable verification
                    # (not recommended for production):
                    # ssl_context.check_hostname = False
                    # ssl_context.verify_mode = ssl.CERT_NONE

                    req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                        output_path.write_bytes(response.read())

                # Verify the PDF was downloaded
                if not output_path.exists() or output_path.stat().st_size == 0:
                    skipped_no_text += 1
                    print(f"  Skipped {doc_id}: PDF download failed (empty file)")
                    if output_path.exists():
                        output_path.unlink()
                    continue

            except Exception as e:
                skipped_no_text += 1
                print(f"  Skipped {doc_id}: PDF download error - {e}")
                if output_path.exists():
                    output_path.unlink()
                continue
        else:
            # Extract text from JSON corpus
            corpus_filename = f"{prefix}corpus/{doc_id}.json"
            try:
                corpus_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=corpus_filename)
            except Exception:
                continue
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)

            # Extract only text content, excluding images and encodings
            context_text = extract_text_only(corpus_data)

            if not context_text or len(context_text) < 50:
                # Skip documents with insufficient text content
                skipped_no_text += 1
                print(f"  Skipped {doc_id}: insufficient text content ({len(context_text) if context_text else 0} chars)")
                continue

            # Save document in requested format (txt or md)
            save_document(
                content=context_text,
                output_path=kb_dir / local_doc_base,
                format=output_format,
                metadata={
                    "source": "open_ragbench",
                    "doc_id": doc_id,
                    "title": f"Open RAGBench Document {doc_id}",
                }
            )

        benchmark_data.append({
            "question": question_text,
            "correct_answers": [answer_text],
            "correct_answer_document_ids": [local_doc_id],
        })
        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{num_samples} documents (skipped {skipped_no_text} with no text)...")

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nOpen RAGBench generation complete:")
    print(f"  Documents created: {len(benchmark_data)}")
    print(f"  Documents skipped: {skipped_no_text}")
    if use_native_pdf:
        print(f"  Format: Native PDFs downloaded from ArXiv")
    else:
        print(f"  Format: {output_format} (text extracted, images/encodings filtered out)")

    return len(list(kb_dir.iterdir())), len(benchmark_data)


register("open_ragbench", prepare, {"num_samples": 50, "repo_id": OPEN_RAGBENCH_REPO_ID})
