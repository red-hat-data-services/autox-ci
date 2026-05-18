"""SlideVQA (NTT/HuggingFace) dataset provider.

SlideVQA is a Visual Question Answering dataset for presentation slides,
containing slide decks with images and corresponding questions.

Dataset: https://huggingface.co/datasets/NTT-hil-insight/SlideVQA
Paper: https://arxiv.org/abs/2301.04883
"""

from __future__ import annotations

import json
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_binary_document, get_file_extension

# Constants
SLIDEVQA_REPO_ID = "NTT-hil-insight/SlideVQA"
SLIDEVQA_DEFAULT_SPLIT = "val"


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    repo_id: str | None = None,
    split: str = SLIDEVQA_DEFAULT_SPLIT,
    output_format: str = "png",
    **_: object,
) -> tuple[int, int]:
    """Download SlideVQA dataset and write kb_dir + bench_path. Returns (num_docs, num_entries).

    Args:
        kb_dir: Directory to write knowledge base slide images
        bench_path: Path to write benchmark JSON file
        num_samples: Number of question samples to generate (default: 50)
        repo_id: HuggingFace repository ID (default: NTT-hil-insight/SlideVQA)
        split: Dataset split - "train", "val", or "test" (default: "val")
        output_format: Output format for slides - "png" or "jpg" (default: "png")
                      Note: SlideVQA provides slide images; PPTX conversion not supported

    Returns:
        (number of slide images written, number of benchmark entries)
    """
    from datasets import load_dataset

    if repo_id is None:
        repo_id = SLIDEVQA_REPO_ID

    # Validate format
    if output_format not in ("png", "jpg"):
        raise ValueError(
            f"SlideVQA supports image formats only (png, jpg). "
            f"Got: {output_format}. "
            f"The dataset provides slide images, not PPTX files."
        )

    # Validate split
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Choose from: {', '.join(valid_splits)}")

    print(f"Loading SlideVQA dataset from {repo_id} (split: {split})...")

    # Note: SlideVQA requires accepting a license agreement on HuggingFace
    # Users must authenticate with HuggingFace Hub and accept the license
    try:
        dataset = load_dataset(repo_id, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SlideVQA dataset. "
            f"You may need to:\n"
            f"1. Accept the dataset license at https://huggingface.co/datasets/{repo_id}\n"
            f"2. Authenticate with HuggingFace: `huggingface-cli login`\n"
            f"Error: {e}"
        ) from e

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    slide_images_written: set[str] = set()
    processed = 0
    skipped = 0

    print(f"Processing SlideVQA dataset (target: {num_samples} samples)...")

    for idx, entry in enumerate(dataset):
        if processed >= num_samples:
            break

        # Extract question and answer
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()

        if not question or not answer:
            skipped += 1
            continue

        # Get deck information
        deck_name = entry.get("deck_name", f"deck_{idx}")
        # Sanitize deck name for filesystem
        deck_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in deck_name)[:100]

        # Get evidence pages (1-indexed in dataset)
        evidence_pages = entry.get("evidence_pages", [])

        # SlideVQA stores slides as page_1, page_2, ..., page_20
        slides = []
        for page_num in range(1, 21):
            page_key = f"page_{page_num}"
            if page_key in entry and entry[page_key] is not None:
                slides.append((page_num, entry[page_key]))

        if not slides:
            skipped += 1
            continue

        slide_doc_ids = []

        for page_num, slide_image in slides:
            slide_doc_id = f"slidevqa_{deck_id}_page_{page_num}"

            # Skip if already written (same deck might be referenced multiple times)
            if slide_doc_id in slide_images_written:
                slide_doc_ids.append((page_num, f"{slide_doc_id}.{output_format}"))
                continue

            try:
                # Convert PIL Image to bytes
                from io import BytesIO
                img_bytes = BytesIO()
                # Determine format for PIL save
                pil_format = "PNG" if output_format == "png" else "JPEG"
                slide_image.save(img_bytes, format=pil_format)
                image_data = img_bytes.getvalue()

                # Save slide image
                output_path = kb_dir / slide_doc_id
                save_binary_document(
                    binary_data=image_data,
                    output_path=output_path,
                    format=output_format,
                )

                slide_images_written.add(slide_doc_id)
                slide_doc_ids.append((page_num, f"{slide_doc_id}.{output_format}"))

            except Exception as e:
                print(f"  Warning: Failed to save slide {slide_doc_id}: {e}")
                continue

        if not slide_doc_ids:
            skipped += 1
            continue

        # Create a mapping from page number to document ID
        page_to_doc = {page_num: doc_id for page_num, doc_id in slide_doc_ids}
        all_doc_ids = [doc_id for _, doc_id in slide_doc_ids]

        # Determine correct answer slides (evidence pages are 1-indexed in the dataset)
        correct_doc_ids = []
        if evidence_pages:
            for page_num in evidence_pages:
                if page_num in page_to_doc:
                    correct_doc_ids.append(page_to_doc[page_num])

        # If no evidence specified, use all slides (conservative approach)
        if not correct_doc_ids:
            correct_doc_ids = all_doc_ids

        benchmark_data.append({
            "question": question,
            "correct_answers": [answer],
            "correct_answer_document_ids": correct_doc_ids,
        })

        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{num_samples} questions (skipped {skipped})...")

    if not benchmark_data:
        raise RuntimeError(
            f"No benchmark entries generated. "
            f"Check that the SlideVQA dataset split '{split}' has valid data."
        )

    # Write benchmark JSON
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nSlideVQA generation complete:")
    print(f"  Questions/answers: {len(benchmark_data)}")
    print(f"  Slide images: {len(slide_images_written)}")
    print(f"  Samples skipped: {skipped}")
    print(f"  Format: {output_format}")

    return len(list(kb_dir.iterdir())), len(benchmark_data)


register("slidevqa", prepare, {
    "num_samples": 50,
    "repo_id": SLIDEVQA_REPO_ID,
    "split": SLIDEVQA_DEFAULT_SPLIT,
    "output_format": "png",
})
