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
SLIDEVQA_DEFAULT_SPLIT = "validation"


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
        split: Dataset split - "train", "validation", or "test" (default: "validation")
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
    valid_splits = ["train", "validation", "test"]
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

        # Get slide images (SlideVQA provides images for each slide in the deck)
        # The exact field names may vary - adjust based on actual dataset structure
        slides = entry.get("slides") or entry.get("images") or []

        if not slides or len(slides) == 0:
            skipped += 1
            continue

        # Get evidence slide indices (which slides answer the question)
        evidence_indices = entry.get("evidence") or entry.get("evidence_indices") or []

        # For simplicity, we'll save all slides in the deck and mark evidence slides
        deck_id = entry.get("deck_id") or entry.get("id") or f"deck_{idx}"
        slide_doc_ids = []

        for slide_idx, slide_image in enumerate(slides):
            slide_doc_id = f"slidevqa_{deck_id}_slide_{slide_idx}"

            # Skip if already written (same deck might be referenced multiple times)
            if slide_doc_id in slide_images_written:
                slide_doc_ids.append(f"{slide_doc_id}.{output_format}")
                continue

            try:
                # Extract image bytes from PIL Image object
                if hasattr(slide_image, "tobytes"):
                    # Convert PIL Image to bytes
                    from io import BytesIO
                    img_bytes = BytesIO()
                    # Determine format for PIL save
                    pil_format = "PNG" if output_format == "png" else "JPEG"
                    slide_image.save(img_bytes, format=pil_format)
                    image_data = img_bytes.getvalue()
                else:
                    # Assume it's already bytes
                    image_data = slide_image

                # Save slide image
                output_path = kb_dir / slide_doc_id
                save_binary_document(
                    binary_data=image_data,
                    output_path=output_path,
                    format=output_format,
                )

                slide_images_written.add(slide_doc_id)
                slide_doc_ids.append(f"{slide_doc_id}.{output_format}")

            except Exception as e:
                print(f"  Warning: Failed to save slide {slide_doc_id}: {e}")
                continue

        if not slide_doc_ids:
            skipped += 1
            continue

        # Determine correct answer slides (evidence slides)
        correct_doc_ids = []
        if evidence_indices:
            for evidence_idx in evidence_indices:
                if 0 <= evidence_idx < len(slide_doc_ids):
                    correct_doc_ids.append(slide_doc_ids[evidence_idx])

        # If no evidence specified, use all slides (conservative approach)
        if not correct_doc_ids:
            correct_doc_ids = slide_doc_ids

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
