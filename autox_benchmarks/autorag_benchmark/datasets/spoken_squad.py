"""Spoken-SQuAD (audio QA) dataset provider.

Spoken-SQuAD is the canonical ASR-robust spoken-document QA benchmark: SQuAD
passages rendered to speech (Google TTS) whose audio serves as the knowledge
base, paired with text questions whose answers are spans in the (spoken)
document. This makes it the audio analogue of the harness's text RAG datasets:
the ``knowledge_base`` is a directory of audio files and ``benchmark_data.json``
holds the questions and gold answers.

Unlike the other providers, the knowledge base written here is BINARY AUDIO
(``.wav`` by default). A downstream RAG pipeline is expected to transcribe the
audio (ASR) before indexing/retrieval — the audio is the document, not text.

Source: https://huggingface.co/datasets/AudioLLMs/spoken_squad_test
    columns: context (Audio — the spoken passage / knowledge-base document),
    instruction (question, str), answer (str). Single ``test`` split of 5,351
    rows (the full Spoken-SQuAD test set).
Original benchmark: https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD
Paper: https://arxiv.org/abs/1804.00320
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_binary_document

# Constants
SPOKEN_SQUAD_REPO_ID = "AudioLLMs/spoken_squad_test"
SPOKEN_SQUAD_DEFAULT_SPLIT = "test"
SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac")


def _audio_doc_id(raw_audio: bytes) -> str:
    """Stable filesystem-safe id for a spoken document, keyed by its audio bytes.

    Multiple questions share the same spoken passage in Spoken-SQuAD; keying the
    audio file by its content deduplicates the knowledge base so retrieval is a
    real task (many questions -> one shared document) rather than a trivial 1:1
    mapping.
    """
    digest = hashlib.sha1(raw_audio).hexdigest()[:16]
    return f"spoken_squad_{digest}"


def _extract_audio_bytes(audio_field: object) -> bytes:
    """Return raw source audio bytes from a Spoken-SQuAD ``context`` cell.

    The audio lives in the ``context`` column (a HuggingFace ``Audio`` feature).
    Recast to a plain struct, each cell is a ``{'bytes', 'path'}`` dict with the
    embedded audio bytes (and, optionally, a source path).
    """
    if not isinstance(audio_field, dict):
        raise ValueError(f"Unexpected context/audio field type: {type(audio_field)!r}")
    raw = audio_field.get("bytes")
    if raw is None:
        path = audio_field.get("path")
        if not path:
            raise ValueError("Audio cell has neither 'bytes' nor 'path'")
        raw = Path(path).read_bytes()
    return raw


def _write_audio(raw_wav: bytes, output_stem: Path, output_format: str) -> Path:
    """Write source WAV bytes to ``output_stem`` in ``output_format``.

    ``wav`` is a byte-for-byte passthrough (no extra dependencies). ``flac`` and
    ``mp3`` transcode the decoded samples and require optional libraries
    (soundfile for flac, pydub+ffmpeg for mp3); a missing dependency raises an
    actionable error rather than silently degrading quality.
    """
    if output_format == "wav":
        return save_binary_document(raw_wav, output_stem, format="wav")

    if output_format == "flac":
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "output_format='flac' requires soundfile. "
                "Install it (`pip install soundfile`) or use --output-format wav."
            ) from exc
        data, sr = sf.read(io.BytesIO(raw_wav))
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="FLAC")
        return save_binary_document(buf.getvalue(), output_stem, format="flac")

    if output_format == "mp3":
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "output_format='mp3' requires pydub + ffmpeg. "
                "Install them (`pip install pydub` and system ffmpeg) or use "
                "--output-format wav."
            ) from exc
        seg = AudioSegment.from_file(io.BytesIO(raw_wav), format="wav")
        buf = io.BytesIO()
        seg.export(buf, format="mp3")
        return save_binary_document(buf.getvalue(), output_stem, format="mp3")

    raise ValueError(
        f"Spoken-SQuAD supports audio formats {SUPPORTED_AUDIO_FORMATS}, got '{output_format}'."
    )


def prepare(
    kb_dir: Path,
    bench_path: Path,
    *,
    num_samples: int = 50,
    repo_id: str | None = None,
    split: str = SPOKEN_SQUAD_DEFAULT_SPLIT,
    output_format: str = "wav",
    **_: object,
) -> tuple[int, int]:
    """Download Spoken-SQuAD and write kb_dir (audio) + bench_path. Returns (num_docs, num_entries).

    Args:
        kb_dir: Directory to write knowledge base audio files.
        bench_path: Path to write benchmark JSON file.
        num_samples: Number of question samples to generate (default: 50).
        repo_id: HuggingFace dataset id (default: AudioLLMs/spoken_squad_test).
        split: Dataset split (default: "test" — the only split available).
        output_format: Audio format for knowledge base files - "wav" (default,
            passthrough), "flac", or "mp3" (transcode; needs optional deps).

    Returns:
        (number of audio documents written, number of benchmark entries)
    """
    from datasets import Features, Value, load_dataset

    if repo_id is None:
        repo_id = SPOKEN_SQUAD_REPO_ID

    if output_format not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Spoken-SQuAD is an audio dataset and supports audio formats only "
            f"{SUPPORTED_AUDIO_FORMATS}. Got: {output_format}."
        )

    print(f"Loading Spoken-SQuAD dataset from {repo_id} (split: {split})...")

    try:
        dataset = load_dataset(repo_id, split=split)
        # In this dataset the spoken passage (the knowledge-base audio) lives in
        # the "context" column, which is a HuggingFace Audio feature; "instruction"
        # is the question and "answer" is the gold answer text. We only need the
        # raw audio bytes, never decoded samples. The Audio feature decodes via
        # torchcodec/ffmpeg on access, and some `datasets` versions ignore
        # Audio(decode=False), so recast "context" to a plain {bytes, path} struct
        # to read the raw storage with no audio-decode backend. (The Audio storage
        # is already a struct<bytes, path>, so this is a schema-only cast.)
        struct_features = Features({**dataset.features})
        struct_features["context"] = {"bytes": Value("binary"), "path": Value("string")}
        dataset = dataset.cast(struct_features)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Spoken-SQuAD dataset from '{repo_id}' (split '{split}').\n"
            f"You may need to authenticate with HuggingFace: `huggingface-cli login`.\n"
            f"Error: {e}"
        ) from e

    kb_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data: list[dict] = []
    audio_written: set[str] = set()
    processed = 0
    skipped = 0

    print(f"Processing Spoken-SQuAD dataset (target: {num_samples} samples)...")

    for entry in dataset:
        if processed >= num_samples:
            break

        # The question is stored under "instruction"; the spoken passage (audio)
        # is the "context" column.
        question = str(entry.get("instruction") or entry.get("question") or "").strip()
        answer = str(entry.get("answer") or "").strip()

        if not question or not answer:
            skipped += 1
            continue

        # Extract the spoken-passage audio bytes and key the document by content.
        try:
            raw_wav = _extract_audio_bytes(entry.get("context"))
        except Exception as e:
            print(f"  Warning: Failed to read audio for a question: {e}")
            skipped += 1
            continue

        doc_id = _audio_doc_id(raw_wav)
        doc_filename = f"{doc_id}.{output_format}"

        if doc_id not in audio_written:
            try:
                _write_audio(raw_wav, kb_dir / doc_id, output_format)
                audio_written.add(doc_id)
            except Exception as e:
                print(f"  Warning: Failed to write audio for {doc_id}: {e}")
                skipped += 1
                continue

        benchmark_data.append({
            "question": question,
            "correct_answers": [answer],
            # Gold id MUST include the extension: the pipeline's context_correctness
            # metric matches the retrieved chunk's document_id (full filename) verbatim.
            "correct_answer_document_ids": [doc_filename],
        })

        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{num_samples} questions (skipped {skipped})...")

    if not benchmark_data:
        raise RuntimeError(
            f"No benchmark entries generated. "
            f"Check that the Spoken-SQuAD split '{split}' has valid data."
        )

    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nSpoken-SQuAD generation complete:")
    print(f"  Questions/answers: {len(benchmark_data)}")
    print(f"  Audio documents: {len(audio_written)}")
    print(f"  Samples skipped: {skipped}")
    print(f"  Format: {output_format}")

    return len(list(kb_dir.iterdir())), len(benchmark_data)


register("spoken_squad", prepare, {
    "num_samples": 50,
    "repo_id": SPOKEN_SQUAD_REPO_ID,
    "split": SPOKEN_SQUAD_DEFAULT_SPLIT,
    "output_format": "wav",
})
