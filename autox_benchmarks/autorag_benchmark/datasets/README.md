# RAG Dataset Generation

This module provides dataset generation for AutoRAG benchmarking, supporting BEIR, OpenRAGBench, and SlideVQA datasets.

## S3 Storage Structure

Generated datasets are organized in S3 with the following structure:

```
s3://<your-input-bucket>/datasets/rag/
├── beir/
│   ├── scifact/
│   │   ├── 50/
│   │   │   ├── knowledge_base/
│   │   │   │   ├── beir_scifact_123.txt
│   │   │   │   └── beir_scifact_456.txt
│   │   │   └── benchmark_data.json
│   │   ├── 100/
│   │   │   ├── knowledge_base/
│   │   │   └── benchmark_data.json
│   │   └── 200/
│   ├── nfcorpus/
│   │   └── 50/
│   └── nq/
│       └── 100/
├── open_ragbench/
│   └── arxiv/
│       ├── 50/
│       │   ├── knowledge_base/
│       │   │   ├── open_ragbench_2401.11899v3.txt
│       │   │   └── open_ragbench_2404.00822v2.txt
│       │   └── benchmark_data.json
│       ├── 100/
│       └── 200/
└── slidevqa/
    ├── val/
    │   ├── png/
    │   │   ├── 50/
    │   │   │   ├── knowledge_base/
    │   │   │   │   ├── slidevqa_deck123_page_1.png
    │   │   │   │   └── slidevqa_deck123_page_2.png
    │   │   │   └── benchmark_data.json
    │   │   ├── 100/
    │   │   └── 200/
    │   └── jpg/
    │       └── 50/
    ├── train/
    │   └── png/
    │       └── 50/
    └── test/
        └── png/
            └── 50/
```

**Granulation levels:**
1. **Dataset type**: `beir`, `open_ragbench`, or `slidevqa`
2. **Dataset variant**: 
   - For BEIR: dataset name (`scifact`, `nfcorpus`, `nq`, etc.)
   - For OpenRAGBench: `arxiv` (corpus type)
   - For SlideVQA: split (`train`, `val`, `test`) and format (`png`, `jpg`)
3. **Number of samples**: `50`, `100`, `200`, etc.

## Quick Start

### FinanceBench (native financial PDFs)

The public [FinanceBench release](https://github.com/patronus-ai/financebench)
contains 150 annotated questions and a document index covering 360 distinct PDF
filings. Use a pinned checkout with `data/` and `pdfs/` present. The generator
checks that every indexed PDF is available and has a PDF header, which also
detects an incomplete Git LFS checkout.

```bash
git clone https://github.com/patronus-ai/financebench.git /tmp/financebench
git -C /tmp/financebench rev-parse HEAD  # record this source revision
python scripts/generate_rag_datasets.py \
  --dataset financebench \
  --financebench-dir /tmp/financebench \
  --output-format pdf \
  --num-samples 150 \
  --output-dir ./generated_datasets/financebench/pdf/150
```

Add `--upload-to-s3` to upload the PDF corpus and benchmark JSON; the script
prints the manifest entry. A smaller `--num-samples` deterministically chooses
questions using seed 42, while retaining **all 360 PDFs**, including distractors.
The emitted gold document names are resolved to S3 object keys during upload.
Use `FINANCEBENCH_DIR` instead of `--financebench-dir` if preferred.

The standard AutoRAG evaluator checks answer correctness and document retrieval,
but does not score FinanceBench's page-level evidence or apply numeric and unit
tolerances. Report its scores as AutoRAG results, not official FinanceBench scores.

### MultiHop-RAG (article retrieval)

The [MultiHop-RAG dataset](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
supplies article text, questions, answers, and evidence article URLs. The provider
keeps the entire article corpus when selecting a smaller number of questions.
Questions without evidence are excluded because the benchmark requires gold
document keys. Downloading from Hugging Face is automatic, or provide a local
directory containing `corpus.json` and `MultiHopRAG.json`:

```bash
python scripts/generate_rag_datasets.py \
  --dataset multihop_rag \
  --multihop-rag-dir /path/to/MultiHopRAG \
  --output-format txt \
  --num-samples 50
```

The provider supports `txt` and `md`. Source articles and the selected question
order are deterministic for a pinned source revision. The AutoRAG scores are not
the official MultiHop-RAG evaluation scores.

### FanOutQA (Wikipedia page retrieval)

The [FanOutQA questions](https://github.com/zhudotexe/fanoutqa) and
[published retrieval corpus](https://huggingface.co/datasets/JinChao1022/fanoutqa-retrieval)
are downloaded automatically, or supplied as local files. The provider keeps the
whole retrieval corpus and selects dev questions with 5–10 available evidence
pages and 4–10 answer values. It excludes time-sensitive questions and rows
whose answer values do not occur in their evidence pages. A requested sample
count above the eligible row count raises an error.

```bash
python scripts/generate_rag_datasets.py \
  --dataset fanoutqa \
  --fanoutqa-dev-json /path/to/fanout-final-dev.json \
  --fanoutqa-corpus-arrow /path/to/data-00000-of-00001.arrow \
  --output-format txt \
  --num-samples 100
```

The provider supports `txt` and `md`. The source files are cached under the
output directory's `source_cache/` so repeated runs do not redownload them.
The composite reference answer is built from the published answer values; report
AutoRAG results separately from official FanOutQA scores.

### Install Dependencies

```bash
cd autox_benchmarks
pip install -e ".[datasets]"
```

### Configure S3 credentials

Copy `.env.example` to `.env` and set AWS keys:

```bash
cp .env.example .env
# AWS_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

### Generate and Upload Datasets

#### OpenRAGBench (50 samples)

```bash
# Generate and upload (uses BENCHMARK_INPUT_DATA_BUCKET_NAME from .env)
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3

# Result: s3://<your-bucket>/datasets/rag/open_ragbench/arxiv/50/
```

#### BEIR SciFact (100 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset scifact \
  --num-samples 100 \
  --upload-to-s3

# Result: s3://<your-bucket>/datasets/rag/beir/scifact/100/
```

#### BEIR NFCorpus (50 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset nfcorpus \
  --num-samples 50 \
  --upload-to-s3

# Result: s3://<your-bucket>/datasets/rag/beir/nfcorpus/50/
```

#### SlideVQA Validation (50 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset slidevqa \
  --slidevqa-split val \
  --num-samples 50 \
  --output-format png \
  --upload-to-s3

# Result: s3://<your-bucket>/datasets/rag/slidevqa/val/png/50/
```

> **Note:** SlideVQA requires accepting the dataset license at [HuggingFace](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) and authenticating with `huggingface-cli login`.

### Generate Locally (without S3 upload)

```bash
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 10 \
  --output-dir ./my_datasets/open_ragbench_10
```

## Adding Datasets to Benchmarking

After generating and uploading datasets, add them to `config/dataset_manifest.yaml`:

```yaml
datasets:
  # OpenRAGBench - 50 samples
  - id: open-ragbench-arxiv-50
    name: "Open RAGBench ArXiv (50 samples)"
    input_data_key: "datasets/rag/open_ragbench/arxiv/50/knowledge_base"
    test_data_key: "datasets/rag/open_ragbench/arxiv/50/benchmark_data.json"
    optimization_metric: "faithfulness"
    embeddings_models:
      - "vllm-embedding/bge-m3"

  # BEIR SciFact - 100 samples
  - id: beir-scifact-100
    name: "BEIR SciFact (100 samples)"
    input_data_key: "datasets/rag/beir/scifact/100/knowledge_base"
    test_data_key: "datasets/rag/beir/scifact/100/benchmark_data.json"
    optimization_metric: "faithfulness"
    embeddings_models:
      - "vllm-embedding/bge-m3"

  # SlideVQA Validation - 50 samples
  - id: slidevqa-val-50
    name: "SlideVQA Validation (50 samples)"
    input_data_key: "datasets/rag/slidevqa/val/png/50/knowledge_base"
    test_data_key: "datasets/rag/slidevqa/val/png/50/benchmark_data.json"
    optimization_metric: "faithfulness"
    embeddings_models:
      - "vllm-embedding/bge-m3"
```

The script automatically prints the correct YAML snippet after generation/upload.

## Available BEIR Datasets

Common BEIR datasets you can use with `--beir-dataset`:

- `scifact` - Scientific fact verification (default)
- `nfcorpus` - Nutrition facts corpus
- `nq` - Natural Questions
- `hotpotqa` - Multi-hop question answering
- `fiqa` - Financial question answering
- `arguana` - Argument search
- `webis-touche2020` - Argument retrieval
- `quora` - Duplicate question detection
- `dbpedia-entity` - Entity retrieval
- `scidocs` - Scientific document retrieval

See [BEIR documentation](https://github.com/beir-cellar/beir) for the full list.

## SlideVQA Dataset

SlideVQA is a Visual Question Answering dataset for presentation slides from the AAAI 2023 paper.

**Dataset**: https://huggingface.co/datasets/NTT-hil-insight/SlideVQA  
**Paper**: https://arxiv.org/abs/2301.04883

**Available splits**:
- `train` - Training set
- `val` - Validation set (default)
- `test` - Test set

**Output formats**:
- `png` - PNG images (default)
- `jpg` - JPEG images

**Requirements**:
1. Accept the dataset license at the HuggingFace link above
2. Authenticate with HuggingFace: `huggingface-cli login`

## Advanced Usage

### Custom .env file

```bash
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3 \
  --env-file /path/to/my.env
```

Shell variables override `.env` values when already set.

### Custom S3 Bucket

```bash
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3 \
  --s3-bucket my-custom-bucket
```

### Custom S3 Prefix

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset scifact \
  --num-samples 50 \
  --upload-to-s3 \
  --s3-prefix my/custom/path/beir/scifact/50
```

### Different BEIR Splits

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset scifact \
  --beir-split train \
  --num-samples 100 \
  --upload-to-s3
```

### SlideVQA with Different Splits and Formats

```bash
# Training set with JPEG format
python scripts/generate_rag_datasets.py \
  --dataset slidevqa \
  --slidevqa-split train \
  --output-format jpg \
  --num-samples 100 \
  --upload-to-s3

# Test set with PNG format
python scripts/generate_rag_datasets.py \
  --dataset slidevqa \
  --slidevqa-split test \
  --output-format png \
  --num-samples 50 \
  --upload-to-s3
```

## Programmatic Usage

**OpenRAGBench example:**
```python
from pathlib import Path
from autorag_benchmark.datasets import get

# Get dataset provider
prepare_fn, default_options = get("open_ragbench")

# Generate dataset
kb_dir = Path("./my_kb")
bench_path = Path("./benchmark.json")
num_docs, num_entries = prepare_fn(
    kb_dir,
    bench_path,
    num_samples=50,
)

print(f"Generated {num_docs} documents and {num_entries} benchmark entries")
```

**SlideVQA example:**
```python
from pathlib import Path
from autorag_benchmark.datasets import get

# Get dataset provider
prepare_fn, default_options = get("slidevqa")

# Generate dataset with PNG format
kb_dir = Path("./slidevqa_kb")
bench_path = Path("./slidevqa_benchmark.json")
num_docs, num_entries = prepare_fn(
    kb_dir,
    bench_path,
    num_samples=50,
    split="val",
    output_format="png",
)

print(f"Generated {num_docs} slide images and {num_entries} benchmark entries")
```

## Output Formats

The dataset generation supports multiple output formats:

### Text Format (`.txt`) - Default

Plain text files compatible with most RAG pipelines.  
**Supported datasets**: BEIR, OpenRAGBench

### Markdown Format (`.md`)

Markdown files with YAML frontmatter:

```markdown
---
source: "open_ragbench"
doc_id: "2404.00822v2"
title: "Open RAGBench Document 2404.00822v2"
---

Document content...
```

**Supported datasets**: BEIR, OpenRAGBench

### PDF Format (`.pdf`)

Formatted PDF files (requires `reportlab`).  
**Supported datasets**: BEIR, OpenRAGBench

### Image Formats (`.png`, `.jpg`)

Slide deck images for visual question answering.  
**Supported datasets**: SlideVQA only

Use `--output-format {txt,md,pdf,png,jpg}` when running the generation script. Note that SlideVQA only supports image formats (png, jpg).

## Dataset Formats

### Knowledge Base Files

Each dataset generates individual files with format-specific extensions:

- **BEIR**: `beir_{dataset}_{doc_id}.{ext}`
- **OpenRAGBench**: `open_ragbench_{doc_id}.{ext}`
- **SlideVQA**: `slidevqa_{deck_id}_page_{page_num}.{ext}`

Where `{ext}` is:
- `txt`, `md`, or `pdf` for BEIR and OpenRAGBench
- `png` or `jpg` for SlideVQA

### Benchmark Data JSON

Standard format used by all datasets:

**BEIR / OpenRAGBench example:**
```json
[
    {
        "question": "What are the challenges in...",
        "correct_answers": [
            "The main challenges are..."
        ],
        "correct_answer_document_keys": [
            "open_ragbench_2404.00822v2.txt"
        ]
    }
]
```

**SlideVQA example:**
```json
[
    {
        "question": "What is shown in the slide?",
        "correct_answers": [
            "The slide shows a comparison chart..."
        ],
        "correct_answer_document_keys": [
            "slidevqa_deck123_page_1.png",
            "slidevqa_deck123_page_2.png"
        ]
    }
]
```

### Document keys and the upload step

A document's *key* is its full S3 object key — that is the value ai4rag carries as
`DoclingDocument.name` through chunking, indexing and evaluation, and the value
`correct_answer_document_keys` is matched against. A bare file name does not match
and silently scores 0.

Generators still write bare file names, as shown above, because the destination
prefix does not exist yet at generation time. `upload_dataset_to_s3` expands them
on the way to S3:

```
doc_0.txt  ->  datasets/rag/beir/scifact/50/knowledge_base/doc_0.txt
```

So the local `benchmark_data.json` holds bare names while the uploaded copy holds
full keys. Names are resolved against the knowledge base directory, so a file
nested inside it keeps its relative path. A name matching no file, or one that is
ambiguous across subdirectories, raises at upload rather than failing silently
mid-pipeline.

## Extending with New Datasets

To add a new dataset:

1. Create a new file in `autorag_benchmark/datasets/` (e.g., `my_dataset.py`)
2. Implement a `prepare()` function:
   ```python
   from pathlib import Path
   from autorag_benchmark.datasets import register

   def prepare(
       kb_dir: Path,
       bench_path: Path,
       *,
       num_samples: int = 50,
       **kwargs
   ) -> tuple[int, int]:
       # Your generation logic here
       return (num_docs, num_entries)

   register("my_dataset", prepare, {"num_samples": 50})
   ```
3. Import the module in `__init__.py`
4. The dataset becomes available via the CLI automatically
