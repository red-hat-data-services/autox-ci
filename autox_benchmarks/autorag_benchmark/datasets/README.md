# RAG Dataset Generation

This module provides dataset generation for AutoRAG benchmarking, supporting BEIR, OpenRAGBench, and SlideVQA datasets.

## S3 Storage Structure

Generated datasets are organized in S3 with the following structure:

```
s3://ai-eng-cracow/datasets/rag/
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

### Install Dependencies

```bash
pip install -r autox_benchmarks/requirements.txt
```

Or with optional dependencies:

```bash
pip install -e ".[dataset-generation]"
```

### Configure S3 Credentials

Create `config/credentials.ini` with your S3 credentials:

```ini
[s3]
endpoint = https://s3.amazonaws.com
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
aws_default_region = us-east-1
```

Or copy the example:

```bash
cp config/credentials.example.dataset-gen.ini config/credentials.ini
# Edit config/credentials.ini with your credentials
```

### Generate and Upload Datasets

#### OpenRAGBench (50 samples)

```bash
# Generate and upload (uses ai-eng-cracow bucket by default)
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3

# Result: s3://ai-eng-cracow/datasets/rag/open_ragbench/arxiv/50/
```

#### BEIR SciFact (100 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset scifact \
  --num-samples 100 \
  --upload-to-s3

# Result: s3://ai-eng-cracow/datasets/rag/beir/scifact/100/
```

#### BEIR NFCorpus (50 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset nfcorpus \
  --num-samples 50 \
  --upload-to-s3

# Result: s3://ai-eng-cracow/datasets/rag/beir/nfcorpus/50/
```

#### SlideVQA Validation (50 samples)

```bash
python scripts/generate_rag_datasets.py \
  --dataset slidevqa \
  --slidevqa-split val \
  --num-samples 50 \
  --output-format png \
  --upload-to-s3

# Result: s3://ai-eng-cracow/datasets/rag/slidevqa/val/png/50/
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

### Custom Credentials File

```bash
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3 \
  --credentials /path/to/my/credentials.ini
```

### Use Environment Variables (instead of credentials.ini)

```bash
export AWS_S3_ENDPOINT=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --upload-to-s3
```

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
        "correct_answer_document_ids": [
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
        "correct_answer_document_ids": [
            "slidevqa_deck123_page_1.png",
            "slidevqa_deck123_page_2.png"
        ]
    }
]
```

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
