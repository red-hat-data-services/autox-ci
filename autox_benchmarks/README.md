# AutoX Benchmarks

End-to-end benchmarking framework for AutoML and AutoRAG pipelines on OpenShift AI / Kubeflow Pipelines.

## Overview

This framework provides automated benchmarking for two pipeline families:

- **AutoML** - Tabular and time-series training with AutoGluon
- **AutoRAG** - Retrieval-Augmented Generation optimization with pattern search

Both benchmark types:
- Submit pipelines to Kubeflow Pipelines (KFP)
- Wait for completion and extract metrics
- Generate CSV results with performance data
- Upload results to S3 for tracking and analysis

## Directory Structure

```
autox_benchmarks/
├── automl_benchmark/          # AutoML-specific orchestration
├── autorag_benchmark/         # AutoRAG-specific orchestration
│   └── datasets/              # Dataset generation providers (BEIR, OpenRAGBench)
├── benchmark_common/          # Shared utilities (KFP client, S3, CSV)
├── config/                    # Configuration files
│   ├── benchmark.yaml         # AutoRAG benchmark config
│   ├── credentials.ini        # Secrets (not in git)
│   └── dataset_manifest.yaml  # AutoRAG dataset registry
├── scripts/                   # Entry point scripts
│   ├── automl_benchmark_orchestrator.py
│   ├── autorag_benchmark_orchestrator.py
│   └── generate_rag_datasets.py
├── templates/                 # Example configurations
│   └── credentials.example.ini
└── results/                   # Generated benchmark CSVs
```

## Installation

**Option 1: Install with optional benchmark dependencies (recommended):**

From the repository root:

```bash
pip install -e ".[benchmarks]"
```

This installs the package plus all benchmark orchestration and dataset generation dependencies.

**Option 2: Install from requirements file:**

```bash
pip install -r autox_benchmarks/requirements.txt
```

## Quick Start

### 1. Configure Credentials

Copy the example credentials file:

```bash
cp templates/credentials.example.ini config/credentials.ini
```

Edit `config/credentials.ini` with your cluster and S3 details:

```ini
[kfp]
host = https://ds-pipeline-dspa-PROJECT.apps.CLUSTER.example.com
namespace = YOUR_PROJECT
token = sha256~...  # or use token_file or KFP_API_TOKEN env var
experiment_name = rag-optimization-benchmark

[storage]
# AutoML
train_data_bucket_name = your-bucket
# AutoRAG
input_data_bucket_name = your-bucket
test_data_bucket_name = your-bucket
# Benchmark result uploads (both AutoML and AutoRAG)
benchmark_s3_prefix = benchmarks
upload_benchmark_results = true

[pipeline]
# AutoML
train_data_secret_name = automl-s3-credentials
# AutoRAG
input_data_secret_name = rag-input-s3-credentials
test_data_secret_name = rag-test-s3-credentials
ogx_secret_name = llama-stack-credentials
vector_io_provider_id = milvus-lite
# Legacy parameter names (llama_stack_secret_name, llama_stack_vector_io_provider_id) also supported

[s3]
endpoint = https://s3.amazonaws.com
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
aws_default_region = us-east-1
```

### 2. Run AutoRAG Benchmark

```bash
python scripts/autorag_benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --credentials config/credentials.ini \
  --output results/rag_benchmark_runs.csv
```

**Options:**
- `--dry-run` - Validate configuration without submitting pipelines
- `--fail-fast` - Stop on first pipeline failure

### 3. Run AutoML Benchmark

```bash
python scripts/automl_benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --credentials config/credentials.ini \
  --output results/automl_benchmark_runs.csv \
  --dataset-filter all
```

**Options:**
- `--dataset-filter PATTERN` - Run only matching datasets: `all`, `tabular`, `timeseries`
- `--dry-run` - Validate configuration without submitting pipelines
- `--fail-fast` - Stop on first pipeline failure
- `--rerun-identical-experiments` - Force new runs even if identical fingerprint exists

## AutoRAG Dataset Generation

Generate benchmark datasets (BEIR, OpenRAGBench) and upload to S3.

### Prerequisites

```bash
pip install -r autox_benchmarks/requirements.txt
```

> **Note:** Dataset generation requires additional dependencies (`beir`, `requests`) which are included in `requirements.txt`.

### Generate and Upload

**OpenRAGBench (ArXiv PDFs):**

```bash
python scripts/generate_rag_datasets.py \
  --dataset open_ragbench \
  --num-samples 50 \
  --output-format pdf \
  --upload-to-s3
```

**BEIR (SciFact):**

```bash
python scripts/generate_rag_datasets.py \
  --dataset beir \
  --beir-dataset scifact \
  --num-samples 100 \
  --output-format txt \
  --upload-to-s3
```

**SlideVQA (Slide Images):**

```bash
python scripts/generate_rag_datasets.py \
  --dataset slidevqa \
  --num-samples 50 \
  --output-format png \
  --slidevqa-split val \
  --upload-to-s3
```

> **Note:** SlideVQA requires accepting the dataset license at [HuggingFace](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) and authenticating with `huggingface-cli login`.

### Supported Datasets

- **BEIR**: `scifact`, `nfcorpus`, `nq`, `fiqa`, `trec-covid`
- **OpenRAGBench**: ArXiv subset (native PDF download)
- **SlideVQA**: Presentation slide images for Visual Question Answering

### Supported Formats

- **txt** - Plain text (BEIR, OpenRAGBench)
- **md** - Markdown (BEIR, OpenRAGBench)
- **pdf** - Native PDFs (OpenRAGBench only - downloads from ArXiv)
- **png** - PNG images (SlideVQA only - slide deck images)
- **jpg** - JPEG images (SlideVQA only - slide deck images)

### S3 Structure

Generated datasets follow this structure:

```
s3://bucket/datasets/rag/
├── beir/
│   └── {dataset_name}/      # e.g., scifact, nfcorpus
│       └── {format}/        # txt, md
│           └── {num_samples}/
│               ├── knowledge_base/
│               └── benchmark_data.json
├── open_ragbench/
│   └── arxiv/
│       └── {format}/        # txt, md, pdf
│           └── {num_samples}/
│               ├── knowledge_base/
│               └── benchmark_data.json
└── slidevqa/
    └── {split}/             # train, val, test
        └── {format}/        # png, jpg
            └── {num_samples}/
                ├── knowledge_base/
                └── benchmark_data.json
```

### Add to Manifest

After generation, the script prints a YAML snippet. Add it to `config/dataset_manifest.yaml`:

**OpenRAGBench example:**
```yaml
datasets:
  - id: open-ragbench-arxiv-50
    name: "Open RAGBench ArXiv (50 samples)"
    input_data_key: "datasets/rag/open_ragbench/arxiv/pdf/50/knowledge_base"
    test_data_key: "datasets/rag/open_ragbench/arxiv/pdf/50/benchmark_data.json"
    optimization_metric: "faithfulness"
    embedding_models:
      - "vllm-embedding/bge-m3"
```

**SlideVQA example:**
```yaml
datasets:
  - id: slidevqa-val-50
    name: "SlideVQA Validation (50 samples)"
    input_data_key: "datasets/rag/slidevqa/val/png/50/knowledge_base"
    test_data_key: "datasets/rag/slidevqa/val/png/50/benchmark_data.json"
    optimization_metric: "faithfulness"
    embedding_models:
      - "vllm-embedding/clip"  # For multimodal image+text
```

## Benchmark Results Upload

Both AutoML and AutoRAG benchmarks upload results to S3 automatically when `upload_benchmark_results = true` (default).

### S3 Result Structure

```
s3://bucket/benchmarks/{batch_id}/
├── datasets/
│   ├── {dataset_id_1}/
│   │   ├── metadata.json       # Run details, arguments, timing
│   │   └── results.csv         # Single-row CSV for this run
│   └── {dataset_id_2}/
│       ├── metadata.json
│       └── results.csv
└── aggregated/
    ├── batch_metadata.json     # Batch summary (all datasets, settings)
    └── benchmark_runs.csv      # Full multi-row CSV
```

**Batch ID format:** `YYYYMMDDTHHMMSSZ` (e.g., `20260514T143527Z`)

### Disabling Upload

Set in `config/credentials.ini`:

```ini
[storage]
upload_benchmark_results = false
```

## Configuration Reference

### AutoRAG Config (`config/benchmark.yaml`)

```yaml
pipeline:
  package_path: "../pipelines/autorag-pipeline.yaml"

run:
  optimization_metric: "faithfulness"
  optimization_max_rag_patterns: 8
  poll_interval_seconds: 30
  timeout_seconds: 86400
  enable_caching: false
  run_name_prefix: "rag-benchmark"

dataset_manifest_path: "dataset_manifest.yaml"
```

### AutoML Config (`config/benchmark.yaml`)

```yaml
pipeline:
  package_path: "../pipelines/autogluon-tabular-training-pipeline.yaml"
  timeseries_package_path: "../pipelines/autogluon-timeseries-training-pipeline.yaml"

run:
  top_n: 3
  poll_interval_seconds: 30
  timeout_seconds: 86400
  enable_caching: false
  run_name_prefix: "benchmark"

dataset_manifest_path: "dataset_manifest.yaml"
```

### Dataset Manifest Examples

**AutoRAG:**

```yaml
datasets:
  - id: example-rag-use-case
    name: "Example RAG Dataset"
    input_data_key: "datasets/rag/example/knowledge_base"
    test_data_key: "datasets/rag/example/benchmark_data.json"
    optimization_metric: "faithfulness"
    embedding_models:
      - "vllm-embedding/bge-m3"
```

**AutoML Tabular:**

```yaml
datasets:
  - id: titanic
    name: "Titanic Survival"
    train_data_file_key: "datasets/tabular/titanic.csv"
    task_type: "binary"
    label_column: "Survived"
```

**AutoML Time Series:**

```yaml
datasets:
  - id: m4-daily
    name: "M4 Daily Forecasts"
    train_data_file_key: "datasets/timeseries/m4_daily.csv"
    task_type: "timeseries"
    id_column: "item_id"
    timestamp_column: "timestamp"
    target: "value"
    prediction_length: 14
```

## Development

### Project Structure

- `benchmark_common/` - Shared utilities
  - `s3_client.py` - S3 client creation and config validation
  - `s3_upload.py` - Common S3 upload functions (batch_id, CSV conversion)
  - `kfp_client.py` - KFP client creation
  - `pipeline_run.py` - Pipeline submission and waiting
  - `run_state.py` - Pipeline state checking
  - `results_csv.py` - CSV output generation

- `automl_benchmark/` - AutoML-specific code
  - `orchestrator.py` - Main orchestration logic
  - `s3_benchmark_upload.py` - AutoML result upload
  - `settings.py` - Settings dataclass
  - `experiment_fingerprint.py` - Deduplication fingerprints

- `autorag_benchmark/` - AutoRAG-specific code
  - `orchestrator.py` - Main orchestration logic
  - `s3_benchmark_upload.py` - AutoRAG result upload
  - `settings.py` - Settings dataclass
  - `pattern_scores.py` - Extract pattern optimization results
  - `datasets/` - Dataset generation providers

### Adding a New RAG Dataset Provider

1. Create `autorag_benchmark/datasets/my_dataset.py`:

```python
from autorag_benchmark.datasets import register
from autorag_benchmark.datasets.document_formats import save_document

def prepare(kb_dir, bench_path, *, num_samples=50, output_format="txt", **_):
    """Generate dataset.
    
    Args:
        kb_dir: Directory to write knowledge base documents
        bench_path: Path to write benchmark JSON file
        num_samples: Number of samples to generate
        output_format: "txt", "md", or "pdf"
        
    Returns:
        (number of documents written, number of benchmark entries)
    """
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Download/generate documents
    for i in range(num_samples):
        content = f"Document {i} content..."
        save_document(
            content=content,
            output_path=kb_dir / f"doc_{i}",
            format=output_format,
            metadata={"source": "my_dataset", "doc_id": str(i)}
        )
    
    # Write benchmark JSON
    benchmark_data = [
        {
            "question": "What is X?",
            "correct_answers": ["Answer to X"],
            "correct_answer_document_ids": ["doc_0.txt"],
        }
    ]
    
    import json
    with open(bench_path, "w") as f:
        json.dump(benchmark_data, f, indent=4)
    
    return (num_samples, len(benchmark_data))

register("my_dataset", prepare, {"num_samples": 50})
```

2. Import in `autorag_benchmark/datasets/__init__.py`:

```python
from autorag_benchmark.datasets import beir, open_ragbench, my_dataset
```

3. Use with generation script:

```bash
python scripts/generate_rag_datasets.py \
  --dataset my_dataset \
  --num-samples 100 \
  --upload-to-s3
```

## Troubleshooting

### S3 Upload Not Working

**Symptom:** No logs about S3 uploads, or warnings about S3 failures.

**Check:**

1. Credentials are set in `config/credentials.ini`:
   ```ini
   [s3]
   endpoint = https://s3.amazonaws.com
   aws_access_key_id = ...
   aws_secret_access_key = ...
   ```

2. Upload is enabled (default is `true`):
   ```ini
   [storage]
   upload_benchmark_results = true
   ```

3. Look for these logs:
   ```
   INFO Benchmark batch_id=20260514T143527Z (results will upload to s3://bucket/benchmarks/)
   INFO Uploaded benchmark artifacts to s3://bucket/benchmarks/.../
   ```

### Pipeline Submission Fails

**Symptom:** Error creating KFP client or submitting pipeline.

**Check:**

1. KFP credentials in `config/credentials.ini`:
   ```ini
   [kfp]
   host = https://ds-pipeline-dspa-PROJECT.apps.CLUSTER.com
   namespace = YOUR_PROJECT
   token = sha256~...
   ```

2. Test KFP connection:
   ```bash
   export KFP_API_TOKEN="sha256~..."
   curl -H "Authorization: Bearer $KFP_API_TOKEN" \
     https://ds-pipeline-dspa-PROJECT.apps.CLUSTER.com/apis/v2beta1/experiments
   ```

3. Run with `--dry-run` to validate config without submitting:
   ```bash
   python scripts/autorag_benchmark_orchestrator.py --dry-run -v
   ```

### Dataset Generation SSL Errors

**Symptom:** `SSL: CERTIFICATE_VERIFY_FAILED` when downloading PDFs from ArXiv.

**Fix:** Install `requests` library for better SSL handling:

```bash
pip install requests>=2.31.0
```

The dataset providers will automatically use `requests` if available, which handles SSL certificates properly on macOS.

### BEIR Dataset Generation Fails

**Symptom:** `ModuleNotFoundError: No module named 'beir'`

**Fix:** Install all dependencies (includes dataset generation):

```bash
pip install -r autox_benchmarks/requirements.txt
```

### SlideVQA Access Denied

**Symptom:** Error loading SlideVQA dataset or "dataset requires authentication"

**Fix:** Accept the dataset license and authenticate:

1. Visit [https://huggingface.co/datasets/NTT-hil-insight/SlideVQA](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA)
2. Accept the dataset license agreement
3. Authenticate with HuggingFace CLI:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

## References

- [S3 Storage Schema](docs/s3-storage-schema.md) - Detailed S3 key layout
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [OpenRAGBench Dataset](https://huggingface.co/datasets/vectara/open_ragbench)
- [SlideVQA Dataset](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA)
- [SlideVQA Paper (AAAI 2023)](https://arxiv.org/abs/2301.04883)

## License

See parent repository for license information.
