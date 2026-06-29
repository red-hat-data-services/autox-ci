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
├── automl_benchmark/          # AutoML orchestration, compare logic, S3 upload
├── autorag_benchmark/         # AutoRAG orchestration, pattern scores, datasets/
├── benchmark_common/          # Shared KFP, S3, pipeline compile, manifest helpers
├── config/                    # Your local config (gitignored secrets)
│   ├── benchmark.yaml         # Run tuning + manifest path (copy from templates/)
│   ├── .env                     # KFP / S3 / pipeline secrets (copy from .env.example)
│   └── dataset_manifest.yaml  # Dataset registry for your suite
├── pipelines/                 # Optional pre-compiled KFP IR (checked-in examples)
├── scripts/                   # CLI entry points (run from this directory)
│   ├── benchmark_orchestrator.py       # AutoML (tabular + time series)
│   ├── autorag_benchmark_orchestrator.py
│   ├── benchmark_compare_app.py        # Streamlit compare UI
│   ├── generate_rag_datasets.py
│   └── ...
├── templates/                 # Example benchmark.yaml, manifests
├── .env.example               # Credentials template (copy to .env)
├── tests/                     # Unit tests (e.g. compare_logic)
├── docs/                      # S3 layout reference
└── results/                   # Default output CSV location
```

## Installation

From the **`autox_benchmarks/`** directory (this package is self-contained):

```bash
cd autox_benchmarks
pip install -e .
```

Optional extras:

```bash
pip install -e ".[compare]"    # Streamlit compare UI
pip install -e ".[datasets]"   # RAG dataset generation (beir, huggingface-hub, etc.)
pip install -e ".[all]"        # compare + datasets + mlflow
pip install -e ".[dev]"        # pytest
pip install -e ".[mlflow]"     # automatic MLflow ingest after benchmarks
```

## Quick Start

All orchestrator commands below assume your shell **current working directory is `autox_benchmarks/`** (so `config/` and `scripts/` resolve correctly).

### 1. Configure credentials

```bash
cd autox_benchmarks
cp .env.example .env
cp templates/benchmark.example.yaml config/benchmark.yaml    # AutoML
# cp templates/benchmark.autorag.example.yaml config/benchmark.yaml   # AutoRAG
cp templates/dataset_manifest.example.yaml config/dataset_manifest.yaml
```

Edit `.env` with your cluster and S3 details (same style as `autox_tests/.env.ml`):

```bash
BENCHMARK_KFP_HOST=https://ds-pipeline-dspa-PROJECT.apps.CLUSTER.example.com
BENCHMARK_KFP_NAMESPACE=YOUR_PROJECT
BENCHMARK_KFP_TOKEN=sha256~...   # or set KFP_API_TOKEN in the shell
BENCHMARK_KFP_EXPERIMENT_NAME=autogluon-benchmark

# AutoML
BENCHMARK_TRAIN_DATA_BUCKET_NAME=your-bucket
BENCHMARK_TRAIN_DATA_SECRET_NAME=automl-s3-credentials

# AutoRAG (when using autorag_benchmark)
# BENCHMARK_INPUT_DATA_BUCKET_NAME=your-bucket
# BENCHMARK_TEST_DATA_BUCKET_NAME=your-bucket
# BENCHMARK_INPUT_DATA_SECRET_NAME=rag-input-s3-credentials
# BENCHMARK_TEST_DATA_SECRET_NAME=rag-test-s3-credentials
# BENCHMARK_OGX_SECRET_NAME=llama-stack-credentials
# BENCHMARK_VECTOR_IO_PROVIDER_ID=milvus-lite

BENCHMARK_S3_PREFIX=benchmarks
BENCHMARK_UPLOAD_RESULTS=true

AWS_S3_ENDPOINT=https://s3.amazonaws.com
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
AWS_DEFAULT_REGION=us-east-1
```

`.env` is loaded automatically. Shell/CI variables take precedence.

### 2. Run AutoML Benchmark

```bash
python scripts/benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --output results/benchmark_runs.csv \
  --dataset-filter all
```

**Options:**
- `--dataset-filter` — `all`, `tabular` (binary/multiclass/regression), or `timeseries`
- `--dry-run` — Build arguments and print them; do not call KFP
- `--fail-fast` — Stop after the first failed dataset run
- `--tabular-package-path` / `--timeseries-package-path` — Use a pre-compiled pipeline YAML (skip Git compile for that slot)
- `--rerun-identical-experiments` — Submit new KFP runs even when S3 experiment dedupe would reuse a prior result

### 3. Run AutoRAG Benchmark

```bash
python scripts/autorag_benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --output results/rag_benchmark_runs.csv
```

**Options:**
- `--dry-run`, `--fail-fast`, `--package-path` (compiled RAG pipeline YAML)

### 4. Compare benchmark results (local UI)

Interactive **Streamlit** app to compare baseline vs batch **`merged_leaderboards.csv`** from S3 or local CSVs. Choose baseline as rolling **`joined_results.csv`** or a specific batch; side-by-side score heatmaps (datasets × models) plus delta tables. Matching is on **`dataset_name` + model**.

```bash
cd autox_benchmarks
pip install -e ".[compare]"
streamlit run scripts/benchmark_compare_app.py
```

Uses `.env` for S3 access (cache: `~/.cache/autox_benchmarks/compare/`). See [docs/s3-storage-schema.md](docs/s3-storage-schema.md).

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `BENCHMARK_ENV_FILE` | Both | Path to `.env` (default: `autox_benchmarks/.env`) |
| `BENCHMARK_CONFIG_PATH` | Both | Path to `benchmark.yaml` (default: `config/benchmark.yaml`) |
| `BENCHMARK_KFP_HOST` | Both | KFP API URL (aliases: `RHOAI_KFP_URL`, `KFP_HOST`) |
| `BENCHMARK_KFP_NAMESPACE` | Both | DSPA namespace (aliases: `RHOAI_PROJECT_NAME`, `KFP_NAMESPACE`) |
| `BENCHMARK_KFP_TOKEN` | Both | KFP bearer token (aliases: `RHOAI_TOKEN`, `KFP_API_TOKEN`) |
| `BENCHMARK_TABULAR_PACKAGE_PATH` | AutoML | Pre-compiled tabular pipeline YAML (same as `--tabular-package-path`) |
| `BENCHMARK_TIMESERIES_PACKAGE_PATH` | AutoML | Pre-compiled time series pipeline YAML |
| `BENCHMARK_PACKAGE_PATH` / `RAG_PACKAGE_PATH` | AutoRAG | Pre-compiled RAG pipeline YAML (same as `--package-path`) |

## Compiled pipeline IR (skip Git compile)

By default the orchestrator clones [pipelines-components](https://github.com/opendatahub-io/pipelines-components) and compiles `pipeline.py` into cached YAML. To use a **fixed compiled IR** instead, set one of (first match wins):

1. **CLI** — `--tabular-package-path`, `--timeseries-package-path` (AutoML) or `--package-path` (AutoRAG)
2. **Environment** — table above (`BENCHMARK_TABULAR_PACKAGE_PATH`, etc.)
3. **`config/benchmark.yaml`** — `pipeline.package_path` / `pipeline.timeseries_package_path` (paths relative to the config file directory)
4. **`.env`** — `BENCHMARK_TABULAR_PACKAGE_PATH` / `BENCHMARK_TIMESERIES_PACKAGE_PATH` (merged over YAML)

Example (AutoML, from `autox_benchmarks/`):

```bash
export BENCHMARK_TABULAR_PACKAGE_PATH="$(pwd)/pipelines/autogluon-tabular-training-pipeline.yaml"
python scripts/benchmark_orchestrator.py --dry-run -v
```

Bundled examples live under `pipelines/`; production IR is often produced by your pipeline CI and pointed at with CLI or env.

## Pipeline parameters

The orchestrator builds a **baseline** argument dict per dataset (secrets, bucket keys, `task_type`, RAG metrics, etc.). You can extend or override per manifest row:

```yaml
datasets:
  - id: breast-w
    train_data_file_key: datasets/classification/breast-w.csv
    task_type: binary
    label_column: Class
    pipeline_arguments:
      top_n: 5
```

(`pipeline_params` is an alias for `pipeline_arguments`.)

Arguments are sent to KFP **as-is**. Names not declared in the compiled pipeline IR are logged at INFO; **invalid or unknown parameters are rejected by KFP / the pipeline run**, not pre-validated (and dropped) in the orchestrator. Use `--dry-run -v` to inspect the payload before submitting.

Declared root inputs are read from the compiled YAML when possible; see `benchmark_common/pipeline_run.py`.

## AutoRAG Dataset Generation

Generate benchmark datasets (BEIR, OpenRAGBench) and upload to S3.

### Prerequisites

```bash
cd autox_benchmarks && pip install -e ".[datasets]"
```

> **Note:** Dataset generation requires the `[datasets]` extra (`beir`, `requests`, `huggingface-hub`, etc.).

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

Results are organized by benchmark type for easy separation and management:

**AutoML Results:**
```
s3://bucket/benchmarks/ml/{batch_id}/
├── datasets/
│   ├── {dataset_id_1}/
│   │   ├── metadata.json         # Run details, arguments, timing
│   │   ├── results.csv           # Single-row CSV for this run
│   │   └── leaderboard.html      # AutoGluon leaderboard (if available)
│   └── {dataset_id_2}/
│       ├── metadata.json
│       └── results.csv
├── aggregated/
│   ├── batch_metadata.json       # Batch summary (all datasets, settings)
│   ├── benchmark_runs.csv        # Full multi-row CSV
│   ├── merged_leaderboards.csv   # Merged with leaderboard details
│   └── autogluon-*-pipeline.yaml # Pipeline definitions
└── joined_results.csv            # Cumulative results across all ML batches
```

**AutoRAG Results:**
```
s3://bucket/benchmarks/rag/{batch_id}/
├── datasets/
│   ├── {dataset_id_1}/
│   │   ├── metadata.json         # Run details, arguments, timing
│   │   └── results.csv           # Single-row CSV for this run
│   └── {dataset_id_2}/
│       ├── metadata.json
│       └── results.csv
└── aggregated/
    ├── batch_metadata.json       # Batch summary (all datasets, settings)
    └── benchmark_runs.csv        # Full multi-row CSV
```

**Batch ID format:** `YYYYMMDDTHHMMSSZ` (e.g., `20260514T143527Z`)

> **Note:** The default structure separates ML and RAG results. You can customize the prefix in `.env` with `BENCHMARK_S3_PREFIX`.

### Customizing S3 Upload

Set in `.env`:

```ini
[storage]
# Disable upload entirely
upload_benchmark_results = false

# Or customize the S3 prefix (defaults: benchmarks/ml for AutoML, benchmarks/rag for AutoRAG)
benchmark_s3_prefix = benchmarks/ml
# benchmark_s3_prefix = benchmarks/rag
# benchmark_s3_prefix = my-custom-prefix
```

## Configuration Reference

### AutoRAG Config (`config/benchmark.yaml`)

```yaml
pipeline:
  compile: {}   # default: compile documents RAG pipeline from Git
  # package_path: "../pipelines/documents-rag-optimization-pipeline.yaml"

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
  compile: {}
  # package_path: "../pipelines/autogluon-tabular-training-pipeline.yaml"
  # timeseries_package_path: "../pipelines/autogluon-timeseries-training-pipeline.yaml"

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

### Tests (AutoML)

From `autox_benchmarks/`:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Dry-run integration tests use `tests/fixtures/automl/` (static pipeline YAML under `pipelines/`, no KFP or Git). They cover CLI flags, dataset filters, manifest `pipeline_arguments`, and package-path resolution.

### MLflow (automatic after each batch)

When enabled, the orchestrator ingests the batch CSV into MLflow with the same **4-level hierarchy** as `MLFlow.ipynb`:

`benchmark → task_type → dataset → entity (model/pattern)`

Add to `.env`:

```bash
BENCHMARK_UPLOAD_MLFLOW=true
BENCHMARK_MLFLOW_KIND=automl          # or autorag
MLFLOW_TRACKING_URI=https://rh-ai.apps.../mlflow
MLFLOW_TRACKING_TOKEN=              # oc whoami -t
MLFLOW_WORKSPACE=ns-automl-benchmarking
BENCHMARK_MLFLOW_EXPERIMENT=automl-benchmarks
```

AutoML reads the merged leaderboard locally (same data as `merged_leaderboards.csv` on S3). AutoRAG uses `benchmark_runs.csv`.

Re-ingest an existing batch from S3:

```bash
python scripts/log_benchmark_mlflow.py 20260529T120000Z --kind automl
```

### Online integration tests (AutoML)

Real KFP + S3 smoke run on **breast-w-smoke** (`top_n: 1`). Only needs `.env` (smoke CSV is auto-uploaded). See [tests/integration/README.md](tests/integration/README.md).

```bash
pytest tests/integration/ -v -s
```

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

1. Credentials are set in `.env`:
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

1. KFP credentials in `.env`:
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
   cd autox_benchmarks
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
cd autox_benchmarks && pip install -e ".[datasets]"
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
