# AutoML benchmarks (KFP orchestrator)

This repository helps to run benchmark suites on **Kubeflow Pipelines** (KFP): **AutoML** (AutoGluon tabular and time series) and **AutoRAG** (RAG optimization). Each suite polls runs to completion and writes a **CSV** of results. Configuration splits **non-secret layout** (YAML) from **cluster and storage identity** (INI).

Python packages:

- **`automl_benchmark`** — AutoGluon benchmarks (dedupe, leaderboard discovery, S3 uploads).
- **`autorag_benchmark`** — RAG optimization pipeline orchestration.
- **`benchmark_common`** — shared loading, KFP client, polling, CSV writer (used by both).

Install from the repo root: `pip install -e .` (see [pyproject.toml](pyproject.toml)) or use `requirements-benchmark.txt`.

## AutoRAG benchmarks

Use a separate YAML + manifest shaped for RAG (`test_data_key` per dataset; optional `input_data_key`). Copy [templates/benchmark.autorag.example.yaml](templates/benchmark.autorag.example.yaml) and [templates/dataset_manifest.autorag.example.yaml](templates/dataset_manifest.autorag.example.yaml). In `credentials.ini`, set **`input_data_bucket_name`**, **`test_data_bucket_name`**, **`input_data_secret_name`**, **`test_data_secret_name`**, **`llama_stack_secret_name`**, and **`llama_stack_vector_io_provider_id`** (see commented blocks in [templates/credentials.example.ini](templates/credentials.example.ini)).

```bash
python scripts/autorag_benchmark_orchestrator.py --config config/benchmark.yaml --credentials config/credentials.ini --output results/rag_benchmark_runs.csv
python scripts/autorag_benchmark_orchestrator.py --dry-run -v
```

## Before running experiments (AutoML)

Complete the checklist below; placeholders in the template commands section mirror these items.

1. **`config/credentials.ini`** (copy from `templates/credentials.example.ini`)
   - **`[kfp]`**: `host` (Data Science Pipelines API URL), `namespace`, and authentication (`token`, or `token_file`, or `KFP_API_TOKEN` / `token_env`).
   - **`[storage]`**: `train_data_bucket_name` where training CSVs live (object keys must match your manifest). Optional: `benchmark_s3_prefix` (default `benchmarks`), `upload_benchmark_results` (default on when not set), and KFP artifact prefix keys—see [docs/s3-storage-schema.md](docs/s3-storage-schema.md).
   - **`[pipeline]`**: `train_data_secret_name` — Kubernetes secret in the project that pipeline pods use for S3 (or equivalent) access.
   - **`[s3]`**: Endpoint and keys for your own records / secret creation; values are not sent to the cluster by this tool. When `upload_benchmark_results` is enabled (see `[storage]`), these credentials also **upload** per-run and batch results to S3 (`s3:PutObject` required on the benchmark prefix).
   - **S3 layout & metadata**: See [docs/s3-storage-schema.md](docs/s3-storage-schema.md) for `benchmarks/{batch_id}/` keys, `metadata.json`, and aggregated `merged_leaderboards.csv`.
   - **Skip duplicate experiments** (default **on**): If `[s3]` can read `benchmarks/experiment_index/v1/{fingerprint}.json`, the orchestrator reuses the stored `results.csv` row instead of submitting KFP again. Pass **`--rerun-identical-experiments`** to force new pipeline runs.

2. **`config/benchmark.yaml`** (copy from `templates/benchmark.example.yaml`)
   - **`pipeline.package_path`**: compiled **tabular** AutoGluon pipeline IR (default: `../pipelines/autogluon-tabular-training-pipeline.yaml` relative to this file’s directory).
   - **`pipeline.timeseries_package_path`**: compiled **time series** pipeline IR (default: `../pipelines/autogluon-timeseries-training-pipeline.yaml`). Override in `credentials.ini` `[pipeline]` if needed.
   - **`dataset_manifest_path`**: manifest of datasets (path relative to this YAML’s directory).
   - **`run`**: optional tuning (`top_n`, timeouts, caching, run name prefix).

3. **Dataset manifest** — each row needs `train_data_file_key` and fields that match the task:
   - **Tabular** (`task_type`: `binary`, `multiclass`, or `regression`): `label_column`, `task_type`.
   - **Time series** (`task_type`: `timeseries`): `id_column` (item id), `timestamp_column`, and `target` **or** `label_column` for the forecast target. Optional: `known_covariates_names` (list of strings), `prediction_length` (integer).

   Start from `templates/dataset_manifest.example.yaml` or generate tabular/regression manifests (see below). Ensure objects exist in the bucket at the keys you declare.

4. **Pipeline packages**: the compiled YAML files referenced by `pipeline.package_path` and `pipeline.timeseries_package_path` must exist. The orchestrator uses the tabular package for non-`timeseries` rows and the time series package when `task_type` is `timeseries`.

## Template commands

Replace `FILL_*` values with your environment. Run from the repository root.

```bash
# One-time config from templates
cp templates/benchmark.example.yaml config/benchmark.yaml
cp templates/credentials.example.ini config/credentials.ini
cp templates/dataset_manifest.example.yaml config/dataset_manifest.example.yaml
# Edit benchmark.yaml and credentials.ini; adjust manifest paths or contents as needed.

# Optional: verbose logging, custom paths
export BENCHMARK_CONFIG_PATH="FILL_PATH_TO/benchmark.yaml"
export BENCHMARK_CREDENTIALS_PATH="FILL_PATH_TO/credentials.ini"

# Validate config wiring without calling KFP
python scripts/benchmark_orchestrator.py --dry-run -v

# Run the benchmark suite; write aggregated CSV
python scripts/benchmark_orchestrator.py --output results/benchmark_runs.csv

# Same entry point with explicit files
python scripts/benchmark_orchestrator.py \
  --config config/benchmark.yaml \
  --credentials config/credentials.ini \
  --output results/benchmark_runs.csv

# Stop on first pipeline failure
python scripts/benchmark_orchestrator.py --fail-fast --output results/benchmark_runs.csv

# Only tabular or only time-series rows (by task_type)
python scripts/benchmark_orchestrator.py --dataset-filter tabular --dry-run
python scripts/benchmark_orchestrator.py --dataset-filter timeseries --dry-run

# Ignore S3 experiment cache and always submit pipelines (same fingerprint as a prior run)
python scripts/benchmark_orchestrator.py --rerun-identical-experiments --output results/benchmark_runs.csv
```

Optional: build a long-form summary from the runs CSV (see `scripts/summarize_benchmark_results.py --help`).

```bash
python scripts/summarize_benchmark_results.py results/benchmark_runs.csv \
  -o results/benchmark_summary.csv
```

## Optional: local datasets and manifest generation

Not required if you already upload CSVs and maintain a manifest.

```bash
pip install scikit-learn openml
python scripts/download_initial_datasets.py --out downloaded_datasets

pip install pandas pyyaml
python scripts/generate_dataset_manifest.py --root downloaded_datasets \
  --s3-key-prefix FILL_S3_PREFIX > config/dataset_manifest.generated.yaml
```

Then point `dataset_manifest_path` in `config/benchmark.yaml` at the generated file (or merge entries into your main manifest) and ensure the same keys exist in your training data bucket.
