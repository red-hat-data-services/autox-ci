# AutoRAG Functional Tests

Parametrized functional tests for the Documents RAG Optimization pipeline on Red Hat OpenShift AI (RHOAI). Each test scenario is declared in `configs/test_configs.json`, submitted to KFP, and validated against expected outcomes.

## Directory layout

```
autorag/
  conftest.py                 Fixtures: env config, KFP client, S3 client, pipeline YAML resolution
  test_pipeline_functional.py Pytest test class (parametrized over configs)
  utils.py                    Run submission, state checks, failure diagnostics, artifact + notebook validation
  configs/
    configs.py                AutoRAGTestConfig dataclass, config loading and tag filtering
    test_configs.json          Scenario definitions (positive / negative)
  data/                       Static benchmark datasets used by pipeline runs
```

## Prerequisites

- A RHOAI cluster with Data Science Pipelines enabled.
- An **S3 Data Connection** created in the dashboard for your namespace (see [Running tests (cluster setup)](../README.md#running-tests-cluster-setup) in the parent README). **DSPA is created automatically** when `RHOAI_KFP_URL` is unset.
- Environment variables for cluster API, bearer token, and pipeline parameters. See `autox_tests/.env.rag.example` for the full template.
- Do not leave duplicate empty `AWS_*` entries at the bottom of `.env.rag` — they overwrite the S3 Storage block when the file is sourced.

## Running tests

All tests are run via the repository-level `run_tests.sh` wrapper:

```bash
# 1. Copy and fill the env template (RHOAI_URL + AWS_* in S3 Storage; match your UI connection name)
cp autox_tests/.env.rag.example autox_tests/.env.rag

# 2. Run all AutoRAG functional tests
./run_tests.sh --suite autorag --env-file autox_tests/.env.rag

# 3. Run only positive scenarios
./run_tests.sh --env-file autox_tests/.env.rag "autorag and positive"

# 4. Run only scenarios tagged "smoke"
./run_tests.sh --env-file autox_tests/.env.rag -t smoke "autorag and positive"

# 5. Dry-run to inspect the generated command
./run_tests.sh --env-file autox_tests/.env.rag --dry-run "autorag"
```

Tests are automatically skipped when required environment variables are missing.

## Test scenarios

Scenarios live in `configs/test_configs.json`. Each entry specifies:

| Field | Description |
|---|---|
| `id` | Short identifier shown in pytest output (e.g. `TC-P-1`) |
| `description` | Human-readable summary |
| `tags` | List of tags for runtime filtering via `-t` / `AUTORAG_FUNCTIONAL_TESTS_TAGS` |
| `expected_result` | `"pass"` or `"fail"` |
| `embedding_models` / `generation_models` | MaaS model IDs (JSON list, or `"env"` to read from `AUTORAG_EMBEDDING_MODELS` / `AUTORAG_GENERATION_MODELS`); required by the pipeline |
| `optimization_metric`, `optimization_max_rag_patterns`, `test_data_key` | Per-scenario parameter overrides |
| `input_data_keys` | JSON list of document-folder paths within the input bucket; the pipeline honours only the first entry |

> The vector-store backend (Milvus / PGVector) is auto-detected by the pipeline from the
> `VECTOR_DB_SECRET_NAME` secret's key prefixes; it is no longer a per-scenario parameter.

### Tag filtering

Pass tags via `--tags` / `-t` on the CLI or set `AUTORAG_FUNCTIONAL_TESTS_TAGS` in the environment. Only scenarios matching **all** specified tags are selected.

### OCR scenario (`ocr` tag)

`TC-P-5` covers OCR text extraction ([pipelines-components#243](https://github.com/opendatahub-io/pipelines-components/pull/243)). Its input is the SlideVQA corpus: 200 slide images (10 decks, ~89 MB of PNGs) with no text layer at all. Without OCR every document extracts to empty output and the run fails the "every uploaded document has extracted text" check (RHOAIENG-91789); with OCR enabled the same corpus produces text and the scenario passes. That check is what gives the scenario its teeth — a pipeline that silently ingests 200 blank documents still reports `SUCCEEDED` on its own.

```bash
./run_tests.sh --env-file autox_tests/.env.rag -t ocr "autorag and positive"
```

Only the benchmark JSON is kept in `data/slidevqa/val/png/10/`; the PNGs live in S3 (`datasets/rag/slidevqa/val/png/10/knowledge_base`) and are never uploaded from the repo. Ground truth uses `correct_answer_document_keys` holding **full S3 object keys** — ai4rag ≥ 0.16.0 validates the record key set with exact equality and matches ingested documents on their object key, so bare filenames silently zero out every retrieval metric. Budget extra runtime: OCR over 200 images is considerably slower than the text-format scenarios.

## Pass / fail criteria

**Expected-pass scenarios:**
1. Pipeline run finishes with state `SUCCEEDED`
2. At least 1 pattern artifact exists in S3
3. Indexing notebook, inference notebook, and `evaluation_results.json` exist in S3
4. When `run_notebook: true`, the best indexing and inference notebooks execute successfully in a Kubernetes Job; `RHOAI_NOTEBOOK_RUNNER_IMAGE` is required

**Expected-fail scenarios:**
1. Pipeline run finishes with state `FAILED` (not `SUCCEEDED`, not timeout)
2. Failure details are logged for observability

Negative scenarios use a capped timeout (600 s) since failures should surface quickly.

## Failure diagnostics

When a pipeline run fails, `utils._collect_failure_details()`:
1. Fetches run-level and task-level metadata from the KFP v2 API
2. Lists pods matching `pipeline/runid=<run_id>` via the Kubernetes API
3. Fetches the last 100 log lines from each container of each failed pod
4. Returns a formatted report appended to the test failure message

Kubernetes API URL is derived from the KFP route URL (OCP and ROSA patterns). Override with `K8S_API_URL` or `K8S_API_PORT` env vars.
