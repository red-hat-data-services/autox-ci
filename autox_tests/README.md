# autox_tests — functional test suites

End-to-end tests that submit real pipeline runs to **OpenShift AI** (Data Science Pipelines / KFP v2) and validate results against a running cluster.

Two independent suites live here, each with its own env file, config JSON, and pytest entry point:

| Suite | Path | Env file |
|---|---|---|
| **AutoML** | `automl/` | `.env.ml` |
| **AutoRAG** | `autorag/` | `.env.rag` |

---

## Running tests (cluster setup)

### Before the first run

1. **S3 Data Connection (manual, once per namespace)** — In the RHOAI dashboard, create an S3 connection in your project (e.g. name `minio`). Set the same name in `.env` as `RHOAI_TRAIN_S3_SECRET_NAME` (AutoML) or `RHOAI_TEST_S3_SECRET_NAME` / `TEST_DATA_SECRET_NAME` (AutoRAG). Tests only ensure labels on that secret; they do not replace credentials from the UI by default.
2. **DSPA (automatic)** — You do **not** need `oc apply` for a pipeline server. Leave `RHOAI_KFP_URL` empty: pytest creates a `DataSciencePipelinesApplication` (`RHOAI_DSPA_NAME`, default `dspa`) with `managedPipelines`, waits for it to become Ready, and uses the `ds-pipeline` route. If a DSPA with that name already exists, tests reuse it (HTTP 409).

To use an existing pipeline server instead: set `RHOAI_KFP_URL` and `RHOAI_CREATE_DSPA=false`.

### Commands

```bash
# AutoML
cp autox_tests/.env.ml.example autox_tests/.env.ml   # edit cluster + S3 + buckets
./run_tests.sh --suite automl --env-file autox_tests/.env.ml -t smoke

# AutoRAG
cp autox_tests/.env.rag.example autox_tests/.env.rag
./run_tests.sh --suite autorag --env-file autox_tests/.env.rag -t smoke
```

Container (same flow):

```bash
podman run --rm -it -v "$(pwd):/workspace:z" -w /workspace \
  --env-file autox_tests/.env.ml python:3.12 \
  bash -c 'pip install uv && ./run_tests.sh --suite automl --env-file autox_tests/.env.ml -t smoke'
```

Managed pipelines are the default (no `pipeline.yaml`). Legacy YAML upload: `RHOAI_USE_MANAGED_PIPELINES=false` or `./run_tests.sh --legacy-pipeline-yaml`.

---

## Custom test configurations

When `autox-ci` is used as a submodule, downstream repos can supply their own test config JSON files to define different datasets, models, or scenario definitions. Override the built-in configs via environment variables or CLI flags:

| Env variable | CLI flag | Overrides |
|---|---|---|
| `AUTORAG_TEST_CONFIGS_PATH` | `--rag-configs` | `autorag/configs/test_configs.json` |
| `AUTOML_TABULAR_TEST_CONFIGS_PATH` | `--tabular-configs` | `automl/configs/tabular_test_configs.json` |
| `AUTOML_TIMESERIES_TEST_CONFIGS_PATH` | `--timeseries-configs` | `automl/configs/timeseries_test_configs.json` |

Custom JSON files must follow the same schema as the built-in configs they replace. The dataclass fields in `configs/configs.py` define the expected keys.

---

## AutoML functional tests

End-to-end tests for the AutoGluon tabular and time series training pipelines. Validates pipeline runs, S3 artifacts, and optionally deploys trained models via KServe for inference scoring.

### Directory layout

```
autox_tests/
├── .env.ml.example                     # env template — copy to .env.ml and fill in
└── automl/
    ├── conftest.py                     # pytest fixtures (KFP client, S3 client, kubeconfig, cleanup)
    ├── test_tabular_functional.py      # tabular positive + negative tests
    ├── test_timeseries_functional.py   # time series positive + negative tests
    ├── utils.py                        # shared helpers (KServe, S3, KFP, scoring)
    └── configs/
        ├── configs.py                  # dataclasses + config loaders
        ├── tabular_test_configs.json   # tabular test scenarios
        └── timeseries_test_configs.json
```

### Prerequisites

Python 3.11+ and `uv` (recommended) or `pip`. Install test dependencies (includes AutoGluon from the RHAI index):

```bash
uv sync --extra test_automl
# or
pip install -e ".[test_automl]"
```

You also need a running OpenShift AI cluster with Data Science Pipelines and an S3-compatible object store reachable from the cluster.

### Environment setup

```bash
cp autox_tests/.env.ml.example autox_tests/.env.ml
# edit .env.ml with your cluster details
```

`autox_tests/.env.ml` is loaded automatically at pytest startup. Shell and CI variables take precedence.

#### Required

| Variable | Purpose |
|---|---|
| `RHOAI_KFP_URL` | Data Science Pipelines HTTP API URL |
| `RHOAI_TOKEN` | Bearer token for KFP and Kubernetes API |
| `RHOAI_PROJECT_NAME` | OpenShift namespace for pipeline runs |
| `RHOAI_TRAIN_DATA_BUCKET` | S3 bucket containing training CSVs (passed to the pipeline as `train_data_bucket_name`) |
| `RHOAI_TRAIN_S3_SECRET_NAME` | Kubernetes secret name with S3 credentials (passed to the pipeline as `train_data_secret_name`; default: `s3-connection`) |

#### S3 artifact validation

| Variable | Purpose |
|---|---|
| `AWS_S3_ENDPOINT` | S3 endpoint URL (e.g. MinIO) |
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_DEFAULT_REGION` | S3 region (default: `us-east-1`) |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Bucket where pipeline outputs are written |

#### Pipeline YAMLs

| Variable | Purpose |
| -------- | ------- |
| `OGX_SECRET_NAME` | Secret with OGX client settings (e.g. API key, base URL). |
| `VECTOR_IO_PROVIDER_ID` | Registered vector I/O provider id in OGX. |

Optional fallbacks for `data_mode=existing_s3` when JSON omits buckets: `TEST_DATA_BUCKET_NAME`, `TEST_DATA_KEY`, `INPUT_DATA_BUCKET_NAME`, `INPUT_DATA_KEY`, or `TEST_DATA_SOURCE_BUCKET` / `TEST_DATA_SOURCE_PREFIX`.

JSON scenarios: `tests/config/autorag_test_configs.json`.

If selected configs use `upload` or `existing_s3`, **S3 env vars** and bucket defaults must satisfy the checks in `tests.lib.settings` (see `describe_autorag_integration_failure`).

## Controlling which tests run (env + pytest)

Scenario lists are read **when test modules import** (see `CONFIGS_FOR_RUN` in each `test_*_rhoai.py`). Set env vars **before** starting pytest (or in `tests/.env` so they load before collection).

### Filter JSON scenarios by tags

**`RHOAI_TEST_CONFIG_TAGS`** — Comma-separated list. Only scenarios whose `tags` in the JSON **intersect** this set (case-insensitive) are included.

Example (only scenarios tagged `smoke`):

```bash
export RHOAI_TEST_CONFIG_TAGS=smoke
pytest tests/scenarios/ -v
```

If no variable is set, all scenarios from the JSON files are eligible (subject to other filters).
|---|---|
| `AUTOML_TABULAR_PIPELINE_PATH` | Local path or `https://` URL to the compiled tabular pipeline YAML |
| `AUTOML_TIMESERIES_PIPELINE_PATH` | Local path or `https://` URL to the compiled time series pipeline YAML |

#### Test filtering, timeouts, caching

| Variable | Default | Purpose |
|---|---|---|
| `AUTOML_FUNCTIONAL_TESTS_TAGS` | — | Comma-separated tags — only scenarios that have **all** requested tags run. Unset = run all. |
| `AUTOML_TABULAR_TEST_CONFIGS_PATH` | — | Path to custom tabular test configs JSON. Overrides built-in `tabular_test_configs.json`. |
| `AUTOML_TIMESERIES_TEST_CONFIGS_PATH` | — | Path to custom timeseries test configs JSON. Overrides built-in `timeseries_test_configs.json`. |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | `3600` | Max seconds to wait for a pipeline run |
| `KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT` | `true` | Disable KFP step caching |
| `AUTOML_FUNCTIONAL_TEST_KEEP_ARTIFACTS` | `false` | Skip S3 artifact cleanup after the session |

#### Model serving (optional)

Set `RHOAI_DEPLOY_AFTER_TRAINING=true` to deploy the top trained model via KServe and run inference after each positive-path pipeline run. Also requires `RHOAI_URL`.

| Variable | Default | Purpose |
|---|---|---|
| `RHOAI_URL` | — | OpenShift API URL (required for KServe deployment) |
| `RHOAI_DEPLOY_AFTER_TRAINING` | `false` | Enable post-training KServe deployment |
| `RHOAI_SERVING_IMAGE` | — | Container image for the AutoGluon ServingRuntime |
| `RHOAI_SERVING_RUNTIME_NAME` | — | Existing ServingRuntime to reuse (skips creation) |
| `RHOAI_CREATE_SERVING_RUNTIME` | `false` | Create the ServingRuntime if missing (requires `RHOAI_SERVING_IMAGE`) |
| `RHOAI_INFERENCE_TIMEOUT` | `300` | Seconds to wait for InferenceService to become Ready |
| `RHOAI_KSERVE_STORAGE_KEY` | — | Existing Data Connection secret for KServe storage; a temporary one is created when unset |
| `RHOAI_HARDWARE_PROFILE_NAME` | `default-profile` | HardwareProfile CR name for the predictor pod |
| `RHOAI_HARDWARE_PROFILE_NAMESPACE` | `redhat-ods-applications` | Namespace of the HardwareProfile CR |
| `RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION` | — | Override `resourceVersion` fetch (useful in air-gapped envs) |
| `RHOAI_PREDICTOR_CPU` | `2` | CPU request/limit for the predictor container |
| `RHOAI_PREDICTOR_MEMORY` | `4Gi` | Memory request/limit for the predictor container |
| `RHOAI_KSERVE_CA_BUNDLE_CONFIGMAP` | — | ConfigMap name for custom CA bundle (MinIO with self-signed TLS) |

Time series deployments automatically set `AUTOGLUON_TS_ID_COLUMN` / `AUTOGLUON_TS_TIMESTAMP_COLUMN` on the predictor container when the test config's column names differ from AutoGluon defaults (`item_id` / `timestamp`).

### Running the tests

```bash
# All AutoML functional tests
pytest autox_tests/automl/ -v

# Tabular only
pytest autox_tests/automl/test_tabular_functional.py -v

# Time series only
pytest autox_tests/automl/test_timeseries_functional.py -v

# Smoke scenarios only
AUTOML_FUNCTIONAL_TESTS_TAGS=smoke pytest autox_tests/automl/ -v

# Negative scenarios only
pytest autox_tests/automl/ -m negative -v

# Single scenario
pytest autox_tests/automl/ -k "TC-A-1_regression" -v
```

### Test scenarios

#### Tabular (`tabular_test_configs.json`)

| ID | Task | Dataset | Label column | top_n | Tags |
|---|---|---|---|---|---|
| TC-A-1_regression | regression | housing pricing | `price` | 1 | smoke |
| TC-A-2_binary_classification | binary | Titanic | `Survived` | 3 | smoke |
| TC-A-3_multiclass | multiclass | car rental | `Action` | 2 | — |
| TC-A-4_energy_regression | regression | UCI Energy Efficiency | `Heating.Load` | 5 | — |
| TC-A-5_credit_default_binary | binary | UCI Credit Default | `default.payment.next.month` | 3 | — |
| TC-A-6_wine_multiclass | multiclass | UCI Wine Quality (red+white) | `quality` | 1 | — |
| TC-NA-1_invalid_task_type | — | — | — | — | negative, validation |
| TC-NA-2_invalid_top_n_zero | — | — | — | — | negative, validation |
| TC-NA-3_label_column_absent | — | — | — | — | negative, data |
| TC-NA-4_missing_s3_object | — | — | — | — | negative, storage |
| TC-NA-5_task_data_mismatch | — | — | — | — | negative, data |
| TC-NA-6_bad_credentials | — | — | — | — | negative, credentials |

#### Time series (`timeseries_test_configs.json`)

| ID | Frequency | Dataset | Tags |
|---|---|---|---|
| TC-B-1_timeseries_fruits_with_covariate | daily | fruits daily price (with covariate) | smoke, renamed_schema, covariate |
| TC-B-2_timeseries_m4_hourly | hourly | M4 hourly subset | hourly, standard_schema |
| TC-NB-1_invalid_target | — | — | negative, data |
| TC-NB-2_invalid_prediction_length | — | — | negative, validation |
| TC-NB-3_missing_s3_object | — | — | negative, storage |
| TC-NB-4_bad_credentials | — | — | negative, credentials |
| TC-NB-5_invalid_top_n_zero | — | — | negative, validation |

### Test data sources

Training datasets used by the positive scenarios are stored in the S3 bucket defined by `RHOAI_TRAIN_DATA_BUCKET`. The files for the original scenarios ship in `automl/data/` for local reference; the three datasets added for RHAIENG-4179 pairwise coverage are described below.

#### Tabular datasets

| Scenario | S3 key | Origin | Preparation |
|---|---|---|---|
| TC-A-4 | `functional-test/tabular/regression/energy_efficiency_regression.csv` | [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) — Tsanas & Xifara (2012) | Exported from the original `.xlsx`; columns renamed with dot notation: `X1`→`Relative.Compactness`, `Y1`→`Heating.Load`, etc. Target is `Heating.Load`; `Cooling.Load` is retained as a feature. Source file: `automl/data/regression/energy_efficiency_regression.xlsx`. |
| TC-A-5 | `functional-test/tabular/classification_binary/credit_default_binary.csv` | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — Yeh & Lien (2009) | Read with `header=1` to skip the secondary `X1…X24` header row present in the original XLS. Label column renamed from `default payment next month` (spaces) to `default.payment.next.month` (dots) to exercise the dot-style label path. Source file: `automl/data/regression/default of credit card clients.xls`. |
| TC-A-6 | `functional-test/tabular/classification_multiclass/wine_quality_multiclass_limited_classes.csv` | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) — Cortez et al. (2009) | Red (`winequality-red.csv`) and white (`winequality-white.csv`) files merged with `pd.concat`. Classes with fewer than 50 samples removed (`quality=3` with 30 rows, `quality=9` with 5 rows) to avoid NaN pseudo-labels during AutoGluon's stacking full-refit. Final dataset: 6,462 rows, classes 4–8. Source files: `automl/data/classification_multi/winequality-red.csv`, `automl/data/classification_multi/winequality-white.csv`. |

#### `inference_sample` format

Positive scenarios include an `inference_sample` sent as the `instances` payload to the KServe `/v1/models/<name>:predict` endpoint. The format differs by suite:

**Time series** — row-oriented list of plain-scalar dicts, sent verbatim:

```json
"inference_sample": [
  { "item_id": "H1", "timestamp": "1750-01-01 00:00:00", "target": 605.0 },
  { "item_id": "H1", "timestamp": "1750-01-01 01:00:00", "target": 586.0 }
]
```

**Tabular** — column-oriented: a single dict per sample where each value is a list; converted to per-row instances before scoring:

```json
"inference_sample": [
  {
    "area": [7420], "bedrooms": [4], "bathrooms": [2],
    "mainroad": ["yes"], "furnishingstatus": ["furnished"]
  }
]
```

When `RHOAI_DEPLOY_AFTER_TRAINING=true` and `inference_sample` is present, the test scores the deployed model and asserts non-empty predictions are returned.

#### `expected_outcome` (negative scenarios)

Negative scenario entries include an `expected_outcome` field: a human-readable description of the expected failure mode (e.g. `"Fail fast with clear validation message"`). This field is informational only — it is not evaluated by the test runner. It exists to document design intent and aid debugging when a scenario passes unexpectedly.

### Pass criteria

**Positive scenarios:**
- Pipeline run reaches `SUCCEEDED` within `RHOAI_PIPELINE_RUN_TIMEOUT`
- At least one model with a metrics JSON exists in S3
- Primary metric present (`r2` for regression, `accuracy` for classification, `MASE` for time series)
- Leaderboard HTML artifact exists in S3
- Sampled test dataset CSV exists in S3
- *(when `RHOAI_DEPLOY_AFTER_TRAINING=true`)* InferenceService becomes Ready and returns non-empty predictions

**Negative scenarios:**
- Pipeline run reaches `FAILED` within 600 s
- At least one of `expected_failing_task` names appears among the run's failed tasks

### Troubleshooting

- **Tests skip with "AutoML functional test env not set"** — one of the required variables is missing; check `.env.ml` against `.env.ml.example`.
- **HardwareProfile 404** — `RHOAI_HARDWARE_PROFILE_NAME` does not exist on the cluster. Run `oc get hardwareprofile -n redhat-ods-applications` to find the correct name, or set `RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION` to skip the live fetch.
- **InferenceService OOMKilled** — increase `RHOAI_PREDICTOR_MEMORY` (default `4Gi`; AutoGluon models can be large).
- **Scoring HTTP 500** — check pod logs; the test captures and prints them automatically on failure.
- **ISVC creation HTTP 500 (`no endpoints available for service "kserve-webhook-server-service"`)** — the KServe webhook pod is down. Run `oc rollout restart deployment/kserve-controller-manager -n redhat-ods-applications` and wait for it to become ready before re-running the test.
- **ISVC creation HTTP 500 (`no endpoints available for service "rhods-operator-service"`)** — the RHODS operator webhook pod is down. Run `oc rollout restart deployment/rhods-operator -n redhat-ods-operator` and wait for it to become ready before re-running the test.
- **`boto3` / `kubernetes` import errors** — re-run `uv sync --extra test_automl`.

---

## AutoRAG functional tests

End-to-end tests for the Documents RAG Optimization pipeline. Submits pipeline runs to KFP, validates S3 artifacts, and optionally executes generated notebooks via papermill.

### Directory layout

```
autox_tests/
├── .env.rag.example                       # env template — copy to .env.rag and fill in
└── autorag/
    ├── conftest.py                        # pytest fixtures (KFP client, S3 client, pipeline YAML)
    ├── test_pipeline_functional.py        # parametrized positive + negative tests
    ├── utils.py                           # run submission, diagnostics, artifact validation
    └── configs/
        ├── configs.py                     # AutoRAGTestConfig dataclass + config loader
        └── test_configs.json             # scenario definitions
```

### Prerequisites

```bash
uv sync --extra test_autorag
# or
pip install -e ".[test_autorag]"
```

You also need a running RHOAI cluster with Data Science Pipelines and a Llama Stack instance with a Milvus vector I/O provider.

### Environment setup

```bash
cp autox_tests/.env.rag.example autox_tests/.env.rag
# edit .env.rag with your cluster details
```

#### Required

| Variable | Purpose |
|---|---|
| `RHOAI_KFP_URL` | Data Science Pipelines HTTP API URL |
| `RHOAI_TOKEN` | Bearer token for KFP |
| `RHOAI_PROJECT_NAME` | OpenShift namespace for pipeline runs |
| `AUTORAG_PIPELINE_PATH` | Local path or `https://` URL to the compiled AutoRAG pipeline YAML |
| `TEST_DATA_SECRET_NAME` | Kubernetes secret with S3 credentials for test data |
| `TEST_DATA_BUCKET_NAME` | S3 bucket for test data |
| `INPUT_DATA_BUCKET_NAME` | S3 bucket for input documents |
| `INPUT_DATA_SECRET_NAME` | Kubernetes secret for input data bucket |
| `LLAMA_STACK_SECRET_NAME` | Kubernetes secret with Llama Stack client settings |

#### S3 artifact validation (optional)

| Variable | Purpose |
|---|---|
| `ARTIFACTS_AWS_ACCESS_KEY_ID` | S3 access key for artifact bucket |
| `ARTIFACTS_AWS_SECRET_ACCESS_KEY` | S3 secret key for artifact bucket |
| `ARTIFACTS_AWS_S3_ENDPOINT` | S3 endpoint for artifact bucket |
| `ARTIFACTS_AWS_DEFAULT_REGION` | S3 region (default: `us-east-1`) |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Bucket where pipeline outputs are written |

#### Notebook execution (optional)

| Variable | Purpose |
|---|---|
| `LLAMA_STACK_CLIENT_BASE_URL` | Llama Stack API base URL for notebook execution |
| `LLAMA_STACK_CLIENT_API_KEY` | Llama Stack API key |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION`, `AWS_S3_BUCKET` | S3 credentials injected into notebook execution environment |

#### Constrained model lists (optional)

| Variable | Purpose |
|---|---|
| `FUNC_TEST_EMBEDDING_MODELS` | Override embeddings models used in tests |
| `FUNC_TEST_GENERATION_MODELS` | Override generation models used in tests |

#### Test filtering and timeouts

| Variable | Default | Purpose |
|---|---|---|
| `TESTS_TAGS` | — | Comma-separated tags — only matching scenarios run. Unset = run all. |
| `AUTORAG_TEST_CONFIGS_PATH` | — | Path to custom AutoRAG test configs JSON. Overrides built-in `test_configs.json`. |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | `3600` | Max seconds to wait for a pipeline run |
| `K8S_API_URL` | — | Kubernetes API URL for pod log fetching (derived from KFP URL when unset) |

### Running the tests

```bash
# All AutoRAG functional tests
pytest autox_tests/autorag/ -v

# Positive scenarios only
pytest autox_tests/autorag/ -m positive -v

# Smoke scenarios only
TESTS_TAGS=smoke pytest autox_tests/autorag/ -v

# Single scenario
pytest autox_tests/autorag/ -k "TC-P-1" -v
```

### Test scenarios

Scenarios are defined in `configs/test_configs.json`. Each entry specifies `id`, `description`, `tags`, `expected_result` (`"pass"` or `"fail"`), `llama_stack_vector_io_provider_id`, and `pipeline_params_overrides`.

### Pass criteria

**Positive scenarios:**
- Pipeline run reaches `SUCCEEDED`
- At least one pattern artifact exists in S3
- Indexing notebook, inference notebook, and `evaluation_results.json` exist in S3
- A randomly selected indexing and inference notebook executes successfully via papermill

**Negative scenarios:**
- Pipeline run reaches `FAILED` within 600 s
- Failure details are logged

### Troubleshooting

- **Tests skip** — check that all required variables are set in `.env.rag`.
- **Pod log fetch fails** — set `K8S_API_URL` explicitly if the automatic derivation from the KFP URL does not match your cluster pattern.
- **Notebook execution fails** — ensure `LLAMA_STACK_CLIENT_BASE_URL`, `LLAMA_STACK_CLIENT_API_KEY`, and AWS vars are set; check the papermill output in the test log.
- **`nbformat` / `papermill` import errors** — re-run `uv sync --extra test_autorag`.
