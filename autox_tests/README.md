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

### Resolve the notebook runner image

The helper below reads the workbench image shipped by the installed RHOAI CSV and prints it for use as `RHOAI_NOTEBOOK_RUNNER_IMAGE`:

```bash
export RHOAI_NOTEBOOK_RUNNER_IMAGE="$(autox_tests/scripts/get_workbench_jupyter_image.sh)"
```

It requires `oc`, `jq`, and an authenticated `oc` session with cluster-admin privileges (or an equivalently scoped role that can read CSVs in `redhat-ods-operator`).

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

### Pipeline source selection

Managed pipelines are the default. To upload a compiled YAML package instead, set `RHOAI_USE_MANAGED_PIPELINES=false` and provide the suite's pipeline-path variable, or use `./run_tests.sh --legacy-pipeline-yaml`. The wrapper sources the selected `.env` files before pytest starts and then resolves the pipeline mode.

For AutoML, package mode requires `AUTOML_TABULAR_PIPELINE_PATH` and/or `AUTOML_TIMESERIES_PIPELINE_PATH`. For AutoRAG optimization, it requires `AUTORAG_PIPELINE_PATH`. AutoRAG indexing selects package mode independently when `AUTORAG_INDEXING_PIPELINE_PATH` is set.

---

## Custom test configurations

When `autox-ci` is used as a submodule, downstream repos can supply their own test config JSON files to define different datasets, models, or scenario definitions. Override the built-in configs via environment variables or CLI flags:

| Env variable | CLI flag | Overrides |
|---|---|---|
| `AUTORAG_TEST_CONFIGS_PATH` | `--rag-configs` | `autorag/configs/optimisation_test_configs.json` |
| `AUTOML_TABULAR_TEST_CONFIGS_PATH` | `--tabular-configs` | `automl/configs/tabular_test_configs.json` |
| `AUTOML_TIMESERIES_TEST_CONFIGS_PATH` | `--timeseries-configs` | `automl/configs/timeseries_test_configs.json` |

Custom JSON files must follow the same schema as the built-in configs they replace. The dataclass fields in `configs/configs.py` define the expected keys.

---

## AutoML functional tests

End-to-end tests for the AutoGluon tabular and time series training pipelines. They validate pipeline runs and S3 artifacts; individual positive scenarios can optionally run a generated notebook or deploy a trained model through KServe.

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

#### Pipeline package mode

| Variable | Purpose |
| -------- | ------- |
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

#### Per-scenario optional checks

Positive entries in both AutoML JSON files support these fields; both default to `false`:

```json
"run_notebook": false,
"deploy": false
```

Set `run_notebook` to `true` to execute the selected model notebook in a Kubernetes Job. Set `deploy` to `true` to perform the KServe inference check described below. Negative scenarios do not use either field.

#### Notebook execution (Kubernetes Job)

`run_notebook: true` needs a runner image. The Job overrides the image's normal command, downloads the generated notebook from S3, and executes it with Papermill. The image must contain Python, `boto3`, `papermill`, the notebook's runtime dependencies, and the configured Jupyter kernel.

| Variable | Default | Purpose |
|---|---|---|
| `RHOAI_NOTEBOOK_RUNNER_IMAGE` | — | Image used for notebook Jobs. Without it, enabled notebook checks are skipped. |
| `RHOAI_NOTEBOOK_JOB_TIMEOUT` | `900` | Maximum seconds to wait for a notebook Job. |
| `RHOAI_NOTEBOOK_KERNEL_NAME` | `python3` | Registered kernel used when a notebook has no kernelspec. |
| `S3_SSL_VERIFY` | `true` | Verify S3 TLS in the test process and notebook Job. Set to `false` only for a trusted development endpoint with a self-signed certificate. |

The test service account needs namespace-scoped permissions to create, read, list, and delete `batch/jobs`; read/list Pods; read `pods/log`; and read the referenced Secrets. Cluster-admin access is not required; the standard namespace `edit` role is normally sufficient.

#### Model serving (optional)

Set a positive scenario's `deploy` field to `true` to deploy its top trained model via KServe and run inference. Deployment also requires `RHOAI_URL`.

| Variable | Default | Purpose |
|---|---|---|
| `RHOAI_URL` | — | OpenShift API URL (required for KServe deployment) |
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

# User-provided test dataset scenarios (requires IR with test_data_bucket_name + test_data_file_key)
AUTOML_FUNCTIONAL_TESTS_TAGS=user_test_data pytest autox_tests/automl/ -v
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
| TC-A-7_user_provided_test_data | binary | German credit (biased) train + test | `Risk` | 1 | user_test_data |
| TC-NA-1_invalid_task_type | — | — | — | — | negative, validation |
| TC-NA-2_invalid_top_n_zero | — | — | — | — | negative, validation |
| TC-NA-3_label_column_absent | — | — | — | — | negative, data |
| TC-NA-4_missing_s3_object | — | — | — | — | negative, storage |
| TC-NA-5_task_data_mismatch | — | — | — | — | negative, data |
| TC-NA-6_bad_credentials | — | — | — | — | negative, credentials |
| TC-NA-7_user_test_missing_object | — | — | — | — | negative, storage, user_test_data |
| TC-NA-8_user_test_schema_mismatch | — | — | — | — | negative, data, user_test_data |

#### Time series (`timeseries_test_configs.json`)

| ID | Frequency | Dataset | Tags |
|---|---|---|---|
| TC-B-1_timeseries_fruits_with_covariate | daily | fruits daily price (with covariate) | smoke, renamed_schema, covariate |
| TC-B-2_timeseries_m4_hourly | hourly | M4 hourly subset | hourly, standard_schema |
| TC-B-3_user_provided_test_data | daily | Poland COVID-19 cases train + test | user_test_data |
| TC-NB-1_invalid_target | — | — | negative, data |
| TC-NB-2_invalid_prediction_length | — | — | negative, validation |
| TC-NB-3_missing_s3_object | — | — | negative, storage |
| TC-NB-4_bad_credentials | — | — | negative, credentials |
| TC-NB-5_invalid_top_n_zero | — | — | negative, validation |
| TC-NB-6_user_test_missing_object | — | — | negative, storage, user_test_data |
| TC-NB-7_user_test_series_too_short | — | — | negative, data, user_test_data |

### Test data sources

Training datasets used by the positive scenarios are stored in the S3 bucket defined by `RHOAI_TRAIN_DATA_BUCKET`. The files for the original scenarios ship in `automl/data/` for local reference; the three datasets added for RHAIENG-4179 pairwise coverage are described below.

#### User-provided test datasets

| Scenario | S3 key | Origin | Preparation |
|---|---|---|---|
| TC-A-7 | `functional-test/tabular/classification_binary/german_credit_data_biased_train.csv` / `german_credit_data_biased_test.csv` | German credit risk with a Sex-biased train/test split (1000 / 400 rows, label `Risk`) | Pre-split user test path. Both files are de-duplicated, so the test artifact has exactly 400 rows; a default 80/20 holdout of train would be 200. Down-sampled from 3500/1500 by stratifying on `Risk` x `Sex`, which keeps the label balance and the intended Sex bias while keeping the scenario short enough for CI. |
| TC-NA-8 | `functional-test/tabular/classification_binary/german_credit_data_biased_test_missing_features.csv` | Same credit-risk label column without training features | Fail-fast schema mismatch in the data loader. |
| TC-B-3 | `functional-test/timeseries/poland_daily_cases_03_03_2021.csv` / `poland_daily_cases_03_04-28_2021.csv` | Poland daily COVID-19 cases: train through 2021-03-03 (407 rows), test 2021-03-04–03-28 (25 rows) | Single-series (`id_column` empty; loader injects `__synthetic_item_id`). 25 test rows is longer than `prediction_length=7` and much smaller than a default ~80-row holdout. Artifact fingerprint is `2021-03-28` after timestamp cleansing. The scoring payload sends `__synthetic_item_id="item_0"` explicitly — the current autogluonserver KServe build rejects instances without the id column even when it was loader-injected; drop that field once a build that infers it ships. |
| TC-NB-7 | Same Poland COVID train/test split | Test series is 25 rows | Fail-fast with `prediction_length=25` (each series must be longer than `prediction_length`). |

These `user_test_data` scenarios require compiled AutoML pipeline IR that declares `test_data_bucket_name` and `test_data_file_key` ([pipelines-components#227](https://github.com/opendatahub-io/pipelines-components/pull/227)+). S3 credentials for the external test object come from `train_data_secret_name` (mounted as both `AWS_*` and `TEST_DATA_AWS_*`); there is no separate `test_data_secret_name` pipeline input. Existing scenarios omit the test-data arguments so they still run against older IR.

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

When a scenario sets `deploy: true` and provides an `inference_sample`, the test scores the deployed model and asserts non-empty predictions are returned.

#### `expected_outcome` (negative scenarios)

Negative scenario entries include an `expected_outcome` field: a human-readable description of the expected failure mode (e.g. `"Fail fast with clear validation message"`). This field is informational only — it is not evaluated by the test runner. It exists to document design intent and aid debugging when a scenario passes unexpectedly.

#### `expected_error_pattern` (negative scenarios)

A regex matched (case-insensitively) against the run's failure details — task errors plus the logs of every failed pod. `expected_failing_task` alone only proves *something* broke in the right component; a missing S3 fixture, bad credentials or an OOM would all satisfy it. Pinning the message means the scenario passes only for the fault it injects:

```json
"expected_failing_task": ["timeseries-data-loader"],
"expected_error_pattern": "prediction_length must be greater than 0"
```

It also makes `expected_failing_task` cheap to keep loose: once the reason is pinned, listing both the loader and the training tasks tolerates validation moving between components across IR versions without ever accepting a wrong-reason failure.

The field is optional. The two `bad_credentials` scenarios leave it unset on purpose — the pod never starts when the secret is missing, so there are no logs to match; they carry an `expected_error_pattern_comment` recording why. When a pattern is set but no pod logs could be collected, the assertion message says so, since Kubernetes connectivity (not the pipeline) is then the likely cause.

#### `missing_object_keys` (negative scenarios)

S3 keys whose *absence* is the injected fault. Everything else referenced by any scenario is treated as a required fixture: before the first run is submitted, the session fixture uploads it (when `AUTOML_UPLOAD_TEST_DATASETS=true`) and then verifies with `head_object` that every required key exists and every `missing_object_keys` entry does not. A fixture that never reached the bucket fails at setup with a clear message instead of surfacing twenty minutes later as a `NoSuchKey` that reads like a product bug. Uploads no longer skip an unmatched key silently — only keys listed here may be absent locally.

### Pass criteria

**Positive scenarios:**
- Pipeline run reaches `SUCCEEDED` within `RHOAI_PIPELINE_RUN_TIMEOUT`
- At least one model with a metrics JSON exists in S3
- Primary metric present (`r2` for regression, `accuracy` for classification, `MASE` for time series)
- Leaderboard HTML artifact exists in S3
- Sampled test dataset CSV exists in S3
- User-provided test scenarios (`user_test_data` tag): `sampled_test_dataset` matches the external CSV row count rather than a default 80/20 holdout
- When `run_notebook: true`, a selected predictor notebook completes in a Kubernetes Job
- *(when `deploy: true`)* InferenceService becomes Ready and returns non-empty predictions

**Negative scenarios:**
- Pipeline run reaches `FAILED` within 600 s
- At least one of `expected_failing_task` names appears among the run's failed tasks (the list also contains DAG nodes — the root pipeline and `condition-*` groups — which fail whenever any child does; ignore them when reading a failure message)
- `expected_error_pattern`, when set, matches the task errors or failed-pod logs

### Troubleshooting

- **Tests skip with "AutoML functional test env not set"** — one of the required variables is missing; check `.env.ml` against `.env.ml.example`.
- **HardwareProfile 404** — `RHOAI_HARDWARE_PROFILE_NAME` does not exist on the cluster. Run `oc get hardwareprofile -n redhat-ods-applications` to find the correct name, or set `RHOAI_HARDWARE_PROFILE_RESOURCE_VERSION` to skip the live fetch.
- **InferenceService OOMKilled** — increase `RHOAI_PREDICTOR_MEMORY` (default `4Gi`; AutoGluon models can be large).
- **Scoring HTTP 500** — check pod logs; the test captures and prints them automatically on failure.
- **Notebook Job fails** — confirm `RHOAI_NOTEBOOK_RUNNER_IMAGE` contains `boto3`, `papermill`, and the generated notebook's runtime dependencies; the failed Job pod log is included in the test failure.
- **ISVC creation HTTP 500 (`no endpoints available for service "kserve-webhook-server-service"`)** — the KServe webhook pod is down. Run `oc rollout restart deployment/kserve-controller-manager -n redhat-ods-applications` and wait for it to become ready before re-running the test.
- **ISVC creation HTTP 500 (`no endpoints available for service "rhods-operator-service"`)** — the RHODS operator webhook pod is down. Run `oc rollout restart deployment/rhods-operator -n redhat-ods-operator` and wait for it to become ready before re-running the test.
- **Every remaining test fails with `KFP API returned 401 Unauthorized`** — `RHOAI_TOKEN` expired part-way through the run. A full AutoML suite takes well over an hour; refresh the token (`oc whoami -t`) in `.env.ml` before starting, or run a tag-filtered subset. (Without the guard in `make_kfp_client`, the KFP SDK reacts to the 401 by trying a GCP token refresh, gets `None`, and every later call dies inside urllib3 with `TypeError: expected string or bytes-like object, got 'NoneType'`.)
- **`boto3` / `kubernetes` import errors** — re-run `uv sync --extra test_automl`.

---

## AutoRAG functional tests

End-to-end tests for the Documents RAG optimization and indexing pipelines. They submit pipeline runs to KFP, validate S3 artifacts, and can execute generated optimization notebooks in Kubernetes Jobs.

### Directory layout

```
autox_tests/
├── .env.rag.example                       # env template — copy to .env.rag and fill in
└── autorag/
    ├── conftest.py                        # pytest fixtures (KFP client, S3 client, pipeline YAML)
    ├── test_pipeline_functional.py        # optimization positive + negative tests
    ├── test_indexing_pipeline_functional.py # indexing positive + negative tests
    ├── utils.py                           # run submission, diagnostics, artifact validation
    └── configs/
        ├── configs.py                     # dataclasses + config loaders
        ├── optimisation_test_configs.json # optimization scenario definitions
        └── indexing_test_configs.json     # indexing scenario definitions
```

### Prerequisites

```bash
uv sync --extra test_autorag
# or
pip install -e ".[test_autorag]"
```

You also need a running RHOAI cluster with Data Science Pipelines, a MaaS (Model-as-a-Service) inference endpoint, and a vector database (Milvus or PGVector).

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
| `TEST_DATA_SECRET_NAME` | Kubernetes secret with S3 credentials for test data |
| `TEST_DATA_BUCKET_NAME` | S3 bucket for test data |
| `INPUT_DATA_BUCKET_NAME` | S3 bucket for input documents |
| `INPUT_DATA_SECRET_NAME` | Kubernetes secret for input data bucket |
| `MAAS_SECRET_NAME` | Kubernetes secret with MaaS inference settings (`MAAS_BASE_URL`, `MAAS_API_KEY`) |
| `VECTOR_DB_SECRET_NAME` | Kubernetes secret with vector DB connection (`MILVUS_*` or `PGVECTOR_*` keys) |

#### S3 artifact validation (optional)

| Variable | Purpose |
|---|---|
| `ARTIFACTS_AWS_ACCESS_KEY_ID` | S3 access key for artifact bucket |
| `ARTIFACTS_AWS_SECRET_ACCESS_KEY` | S3 secret key for artifact bucket |
| `ARTIFACTS_AWS_S3_ENDPOINT` | S3 endpoint for artifact bucket |
| `ARTIFACTS_AWS_DEFAULT_REGION` | S3 region (default: `us-east-1`) |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Bucket where pipeline outputs are written |

#### Pipeline package mode

Managed AutoRAG pipelines are used by default. `AUTORAG_PIPELINE_PATH` is required only when package mode is selected with `RHOAI_USE_MANAGED_PIPELINES=false` or `./run_tests.sh --legacy-pipeline-yaml`. Indexing does not use that switch: set `AUTORAG_INDEXING_PIPELINE_PATH` to use a compiled indexing YAML, otherwise its managed pipeline is used.

#### Notebook execution (optimization only)

| Variable | Default | Purpose |
|---|---|---|
| `RHOAI_NOTEBOOK_RUNNER_IMAGE` | — | Image containing Python, `boto3`, `papermill`, and notebook dependencies. Required when an optimization scenario enables `run_notebook`. |
| `RHOAI_NOTEBOOK_JOB_TIMEOUT` | `900` | Maximum seconds to wait for each notebook Job. |
| `RHOAI_NOTEBOOK_KERNEL_NAME` | `python3` | Jupyter kernel registered in the runner image. |
| `S3_SSL_VERIFY` | `true` | Verify S3 TLS in the notebook Job; use `false` only for a trusted development endpoint with a self-signed certificate. |
| `MAAS_SECRET_NAME`, `VECTOR_DB_SECRET_NAME` | — | Existing secrets injected into AutoRAG notebook Jobs. |
| `RHOAI_TEST_S3_SECRET_NAME` | — | Existing S3 secret injected into notebook Jobs. |

#### Model lists (required by the MaaS pipeline)

| Variable | Purpose |
|---|---|
| `AUTORAG_EMBEDDING_MODELS` | Embedding model IDs (JSON array / comma-separated) for optimization configs using `"env"` |
| `AUTORAG_GENERATION_MODELS` | Generation model IDs (JSON array / comma-separated) for optimization configs using `"env"` |
| `AUTORAG_INDEXING_EMBEDDING_MODEL_ID` | Single embedding model ID for indexing positive tests using `"env"` |

#### Test filtering and timeouts

| Variable | Default | Purpose |
|---|---|---|
| `AUTORAG_FUNCTIONAL_TESTS_TAGS` | — | Comma-separated tags — only matching scenarios run. Unset = run all. |
| `AUTORAG_TEST_CONFIGS_PATH` | — | Path to custom optimization configs JSON. Overrides built-in `optimisation_test_configs.json`. |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | `3600` | Max seconds to wait for a pipeline run |
| `K8S_API_URL` | — | Kubernetes API URL for pod log fetching (derived from KFP URL when unset) |

### Running the tests

```bash
# All AutoRAG functional tests
pytest autox_tests/autorag/ -v

# Positive scenarios only
pytest autox_tests/autorag/ -m positive -v

# Optimization tests only
pytest autox_tests/autorag/test_pipeline_functional.py -v

# Indexing tests only
pytest autox_tests/autorag/test_indexing_pipeline_functional.py -v

# Smoke scenarios only
AUTORAG_FUNCTIONAL_TESTS_TAGS=smoke pytest autox_tests/autorag/ -v

# Single scenario
pytest autox_tests/autorag/ -k "TC-P-1" -v
```

### Test scenarios

Scenarios are defined in `configs/optimisation_test_configs.json` (optimization) and `configs/indexing_test_configs.json` (indexing). Each entry specifies `id`, `description`, `tags`, `expected_result` (`"pass"` or `"fail"`), the required model lists (`embedding_models` / `generation_models`, or `"env"`), and per-scenario parameter overrides. The vector-store backend is auto-detected from `VECTOR_DB_SECRET_NAME`; it is no longer a scenario field.

Only positive optimization scenarios accept `"run_notebook": true`; it defaults to `false`. This runs the best pattern's indexing and inference notebooks sequentially in one Kubernetes Job pod. AutoRAG has no `deploy` field, and indexing scenarios do not run notebook Jobs.

### Pass criteria

**Positive scenarios:**
- Pipeline run reaches `SUCCEEDED`
- At least one pattern artifact exists in S3
- Indexing notebook, inference notebook, and `evaluation_results.json` exist in S3
- When the optimization scenario has `run_notebook: true`, the best pattern's indexing and inference notebooks complete sequentially in one Kubernetes Job pod

**Negative scenarios:**
- Pipeline run reaches `FAILED` within 600 s
- Failure details are logged

### Troubleshooting

- **Tests skip** — check that all required variables are set in `.env.rag`.
- **Pod log fetch fails** — set `K8S_API_URL` explicitly if the automatic derivation from the KFP URL does not match your cluster pattern.
- **Notebook Job fails** — confirm the runner image includes `boto3`, `papermill`, and the notebook's dependencies; the failed Job pod log is included in the test failure.
