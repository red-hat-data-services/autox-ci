# Root `tests/` suite — OpenShift AI / Kubeflow Pipelines integration

This directory contains **optional** end-to-end tests that submit real pipeline runs to **OpenShift AI** (Data Science Pipelines / KFP v2) using cluster credentials and object storage. They are separate from per-component tests under `components/` and `pipelines/`.

## Prerequisites

- Python 3.11+ and a project virtualenv (see repository `CONTRIBUTING.md`).
- Install extras for these tests (KFP client, S3, Kubernetes client, `python-dotenv`):

  ```bash
  pip install -e ".[test_rhoai]"
  ```

  (`test_automl` pulls the same stack; `test_rhoai` is the name aligned with this suite.)

- A reachable OpenShift/Kubernetes API (`RHOAI_URL`), bearer token, S3-compatible storage, and (for most flows) a KFP API URL or DSPA creation enabled — see [Environment variables](#environment-variables).

## Environment variables

### Loading order

1. **`tests/.env`** — If present, it is loaded at pytest startup (`pytest_configure` in `tests/scenarios/conftest.py`) via `tests.lib.env.load_tests_env`. **Variables already set in the process environment are not overwritten** (exports and CI secrets win).
2. **Shell / CI** — Export variables or inject them in your runner; they take precedence over `tests/.env`.

Copy the template and edit:

```bash
cp tests/.env.example tests/.env
```

Canonical field descriptions and optional knobs are in **`tests/.env.example`**. Below is a condensed map of what each test family needs.

### Shared (cluster + object storage)

| Variable | Purpose |
| -------- | ------- |
| `RHOAI_URL` | OpenShift/Kubernetes API URL (used to ensure project and apply the S3 secret). |
| `RHOAI_TOKEN` | Bearer token for API and KFP. |
| `RHOAI_PROJECT_NAME` | Namespace for runs and secrets (default in example: `kfp-integration-test`). |
| `RHOAI_KFP_URL` | Data Science Pipelines HTTP API base URL (optional if DSPA is created — see below). |
| `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | S3 client and secret data for `RHOAI_TEST_S3_SECRET_NAME`. |
| `RHOAI_TEST_DATA_BUCKET` | Bucket for uploaded fixtures (AutoML CSVs; AutoRAG docs + benchmark when `data_mode=upload`). |
| `RHOAI_TEST_S3_SECRET_NAME` | Kubernetes secret name in the project (default `s3-connection`). |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Optional; used when creating DSPA with object storage wiring. |

**HTTPS verification for S3 and KFP** (boto3 uploads, `kfp.Client`): By default TLS certificates are verified. For self-signed endpoints (common lab MinIO gateways), set **`RHOAI_HTTPS_VERIFY=false`** (or `0` / `no` / `off`). The legacy alias **`KFP_VERIFY_SSL`** is still honored when `RHOAI_HTTPS_VERIFY` is unset. This does not change how pipeline runs inside the cluster talk to OGX; those use the component’s own retry logic.

**TLS for the Kubernetes client** (`RHOAI_URL`): By default the suite may skip TLS verify for typical lab clusters. For enterprise CA or stricter verification, use `RHOAI_OPENSHIFT_CA_BUNDLE_PATH` or `RHOAI_OPENSHIFT_CA_DATA`, and read the comments in `tests/.env.example` for `RHOAI_OPENSHIFT_API_INSECURE_TLS`.

**Aliases** (AutoRAG also accepts): `KFP_HOST` → KFP URL, `KFP_TOKEN` → token, `KFP_NAMESPACE` → project.

### Optional: create a DataSciencePipelinesApplication (DSPA)

If the operator is available and you set **`RHOAI_CREATE_DSPA=true`**, the suite can create a DSPA and discover the pipeline API from the `ds-pipeline` route, so **`RHOAI_KFP_URL` may be omitted**. Tunables: `RHOAI_DSPA_*` (API group/version, route prefix, wait timeouts, etc.) — see `tests/.env.example`.

### AutoML (tabular + time series)

Requires the **shared** block plus:

- `RHOAI_TEST_DATA_BUCKET` (training data uploads / defaults).
- Either `RHOAI_KFP_URL` **or** `RHOAI_CREATE_DSPA=true` with a working DSPA route.

JSON scenarios: `tests/config/automl_tabular_test_configs.json`, `tests/config/automl_timeseries_test_configs.json`.

### AutoRAG (documents RAG optimization)

Requires the **shared** block (token, S3 secret in cluster, KFP URL or DSPA) plus **OGX**:

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

**Negative pipeline tests** (`tests/scenarios/test_pipeline_negative_rhoai.py`) use the **same tag names** in `RHOAI_TEST_CONFIG_TAGS`: `tabular`, `timeseries`, and `autorag` (they also appear on JSON scenarios). If `RHOAI_TEST_CONFIG_TAGS` is unset or empty, all three negative suites run. If it is set, tests for a family are **deselected** (not collected) unless that tag is listed — for example `smoke` alone does not include `tabular` / `timeseries` / `autorag`, so no negative tests are collected unless you add them (e.g. `smoke,tabular`).

**Positive** `test_automl_*_rhoai.py` / `test_autorag_rhoai.py` tests are also **deselected** when tag filtering yields **no** JSON scenarios for that pipeline (so you do not see empty parametrization or `NOTSET` ids). This uses `pytest_collection_modifyitems` in `tests/scenarios/conftest.py` together with `rhoai_negative_pipeline_family_allowed` and config loaders in `tests/lib/settings.py` / `tests/lib/config_loaders.py`.

### Precompiled pipeline YAML (tabular, timeseries, AutoRAG)

Each suite uses a **`pipeline.yaml`** package path from:

- **`RHOAI_PIPELINE_YAML_TABULAR`**, **`RHOAI_PIPELINE_YAML_TIMESERIES`**, **`RHOAI_PIPELINE_YAML_AUTORAG`** — absolute or relative path to a local file when set; or
- If unset, the file is **downloaded once per session** from
  [pipelines-components](https://github.com/red-hat-data-services/pipelines-components/tree/rhoai-3.4/pipelines/training)
  via **`RHOAI_PIPELINES_COMPONENTS_REPO`** (default `red-hat-data-services/pipelines-components`) and **`RHOAI_PIPELINES_COMPONENTS_REF`** (default `rhoai-3.4`), under `pipelines/training/.../pipeline.yaml`.

### Run timing

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | `3600` | Max seconds to wait for a pipeline run. |
| `RHOAI_PIPELINE_NEGATIVE_RUN_TIMEOUT` | (same as `RHOAI_PIPELINE_RUN_TIMEOUT`) | Max seconds to wait for **negative** pipeline tests (`tests/scenarios/test_pipeline_negative_rhoai.py`). |
| `RHOAI_KFP_POLL_INTERVAL_SECONDS` | `25` | Seconds between KFP ``get_run`` polls. The same interval controls how often run/task status is printed (lines go to the controlling terminal, ``/dev/tty``, when available so pytest does not buffer them until the test ends). |
| `RHOAI_KFP_PIPELINE_DISPLAY_NAME` | (unset) | Optional. Pipeline ``name=`` from ``@dsl.pipeline`` so progress output can hide the compiled root-DAG task (``{name}-<suffix>``). Scenario tests pass this per pipeline; set in env if you call ``wait_for_run_with_progress`` yourself. |

### Negative pipeline tests

Tests in `tests/scenarios/test_pipeline_negative_rhoai.py` assert that invalid inputs do **not** produce a successful run: either run creation fails (API/client error) or the run ends in **FAILED**/**ERROR** (not **SUCCEEDED**). They are marked **`pipeline_negative`** in addition to **`integration`** and **`openshift_ai`**.

- **Unknown parameter names:** If your KFP backend ignores extra keys, a run could still **SUCCEEDED**; the test fails with an explicit message so you can treat that as an environment limitation.
- **Invalid non-data parameters** (separate from S3 / `train_data_*` inputs) are covered per pipeline: tabular — `task_type`, `top_n`, `label_column`; time series — `target`, `id_column`, `timestamp_column`, `prediction_length`, `top_n`; AutoRAG — `optimization_metric`, `optimization_max_rag_patterns`, `embeddings_models`, `generation_models`.
- **AutoRAG:** At least one scenario must be selected by `tests/config/autorag_test_configs.json` and `RHOAI_TEST_CONFIG_TAGS` (if set); otherwise AutoRAG negative tests **skip** (uploads are keyed by selected config ids).

Run only negative tests:

```bash
pytest tests/scenarios/test_pipeline_negative_rhoai.py -m "pipeline_negative" -v
```

Run negative tests for one pipeline family using **pytest markers** `tabular`, `timeseries`, `autorag`:

```bash
pytest tests/scenarios/test_pipeline_negative_rhoai.py -m "pipeline_negative and tabular" -v
pytest tests/scenarios/test_pipeline_negative_rhoai.py -m "pipeline_negative and timeseries" -v
pytest tests/scenarios/test_pipeline_negative_rhoai.py -m "pipeline_negative and autorag" -v
```

The same scope can be driven by **`RHOAI_TEST_CONFIG_TAGS`** (no `-m` needed for family selection):

```bash
export RHOAI_TEST_CONFIG_TAGS=tabular
pytest tests/scenarios/test_pipeline_negative_rhoai.py -v
```

### Pytest selection (markers and paths)

Integration tests are marked **`integration`** and **`openshift_ai`**.

Run **only** this suite (not `scripts/` tests):

```bash
pytest tests/scenarios/ -v
```

Run by marker:

```bash
pytest tests/scenarios/ -m "integration and openshift_ai" -v
```

Run a single module:

```bash
pytest tests/scenarios/test_automl_tabular_rhoai.py -v
pytest tests/scenarios/test_automl_timeseries_rhoai.py -v
pytest tests/scenarios/test_autorag_rhoai.py -v
pytest tests/scenarios/test_pipeline_negative_rhoai.py -v
```

Narrow by test name / parametrized id with `-k` (examples):

```bash
pytest tests/scenarios/test_automl_tabular_rhoai.py -k "smoke and compile" -v
```

Note: Default pytest `testpaths` in `pyproject.toml` includes `scripts` and `tests`. From the repo root, `pytest` runs both; use `tests/scenarios/` (or `-m openshift_ai`) to focus on this integration suite.

## Configuration files

| File | Role |
| ---- | ---- |
| `tests/config/automl_tabular_test_configs.json` | Tabular AutoML scenarios (`tags`, `data_mode`: `upload` / `existing_s3`). |
| `tests/config/automl_timeseries_test_configs.json` | Time series AutoML scenarios. |
| `tests/config/autorag_test_configs.json` | AutoRAG scenarios (`argument_overrides`, data modes). |
| `tests/data/` | Local datasets and AutoRAG fixtures referenced from JSON. |

## Troubleshooting

- **`pytest.fail` with “OpenShift AI … integration is not configured”** — See the message body: it lists missing or invalid variables. Cross-check `tests/.env` against `tests/.env.example`.
- **`RHOAI_KFP_URL` / route errors** — Confirm the Data Science Pipelines route is reachable, or enable `RHOAI_CREATE_DSPA=true` and wait settings if the operator creates the route asynchronously.
- **`kubernetes` / `boto3` import errors** — Install `pip install -e ".[test_rhoai]"`.
- **Empty selection** — If `RHOAI_TEST_CONFIG_TAGS` matches no scenario tags, parametrization may yield no tests for that file; adjust tags or unset the variable.

For deeper behavior (fixtures, DSPA creation, S3 uploads), see `tests/scenarios/conftest.py` and `tests/lib/settings.py`.
