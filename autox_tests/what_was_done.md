# autox-ci: Summary of Changes

## Overview

This repository was prepared to serve as a **git submodule** for all AutoRAG and AutoML functional tests, as described in [RHOAIENG-60746](https://redhat.atlassian.net/browse/RHOAIENG-60746). The goal is to allow other repositories (e.g. `opendatahub-tests`, `pipelines-components`) to add `autox-ci` as a submodule and run tests with a single shell script.

## What was done

### 1. Submodule runner (`run_tests.sh`)
- Created a self-contained shell script at the repo root
- Handles virtualenv creation, dependency installation, `.env` file sourcing, and pytest execution
- Accepts `--env-file` flag to point to any `.env` location
- Accepts pytest marker expressions (e.g. `"functional and tabular"`)
- Supports `--` separator to pass extra pytest flags
- Resolves its own location via `BASH_SOURCE` so it works from any working directory
- Changes into `autox-ci` directory before running pytest to avoid picking up parent repo's `conftest.py`

### 2. AutoRAG functional tests
- **Source:** Ported from `pipelines-components` repository (branch with AutoRAG functional tests)
- **Location:** `autox_tests/autorag/functional/`
- **Test scenarios:** 3 negative (TC-F-1, TC-F-2, TC-F-3) + 2 positive (TC-P-1, TC-P-2)
- **Configs:** `autox_tests/autorag/functional/configs/test_configs.json`
- **Data paths:** S3 keys are defined per test scenario in `test_configs.json`
- Fixed trailing comma bugs in dataclass fields
- Fixed `find_dotenv()` (cwd-relative) replaced with `load_tests_env()` (file-relative) for submodule compatibility
- Pipeline YAML resolution supports local file path or URL

### 3. AutoML functional tests
- **Source:** Ported from [`Mateusz-Switala/pipelines-components`](https://github.com/Mateusz-Switala/pipelines-components/tree/tests_functional_for_automl) branch `tests_functional_for_automl`
- **Location:** `autox_tests/automl/functional/`
- **Test scenarios:**
  - Tabular: 3 scenarios (regression, binary classification, multiclass) in `configs/tabular_test_configs.json`
  - Timeseries: 2 scenarios (sales, m4_hourly) in `configs/timeseries_test_configs.json`
- **Test data:** CSV datasets stored in `autox_tests/automl/functional/data/` — uploaded to S3 automatically by test fixtures before pipeline runs
- Adapted to use `autox-ci` shared infrastructure (`load_tests_env()`, `resolve_precompiled_pipeline_yaml()`)

### 4. Pipeline YAML resolution
- Unified env var naming: `AUTOML_TABULAR_PIPELINE_PATH`, `AUTOML_TIMESERIES_PIPELINE_PATH`, `AUTORAG_PIPELINE_PATH`
- Supports local file path or URL (including GitHub raw URLs)
- User must explicitly provide the path — no silent auto-download

### 5. Configuration
- **Separate `.env` files per cluster:**
  - `.env.ml` — AutoML cluster credentials, S3, pipeline paths
  - `.env.rag` — AutoRAG cluster credentials, S3, pipeline paths, Llama Stack
- `.env.example` — template with all variables and safe defaults
- All `.env*` files are gitignored
- `pyproject.toml` updated with all testpaths and markers

### 6. Config path fixes
- `config_loaders.py` updated to look in `autox_tests/automl/config/` and `autox_tests/autorag/config/` instead of non-existent `autox_tests/config/`

## Environment files

| File | Purpose |
|------|---------|
| `.env.ml` | AutoML tests — cluster URL, token, MinIO/S3 creds, pipeline YAML path |
| `.env.rag` | AutoRAG tests — cluster URL, token, AWS S3 creds, Llama Stack, pipeline YAML path |
| `.env.example` | Template with all variables (empty values, safe defaults) |
| `.env` | Legacy combined file (still works but separate files recommended) |

AutoML and AutoRAG typically target **different clusters** with different S3 backends, so they cannot share the same `.env` file (env var names like `RHOAI_KFP_URL`, `AWS_*` overlap).

## How to run tests

```bash
# AutoRAG functional tests
./run_tests.sh --env-file autox_tests/.env.rag "functional" -- autox_tests/autorag/functional/

# AutoML tabular functional tests
./run_tests.sh --env-file autox_tests/.env.ml "functional and tabular"

# AutoML timeseries functional tests
./run_tests.sh --env-file autox_tests/.env.ml "functional and timeseries"

# From a parent repo using autox-ci as submodule
./submodules/autox-ci/run_tests.sh --env-file .env.rag "functional" -- autox_tests/autorag/functional/
```

---

## TODO

### Test data should be defined in `test_configs.json`, not in `.env`
Currently, S3 data paths (`test_data_key`, `input_data_key`) are hardcoded per test scenario in `test_configs.json`. This is intentional — as the number of test scenarios grows, each will need different data paths, and putting them all in `.env` would lead to an unmanageable number of env variables. The `.env` file should only contain cluster-level configuration (URLs, tokens, secrets, bucket names), while per-scenario data paths belong in the JSON configs.

However, the current test configs use hardcoded S3 keys that must match what is actually uploaded to the S3 bucket on the target cluster. When moving to a new cluster or bucket, these paths need to be updated in `test_configs.json`.

### Pipeline YAML must match test configs
The test configs reference specific pipeline parameters (e.g. `llama_stack_vector_io_provider_id`). If the pipeline YAML version changes and adds/removes parameters, the test configs must be updated to match. There is no validation at test time — mismatched parameters result in a 400 rejection from KFP.

### AutoML functional tests need real cluster validation
The AutoML functional tests have been ported and collect correctly, but have not yet been validated end-to-end on a live cluster with the correct pipeline YAML and S3 data.

### Negative tests and submission rejections
Some negative test scenarios (e.g. TC-F-1 which omits `llama_stack_vector_io_provider_id`) cause a 400 submission rejection instead of a runtime pipeline failure. The test currently fails with an unhandled `ApiException`. Consider handling 400 errors in negative tests as valid "expected failures".
