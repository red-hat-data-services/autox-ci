# AutoX CI

End-to-end test suite and CI tooling for AutoX components (AutoRAG, AutoML) running on Red Hat OpenShift AI with Kubeflow Pipelines.

## Project structure

```
autox_tests/
  autorag/              AutoRAG functional tests (pipeline submission, artifact + notebook validation)
  automl/               AutoML functional tests (tabular, time series)
  lib/                  Shared utilities (env loading, KFP helpers, S3, failure diagnostics)
  .env.rag.example      Env template for AutoRAG tests
  .env.ml.example       Env template for AutoML tests
run_tests.sh            Test runner (uv + pytest wrapper)
pyproject.toml          Dependencies, extras, and pytest markers
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A RHOAI cluster with Data Science Pipelines enabled
- Environment variables for cluster access and pipeline parameters (see `.env.*.example` files)

## Quick start

1. In the RHOAI UI, create an **S3 Data Connection** in your test namespace (e.g. `minio`) and point `.env` at that secret name.
2. Copy and fill the env template — leave **`RHOAI_KFP_URL` unset** so tests auto-create **DSPA** and import managed pipelines.
3. Run via `run_tests.sh` (details: [autox_tests/README.md](autox_tests/README.md#running-tests-cluster-setup)).

```bash
# AutoML
cp autox_tests/.env.ml.example autox_tests/.env.ml
./run_tests.sh --suite automl --env-file autox_tests/.env.ml -t smoke

# AutoRAG
cp autox_tests/.env.rag.example autox_tests/.env.rag
./run_tests.sh --suite autorag --env-file autox_tests/.env.rag -t smoke
```

## Test runner (`run_tests.sh`)

```
./run_tests.sh [OPTIONS] [MARKER_EXPR] [-- PYTEST_ARGS...]
```

| Option | Description |
|---|---|
| `-t, --tags TAGS` | Comma-separated tags for scenario filtering (matched against test config JSON) |
| `--env-file FILE` | Source a `.env` file before running (shell exports take precedence) |
| `--extras NAME` | uv extras to install (default: `test_autorag`; comma-separated for multiple) |
| `--rag-configs PATH` | Custom AutoRAG test configs JSON (sets `AUTORAG_TEST_CONFIGS_PATH`) |
| `--tabular-configs PATH` | Custom AutoML tabular test configs JSON (sets `AUTOML_TABULAR_TEST_CONFIGS_PATH`) |
| `--timeseries-configs PATH` | Custom AutoML timeseries test configs JSON (sets `AUTOML_TIMESERIES_TEST_CONFIGS_PATH`) |
| `--dry-run` | Print the command without executing |

Everything after `--` is forwarded to pytest.

### Examples

```bash
# AutoRAG smoke tests
./run_tests.sh --env-file autox_tests/.env.rag -t smoke "autorag and positive"

# AutoML tabular tests
./run_tests.sh --env-file autox_tests/.env.ml --extras test_automl "tabular"

# Dry-run to inspect the command
./run_tests.sh --dry-run "autorag and negative"

# Pass extra pytest flags
./run_tests.sh --env-file autox_tests/.env.rag "autorag" -- -v -x
```

## Pytest markers

| Marker | Description |
|---|---|
| `autorag` | AutoRAG pipeline tests |
| `tabular` | Tabular AutoML tests |
| `timeseries` | Time series AutoML tests |
| `positive` | Scenarios expected to pass |
| `negative` | Scenarios expected to fail |
| `integration` | Integration tests hitting a live cluster |
| `openshift_ai` | RHOAI-specific scenarios |

Combine markers with `and` / `or` / `not` in the marker expression:

```bash
./run_tests.sh "autorag and positive and not integration"
```

## Submodule usage

This repo can be embedded as a git submodule:

```bash
./submodules/autox-ci/run_tests.sh --env-file my.env "autorag"
```

`run_tests.sh` resolves paths relative to its own location, so it works from any calling directory.

### Custom test configurations

Downstream repos can provide their own test config JSONs to override the built-in scenarios (e.g. different datasets, models, or S3 paths). Custom files must follow the same JSON schema as the built-in configs.

```bash
# AutoRAG with custom scenarios
./submodules/autox-ci/run_tests.sh --suite autorag --env-file my.env \
    --rag-configs my_configs/autorag_scenarios.json

# AutoML with custom tabular + timeseries scenarios
./submodules/autox-ci/run_tests.sh --suite automl --env-file my.env \
    --tabular-configs my_configs/tabular.json \
    --timeseries-configs my_configs/timeseries.json
```

The same overrides work via environment variables (useful in CI):

```bash
export AUTORAG_TEST_CONFIGS_PATH=my_configs/autorag_scenarios.json
./submodules/autox-ci/run_tests.sh --suite autorag --env-file my.env
```
