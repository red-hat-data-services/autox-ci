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

```bash
# AutoRAG
cp autox_tests/.env.rag.example autox_tests/.env.rag
./run_tests.sh --env-file autox_tests/.env.rag "autorag and positive"

# AutoML
cp autox_tests/.env.ml.example autox_tests/.env.ml
./run_tests.sh --env-file autox_tests/.env.ml --extras test_automl "tabular or timeseries"
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
