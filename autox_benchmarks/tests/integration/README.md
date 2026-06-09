# Online integration tests (AutoML)

Exercises **real KFP and S3** with the smallest tabular smoke path: **breast-w-smoke** (200-row bundled sample), **`top_n: 1`**.

## Prerequisite

Copy [`.env.example`](../../.env.example) to **`autox_benchmarks/.env`** and set:

- `BENCHMARK_KFP_HOST`, `BENCHMARK_KFP_NAMESPACE`, `BENCHMARK_KFP_EXPERIMENT_NAME`, `BENCHMARK_KFP_TOKEN` (or `KFP_API_TOKEN`)
- `BENCHMARK_TRAIN_DATA_BUCKET_NAME`, `BENCHMARK_TRAIN_DATA_SECRET_NAME`
- `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Names align with `autox_tests` (e.g. `RHOAI_KFP_URL`, `RHOAI_TOKEN` work as aliases).

No manual S3 upload: the suite uploads `tests/fixtures/automl/integration/breast-w_n200.csv` when that object is missing on your train-data bucket.

## Run

```bash
cd autox_benchmarks
pytest tests/integration/ -v -s
```

Optional:

```bash
export BENCHMARK_INTEGRATION_TIMEOUT_SECONDS=3600
export KFP_API_TOKEN="$(oc whoami -t)"   # if not in .env
```

`pytest tests/` does **not** collect integration tests unless you pass `tests/integration/` or set `BENCHMARK_RUN_INTEGRATION=1`.

## What is tested

| Test | Checks |
|------|--------|
| `test_credentials_kfp_and_s3_ready` | `.env`, KFP connectivity, smoke CSV on bucket (upload if needed) |
| `test_orchestrator_smoke_run_succeeds` | Full orchestrator smoke run |
| `test_cli_smoke_run_subprocess` | Same via `scripts/benchmark_orchestrator.py` |

Session setup validates prerequisites once. If anything is missing or unreachable, the suite fails immediately.

Expect several minutes to tens of minutes depending on cluster load (AutoGluon with `top_n=1` on 200 rows).
