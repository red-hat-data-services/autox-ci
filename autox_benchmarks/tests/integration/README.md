# Online integration tests (AutoML)

Exercises **real KFP and S3** with the smallest tabular smoke path: **breast-w-smoke** (200-row bundled sample), **`top_n: 1`**.

## Prerequisite

**`config/credentials.ini`** (or `BENCHMARK_CREDENTIALS_PATH`) with working sections:

- `[kfp]` — host, namespace, experiment_name, and token (`token`, `token_file`, or `KFP_API_TOKEN`)
- `[storage]` — `train_data_bucket_name`
- `[s3]` — access key, secret, endpoint/region as needed for your cluster

No manual S3 upload: the suite uploads `tests/fixtures/automl/integration/breast-w_n200.csv` to  
`benchmark/smoke/breast-w_n200.csv` on your train-data bucket when that object is missing.

## Run

```bash
cd autox_benchmarks
pytest tests/integration/ -v -s
```

Optional:

```bash
export BENCHMARK_INTEGRATION_TIMEOUT_SECONDS=3600   # cap KFP wait (seconds)
export KFP_API_TOKEN="$(oc whoami -t)"              # if not in credentials.ini
```

`pytest tests/` (default offline suite) does **not** collect integration tests. To include them in a full-tree run, set `BENCHMARK_RUN_INTEGRATION=1`.

## What is tested

| Test | Checks |
|------|--------|
| `test_credentials_kfp_and_s3_ready` | Config, KFP connectivity, smoke CSV on bucket (upload if needed) |
| `test_orchestrator_smoke_run_succeeds` | Full orchestrator smoke run |
| `test_cli_smoke_run_subprocess` | Same via `scripts/benchmark_orchestrator.py` |

Session setup validates prerequisites once. If anything is missing or unreachable, the suite fails immediately and remaining tests do not run.

Expect several minutes to tens of minutes depending on cluster load (AutoGluon with `top_n=1` on 200 rows).
