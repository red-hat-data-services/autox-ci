# AutoRAG Functional Tests

Parametrized functional tests for the Documents RAG Optimization pipeline on Red Hat OpenShift AI (RHOAI). Each test scenario is declared in `configs/test_configs.json`, submitted to KFP, and validated against expected outcomes.

## Directory layout

```
autorag/
  conftest.py                 Fixtures: env config, KFP client, S3 client, pipeline YAML resolution
  test_pipeline_functional.py Pytest test class (parametrized over configs)
  response_validation.py        Score/prompt/Responses API export validators
  test_response_validation.py   Unit tests for response_validation helpers
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
| `tags` | List of tags for runtime filtering via `-t` / `TESTS_TAGS` |
| `expected_result` | `"pass"` or `"fail"` |
| `vector_io_provider_id` | Milvus provider ID for this scenario |
| `pipeline_params_overrides` | Per-scenario parameter overrides |

### Tag filtering

Pass tags via `--tags` / `-t` on the CLI or set `TESTS_TAGS` in the environment. Only scenarios matching **all** specified tags are selected.

## Pass / fail criteria

**Expected-pass scenarios:**
1. Pipeline run finishes with state `SUCCEEDED`
2. At least 1 pattern artifact exists in S3
3. Indexing notebook, inference notebook, and `evaluation_results.json` exist in S3
4. A randomly selected indexing and inference notebook executes successfully via papermill
5. When tagged `response_quality`: Unitxt scores in `pattern.json`, generation prompt template, `evaluation_results.json` content, leaderboard artifact, and answer-quality stats are validated
6. When additionally tagged `responses_api`: `responses_template` and `vector_store_binding` parity are validated in the best pattern export, then a live `POST /v1/responses` probe checks `file_search` hits and answer text (requires `OGX_CLIENT_BASE_URL` / `OGX_CLIENT_API_KEY`)

Scenarios tagged `response_quality` perform deeper artifact checks. Filter with `-t response_quality` or combine tags (e.g. `-t "smoke and response_quality"`).

Optional LLM-as-a-Judge sampling: add the `llm_judge` tag to a scenario, set `AUTORAG_RUN_LLM_JUDGE=true`, and configure `AUTORAG_LLM_JUDGE_MODEL` to a foundation model available on your cluster (no default is assumed). Also requires `OGX_CLIENT_BASE_URL` / `OGX_CLIENT_API_KEY`.

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
