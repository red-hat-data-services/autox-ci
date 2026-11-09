# TLS certificate integration tests

Integration tests for CA selection and certificate verification in the installed
ai4rag clients: MaaS, S3, vector databases, DSPA CA injection and the AutoRAG pipeline.
Tests run in cluster Pods using the product's installed dependencies.

## Configuration

Use `autox_tests/.env.rag` (see [template](../../.env.rag.example)) and a verified
kubeconfig. Shell exports take precedence over `.env.rag`.

```dotenv
AUTOX_TLS_ENABLED=true
AUTOX_TLS_AUTORAG_IMAGE=<product image containing ai4rag>
```

`RHOAI_PROJECT_NAME` must identify an existing namespace. DSPA and N01 also require
an existing pipeline server; the TLS suite does not create a namespace or DSPA.
`AUTOX_TLS_AUTORAG_IMAGE` runs runtime/DSPA probes. N01 uses its source pipeline's
images. `RHOAI_NOTEBOOK_RUNNER_IMAGE` is not a substitute for the product image.

| Tests | Additional configuration |
| --- | --- |
| MaaS runtime | None; uses a generated endpoint inside the Pod |
| S3 | `INPUT_DATA_BUCKET_NAME`, input data connection or `AWS_*`, trusted CA; read/upload/delete permissions |
| Vector databases | `VECTOR_DB_SECRET_NAME` or `NEO4J_DB_SECRET_NAME`, trusted CA |
| DSPA | `AUTOX_TLS_DSPA_ENABLED=true`, `RHOAI_KFP_URL`, `RHOAI_TOKEN`, `MAAS_SECRET_NAME`, model lists |
| N01 | `AUTOX_TLS_FLOWS_ENABLED=true`, KFP/MaaS/model settings, input/test data connections and buckets, vector connection, S3 CA |

Model lists reuse `AUTORAG_GENERATION_MODELS` and `AUTORAG_EMBEDDING_MODELS`.
Connections reuse `INPUT_DATA_SECRET_NAME`, `TEST_DATA_SECRET_NAME`,
`MAAS_SECRET_NAME`, `VECTOR_DB_SECRET_NAME` and `NEO4J_DB_SECRET_NAME`.
S3 Secret names fall back to `RHOAI_TRAIN_S3_SECRET_NAME`, then
`RHOAI_TEST_S3_SECRET_NAME`. Existing Secrets take precedence over local credentials.
Without Secrets, probes can use `AWS_*`, `MAAS_BASE_URL`/`MAAS_API_KEY`,
`MILVUS_URI`/`MILVUS_TOKEN`, `PGVECTOR_HOST`/`PGVECTOR_PASSWORD` or
`NEO4J_URI`/`NEO4J_PASSWORD`. N01 requires exactly one configured database backend
when no database Secret is selected.

### Trusted CA

Milvus probes first read CA PEM from `MILVUS_SERVER_CERT` or `MILVUS_CA_CERT`
in the selected database Secret. An invalid nonempty CA fails setup.

Otherwise, probes use `RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP`, with key
`RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP_KEY` (default `ca-bundle.crt`). Without this
ConfigMap, they use `AWS_CA_BUNDLE` for S3, or `MILVUS_SERVER_CERT`,
`PGVECTOR_CA_CERT`, `NEO4J_CA_CERT` for databases, falling back to `AWS_CA_BUNDLE`.
Database CA settings accept a local PEM path or inline PEM; `AWS_CA_BUNDLE`
should point to a local PEM file. Use `KFP_CA_BUNDLE` for a private KFP API issuer.
Certificate verification must remain enabled.

## Running tests

Run commands from the repository root after configuring `.env.rag`.

```bash
# All enabled TLS scenarios:
./run_tests.sh --suite tls -- -vv -rs

# MaaS runtime:
./run_tests.sh --suite tls -- -k test_maas_runtime -vv -rs

# S3:
./run_tests.sh --suite tls --tags s3 -- -k test_service_runtime -vv -rs

# Milvus:
./run_tests.sh --suite tls -- -k 'test_service_runtime and milvus' -vv -rs

# DSPA CA injection:
AUTOX_TLS_DSPA_ENABLED=true ./run_tests.sh --suite tls -- -k test_dspa_plain_ca_injection -vv -rs

# AutoRAG pipeline with an intentionally wrong MaaS CA:
AUTOX_TLS_FLOWS_ENABLED=true ./run_tests.sh --suite tls -- -k N01-maas -vv -rs

# Jenkins report:
./run_tests.sh --suite tls -- --junitxml=results/tls.xml -v -rs
```

Currently `--suite tls` runs only AutoRAG and loads `.env.rag`.
`--tags` reuses `AUTORAG_FUNCTIONAL_TESTS_TAGS` and requires all supplied tags.
Missing external-service credentials and disabled DSPA/N01 scenarios are reported
as skipped; `-rs` shows why. For direct pytest runs, install host dependencies
with `uv sync --extra tls` and select `autox_tests/autorag/tls/ -m tls`.

`./run_tests.sh --suite autorag` excludes TLS by default. A custom marker or direct
pytest invocation can include TLS: use `positive and not tls` for ordinary positive
tests, or `-m 'not tls'` with pytest.

## Certificate scenarios

Wrong-host and expired cases must fail for the matching reason. An unknown-issuer
error, timeout or connection failure does not satisfy those cases. Expected
certificate rejection produces a **PASSED** test result.

S3 and Milvus wrong-host/expired endpoints are generated inside the Pod and need
no external credentials. PGVector and Neo4j require separate endpoints configured
with `AUTOX_TLS_<BACKEND>_WRONG_HOST_SECRET` / `..._CA_CONFIGMAP` and
`AUTOX_TLS_<BACKEND>_EXPIRED_SECRET` / `..._CA_CONFIGMAP`, where `<BACKEND>` is
`PGVECTOR` or `NEO4J`. The CA must be their actual issuer.

S04-s3-plain-dir and S05-s3-missing require an issuer absent from the image's
default trust; publicly trusted endpoints skip those cases. PGVector missing-CA
cases require a non-local, non-cluster hostname. For cluster-local PGVector use
`--tags pgvector,cluster-local`. Milvus must be remote, not Lite; PostgreSQL needs
pgvector and Neo4j needs vector-index support.

DSPA probes expect the platform to supply `SSL_CERT_DIR` with `/kfp/certs/ca.crt`.
N01 injects a wrong MaaS CA into a local copy of the selected managed pipeline,
or `AUTORAG_PIPELINE_PATH` in YAML mode (`RHOAI_USE_MANAGED_PIPELINES=false`).
The registered pipeline and its versions are not patched.

## Resources and cleanup

Tests create temporary Jobs/Pods, Secrets and ConfigMaps, and send requests to the
configured services. Database probes write to unique `ai4rag_tls_<uuid>` collections
and remove them afterward. Real S3 probes upload a small file; N01 uploads its own
documents and benchmark under `autox-tls-datasets/<random-session>/`. No manual S3
object key is required. Uploads do not overwrite existing objects, and cleanup
removes only unchanged uploads owned by the tests.

Existing Secrets, ConfigMaps and shared cluster trust are not modified. KFP run
history and pipeline artifacts can remain. Interrupted runs or cleanup failures
can leave resources, including a partially created database collection. Use a test
namespace and test data services.

| Optional setting | Purpose |
| --- | --- |
| `RHOAI_NOTEBOOK_JOB_TIMEOUT` | Runtime Job timeout; default 180s if unset |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | DSPA/N01 timeout; defaults 600s/3600s if unset |
| `AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true` | Retain owned runtime resources and uploads; temporary DSPA/N01 credential Secrets are still removed |
| `AUTOX_TLS_PULL_SECRET` | Existing image-pull Secret for private registries |
| `AUTOX_TLS_REPORT_DIR` | Sanitized JSON diagnostics; default `tls-results` |
