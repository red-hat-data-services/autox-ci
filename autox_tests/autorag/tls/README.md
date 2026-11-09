# TLS certificate integration tests

This opt-in suite tests CA selection and certificate verification in the installed
ai4rag clients, S3/vector stores, real DSPA injection and a product MaaS CA fault.
Ordinary AutoML/AutoRAG success and model quality belong to their functional suites.
The suite never installs a different ai4rag into the runtime image.

Tracked scenario catalogs live in `autorag/tls/configs/`; shared helpers and the
standalone container probe live in `lib/tls/`. There are no environment overrides
for scenario catalogs.

## Run

```bash
# Reuse autox_tests/.env.rag (template: autox_tests/.env.rag.example).
# Add the TLS settings from that template; cluster execution is opt-in.
./run_tests.sh --suite tls -- -k 'not pgvector and not neo4j' -vv -rs
```

Pytest loads `.env.rag`; existing shell exports take precedence. `--tags`
requires every supplied tag to match. `-k` selects tests but does not remove
credentials for other backends. Unconfigured external services and disabled
DSPA/product flows skip with explicit reasons. Skips are not certificate coverage.

```bash
# Self-contained MaaS trust matrix:
./run_tests.sh --suite tls -- -k test_maas_runtime -vv -rs
# Actual S3 endpoint and generated S3 certificate fixtures:
./run_tests.sh --suite tls --tags s3 -- -k test_service_runtime -vv -rs
# Generated Milvus certificate fixtures, no real Milvus credentials needed:
./run_tests.sh --suite tls -- -k 'test_service_runtime and D06-milvus' -vv -rs
# Real DSPA CA injection, enabled separately:
./run_tests.sh --suite tls --tags dspa -- -k test_dspa_plain_ca_injection -vv -rs
# Product certificate rejection, enabled separately:
./run_tests.sh --suite tls -- -k 'test_product_pipeline_tls and N01' -vv -rs
```

`--suite tls` runs each product's `tls/` tests in a separate pytest process,
loading only that product's environment (`.env.ml` for AutoML, `.env.rag` for
AutoRAG). Currently only AutoRAG has TLS cases; AutoML is not run by the alias.
The two environment files are never combined by this alias. Exported Jenkins
settings apply to every process; use the product commands for separate settings.

```bash
# Equivalent AutoRAG entry point:
./run_tests.sh --suite autorag tls
# Positive certificate cases:
./run_tests.sh --suite autorag 'tls and positive'
# JUnit results for Jenkins:
./run_tests.sh --suite tls -- --junitxml=results/tls.xml -v -rs
# Ordinary functional tests (TLS is excluded by default without a marker):
./run_tests.sh --suite autorag
```

The TLS alias preserves extra filters, for example `--suite tls positive`
selects `tls and (positive)`. `--tags` uses the product's existing tag setting.
If both products have TLS directories, JUnit report filenames receive a product
prefix, such as `automl-tls.xml` and `autorag-tls.xml`, to preserve both results.
For direct pytest runs, install extras and select the product path with `-m tls`.

The suite contains only integration tests in `test_runtime.py`, `test_dspa.py`
and `test_flows.py`. Install the host dependencies with `uv sync --extra tls`.
Generated endpoints run inside the probe pods; gRPC/pymilvus come from the product
image. Runtime image references and IDs are recorded for diagnostics, without
comparing them with an expected digest.

## Minimal settings

The `.env.rag.example` template adds only the TLS enable flag and product image.
Namespace, connections, model lists and timeouts reuse the existing AutoRAG
settings. Set `AUTOX_TLS_ENABLED=true` and provide the image to run the
self-contained MaaS cases. S3 probes prepare their own object automatically.
DSPA/N01 opt-ins and diagnostic options below are omitted from the env
template and keep their defaults until explicitly configured.

| Scope | Settings |
| --- | --- |
| Every cluster test | `AUTOX_TLS_ENABLED=true`, existing `RHOAI_PROJECT_NAME`, `AUTOX_TLS_AUTORAG_IMAGE`, verified kubeconfig |
| Actual S3 endpoint | Existing input data connection or `AWS_*`, `INPUT_DATA_BUCKET_NAME`, trusted CA source; read/upload/delete access |
| DSPA | `AUTOX_TLS_DSPA_ENABLED=true`, `RHOAI_KFP_URL`, `RHOAI_TOKEN`, MaaS credentials and model lists |
| MaaS credentials | `MAAS_BASE_URL`, `MAAS_API_KEY`, or existing `MAAS_SECRET_NAME` |
| Model selection | `AUTORAG_GENERATION_MODELS`, `AUTORAG_EMBEDDING_MODELS`; JSON lists or comma-separated IDs |
| Vector databases | Existing `VECTOR_DB_SECRET_NAME` / `NEO4J_DB_SECRET_NAME`, or local backend credentials, and trusted CA source |
| N01 | `AUTOX_TLS_FLOWS_ENABLED=true`, existing KFP/MaaS/model settings, input/test data connections and buckets, vector connection and S3 CA |

The namespace must already exist. The suite does not create a namespace or DSPA,
or update shared cluster trust. Resources use unique owned names within
`RHOAI_PROJECT_NAME`. `AUTOX_TLS_AUTORAG_IMAGE` is the single runtime/DSPA image;
N01 uses its source pipeline image. The image must contain product ai4rag clients;
`RHOAI_NOTEBOOK_RUNNER_IMAGE` serves a different purpose and is not substituted.

`AWS_DEFAULT_REGION` defaults to `us-east-1`; PostgreSQL port/database/user default
to `5432`/`postgres`/`postgres`; Neo4j username/database to `neo4j`/`neo4j`.
Real S3 probes upload a small non-empty object once per pytest session to
`INPUT_DATA_BUCKET_NAME`, under `autox-tls-datasets/<random-session>/runtime/`.
They use the input data connection and trusted CA settings for setup, then pass
the generated key directly to each probe scenario. No manual object or key
setting is needed. The product probe calls `get_object` and reads one byte;
it does not list objects or perform a separate HEAD request.

The upload and cleanup use the same ownership tracking as N01. Objects are
created with `IfNoneMatch="*"` and removed at session teardown, including after
probe failures, unless retention is enabled. Cleanup checks the recorded owner
and ETag with HEAD before deleting; it leaves changed objects untouched.
Permissions must allow read/upload/delete under the generated prefix. Setup
upload failures fail the test and cannot count as expected certificate rejection.
Generated wrong-host/expired S3 endpoints and non-S3 probes do not upload to real S3.

## CA and existing credentials

Use `AWS_CA_BUNDLE` for the S3 CA and as a shared fallback for database CAs.
Databases with different issuers use `MILVUS_SERVER_CERT`, `PGVECTOR_CA_CERT` or
`NEO4J_CA_CERT`; those settings accept a local PEM path or inline PEM. S3's host
SDK setting `AWS_CA_BUNDLE` must be a local file path. Use `KFP_CA_BUNDLE` if the
KFP API requires another private issuer. Trust is never downloaded from the target.

Tests reuse existing namespace Secrets: `INPUT_DATA_SECRET_NAME`,
`TEST_DATA_SECRET_NAME`, `MAAS_SECRET_NAME`, `VECTOR_DB_SECRET_NAME` and
`NEO4J_DB_SECRET_NAME`. S3 connections fall back to `RHOAI_TRAIN_S3_SECRET_NAME`,
then `RHOAI_TEST_S3_SECRET_NAME`, matching functional tests. Existing references
take precedence over local credentials and are read-only. Database probes read
the vector Secret's keys and only run the matching backend. Neo4j prefers its
dedicated `NEO4J_DB_SECRET_NAME`. No normal-endpoint TLS-specific Secret aliases
are needed.

Without existing Secrets, local settings can generate temporary credentials:
`AWS_*`, `MAAS_BASE_URL` / `MAAS_API_KEY`, `MILVUS_URI` / `MILVUS_TOKEN`,
`PGVECTOR_HOST` / `PGVECTOR_PASSWORD`, or `NEO4J_URI` / `NEO4J_PASSWORD`.
Existing MaaS Secrets need only `MAAS_BASE_URL` and `MAAS_API_KEY`. DSPA takes
the first models from `AUTORAG_GENERATION_MODELS` / `AUTORAG_EMBEDDING_MODELS`
and passes them in the probe scenario; no extra model keys are added to Secrets.

Milvus runtime probes first use CA PEM from `MILVUS_SERVER_CERT` (or
`MILVUS_CA_CERT`) in the selected database Secret. This keeps the CA paired with
the connection without duplicating it in `.env.rag`. An invalid nonempty Secret
CA fails setup; it is not silently replaced. Negative trust cases still substitute
or remove this CA inside the probe.

The shared CA ConfigMap uses the existing `RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP`
and `RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP_KEY` (default `ca-bundle.crt`). When set,
it takes precedence over local CA values. Milvus probes use this fallback only
when the selected Secret has no CA. For N01, the dataset uploader reads
each selected S3 Secret and this CA ConfigMap, so upload and pipeline credentials
address the same services, including separate input/test connections.
Local CA file paths are never exported as product Secret values. For Milvus,
the harness detects whether the installed product consumes `server_cert` or
`ca_cert`; both fields receive the same trusted PEM when building a flow Secret.

## Generated certificate failures

S05-s3-wrong-host/S05-s3-expired always generate a loopback HTTPS endpoint inside
the probe pod. D06-milvus-wrong-host/D06-milvus-expired always generate a loopback
gRPC TLS endpoint. These four cases ignore real endpoint credentials and use
session-specific CA/leaf certificates and dummy credentials. No additional env
settings, Secret references, Service or Route are needed or supported.

The wrong-host leaf has a SAN for another name; the expired leaf has the correct
name and a past expiry date. The client receives the valid test issuer so that
unknown CA cannot substitute for hostname/expiry rejection. The test calls the
installed ai4rag client/adapter and requires the matching rejection signal.
UNAVAILABLE, timeout, connection failure, unknown issuer, protocol errors or
unexpected success cannot pass. Allowlisted native gRPC diagnostics supplement
SDK exception chains; raw native stderr is discarded. The local gRPC listener
is a TLS fixture, not a functional Milvus instance. These checks cover client
validation, not the certificate configuration of a deployed MinIO/Milvus.

PGVector and Neo4j wrong-host/expired cases still need separate real-protocol
endpoints. Configure `AUTOX_TLS_<BACKEND>_WRONG_HOST_SECRET` and corresponding
`..._CA_CONFIGMAP`, or `..._EXPIRED_SECRET`/`..._CA_CONFIGMAP`. Their CA must be the
actual issuer. Unconfigured cases skip; the healthy endpoint is never substituted.

S04-s3-plain-dir and S05-s3-missing require an issuer absent from the image's SDK
default trust. An independent verified TLS handshake checks this prerequisite.
Public/default-trusted endpoints skip these cases; this does not prove private-CA
consumption. Private-issuer scenarios continue through the product client, where
unexpected success fails. Preflight network, hostname and expiry errors fail the
test. The preflight connects directly to the endpoint, without an HTTP proxy.
S3's supported explicit trust path is `AWS_CA_BUNDLE` or the factory's `verify`.

PGVector missing-CA cases need a hostname outside the library's local/
cluster category; its default local connection is opportunistic and cannot prove
verified TLS. For cluster-local PGVector run D01/D02/D04 with
`--tags pgvector,cluster-local`. Milvus must be remote, not Lite. PostgreSQL needs
pgvector and Neo4j vector-index support.

Database CA coverage uses explicit CA settings and certificate rejection.
Every wrong-host and expired scenario requires evidence of the matching rejection
reason. Unknown-issuer errors cannot satisfy hostname or expiry coverage.
Generic database success with `SSL_CERT_FILE` is not a separate scenario because
it does not establish that the selected driver consumed that CA source.

## DSPA and N01

DSPA submits a minimal KFP task without adding a CA volume. The platform must
supply `SSL_CERT_DIR` containing `/kfp/certs` and its `ca.crt`. One case preserves
platform settings, another removes process-local `MAAS_CA_BUNDLE`/`SSL_CERT_FILE`
so the plain-directory fallback is isolated. Neither changes shared trust.

N01 generates an unrelated valid MaaS CA and injects it into a local copy of the
selected AutoRAG pipeline. It uses the shared managed resolver, or
`AUTORAG_PIPELINE_PATH` with `RHOAI_USE_MANAGED_PIPELINES=false` for YAML mode.
Registered pipelines and their versions are never patched. Only executors
consuming both MaaS credential keys are wrapped. A failed task must show the
same generated CA hash marker and a certificate-specific error in the same
container. No manual wrong-CA Secret or image assertion is used.

N01 uses `VECTOR_DB_SECRET_NAME`, falling back to `NEO4J_DB_SECRET_NAME`.
Without either reference, exactly one local backend must be configured. N01
reuses the selected backend's credentials; locally supplied backend CA PEM is embedded in
temporary product credentials rather than exporting a host file path.

After the explicit flow opt-in, N01 always uploads two bundled Aster documents
to `INPUT_DATA_BUCKET_NAME` and their benchmark to `TEST_DATA_BUCKET_NAME`, under
`autox-tls-datasets/<random-session>/rag-light/`. References in the benchmark match
the generated keys. S3 permissions must allow read/upload/delete there. No manual
dataset-key override is needed. Input and test uploads use their respective
existing data-connection Secrets, including when the S3 endpoints differ.
Uploads use `IfNoneMatch="*"`, never create buckets or overwrite objects, and
record ownership, ETag and VersionId. Cleanup removes only recorded unchanged
uploads; changed objects remain. No recursive prefix deletion or product artifact
cleanup occurs. Upload failures cannot count as certificate rejection.

## Operational options and safety

| Optional setting | Purpose |
| --- | --- |
| `RHOAI_NOTEBOOK_JOB_TIMEOUT` | Existing Job timeout reused for runtime probes; default 180s if unset |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | Existing pipeline timeout reused for DSPA/N01; defaults 600s/3600s if unset |
| `AUTOX_TLS_REPORT_DIR` | Sanitized JSON evidence; defaults to `tls-results` |
| `AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS=true` | Existing retention option also retains owned TLS Jobs/material/uploads |
| `AUTOX_TLS_PULL_SECRET` | Existing image-pull Secret for a private registry |
| `AUTORAG_FUNCTIONAL_TESTS_TAGS` | Selection normally set with runner `--tags` |

Temporary DSPA/N01 credential Secrets are always cleaned even when resources are
retained. Existing Secrets/ConfigMaps are never changed or deleted. Runtime
resources have unique owned names. Real database probes create only a unique
`ai4rag_tls_<uuid>` collection and remove it afterward; a constructor failure may
leave its own partially created collection. They do not test query quality.

Reports contain versions, selected CA hash, stage, image IDs and allowlisted
error categories, without raw SDK logs, passwords, tokens or private keys.
Generic diagnostic categories are hints, not proof of TLS rejection. Malformed
Milvus CA passes only with native CA-parser evidence, independently malformed
PEM and confirmation that the product consumed it. Verification-disabled settings
(`S3_SSL_VERIFY=false` / `KFP_VERIFY_SSL=false` / `RHOAI_HTTPS_VERIFY=false`)
are rejected by the suite.
