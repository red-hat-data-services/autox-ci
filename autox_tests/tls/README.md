# TLS certificate integration tests

This opt-in suite tests CA selection and certificate verification in the installed
ai4rag clients, S3/vector stores, real DSPA injection and a product MaaS CA fault.
Ordinary AutoML/AutoRAG success and model quality belong to their functional suites.
The suite never installs a different ai4rag into the runtime image.

Tracked scenario catalogs live in `configs/`; helpers and the standalone container
probe live in `utils/`. There are no environment overrides for scenario catalogs.

## Run

```bash
cp autox_tests/.env.tls.example autox_tests/.env.tls
# Edit .env.tls; cluster execution is disabled in the example.
./run_tests.sh --suite tls -- -k 'not pgvector and not neo4j' -vv -rs
```

The runner loads `.env.tls`; existing shell exports take precedence. `--tags`
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

The suite contains only integration tests in `test_runtime.py`, `test_dspa.py`
and `test_flows.py`. Install the host dependencies with `uv sync --extra tls`.
Generated endpoints run inside the probe pods; gRPC/pymilvus come from the product
image. Runtime image references and IDs are recorded for diagnostics, without
comparing them with an expected digest.

## Minimal settings

| Scope | Settings |
| --- | --- |
| Every cluster test | `AUTOX_TLS_ENABLED=true`, `AUTOX_TLS_NAMESPACE`, `AUTOX_TLS_AUTORAG_IMAGE`, verified kubeconfig |
| Actual S3 endpoint | `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `TLS_S3_BUCKET`, `TLS_S3_KEY`, trusted CA source |
| DSPA | `AUTOX_TLS_DSPA_ENABLED=true`, `RHOAI_KFP_URL`, `RHOAI_TOKEN`, MaaS credentials and model lists |
| MaaS credentials | `MAAS_BASE_URL`, `MAAS_API_KEY`, or existing `MAAS_SECRET_NAME` |
| Model selection | `AUTORAG_GENERATION_MODELS`, `AUTORAG_EMBEDDING_MODELS`; JSON lists or comma-separated IDs |
| Milvus | `MILVUS_URI`, optional `MILVUS_TOKEN`, trusted CA source |
| PGVector | `PGVECTOR_HOST`, `PGVECTOR_PASSWORD`, trusted CA source |
| Neo4j | `NEO4J_URI`, `NEO4J_PASSWORD`, trusted CA source |
| N01 | `AUTOX_TLS_FLOWS_ENABLED=true`, KFP settings, MaaS/model settings, S3 credentials/CA/`TLS_S3_BUCKET`, one selected vector backend |

The namespace must already exist. The suite does not create a namespace or DSPA,
or update shared cluster trust. `AUTOX_TLS_NAMESPACE` is explicit to keep TLS
resources isolated from other suites. `AUTOX_TLS_AUTORAG_IMAGE` is the single
runtime/DSPA image; N01 uses its source pipeline image.

`AWS_DEFAULT_REGION` defaults to `us-east-1`; PostgreSQL port/database/user default
to `5432`/`postgres`/`postgres`; Neo4j username/database to `neo4j`/`neo4j`.
`TLS_S3_KEY` must identify an existing non-empty object. Runtime S3 probes use
only `get_object` and read one byte; they do not list objects or perform a
separate HEAD request. N01 does not use this key, and always prepares its own corpus.

## CA and existing credentials

Use `AWS_CA_BUNDLE` for the S3 CA and as a shared fallback for database CAs.
Databases with different issuers use `MILVUS_SERVER_CERT`, `PGVECTOR_CA_CERT` or
`NEO4J_CA_CERT`; those settings accept a local PEM path or inline PEM. S3's host
SDK setting `AWS_CA_BUNDLE` must be a local file path. Use `KFP_CA_BUNDLE` if the
KFP API requires another private issuer. Trust is never downloaded from the target.

Alternatively, reuse existing namespace Secrets through `AUTOX_TLS_S3_SECRET`,
`MAAS_SECRET_NAME`, `AUTOX_TLS_MILVUS_SECRET`, `AUTOX_TLS_PGVECTOR_SECRET` or
`AUTOX_TLS_NEO4J_SECRET`. Explicit references take precedence over local credentials
and are read-only. No manual Secret is required when standard endpoint values
are supplied: tests generate and clean isolated credential Secrets automatically.

Existing MaaS Secrets must contain `MAAS_BASE_URL`, `MAAS_API_KEY`,
`TLS_GENERATION_MODEL` and `TLS_EMBEDDING_MODEL`. The latter two are internal probe
fields, generated automatically from the first model in each host model list.
They are not additional host `.env.tls` settings. N01 uses the host model lists.

Existing CA ConfigMaps can be referenced with `AUTOX_TLS_<BACKEND>_CA_CONFIGMAP`
and `AUTOX_TLS_<BACKEND>_CA_KEY` (default `ca-bundle.crt`). They take precedence
over local CA values. For N01, the dataset uploader reads the selected S3 Secret
and CA ConfigMap too, so upload and pipeline credentials address the same service.
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

Set `AUTOX_TLS_VECTOR_DB=milvus`/`pgvector`/`neo4j` when several databases are
configured. Otherwise exactly one backend must be configured. N01 reuses the
selected backend's credentials; locally supplied backend CA PEM is embedded in
temporary product credentials rather than exporting a host file path.

After the explicit flow opt-in, N01 always uploads two bundled Aster documents
and their benchmark to `TLS_S3_BUCKET`, under
`autox-tls-datasets/<random-session>/rag-light/`. References in the benchmark match
the generated keys. S3 permissions must allow read/upload/delete there. No manual
dataset-key override or separate input/test bucket/credential selection exists.
Uploads use `IfNoneMatch="*"`, never create buckets or overwrite objects, and
record ownership, ETag and VersionId. Cleanup removes only recorded unchanged
uploads; changed objects remain. No recursive prefix deletion or product artifact
cleanup occurs. Upload failures cannot count as certificate rejection.

## Operational options and safety

| Optional setting | Purpose |
| --- | --- |
| `AUTOX_TLS_TIMEOUT` | One positive-seconds override; defaults are runtime 180s, DSPA 600s, N01 3600s |
| `AUTOX_TLS_REPORT_DIR` | Sanitized JSON evidence; defaults to `tls-results` |
| `AUTOX_TLS_KEEP_RESOURCES=true` | Retain owned runtime Jobs/material/uploads for debugging |
| `AUTOX_TLS_PULL_SECRET` | Existing image-pull Secret for a private registry |
| `AUTOX_TLS_TESTS_TAGS` | Selection normally set with runner `--tags` |

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
