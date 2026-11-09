"""TLS settings reuse existing product configuration and data connections."""

import base64
from io import BytesIO
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from autox_tests.autorag.tls import conftest as fixtures
from autox_tests.autorag.tls.configs.configs import (
    TLSRuntimeConfig, get_flow_configs,
)
from autox_tests.lib.tls import settings
from autox_tests.lib.tls.datasets import OwnedDatasets, build_dataset_plan
from autox_tests.lib.tls.material import generate_material

pytestmark = pytest.mark.config


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(fixtures, "load_tests_env", lambda component: None)


def test_namespace_uses_existing_product_name():
    os.environ.update(RHOAI_PROJECT_NAME="product-project", AUTOX_TLS_NAMESPACE="obsolete")
    assert fixtures.get_tls_environment_config()["namespace"] == "product-project"


def test_runtime_probe_uses_the_existing_input_bucket_without_manual_key():
    os.environ.update(
        INPUT_DATA_BUCKET_NAME="documents", TLS_S3_BUCKET="obsolete",
        INPUT_DATA_SECRET_NAME="input-connection", AUTOX_TLS_AUTORAG_IMAGE="product:latest",
        RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP="trust",
    )
    request = Mock()
    request.getfixturevalue.return_value = "autox-tls-datasets/session/runtime/probe.txt"
    profile = fixtures.service_profile.__wrapped__(
        TLSRuntimeConfig(id="s3", engine="s3"), (Mock(), "project"), request,
    )
    assert profile["secret_name"] == "input-connection"
    assert profile["environment"] == {
        "INPUT_DATA_BUCKET_NAME": "documents",
    }
    assert profile["s3_key"] == request.getfixturevalue.return_value
    request.getfixturevalue.assert_called_once_with("tls_runtime_s3_dataset")


def test_generated_endpoint_needs_no_real_data_connection():
    os.environ.update(AUTOX_TLS_AUTORAG_IMAGE="product:latest")
    request = Mock()
    profile = fixtures.service_profile.__wrapped__(
        TLSRuntimeConfig(id="wrong-host", engine="s3", certificate="wrong-host", expected="tls_error"),
        (Mock(), "project"), request,
    )
    assert profile["secret_name"] is None
    assert profile["environment"]["INPUT_DATA_BUCKET_NAME"] == "tls-fixture"
    assert profile["s3_key"] == "probe.txt"
    request.getfixturevalue.assert_not_called()


def test_database_probe_does_not_upload_s3_data():
    os.environ.update(
        PGVECTOR_HOST="postgres.example", PGVECTOR_PASSWORD="dummy",
        AUTOX_TLS_AUTORAG_IMAGE="product:latest", RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP="trust",
    )
    request = Mock()
    fixtures.service_profile.__wrapped__(
        TLSRuntimeConfig(id="postgres", engine="pgvector"), (Mock(), "project"), request,
    )
    request.getfixturevalue.assert_not_called()


def test_upload_failure_cannot_pass_as_expected_certificate_rejection():
    os.environ.update(
        INPUT_DATA_BUCKET_NAME="documents", INPUT_DATA_SECRET_NAME="input-connection",
        AUTOX_TLS_AUTORAG_IMAGE="product:latest", RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP="trust",
    )
    request = Mock()
    request.getfixturevalue.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        fixtures.service_profile.__wrapped__(
            TLSRuntimeConfig(id="wrong-ca", engine="s3", trust="wrong", expected="tls_error"),
            (Mock(), "project"), request,
        )


@pytest.mark.parametrize("kind", ["input", "test"])
def test_s3_connections_use_functional_precedence(kind):
    os.environ.update({
        f"{kind.upper()}_DATA_SECRET_NAME": "selected-data-connection",
        "RHOAI_TRAIN_S3_SECRET_NAME": "train-fallback",
        "RHOAI_TEST_S3_SECRET_NAME": "default-fallback",
        "AUTOX_TLS_S3_SECRET": "obsolete",
    })
    assert settings.endpoint_credentials("s3", data_kind=kind) == (
        "selected-data-connection", {},
    )
    del os.environ[f"{kind.upper()}_DATA_SECRET_NAME"]
    assert settings.s3_secret_name(kind) == "train-fallback"
    del os.environ["RHOAI_TRAIN_S3_SECRET_NAME"]
    assert settings.s3_secret_name(kind) == "default-fallback"


def test_shared_ca_configmap_uses_existing_name_and_key():
    os.environ.update(
        RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP="product-trust",
        RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP_KEY="trusted.pem",
        AUTOX_TLS_S3_CA_CONFIGMAP="obsolete",
    )
    assert settings.service_ca("s3") == {
        "ca_configmap": "product-trust", "ca_key": "trusted.pem",
    }


def test_negative_endpoint_does_not_use_the_healthy_connection():
    os.environ.update(VECTOR_DB_SECRET_NAME="healthy-database")
    with pytest.raises(settings.MissingTLSConfiguration, match="WRONG_HOST_SECRET"):
        settings.endpoint_credentials("pgvector", "wrong-host")


def test_runtime_selects_only_the_matching_existing_vector_secret():
    os.environ.update(VECTOR_DB_SECRET_NAME="vectors", NEO4J_DB_SECRET_NAME="graph")
    core = Mock()
    core.read_namespaced_secret.side_effect = lambda name, namespace: SimpleNamespace(
        data={"MILVUS_URI": "encoded"} if name == "vectors" else {"NEO4J_URI": "encoded"}
    )
    assert settings.backend_secret_name("milvus", core, "project") == "vectors"
    assert settings.backend_secret_name("neo4j", core, "project") == "graph"
    assert settings.backend_secret_name("pgvector", core, "project") is None


@pytest.fixture(scope="module")
def milvus_ca():
    return generate_material()["ca.pem"]


@pytest.mark.parametrize("key", ["MILVUS_SERVER_CERT", "MILVUS_CA_CERT"])
@pytest.mark.parametrize("trust", ["explicit", "wrong", "invalid", "missing"])
def test_milvus_probe_pairs_secret_ca_with_connection(milvus_ca, key, trust):
    os.environ.update(
        VECTOR_DB_SECRET_NAME="milvus", AUTOX_TLS_AUTORAG_IMAGE="product:latest",
        RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP="unrelated-trust",
        AWS_CA_BUNDLE="/unrelated/public-roots.pem",
    )
    core = Mock()
    core.read_namespaced_secret.return_value = SimpleNamespace(data={
        "MILVUS_URI": base64.b64encode(b"https://milvus.example").decode(),
        key: base64.b64encode(milvus_ca.encode()).decode(),
    })
    case = TLSRuntimeConfig(
        id="milvus", engine="milvus", trust=trust,
        expected="success" if trust == "explicit" else "tls_error",
    )
    profile = fixtures.service_profile.__wrapped__(case, (core, "project"), Mock())
    assert profile["secret_name"] == "milvus"
    assert profile["ca_pem"] == milvus_ca
    assert "ca_configmap" not in profile
    # Trust manipulation remains a probe operation, separate from source selection.
    assert case.payload()["trust"] == trust
    core.read_namespaced_secret.assert_called_with("milvus", "project")


@pytest.mark.parametrize("value", [b"not PEM", b"/host/path/ca.pem", b"\xff"])
def test_milvus_secret_invalid_ca_does_not_fall_back(value):
    os.environ["RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP"] = "unrelated-trust"
    core = Mock()
    core.read_namespaced_secret.return_value = SimpleNamespace(data={
        "MILVUS_SERVER_CERT": base64.b64encode(value).decode(),
    })
    with pytest.raises(ValueError, match="must contain CA PEM"):
        settings.service_ca("milvus", backend_secret="milvus", core=core, namespace="project")


def test_milvus_secret_without_ca_uses_shared_fallback():
    os.environ["RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP"] = "shared-trust"
    core = Mock()
    core.read_namespaced_secret.return_value = SimpleNamespace(data={"MILVUS_URI": "encoded"})
    assert settings.service_ca("milvus", backend_secret="milvus", core=core, namespace="project") == {
        "ca_configmap": "shared-trust", "ca_key": "ca-bundle.crt",
    }


@pytest.mark.parametrize("certificate", ["wrong-host", "expired"])
def test_generated_milvus_endpoint_does_not_read_real_secret(certificate):
    os.environ.update(VECTOR_DB_SECRET_NAME="milvus", AUTOX_TLS_AUTORAG_IMAGE="product:latest")
    core = Mock()
    profile = fixtures.service_profile.__wrapped__(
        TLSRuntimeConfig(id="fixture", engine="milvus", certificate=certificate, expected="tls_error"),
        (core, "project"), Mock(),
    )
    assert profile["local_milvus_fixture"]
    assert profile["secret_name"] is None
    assert "ca_pem" not in profile
    core.read_namespaced_secret.assert_not_called()


def test_flow_prefers_the_existing_vector_connection():
    os.environ.update(VECTOR_DB_SECRET_NAME="product-vectors", AUTOX_TLS_VECTOR_DB="obsolete")
    assert settings.flow_secret_credentials("vector_db") == ("product-vectors", {})


def test_existing_timeout_and_retention_settings():
    os.environ.update(
        RHOAI_PIPELINE_RUN_TIMEOUT="45", RHOAI_NOTEBOOK_JOB_TIMEOUT="25",
        AUTOX_TLS_TIMEOUT="999", AUTORAG_FUNCTIONAL_TEST_KEEP_ARTIFACTS="yes",
    )
    assert settings.timeout_seconds(600) == 45
    assert settings.timeout_seconds(180, "RHOAI_NOTEBOOK_JOB_TIMEOUT") == 25
    assert settings.keep_resources()
    os.environ["RHOAI_NOTEBOOK_JOB_TIMEOUT"] = "0"
    with pytest.raises(ValueError, match="RHOAI_NOTEBOOK_JOB_TIMEOUT"):
        settings.timeout_seconds(180, "RHOAI_NOTEBOOK_JOB_TIMEOUT")


def test_flow_reuses_separate_input_and_benchmark_buckets_and_secrets():
    os.environ.update(
        INPUT_DATA_BUCKET_NAME="documents", TEST_DATA_BUCKET_NAME="benchmarks",
        INPUT_DATA_SECRET_NAME="input-connection", TEST_DATA_SECRET_NAME="test-connection",
        MAAS_SECRET_NAME="maas", VECTOR_DB_SECRET_NAME="vectors",
        AUTORAG_GENERATION_MODELS='["generation"]', AUTORAG_EMBEDDING_MODELS='["embedding"]',
        TLS_S3_BUCKET="obsolete",
    )
    definition = get_flow_configs(filter_tags=False)[0]
    plan = build_dataset_plan(definition.definition, "test-prefix")
    flow = definition.resolve(
        secret_resolver=lambda name: settings.flow_secret_credentials(name)[0],
        dataset_values=plan.values,
    )
    args = flow["arguments"]
    assert args["input_data_bucket_name"] == "documents"
    assert args["test_data_bucket_name"] == "benchmarks"
    assert args["input_data_secret_name"] == "input-connection"
    assert args["test_data_secret_name"] == "test-connection"


@pytest.mark.parametrize("benchmark_fails", [False, True])
def test_datasets_use_and_clean_their_own_connections(benchmark_fails):
    clients = {kind + "_data_bucket_name": Mock() for kind in ("input", "test")}
    owned = OwnedDatasets(clients)
    for client in clients.values():
        client.put_object.return_value = {"ETag": "etag"}
        client.head_object.return_value = {
            "ETag": "etag", "Metadata": {"autox-tls-owner": owned.owner},
        }
    definition = get_flow_configs(filter_tags=False)[0].definition
    plan = build_dataset_plan(definition, owned.prefix)
    flow = {"arguments": {key: "same-bucket" for key in clients}}
    if benchmark_fails:
        clients["test_data_bucket_name"].put_object.side_effect = RuntimeError("upload failed")
        with pytest.raises(RuntimeError, match="upload failed"):
            owned.prepare(flow, plan)
    else:
        owned.prepare(flow, plan)
    owned.close()
    assert clients["input_data_bucket_name"].put_object.call_count == 2
    assert clients["input_data_bucket_name"].delete_object.call_count == 2
    assert clients["test_data_bucket_name"].put_object.call_count == 1
    assert clients["test_data_bucket_name"].delete_object.call_count == (0 if benchmark_fails else 1)


def test_uploader_reads_both_existing_data_connections(monkeypatch, tmp_path):
    os.environ.update(INPUT_DATA_SECRET_NAME="input", TEST_DATA_SECRET_NAME="test")
    core = Mock()
    core.read_namespaced_secret.side_effect = lambda name, namespace: SimpleNamespace(data={
        key: base64.b64encode(value.encode()).decode()
        for key, value in {
            "AWS_S3_ENDPOINT": f"https://{name}.example", "AWS_ACCESS_KEY_ID": name,
            "AWS_SECRET_ACCESS_KEY": "dummy",
        }.items()
    })
    clients = {name: Mock() for name in ("input", "test")}
    monkeypatch.setattr(fixtures, "service_ca", lambda engine: {"ca_pem": "test-ca"})
    factory = Mock(side_effect=lambda service, **kwargs: clients[kwargs["aws_access_key_id"]])
    monkeypatch.setattr("boto3.client", factory)
    paths = SimpleNamespace(mktemp=lambda name: tmp_path)
    generator = fixtures.tls_owned_datasets.__wrapped__((core, "project"), paths)
    owned = next(generator)
    assert owned.clients["input_data_bucket_name"] is clients["input"]
    assert owned.clients["test_data_bucket_name"] is clients["test"]
    generator.close()
    for client in clients.values():
        client.close.assert_called_once()


@pytest.mark.parametrize("failure", ["", "probe", "upload"])
def test_runtime_object_is_uploaded_and_cleaned_with_the_input_connection(monkeypatch, tmp_path, failure):
    os.environ.update(INPUT_DATA_BUCKET_NAME="inputs", INPUT_DATA_SECRET_NAME="input")
    core = Mock()
    core.read_namespaced_secret.return_value = SimpleNamespace(data={
        key: base64.b64encode(value.encode()).decode()
        for key, value in {
            "AWS_S3_ENDPOINT": "https://input.example", "AWS_ACCESS_KEY_ID": "input",
            "AWS_SECRET_ACCESS_KEY": "dummy",
        }.items()
    })
    client = Mock()
    if failure == "upload":
        client.put_object.side_effect = RuntimeError("upload failed")
    else:
        client.put_object.return_value = {"ETag": "etag"}
    client.head_object.side_effect = lambda **kwargs: {
        "ETag": "etag", "Metadata": client.put_object.call_args.kwargs["Metadata"],
    }
    factory = Mock(return_value=client)
    monkeypatch.setattr("boto3.client", factory)
    monkeypatch.setattr(fixtures, "service_ca", lambda engine: {"ca_pem": "test-ca"})
    paths = SimpleNamespace(mktemp=lambda name: tmp_path)
    generator = fixtures.tls_runtime_s3_dataset.__wrapped__((core, "project"), paths)

    if failure == "upload":
        with pytest.raises(RuntimeError, match="upload failed"):
            next(generator)
        client.delete_object.assert_not_called()
    else:
        key = next(generator)
        assert key.startswith("autox-tls-datasets/")
        assert key.endswith("/runtime/probe.txt")
        upload = client.put_object.call_args.kwargs
        assert upload["Key"] == key
        assert upload["Bucket"] == "inputs"
        assert upload["Body"]
        assert upload["IfNoneMatch"] == "*"
        if failure == "probe":
            with pytest.raises(RuntimeError, match="probe failed"):
                generator.throw(RuntimeError("probe failed"))
        else:
            generator.close()
        client.delete_object.assert_called_once_with(Bucket="inputs", Key=key)
    core.read_namespaced_secret.assert_called_once_with("input", "project")
    factory.assert_called_once()
    assert factory.call_args.kwargs["endpoint_url"] == "https://input.example"
    client.close.assert_called_once()


def test_standalone_s3_probe_reads_the_generated_object_key(monkeypatch):
    from autox_tests.lib.tls.probe import service_request

    os.environ.update(AWS_S3_ENDPOINT="https://input.example", INPUT_DATA_BUCKET_NAME="inputs")
    client = Mock()
    body = BytesIO(b"probe")
    client.get_object.return_value = {"Body": body}
    for name in ("ai4rag", "ai4rag.utils", "ai4rag.utils.clients", "ai4rag.utils.clients.s3"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["ai4rag.utils.clients.s3"].create_s3_client = Mock(return_value=client)
    key = "autox-tls-datasets/session/runtime/probe.txt"

    service_request({"engine": "s3", "trust": "explicit", "s3_key": key}, "ca.pem", "wrong.pem")

    client.get_object.assert_called_once_with(Bucket="inputs", Key=key)
    assert body.closed
    client.close.assert_called_once()
