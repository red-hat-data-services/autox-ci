"""TLS fixtures using existing platform settings and isolated test-owned credentials."""

import os
from contextlib import contextmanager

import pytest

from autox_tests.lib.env import load_tests_env
from autox_tests.autorag.tls.configs.configs import (
    MissingTLSConfiguration,
    resolve_service_profile,
)
from autox_tests.lib.tls.material import generate_material
from autox_tests.lib.tls.resources import OwnedCredentialSecrets
from autox_tests.lib.tls.settings import (
    flow_secret_credentials,
    backend_secret_name,
    model_id,
    timeout_seconds,
    endpoint_credentials,
    service_ca,
    CredentialValues,
)


def pytest_collection_modifyitems(items):
    """Use the same positive/negative marker selection as AutoML and AutoRAG."""
    for item in items:
        if "/tls/" not in str(item.path):
            continue
        item.add_marker(pytest.mark.autorag)
        callspec = getattr(item, "callspec", None)
        test_config = callspec.params.get("test_config") if callspec else None
        if test_config is not None and hasattr(test_config, "expected"):
            marker = "positive" if test_config.expected == "success" else "negative"
            item.add_marker(getattr(pytest.mark, marker))


def get_tls_environment_config():
    """Build shared TLS settings from .env.rag without contacting any service."""
    load_tests_env("autorag")
    return {
        "enabled": os.environ.get("AUTOX_TLS_ENABLED", "false").lower() == "true",
        "namespace": os.environ.get("RHOAI_PROJECT_NAME", "").strip(),
        "image": os.environ.get("AUTOX_TLS_AUTORAG_IMAGE", "").strip(),
        "dspa_enabled": os.environ.get("AUTOX_TLS_DSPA_ENABLED", "false").lower()
        == "true",
        "flows_enabled": os.environ.get("AUTOX_TLS_FLOWS_ENABLED", "false").lower()
        == "true",
        "ca_dir": "/kfp/certs",
        "dspa_timeout": timeout_seconds(600),
        "rhoai_kfp_url": (os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST") or "").strip(),
        "rhoai_token": os.environ.get("RHOAI_TOKEN", "").strip(),
    }


@pytest.fixture(scope="session")
def tls_environment_config():
    return get_tls_environment_config()


@pytest.fixture(scope="session")
def tls_cluster(tls_environment_config):
    if not tls_environment_config["enabled"]:
        pytest.skip("Set AUTOX_TLS_ENABLED=true to run certificate tests on a cluster")
    for setting in ("S3_SSL_VERIFY", "KFP_VERIFY_SSL"):
        if os.environ.get(setting, "true").strip().lower() in {"false", "0", "no", "off"}:
            pytest.fail(f"{setting}=false is not permitted in the certificate suite")
    from autox_tests.lib.settings import get_rhoai_integration_https_verify

    if not get_rhoai_integration_https_verify():
        pytest.fail("TLS suite requires verified KFP API access (RHOAI_HTTPS_VERIFY)")
    from kubernetes import client, config

    namespace = tls_environment_config["namespace"]
    if not namespace:
        pytest.fail("Set RHOAI_PROJECT_NAME to an existing test namespace")
    configuration = client.Configuration()
    config.load_kube_config(
        config_file=os.environ.get("KUBECONFIG"), client_configuration=configuration
    )
    if not configuration.verify_ssl:
        pytest.fail("TLS suite requires verified Kubernetes API access")
    api = client.CoreV1Api(client.ApiClient(configuration))
    api.read_namespace(namespace)
    return api, namespace


@pytest.fixture(scope="session")
def tls_material(tls_cluster):
    return generate_material()


@pytest.fixture(scope="session")
def tls_image(tls_cluster, tls_environment_config):
    image = tls_environment_config["image"]
    if not image:
        pytest.fail("AUTOX_TLS_AUTORAG_IMAGE is required to launch runtime/DSPA probes")
    return image


@pytest.fixture
def service_profile(test_config, tls_cluster, request):
    try:
        existing = None
        if test_config.engine in {"milvus", "pgvector", "neo4j"} and test_config.certificate == "server":
            existing = backend_secret_name(test_config.engine, *tls_cluster)
        profile = resolve_service_profile(
            test_config, backend_secret=existing,
            core=tls_cluster[0], namespace=tls_cluster[1],
        )
        if test_config.engine == "s3" and not profile["local_s3_fixture"]:
            profile["s3_key"] = request.getfixturevalue("tls_runtime_s3_dataset")
        return profile
    except MissingTLSConfiguration as exc:
        pytest.skip(str(exc))
    except ValueError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def tls_kfp_session(tls_cluster, tls_environment_config):
    """Use existing KFP only; deliberately avoid namespace/DSPA creation fixtures."""
    from autox_tests.conftest import make_kfp_client_for_session

    config = tls_environment_config
    if not config["rhoai_kfp_url"] or not config["rhoai_token"]:
        pytest.fail("RHOAI_KFP_URL and RHOAI_TOKEN are required for KFP TLS tests")
    return make_kfp_client_for_session(
        namespace_config={
            "rhoai_project": config["namespace"],
            "rhoai_token": config["rhoai_token"],
        },
        configured_kfp_url=config["rhoai_kfp_url"],
        datascience_pipelines_application=None,
        kubeconfig_path=os.environ.get("KUBECONFIG"),
    )


@pytest.fixture(scope="session")
def tls_dspa_config(tls_environment_config):
    if not tls_environment_config["dspa_enabled"]:
        pytest.skip("Set AUTOX_TLS_DSPA_ENABLED=true for real DSPA injection coverage")
    try:
        endpoint_credentials("maas")
        model_id("GENERATION")
        model_id("EMBEDDING")
    except ValueError as exc:
        pytest.fail(str(exc))
    return tls_environment_config


@pytest.fixture(scope="session")
def tls_owned_credentials(tls_cluster):
    core, namespace = tls_cluster
    owned = OwnedCredentialSecrets(core, namespace)
    try:
        yield owned
    finally:
        owned.close()


@pytest.fixture(scope="session")
def tls_dspa_secret(tls_dspa_config, tls_owned_credentials):
    return tls_owned_credentials.reference(*endpoint_credentials("maas"))


@pytest.fixture(scope="session")
def kfp_client_dspa(tls_dspa_config, request):
    return request.getfixturevalue("tls_kfp_session")


@pytest.fixture
def tls_flow(test_config, tls_environment_config, request):
    if not tls_environment_config["flows_enabled"]:
        pytest.skip("Set AUTOX_TLS_FLOWS_ENABLED=true to run product pipeline flows")
    try:

        def reference(name):
            existing, values = flow_secret_credentials(name)
            if existing:
                return existing
            return request.getfixturevalue("tls_owned_credentials").reference(
                None, values
            )

        from autox_tests.lib.tls.datasets import build_dataset_plan

        uploader = request.getfixturevalue("tls_owned_datasets")
        plan = build_dataset_plan(test_config.definition, uploader.prefix)
        flow = test_config.resolve(
            secret_resolver=reference, dataset_values=plan.values
        )
        uploader.prepare(flow, plan)
        return flow
    except MissingTLSConfiguration as exc:
        pytest.skip(str(exc))
    except ValueError as exc:
        pytest.fail(str(exc))


@pytest.fixture
def kfp_client_tls(tls_flow, request):
    # Request the session client only after the per-flow opt-in/configuration gate.
    # A declared session dependency would run before function-scoped tls_flow.
    return request.getfixturevalue("tls_kfp_session")


@pytest.fixture
def tls_pipeline_source_target(tls_flow, kfp_client_tls, tmp_path_factory):
    from autox_tests.lib.managed_pipelines import (
        PipelineRunTarget,
        resolve_managed_pipeline_target,
    )
    from autox_tests.lib.pipeline_yaml_sources import PIPELINE_YAML_AUTORAG_ENV

    flow = tls_flow
    if flow.get("package_path") or flow.get("pipeline_id"):
        return PipelineRunTarget(
            mode="package" if flow.get("package_path") else "managed",
            artifact_prefix=flow["family"],
            package_path=flow.get("package_path"),
            pipeline_id=flow.get("pipeline_id"),
            pipeline_version_id=flow.get("pipeline_version_id"),
        )
    directory = tmp_path_factory.mktemp("tls-pipeline")
    return resolve_managed_pipeline_target(
        kfp_client_tls,
        kind=flow["family"],
        path_env_var=PIPELINE_YAML_AUTORAG_ENV,
        cache_dir=directory,
        cache_file_name=flow["family"] + "-pipeline.yaml",
    )


@pytest.fixture
def tls_pipeline_run_target(
    tls_flow, tls_pipeline_source_target, kfp_client_tls, tmp_path_factory, request
):
    from autox_tests.lib.tls.faults import prepare_maas_ca_fault

    target, evidence = prepare_maas_ca_fault(
        kfp_client_tls,
        tls_pipeline_source_target,
        tmp_path_factory.mktemp("tls-maas-fault"),
        request.getfixturevalue("tls_material")["other.pem"],
    )
    tls_flow["fault_evidence"] = evidence
    return target


@contextmanager
def _owned_s3_datasets(tls_cluster, tmp_path_factory, data_kinds):
    import base64
    import boto3
    from autox_tests.lib.tls.datasets import OwnedDatasets

    core, namespace = tls_cluster
    ca = service_ca("s3")
    pem = ca.get("ca_pem")
    if pem is None:
        configmap = core.read_namespaced_config_map(ca["ca_configmap"], namespace)
        pem = (configmap.data or {}).get(ca["ca_key"])
        if not pem:
            pytest.fail("S3 CA ConfigMap does not contain the configured certificate key")
    ca_file = tmp_path_factory.mktemp("tls-s3-ca") / "ca.pem"
    ca_file.write_text(pem)
    clients = {}
    owned = OwnedDatasets(clients)
    try:
        for kind in data_kinds:
            existing, values = endpoint_credentials("s3", data_kind=kind)
            required = {"AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
            if existing:
                secret = core.read_namespaced_secret(existing, namespace)
                values = CredentialValues({
                    key: base64.b64decode(value).decode()
                    for key, value in (secret.data or {}).items()
                    if key in required | {"AWS_DEFAULT_REGION"}
                })
            if any(not values.get(name) for name in required):
                pytest.fail("S3 dataset credentials must contain endpoint, access key and secret key")
            if not values["AWS_S3_ENDPOINT"].startswith("https://"):
                pytest.fail("TLS dataset endpoint must use HTTPS")
            clients[kind + "_data_bucket_name"] = boto3.client(
                "s3",
                endpoint_url=values["AWS_S3_ENDPOINT"],
                aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"],
                region_name=values.get("AWS_DEFAULT_REGION") or "us-east-1",
                verify=str(ca_file),
            )
        yield owned
    finally:
        try:
            owned.close()
        finally:
            for client in clients.values():
                client.close()


@pytest.fixture(scope="session")
def tls_owned_datasets(tls_cluster, tmp_path_factory):
    with _owned_s3_datasets(tls_cluster, tmp_path_factory, ("input", "test")) as owned:
        yield owned


@pytest.fixture(scope="session")
def tls_runtime_s3_dataset(tls_cluster, tmp_path_factory):
    """Upload one small object once, only when a real S3 probe is selected."""
    from autox_tests.lib.tls.datasets import DatasetObject, DatasetPlan

    bucket = os.environ.get("INPUT_DATA_BUCKET_NAME", "").strip()
    if not bucket:
        pytest.fail("Set INPUT_DATA_BUCKET_NAME for S3 certificate tests")
    with _owned_s3_datasets(tls_cluster, tmp_path_factory, ("input",)) as owned:
        key = owned.prefix + "/runtime/probe.txt"
        plan = DatasetPlan(
            values={"s3_key": key},
            objects=[DatasetObject(
                "input_data_bucket_name", key, b"AutoX TLS S3 probe\n", "text/plain",
            )],
        )
        owned.prepare({"arguments": {"input_data_bucket_name": bucket}}, plan)
        yield key
