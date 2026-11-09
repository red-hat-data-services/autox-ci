"""Parametrized TLS runtime scenarios, configured like AutoML/AutoRAG tests."""

import pytest

from autox_tests.autorag.tls.configs.configs import get_runtime_configs
from autox_tests.lib.tls.runtime import run_job

pytestmark = [pytest.mark.tls, pytest.mark.integration]


@pytest.mark.parametrize(
    "test_config", get_runtime_configs("maas"), ids=lambda config: config.id
)
def test_maas_runtime(tls_cluster, tls_material, tls_image, test_config):
    core, namespace = tls_cluster
    run_job(core, namespace, tls_image, test_config.payload(), tls_material)


@pytest.mark.parametrize(
    "test_config", get_runtime_configs("service"), ids=lambda config: config.id
)
def test_service_runtime(tls_cluster, tls_material, service_profile, test_config):
    core, namespace = tls_cluster
    run_job(
        core,
        namespace,
        service_profile["image"],
        {
            **test_config.payload(),
            "local_s3_fixture": service_profile["local_s3_fixture"],
            "local_milvus_fixture": service_profile["local_milvus_fixture"],
            "s3_key": service_profile["s3_key"],
        },
        tls_material,
        profile=service_profile,
        timeout=service_profile["timeout"],
    )
