"""Product CA rejection with injected faults and certificate-specific evidence."""

import hashlib
import uuid
from pathlib import Path

import pytest

from autox_tests.tls.configs.configs import get_flow_configs
from autox_tests.tls.utils.settings import timeout_seconds
from autox_tests.tls.utils.flows import (
    assert_tls_failure,
    collect_run_evidence,
    write_evidence,
)

pytestmark = [pytest.mark.tls, pytest.mark.integration]


@pytest.mark.parametrize(
    "test_config", get_flow_configs(), ids=lambda config: config.id
)
def test_product_pipeline_tls(
    tls_cluster,
    test_config,
    tls_flow,
    kfp_client_tls,
    tls_pipeline_run_target,
):
    from autox_tests.lib.kfp_run_state import _run_failed
    from autox_tests.lib.managed_pipelines import submit_pipeline_run_and_wait

    core, namespace = tls_cluster
    flow = tls_flow
    target = tls_pipeline_run_target
    run_id, detail = submit_pipeline_run_and_wait(
        kfp_client_tls,
        target,
        flow["arguments"],
        run_name="autox-ca-" + uuid.uuid4().hex[:12],
        timeout=timeout_seconds(3600),
        namespace=namespace,
    )
    evidence, logs = collect_run_evidence(core, namespace, run_id)
    evidence.update(
        {
            "flow": test_config.id,
            "family": flow["family"],
            "pipeline_id": target.pipeline_id,
            "pipeline_version_id": target.pipeline_version_id,
        }
    )
    if target.package_path:
        evidence["yaml_sha256"] = hashlib.sha256(
            Path(target.package_path).read_bytes()
        ).hexdigest()
    if flow.get("fault_evidence"):
        evidence["fault"] = flow["fault_evidence"]
    write_evidence(run_id, evidence)
    assert _run_failed(detail), f"Expected a CA-rejected pipeline run: {run_id}"
    assert_tls_failure(logs, flow["expected_failed_task"])
    from autox_tests.tls.utils.faults import assert_injected_maas_rejection

    assert_injected_maas_rejection(
        logs, flow["expected_failed_task"], flow["fault_evidence"]
    )
