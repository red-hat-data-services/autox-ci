"""Validate operator-provided trust using a real KFP task, without adding a CA mount."""

import json

import pytest

from autox_tests.tls.configs.configs import get_runtime_configs
from autox_tests.tls.utils.runtime import PROBE_SOURCE, parse_result


def probe_component(source: str, scenario: str) -> str:
    import json

    scope = {"__name__": "autox_tls_dspa"}
    exec(source, scope)
    result = scope["execute"](json.loads(scenario))
    print("AUTOX_TLS_RESULT=" + json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("DSPA TLS probe failed: " + json.dumps(result))
    return json.dumps(result)


@pytest.mark.tls
@pytest.mark.integration
@pytest.mark.parametrize(
    "test_config", get_runtime_configs("dspa"), ids=lambda config: config.id
)
def test_dspa_plain_ca_injection(
    tls_cluster,
    tls_image,
    tmp_path,
    test_config,
    tls_dspa_config,
    kfp_client_dspa,
    tls_dspa_secret,
):
    """Check platform trust unchanged and the isolated directory fallback separately."""
    from kfp import compiler, dsl, kubernetes
    from autox_tests.lib.managed_pipelines import (
        PipelineRunTarget,
        submit_pipeline_run_and_wait,
    )
    from autox_tests.lib.kfp_run_state import _run_succeeded

    core, namespace = tls_cluster
    secret = tls_dspa_secret
    case = test_config.payload()
    case["id"] = "DSPA-" + test_config.id
    case["mount_dir"] = tls_dspa_config["ca_dir"]
    component = dsl.component(base_image=tls_image, install_kfp_package=False)(
        probe_component
    )

    @dsl.pipeline(name="autox-tls-dspa")
    def pipeline():
        task = component(source=PROBE_SOURCE, scenario=json.dumps(case))
        task.set_caching_options(False)
        kubernetes.use_secret_as_env(
            task,
            secret,
            {
                "MAAS_BASE_URL": "MAAS_BASE_URL",
                "MAAS_API_KEY": "MAAS_API_KEY",
                "TLS_GENERATION_MODEL": "TLS_GENERATION_MODEL",
                "TLS_EMBEDDING_MODEL": "TLS_EMBEDDING_MODEL",
            },
        )

    package = tmp_path / "dspa-tls.yaml"
    compiler.Compiler().compile(pipeline, str(package))
    run_id, detail = submit_pipeline_run_and_wait(
        kfp_client_dspa,
        PipelineRunTarget(
            mode="package", artifact_prefix="autox-tls-dspa", package_path=str(package)
        ),
        {},
        run_name="autox-tls-dspa-" + test_config.id,
        timeout=tls_dspa_config["dspa_timeout"],
        namespace=namespace,
    )
    from autox_tests.tls.utils.flows import (
        collect_run_evidence,
        write_evidence,
    )

    evidence, logs = collect_run_evidence(core, namespace, run_id)
    write_evidence("dspa-" + run_id, evidence)
    probe_logs = "\n".join(content for _, _, _, content in logs)
    if any(line.startswith("AUTOX_TLS_RESULT=") for line in probe_logs.splitlines()):
        # Surface the actual configuration/request failure before checking the overall run state.
        parse_result(probe_logs, case)
    assert _run_succeeded(detail), f"DSPA diagnostic pipeline failed; run ID: {run_id}"
    parse_result(probe_logs, case)
