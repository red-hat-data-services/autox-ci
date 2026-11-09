"""Inject CA faults into a local pipeline copy without changing managed resources."""

import copy
import hashlib
import ssl
from dataclasses import replace
from pathlib import Path

import yaml

from autox_tests.lib.tls.flows import TLS_ERROR

CA_MARKER = "AUTOX_TLS_MAAS_CA_SHA256="


def _pipeline_documents(client, target):
    if target.mode == "package":
        return list(yaml.safe_load_all(Path(target.package_path).read_text()))
    if not target.pipeline_id or not target.pipeline_version_id:
        raise ValueError("Automatic CA faults require a pinned source pipeline version")
    version = client.get_pipeline_version(
        pipeline_id=target.pipeline_id, pipeline_version_id=target.pipeline_version_id
    )
    value = (
        version.get("pipeline_spec")
        if isinstance(version, dict)
        else version.pipeline_spec
    )
    if isinstance(value, str):
        return list(yaml.safe_load_all(value))
    if not isinstance(value, dict):
        raise ValueError(
            "Source pipeline version did not provide a pipeline specification"
        )
    value = copy.deepcopy(value)
    if "pipeline_spec" in value or "pipelineSpec" in value:
        spec = value.get("pipeline_spec", value.get("pipelineSpec"))
        platform = value.get("platform_spec", value.get("platformSpec"))
        return [spec, platform] if platform else [spec]
    platform = value.pop("platforms", None)
    return [value, {"platforms": platform}] if platform else [value]


def maas_ca_wrapper(pem):
    """The product runs unchanged after setting its supported explicit CA variable."""
    # Validate the fixture locally; malformed PEM must not masquerade as wrong issuer.
    ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=pem)
    digest = hashlib.sha256(pem.encode()).hexdigest()
    return (
        "import os, sys, tempfile\n"
        f"pem = {pem!r}\n"
        "descriptor, path = tempfile.mkstemp(prefix='autox-maas-wrong-', suffix='.pem')\n"
        "with os.fdopen(descriptor, 'w') as output:\n"
        "    output.write(pem)\n"
        "os.environ['MAAS_CA_BUNDLE'] = path\n"
        f"print({CA_MARKER + digest!r}, flush=True)\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n"
    )


def prepare_maas_ca_fault(client, target, directory, pem):
    """Preserve images, args, credentials and platform settings in a per-run package."""
    documents = copy.deepcopy(_pipeline_documents(client, target))
    specs = [
        doc for doc in documents if isinstance(doc, dict) and "deploymentSpec" in doc
    ]
    platforms = [
        doc for doc in documents if isinstance(doc, dict) and "platforms" in doc
    ]
    if len(specs) != 1 or len(platforms) != 1:
        raise ValueError(
            "Automatic MaaS CA injection needs one pipeline spec and Kubernetes platform spec"
        )
    spec = specs[0]
    kubernetes = platforms[0]["platforms"].get("kubernetes", {})
    bindings = kubernetes.get("deploymentSpec", {}).get("executors", {})
    executors = spec["deploymentSpec"].get("executors", {})
    affected = []
    wrapper = maas_ca_wrapper(pem)
    for name, binding in bindings.items():
        keys = {
            item.get("envVar")
            for secret in binding.get("secretAsEnv", [])
            for item in secret.get("keyToEnv", [])
        }
        if not {"MAAS_BASE_URL", "MAAS_API_KEY"} <= keys:
            continue
        container = executors.get(name, {}).get("container", {})
        command = container.get("command")
        if not command or any(not isinstance(arg, str) for arg in command):
            raise ValueError(f"MaaS executor {name} has no supported explicit command")
        container["command"] = ["python3", "-c", wrapper, *command]
        affected.append(name)
    if not affected:
        raise ValueError(
            "No executors consume MaaS credentials; refusing an ineffective CA fault"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "maas-wrong-ca.yaml"
    package.write_text(yaml.safe_dump_all(documents, sort_keys=False))
    evidence = {
        "kind": "maas_wrong_ca",
        "ca_sha256": hashlib.sha256(pem.encode()).hexdigest(),
        "executors": sorted(affected),
        "source_pipeline_id": target.pipeline_id,
        "source_pipeline_version_id": target.pipeline_version_id,
    }
    return replace(target, mode="package", package_path=str(package)), evidence


def assert_injected_maas_rejection(logs, task_fragment, evidence):
    marker = CA_MARKER + evidence["ca_sha256"]
    assert any(
        task_fragment in task
        and failed
        and marker in content.splitlines()
        and TLS_ERROR.search(content)
        for task, _, failed, content in logs
    ), (
        "Expected failed MaaS task must show both generated CA injection and certificate rejection"
    )
