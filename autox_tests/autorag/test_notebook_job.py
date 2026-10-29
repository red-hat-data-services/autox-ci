"""Unit coverage for connected and disconnected notebook Job secret injection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kubernetes import client as k8s_client

from autox_tests.lib import notebooks

pytestmark = pytest.mark.config


class _BatchApi:
    def __init__(self, *args, **kwargs):
        self.job = None

    def create_namespaced_job(self, **kwargs):
        self.job = kwargs["body"]

    def read_namespaced_job_status(self, **kwargs):
        return SimpleNamespace(status=SimpleNamespace(succeeded=1, failed=0))

    def delete_namespaced_job(self, **kwargs):
        return None


@pytest.mark.parametrize(
    ("docling_secret_name", "expected_secrets"),
    [
        (None, ["s3", "maas", "vector", "pip"]),
        ("docling", ["s3", "maas", "vector", "pip", "docling"]),
    ],
)
def test_notebook_job_only_injects_docling_for_disconnected_autorag(
    monkeypatch: pytest.MonkeyPatch,
    docling_secret_name: str | None,
    expected_secrets: list[str],
) -> None:
    """Connected and AutoML Jobs never require the optional Docling Secret."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("RHOAI_NOTEBOOK_PIP_SECRET_NAME", "pip")
    if docling_secret_name:
        monkeypatch.setenv("RHOAI_NOTEBOOK_DOCLING_SECRET_NAME", docling_secret_name)
    else:
        monkeypatch.delenv("RHOAI_NOTEBOOK_DOCLING_SECRET_NAME", raising=False)
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=["s3", "maas", "vector"],
    )

    injected = [
        source.secret_ref.name
        for source in batch_api.job.spec.template.spec.containers[0].env_from
    ]
    assert injected == expected_secrets


def test_notebook_job_uses_docling_path_for_offline_cell_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner only skips disconnected validation cells without Docling artifacts."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=[],
    )

    env = {
        variable.name: variable.value
        for variable in batch_api.job.spec.template.spec.containers[0].env
    }
    assert "NOTEBOOK_EXCLUDE_FIRST_CODE_CELLS_AFTER_SECTIONS" not in env
    assert "_exclude_code_cells_before_next_markdown_after_sections" in (
        notebooks._NOTEBOOK_JOB_PROGRAM
    )
    assert 'cell["source"] = "# Skipped by the AutoX notebook runner.' in (
        notebooks._NOTEBOOK_JOB_PROGRAM
    )
    assert "if not docling_path:" in notebooks._NOTEBOOK_JOB_PROGRAM
    assert '"Validate Offline Configuration"' in notebooks._NOTEBOOK_JOB_PROGRAM
    assert '"Configure Models for Disconnected Environments"' not in (
        notebooks._NOTEBOOK_JOB_PROGRAM
    )
    compile(notebooks._NOTEBOOK_JOB_PROGRAM, "notebook-runner", "exec")


def test_notebook_job_passes_pip_index_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly configured package index is available to notebook cells."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("NOTEBOOK_PIP_INDEX_URL", "https://pypi.example.invalid/simple")
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=[],
    )

    env = {
        variable.name: variable.value
        for variable in batch_api.job.spec.template.spec.containers[0].env
    }
    assert env["NOTEBOOK_PIP_INDEX_URL"] == "https://pypi.example.invalid/simple"
    assert "not os.environ.get(\"PIP_INDEX_URL\", \"\").strip()" in (
        notebooks._NOTEBOOK_JOB_PROGRAM
    )


def test_notebook_job_mounts_ca_bundle_configmap_at_workbench_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured CA-bundle ConfigMap is mounted at the workbench trust path."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP", "trusted-ca")
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=[],
    )

    pod_spec = batch_api.job.spec.template.spec
    volume = pod_spec.volumes[0]
    assert volume.name == "notebook-ca-bundle"
    assert volume.config_map.name == "trusted-ca"
    assert volume.config_map.items[0].key == "ca-bundle.crt"
    assert volume.config_map.items[0].path == "ca-bundle.crt"
    mount = pod_spec.containers[0].volume_mounts[0]
    assert mount.name == "notebook-ca-bundle"
    assert mount.mount_path == notebooks._WORKBENCH_CA_BUNDLE_DIR
    assert mount.read_only is True


def test_notebook_job_mounts_ca_bundle_configmap_with_custom_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-default ConfigMap key is remapped to ca-bundle.crt in the mount."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP", "router-certs")
    monkeypatch.setenv("RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP_KEY", "tls.crt")
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=[],
    )

    volume = batch_api.job.spec.template.spec.volumes[0]
    assert volume.config_map.items[0].key == "tls.crt"
    assert volume.config_map.items[0].path == "ca-bundle.crt"


def test_notebook_job_without_ca_bundle_configmap_mounts_no_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the optional CA-bundle ConfigMap the pod keeps its default spec."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.delenv("RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP", raising=False)
    monkeypatch.delenv("RHOAI_NOTEBOOK_CA_BUNDLE_CONFIGMAP_KEY", raising=False)
    monkeypatch.setattr(
        notebooks,
        "make_k8s_core_api_from_config",
        lambda config: SimpleNamespace(api_client=object()),
    )
    monkeypatch.setattr(k8s_client, "BatchV1Api", lambda *args, **kwargs: batch_api)

    notebooks.run_notebooks_as_k8s_job(
        bucket="artifacts",
        notebook_keys=["indexing.ipynb"],
        config={"rhoai_project": "test-project"},
        secret_names=[],
    )

    pod_spec = batch_api.job.spec.template.spec
    assert pod_spec.volumes is None
    assert pod_spec.containers[0].volume_mounts is None


def test_notebook_runner_program_materializes_ca_bundle() -> None:
    """The runner trusts a private CA without disabling TLS verification."""
    program = notebooks._NOTEBOOK_JOB_PROGRAM
    # ai4rag's create_maas_client reads MAAS_CA_BUNDLE as a PEM file path.
    assert 'os.environ.get("MAAS_CA_BUNDLE", "")' in program
    # PEM data delivered by an env_from secret becomes a file on disk.
    assert "ca_data.startswith(\"-----BEGIN\")" in program
    assert 'os.environ["MAAS_CA_BUNDLE"] = str(materialized)' in program
    # The mounted workbench trust bundle is used when no secret data is set,
    # at the exact path the generated notebooks read for AWS_CA_BUNDLE.
    assert (
        f"Path(\"{notebooks._WORKBENCH_CA_BUNDLE_DIR}/ca-bundle.crt\")" in program
    )
    assert 'os.environ.setdefault("AWS_CA_BUNDLE", str(workbench_ca))' in program
    # Generic clients get a merged bundle that keeps the system CAs.
    assert 'os.environ.setdefault("SSL_CERT_FILE", str(merged))' in program
    assert 'os.environ.setdefault("REQUESTS_CA_BUNDLE", str(merged))' in program
    compile(program, "notebook-runner", "exec")

