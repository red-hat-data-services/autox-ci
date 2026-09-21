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
