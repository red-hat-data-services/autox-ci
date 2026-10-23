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


def _classify(monkeypatch: pytest.MonkeyPatch, *, pods, maas_bound, timed_out):
    monkeypatch.setattr(notebooks, "_list_job_pods", lambda *args, **kwargs: pods)
    monkeypatch.setattr(notebooks, "_job_pod_logs", lambda *args, **kwargs: "(logs)")
    return notebooks._classify_notebook_job_failure(
        None, "test-project", "job-1", 300, maas_bound=maas_bound, timed_out=timed_out
    )


def _unschedulable_pod() -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name="job-1-abc"),
        status=SimpleNamespace(
            phase="Pending",
            reason=None,
            conditions=[
                SimpleNamespace(
                    type="PodScheduled",
                    status="False",
                    reason="Unschedulable",
                    message="0/3 nodes are available: 3 Insufficient memory.",
                )
            ],
            container_statuses=[],
        ),
    )


def test_notebook_job_timeout_maas_bound_is_maas_slowness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MaaS-bound Job (AutoRAG notebooks) timing out while running is rerunnable."""
    from autox_tests.lib.failure_classes import MaasSlownessTimeout

    exc = _classify(
        monkeypatch, pods=[SimpleNamespace()], maas_bound=True, timed_out=True
    )
    assert isinstance(exc, MaasSlownessTimeout)


def test_notebook_job_timeout_resource_signals_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unschedulable Job pod is a resource shortage and must not be rerun."""
    from autox_tests.lib.failure_classes import ResourceShortageTimeout

    exc = _classify(
        monkeypatch,
        pods=[_unschedulable_pod()],
        maas_bound=True,
        timed_out=True,
    )
    assert isinstance(exc, ResourceShortageTimeout)
    assert "Unschedulable" in str(exc)


def test_notebook_job_timeout_not_maas_bound_keeps_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AutoML Jobs (not MaaS-bound) keep the original, non-rerun failure."""
    exc = _classify(
        monkeypatch, pods=[SimpleNamespace()], maas_bound=False, timed_out=True
    )
    assert type(exc) is AssertionError


def test_notebook_job_failure_oomkilled_is_resource_shortage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed Job (OOMKilled) is classified as resource shortage."""
    from autox_tests.lib.failure_classes import ResourceShortageTimeout

    oom_pod = SimpleNamespace(
        metadata=SimpleNamespace(name="job-1-abc"),
        status=SimpleNamespace(
            phase="Running",
            reason=None,
            conditions=[],
            container_statuses=[
                SimpleNamespace(
                    name="notebook-runner",
                    state=SimpleNamespace(
                        terminated=SimpleNamespace(reason="OOMKilled", exit_code=137),
                        waiting=None,
                    ),
                )
            ],
        ),
    )
    exc = _classify(
        monkeypatch, pods=[oom_pod], maas_bound=True, timed_out=False
    )
    assert isinstance(exc, ResourceShortageTimeout)
    assert "OOMKilled" in str(exc)
