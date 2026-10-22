"""Unit coverage for connected and disconnected notebook Job secret injection."""

from __future__ import annotations

import ast
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
def test_notebook_job_injects_docling_secret_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    docling_secret_name: str | None,
    expected_secrets: list[str],
) -> None:
    """Either cluster mode can use the optional Docling artifacts Secret."""
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


@pytest.fixture
def filter_notebook_cells():
    """Load the embedded filter without starting S3 or notebook execution."""
    program = ast.parse(notebooks._NOTEBOOK_JOB_PROGRAM)
    helper = next(
        node
        for node in program.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_exclude_code_cells_for_environment"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "notebook-filter", "exec"),
        namespace,
    )
    return namespace[helper.name]


@pytest.mark.parametrize("disconnected", [False, True])
@pytest.mark.parametrize(
    ("tags", "environment"),
    [
        ([], None),
        (["setup"], None),
        (["disconnected"], "disconnected"),
        (["connected"], "connected"),
        (["DISCONNECTED", "setup"], "disconnected"),
        ([" CONNECTED "], "connected"),
    ],
)
def test_notebook_environment_filter_executes_matching_and_shared_cells(
    filter_notebook_cells,
    disconnected,
    tags,
    environment,
    capsys,
):
    """Only the matching environment executes, irrespective of surrounding Markdown."""
    source = 'executed.append("selected")'
    notebook = SimpleNamespace(
        cells=[
            {
                "cell_type": "markdown",
                "source": "### A renamed section",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": source,
                "metadata": {"tags": tags},
                "execution_count": 7,
                "outputs": [{"output_type": "stream", "text": "old output"}],
            },
            {"cell_type": "code", "source": 'executed.append("shared")'},
            {
                "cell_type": "markdown",
                "source": "### Validate Offline Configuration",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": 'executed.append("after heading")',
                "metadata": {},
            },
        ]
    )
    original = deepcopy(notebook.cells)
    filter_notebook_cells(notebook, disconnected=disconnected)
    current_environment = "disconnected" if disconnected else "connected"
    should_run = environment is None or environment == current_environment

    namespace = {"executed": []}
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    assert namespace["executed"] == (["selected"] if should_run else []) + [
        "shared",
        "after heading",
    ]
    assert len(notebook.cells) == len(original)
    for index in (0, 2, 3, 4):
        assert notebook.cells[index] == original[index]
    if should_run:
        assert notebook.cells[1] == original[1]
        assert not capsys.readouterr().out
    else:
        assert notebook.cells[1]["metadata"] == original[1]["metadata"]
        assert notebook.cells[1]["execution_count"] is None
        assert notebook.cells[1]["outputs"] == []
        output = capsys.readouterr().out
        assert "Skipping code cell 1" in output
        assert f"tagged {environment!r}" in output
        assert f"in {current_environment} environment" in output


@pytest.mark.parametrize("disconnected", [False, True])
def test_notebook_environment_filter_rejects_conflicting_tags(
    filter_notebook_cells, disconnected
):
    """Conflicting tags must fail before Papermill executes the notebook."""
    notebook = SimpleNamespace(
        cells=[
            {
                "cell_type": "code",
                "source": "pass",
                "metadata": {"tags": ["CONNECTED", "disconnected"]},
            }
        ]
    )
    with pytest.raises(
        ValueError, match="Code cell 0 has both connected and disconnected tags"
    ):
        filter_notebook_cells(notebook, disconnected=disconnected)


def test_notebook_environment_filter_preserves_non_code_cells(filter_notebook_cells):
    """Environment tags on Markdown and raw cells do not affect their contents."""
    notebook = SimpleNamespace(
        cells=[
            {
                "cell_type": cell_type,
                "source": "keep",
                "metadata": {"tags": ["disconnected", "connected"]},
            }
            for cell_type in ("markdown", "raw")
        ]
    )
    original = deepcopy(notebook.cells)
    filter_notebook_cells(notebook, disconnected=False)
    assert notebook.cells == original


@pytest.mark.parametrize("disconnected", [False, True])
@pytest.mark.parametrize("inject_mock_input", [False, True])
@pytest.mark.parametrize("artifact_source", ["none", "image", "s3"])
def test_notebook_job_filters_tags_using_explicit_cluster_mode(
    monkeypatch,
    tmp_path,
    disconnected,
    inject_mock_input,
    artifact_source,
):
    """Run the embedded Job program with local S3 and notebook execution substitutes."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
    cells = [
        {
            "cell_type": "code",
            "source": f'executed.append("{label}")',
            "metadata": {"tags": tags},
        }
        for label, tags in (
            ("shared", []),
            ("offline", ["disconnected"]),
            ("online", ["connected"]),
        )
    ]
    downloaded = []
    executed = []

    def download_file(bucket, key, destination):
        downloaded.append((bucket, key))
        Path(destination).write_text(
            json.dumps({"cells": cells}) if key.endswith(".ipynb") else "model"
        )

    def read_notebook(stream, as_version):
        assert as_version == 4
        return SimpleNamespace(cells=json.load(stream)["cells"])

    def write_notebook(notebook, stream):
        json.dump({"cells": notebook.cells}, stream)

    def execute_notebook(input_path, output_path, **kwargs):
        assert Path(output_path).parent == Path(input_path).parent
        assert kwargs["cwd"] == str(Path(input_path).parent)
        notebook = json.loads(Path(input_path).read_text())
        namespace = {"executed": executed}
        for cell in notebook["cells"]:
            exec(cell["source"], namespace)
        assert ("input" in namespace) == inject_mock_input
        if inject_mock_input:
            assert namespace["input"]() == "Sample query?"

    fake_s3 = SimpleNamespace(
        download_file=download_file,
        get_paginator=lambda name: SimpleNamespace(
            paginate=lambda **kwargs: [{"Contents": [{"Key": "models/model.bin"}]}]
        ),
    )
    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: fake_s3)
    )
    monkeypatch.setitem(
        sys.modules,
        "nbformat",
        SimpleNamespace(
            read=read_notebook,
            write=write_notebook,
            v4=SimpleNamespace(
                new_code_cell=lambda source: {
                    "cell_type": "code",
                    "source": source,
                    "metadata": {},
                }
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "papermill", SimpleNamespace(execute_notebook=execute_notebook)
    )
    monkeypatch.setitem(
        sys.modules,
        "pathlib",
        SimpleNamespace(
            Path=lambda value: (
                tmp_path / "notebooks" if value == "/tmp/notebooks" else Path(value)
            ),
        ),
    )
    for name in (
        "DOCLING_ARTIFACTS_PATH",
        "DOCLING_ARTIFACTS_S3_BUCKET",
        "DOCLING_ARTIFACTS_S3_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    if artifact_source != "none":
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "models"))
    if artifact_source == "s3":
        monkeypatch.setenv("DOCLING_ARTIFACTS_S3_BUCKET", "models-bucket")
        monkeypatch.setenv("DOCLING_ARTIFACTS_S3_PREFIX", "models/")
    monkeypatch.setenv("NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("NOTEBOOK_S3_BUCKET", "notebooks-bucket")
    monkeypatch.setenv("NOTEBOOK_S3_KEYS", json.dumps(["indexing.ipynb"]))
    monkeypatch.setenv("NOTEBOOK_INJECT_MOCK_INPUT", str(inject_mock_input).lower())
    monkeypatch.setenv(
        "NOTEBOOK_CLUSTER_MODE", "disconnected" if disconnected else "connected"
    )

    exec(compile(notebooks._NOTEBOOK_JOB_PROGRAM, "notebook-runner", "exec"), {})

    assert executed == ["shared", "offline" if disconnected else "online"]
    assert ("notebooks-bucket", "indexing.ipynb") in downloaded
    assert (("models-bucket", "models/model.bin") in downloaded) == (
        artifact_source == "s3"
    )


@pytest.mark.parametrize(
    ("configured_mode", "expected_mode"),
    [
        (None, "connected"),
        ("connected", "connected"),
        ("disconnected", "disconnected"),
        (" DISCONNECTED ", "disconnected"),
    ],
)
def test_notebook_job_passes_cluster_mode(monkeypatch, configured_mode, expected_mode):
    """The explicit mode reaches the pod independently of model secrets."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("RHOAI_NOTEBOOK_DOCLING_SECRET_NAME", "models")
    if configured_mode is None:
        monkeypatch.delenv("RHOAI_NOTEBOOK_CLUSTER_MODE", raising=False)
    else:
        monkeypatch.setenv("RHOAI_NOTEBOOK_CLUSTER_MODE", configured_mode)
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
    assert env["NOTEBOOK_CLUSTER_MODE"] == expected_mode


@pytest.mark.parametrize("mode", ["invalid", "auto", ""])
def test_notebook_job_rejects_invalid_cluster_mode(monkeypatch, mode):
    """Typos fail before contacting Kubernetes rather than selecting unintended cells."""
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("RHOAI_NOTEBOOK_CLUSTER_MODE", mode)
    make_api = Mock()
    monkeypatch.setattr(notebooks, "make_k8s_core_api_from_config", make_api)
    with pytest.raises(
        ValueError,
        match="RHOAI_NOTEBOOK_CLUSTER_MODE must be connected or disconnected",
    ):
        notebooks.run_notebooks_as_k8s_job(
            bucket="artifacts",
            notebook_keys=["indexing.ipynb"],
            config={"rhoai_project": "test-project"},
            secret_names=[],
        )
    make_api.assert_not_called()


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
    assert 'not os.environ.get("PIP_INDEX_URL", "").strip()' in (
        notebooks._NOTEBOOK_JOB_PROGRAM
    )
