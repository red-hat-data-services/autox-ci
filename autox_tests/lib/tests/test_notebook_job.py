"""Unit coverage for connected and disconnected notebook Job secret injection."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from kubernetes import client as k8s_client

from autox_tests.lib import notebooks

pytestmark = pytest.mark.config


@pytest.mark.parametrize("has_pip_secret", [False, True])
@pytest.mark.parametrize(
    "install_command",
    [
        "%pip install autogluon.tabular[lightgbm,xgboost,tabm,fastai]==1.5.0+rhaiv.7 | tail -n 1",
        "%pip install autogluon.timeseries==1.5.0+rhaiv.7 | tail -n 1",
    ],
)
@pytest.mark.parametrize("blank_before_install", [False, True])
def test_notebook_job_rewrites_automl_extra_index_cell_only_with_pip_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    has_pip_secret: bool,
    install_command: str,
    blank_before_install: bool,
) -> None:
    """Patch the AutoML install cell only when a pip Secret is configured."""
    original_cell = (
        "import os\n\n"
        'os.environ["PIP_EXTRA_INDEX_URL"] = (\n'
        '    "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6-EA2/cpu-ubi9-test/simple/"\n'
        ")\n"
        + ("\n" if blank_before_install else "")
        + install_command
    )
    captured: dict[str, object] = {}

    class _S3:
        def download_file(self, bucket: str, key: str, path: str) -> None:
            del bucket, key
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cells": [{"cell_type": "code", "source": original_cell}]}, f)

    class _Nbformat:
        @staticmethod
        def read(file, as_version: int):
            del as_version
            return SimpleNamespace(
                cells=[SimpleNamespace(**cell) for cell in json.load(file)["cells"]]
            )

        @staticmethod
        def write(notebook, file) -> None:
            json.dump(
                {"cells": [vars(cell) for cell in notebook.cells]}, file
            )

    def execute_notebook(input_path: str, output_path: str, **kwargs) -> None:
        del output_path, kwargs
        with open(input_path, encoding="utf-8") as f:
            captured["source"] = json.load(f)["cells"][0]["source"]

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *_, **__: _S3()))
    monkeypatch.setitem(sys.modules, "nbformat", _Nbformat)
    monkeypatch.setitem(sys.modules, "papermill", SimpleNamespace(execute_notebook=execute_notebook))
    monkeypatch.setenv("NOTEBOOK_S3_BUCKET", "artifacts")
    monkeypatch.setenv(
        "NOTEBOOK_S3_KEYS", json.dumps(["model/notebooks/automl_predictor_notebook.ipynb"])
    )
    monkeypatch.setenv("NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("NOTEBOOK_WORKDIR", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_HAS_PIP_SECRET", str(has_pip_secret).lower())
    monkeypatch.delenv("DOCLING_ARTIFACTS_S3_PREFIX", raising=False)
    monkeypatch.delenv("DOCLING_ARTIFACTS_S3_BUCKET", raising=False)
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)

    program = notebooks._NOTEBOOK_JOB_PROGRAM.replace(
        'Path("/tmp/notebooks")', 'Path(os.environ["NOTEBOOK_WORKDIR"])'
    )
    exec(compile(program, "notebook-job-program", "exec"), {})

    expected_cell = original_cell.replace("PIP_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL_DEV")
    assert captured["source"] == (expected_cell if has_pip_secret else original_cell)


def test_notebook_job_prints_pip_output_when_papermill_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pip diagnostics saved in output.ipynb must reach the failed Job's logs."""

    class _S3:
        def download_file(self, bucket: str, key: str, path: str) -> None:
            del bucket, key
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cells": []}, f)

    class _Nbformat:
        @staticmethod
        def read(file, as_version: int):
            del as_version
            return SimpleNamespace(cells=json.load(file)["cells"])

    def execute_notebook(input_path: str, output_path: str, **kwargs) -> None:
        del input_path, kwargs
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "source": "%pip install ai4rag",
                            "outputs": [
                                {
                                    "output_type": "stream",
                                    "text": "ERROR: No matching distribution found for ai4rag\\n",
                                }
                            ],
                        },
                        {
                            "cell_type": "code",
                            "source": "from ai4rag import missing",
                            "outputs": [
                                {
                                    "output_type": "error",
                                    "traceback": ["ModuleNotFoundError: No module named 'ai4rag'"],
                                }
                            ],
                        },
                    ]
                },
                f,
            )
        error = RuntimeError("notebook execution failed")
        error.cell_index = 1
        raise error

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *_, **__: _S3()))
    monkeypatch.setitem(sys.modules, "nbformat", _Nbformat)
    monkeypatch.setitem(sys.modules, "papermill", SimpleNamespace(execute_notebook=execute_notebook))
    monkeypatch.setenv("NOTEBOOK_S3_BUCKET", "artifacts")
    monkeypatch.setenv("NOTEBOOK_S3_KEYS", json.dumps(["rag_patterns/indexing.ipynb"]))
    monkeypatch.setenv("NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    monkeypatch.setenv("NOTEBOOK_WORKDIR", str(tmp_path))
    monkeypatch.setenv("NOTEBOOK_INJECT_MOCK_INPUT", "false")
    monkeypatch.setenv("NOTEBOOK_HAS_PIP_SECRET", "false")
    monkeypatch.delenv("DOCLING_ARTIFACTS_S3_PREFIX", raising=False)
    monkeypatch.delenv("DOCLING_ARTIFACTS_S3_BUCKET", raising=False)
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)

    program = notebooks._NOTEBOOK_JOB_PROGRAM.replace(
        'Path("/tmp/notebooks")', 'Path(os.environ["NOTEBOOK_WORKDIR"])'
    )
    with pytest.raises(RuntimeError, match="notebook execution failed"):
        exec(compile(program, "notebook-job-program", "exec"), {})

    output = capsys.readouterr().out
    assert "NOTEBOOK EXECUTION DIAGNOSTICS" in output
    assert "package-install output" in output
    assert "No matching distribution found for ai4rag" in output
    assert "ModuleNotFoundError: No module named 'ai4rag'" in output


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
    ("docling_secret_name", "pip_secret_name", "expected_secrets"),
    [
        (None, None, ["s3", "maas", "vector"]),
        (None, "pip", ["s3", "maas", "vector", "pip"]),
        ("docling", "pip", ["s3", "maas", "vector", "pip", "docling"]),
    ],
)
def test_notebook_job_only_injects_docling_for_disconnected_autorag(
    monkeypatch: pytest.MonkeyPatch,
    docling_secret_name: str | None,
    pip_secret_name: str | None,
    expected_secrets: list[str],
) -> None:
    """Connected and AutoML Jobs never require the optional Docling Secret."""
    batch_api = _BatchApi()
    monkeypatch.setenv("RHOAI_NOTEBOOK_RUNNER_IMAGE", "example.invalid/notebook:latest")
    if pip_secret_name:
        monkeypatch.setenv("RHOAI_NOTEBOOK_PIP_SECRET_NAME", pip_secret_name)
    else:
        monkeypatch.delenv("RHOAI_NOTEBOOK_PIP_SECRET_NAME", raising=False)
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
    env = {item.name: item.value for item in batch_api.job.spec.template.spec.containers[0].env}
    assert env["NOTEBOOK_HAS_PIP_SECRET"] == str(bool(pip_secret_name)).lower()
