"""Local regressions for product environment isolation and TLS runner selection."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from autox_tests.lib import env

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.config
ROOT = Path(__file__).resolve().parents[2]


def test_product_env_does_not_load_the_other_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "_TESTS_DIR", tmp_path)
    monkeypatch.delenv("TLS_SELECTED_PRODUCT", raising=False)
    monkeypatch.delenv("TLS_ML_ONLY", raising=False)
    monkeypatch.setenv("TLS_JENKINS_VALUE", "jenkins")
    (tmp_path / ".env.ml").write_text("TLS_SELECTED_PRODUCT=automl\nTLS_ML_ONLY=yes\n")
    (tmp_path / ".env.rag").write_text(
        "TLS_SELECTED_PRODUCT=autorag\nTLS_JENKINS_VALUE=file\n"
    )

    env.load_tests_env("autorag")

    assert os.environ["TLS_SELECTED_PRODUCT"] == "autorag"
    assert os.environ["TLS_JENKINS_VALUE"] == "jenkins"
    assert "TLS_ML_ONLY" not in os.environ


def test_tls_does_not_initialize_functional_s3_clients(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    pytester.makeconftest("""
        import pytest
        from autox_tests.autorag.conftest import (
            upload_datasets, s3_teardown, s3_teardown_indexing, s3_cleanup_tracker,
        )

        @pytest.fixture(scope="session")
        def s3_client_functional():
            pytest.fail("TLS must not create the optimization S3 client")

        @pytest.fixture(scope="session")
        def s3_client_indexing_functional():
            pytest.fail("TLS must not create the indexing S3 client")
    """)
    pytester.makepyfile("""
        import pytest
        from autox_tests.autorag.tls.configs.configs import TLSRuntimeConfig

        @pytest.mark.parametrize("test_config", [TLSRuntimeConfig(id="tls-local")])
        def test_tls(test_config):
            assert test_config.id == "tls-local"
    """)
    pytester.runpytest_subprocess("-q").assert_outcomes(passed=1)


@pytest.mark.parametrize("automl_fails", [False, True])
@pytest.mark.parametrize("junit_option", ["--junitxml", "--junit-xml="])
def test_tls_alias_runs_products_with_isolated_envs(
    tmp_path, monkeypatch, automl_fails, junit_option
):
    shutil.copy2(ROOT / "run_tests.sh", tmp_path / "run_tests.sh")
    tests = tmp_path / "autox_tests"
    tests.mkdir()
    (tests / "conftest.py").write_text(
        "from pathlib import Path\n"
        "from autox_tests.lib import env\n"
        "env._TESTS_DIR = Path(__file__).parent\n"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    tls: certificate tests\n    positive: positive cases\n"
    )
    for product, suffix in (("automl", "ml"), ("autorag", "rag")):
        directory = tests / product / "tls"
        directory.mkdir(parents=True)
        (tests / f".env.{suffix}").write_text(
            f"TLS_SELECTED_PRODUCT={product}\nTLS_JENKINS_VALUE=file\n"
        )
        (directory.parent / "conftest.py").write_text(
            "from autox_tests.lib.env import load_tests_env\n"
            f"def pytest_configure(config): load_tests_env({product!r})\n"
        )
        should_pass = not (product == "automl" and automl_fails)
        (directory / "test_certificate.py").write_text(f"""
import os
import pytest

@pytest.mark.tls
@pytest.mark.positive
def test_certificate():
    assert os.environ['TLS_SELECTED_PRODUCT'] == {product!r}
    assert os.environ['TLS_JENKINS_VALUE'] == 'jenkins'
    assert os.environ[{product.upper() + '_FUNCTIONAL_TESTS_TAGS'!r}] == 's3'
    assert {should_pass!r}, 'simulated certificate rejection regression'

@pytest.mark.tls
def test_excluded_by_positive_filter():
    pytest.fail('the alias lost the positive marker filter')
""")
        (directory.parent / "test_functional.py").write_text(
            "def test_functional(): raise AssertionError('TLS selected a functional test')\n"
        )

    # Exercise the real shell runner and pytest, with no dependency downloads.
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path('uv-calls.jsonl').open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "args = sys.argv[sys.argv.index('pytest') + 1:]\n"
        "os.execv(sys.executable, [sys.executable, '-m', 'pytest', *args])\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("TLS_JENKINS_VALUE", "jenkins")
    monkeypatch.delenv("TLS_SELECTED_PRODUCT", raising=False)
    report = str(tmp_path / "results/certificates.xml")
    report_args = (
        [junit_option + report] if junit_option.endswith("=") else [junit_option, report]
    )

    result = subprocess.run(
        ["bash", str(tmp_path / "run_tests.sh"), "--suite", "tls", "positive",
         "--tags", "s3", "--", "-q", *report_args],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == int(automl_fails), result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "uv-calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    for product in ("automl", "autorag"):
        assert (tmp_path / "results" / f"{product}-certificates.xml").is_file()
