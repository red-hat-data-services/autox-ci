"""Local tests for proxy propagation into the harness HTTP clients.

Jenkins agents reach the OpenShift API and the KFP route through a corporate
egress proxy. The kubernetes Python client and the KFP SDK ignore
``HTTPS_PROXY``/``NO_PROXY`` unless they are copied into their own
``Configuration`` objects (``oc`` and boto3 read the environment), so these
tests pin that behavior.
"""

import os

import urllib3
import pytest
from requests.utils import get_environ_proxies, should_bypass_proxies

from autox_tests.conftest import make_kfp_client_for_session
from autox_tests.lib import k8s_utils
from autox_tests.lib.kfp_retry import _find_api_client

_PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy")


def _clear_proxy_env(monkeypatch) -> None:
    for name in _PROXY_ENV:
        monkeypatch.delenv(name, raising=False)


def _blank_configuration():
    from kubernetes import client

    return client.Configuration()


def _api_client(configuration):
    from kubernetes import client

    return client.ApiClient(configuration)


def test_apply_env_proxy_populates_configuration(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    monkeypatch.setenv("NO_PROXY", "*.rhai-wx.com, example.internal")
    configuration = _blank_configuration()

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert configuration.proxy == "http://squid.corp.redhat.com:3128"
    assert configuration.no_proxy.split(",") == [
        ".rhai-wx.com",
        "example.internal",
        "localhost",
        "127.0.0.1",
        "::1",
        ".svc",
        ".cluster.local",
    ]


def test_apply_env_proxy_adds_scheme_to_bare_host_port(monkeypatch) -> None:
    """Jenkins may export ``https_proxy=squid.corp.redhat.com:3128``."""
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("https_proxy", "squid.corp.redhat.com:3128")
    configuration = _blank_configuration()
    configuration.host = "https://api.atris-con-2.rhai-wx.com:6443"

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert configuration.proxy == "http://squid.corp.redhat.com:3128"
    assert isinstance(_api_client(configuration).rest_client.pool_manager, urllib3.ProxyManager)


def test_apply_env_proxy_is_noop_without_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    configuration = _blank_configuration()

    assert k8s_utils.apply_env_proxy(configuration) is False
    assert configuration.proxy is None


def test_normalized_no_proxy_matches_wildcard_subdomains() -> None:
    normalized = k8s_utils.normalize_no_proxy("*.rhai-wx.com")
    assert should_bypass_proxies(
        "https://api.atris-con-2.rhai-wx.com:6443", no_proxy=normalized
    )


def test_configured_client_uses_proxy_manager(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    configuration = _blank_configuration()
    configuration.host = "https://api.atris-con-2.rhai-wx.com:6443"

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert isinstance(_api_client(configuration).rest_client.pool_manager, urllib3.ProxyManager)


def test_api_client_bypasses_proxy_for_in_cluster_host(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    configuration = _blank_configuration()
    configuration.host = "https://kubernetes.default.svc:443"

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert type(_api_client(configuration).rest_client.pool_manager) is urllib3.PoolManager


def test_lowercase_no_proxy_is_honored(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    monkeypatch.setenv("no_proxy", "*.rhai-wx.com")

    assert k8s_utils.proxy_for_url("https://api.atris-con-2.rhai-wx.com:6443") is None
    assert k8s_utils.proxy_for_url("https://s3.example.com") == "http://squid.corp.redhat.com:3128"


def test_normalize_proxy_environment_rewrites_bare_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "squid.corp.redhat.com:3128")
    monkeypatch.setenv("NO_PROXY", "*.rhai-wx.com")

    assert k8s_utils.normalize_proxy_environment() is True
    assert os.environ["HTTPS_PROXY"] == "http://squid.corp.redhat.com:3128"
    assert os.environ["NO_PROXY"].split(",") == [
        ".rhai-wx.com",
        "localhost",
        "127.0.0.1",
        "::1",
        ".svc",
        ".cluster.local",
    ]


def test_normalize_proxy_environment_sets_default_no_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("https_proxy", "squid.corp.redhat.com:3128")

    assert k8s_utils.normalize_proxy_environment() is True
    assert os.environ["https_proxy"] == "http://squid.corp.redhat.com:3128"
    assert os.environ["NO_PROXY"].split(",") == [
        "localhost",
        "127.0.0.1",
        "::1",
        ".svc",
        ".cluster.local",
    ]


def test_requests_honors_normalized_wildcard_no_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "squid.corp.redhat.com:3128")
    monkeypatch.setenv("NO_PROXY", "*.rhai-wx.com")

    k8s_utils.normalize_proxy_environment()

    assert get_environ_proxies("https://api.atris-con-2.rhai-wx.com:6443") == {}


def test_kfp_client_uses_proxy_manager(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "squid.corp.redhat.com:3128")
    namespace_config = {"rhoai_project": "test-project", "rhoai_token": "token"}

    client = make_kfp_client_for_session(
        namespace_config=namespace_config,
        configured_kfp_url="https://ds-pipeline.example.invalid",
        datascience_pipelines_application=None,
        kubeconfig_path=None,
    )

    api_client = _find_api_client(client)
    assert api_client is not None
    assert isinstance(api_client.rest_client.pool_manager, urllib3.ProxyManager)


def test_kfp_client_bypasses_proxy_for_in_cluster_host(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    namespace_config = {"rhoai_project": "test-project", "rhoai_token": "token"}

    client = make_kfp_client_for_session(
        namespace_config=namespace_config,
        configured_kfp_url="https://ds-pipeline.test-project.svc:8888",
        datascience_pipelines_application=None,
        kubeconfig_path=None,
    )

    api_client = _find_api_client(client)
    assert api_client is not None
    assert not isinstance(api_client.rest_client.pool_manager, urllib3.ProxyManager)


def test_make_k8s_core_api_from_config_applies_proxy(monkeypatch) -> None:
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    config = {
        "rhoai_token": "token",
        "rhoai_kfp_url": "https://ds-pipeline.apps.example.invalid",
    }

    api = k8s_utils.make_k8s_core_api_from_config(config)

    assert api is not None
    assert isinstance(api.api_client.rest_client.pool_manager, urllib3.ProxyManager)
