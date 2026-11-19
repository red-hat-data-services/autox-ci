"""Local tests for proxy propagation into the kubernetes client.

Jenkins agents reach the OpenShift API through a corporate egress proxy. The
kubernetes Python client ignores ``HTTPS_PROXY``/``NO_PROXY`` unless they are
copied into ``Configuration.proxy``/``Configuration.no_proxy`` (``oc`` and
boto3 do read them), so these tests pin that behavior.
"""

import urllib3
import pytest
from requests.utils import should_bypass_proxies

from autox_tests.lib import k8s_utils

pytestmark = pytest.mark.config


def _blank_configuration():
    from kubernetes import client

    return client.Configuration()


def _api_client(configuration):
    from kubernetes import client

    configuration.host = "https://api.atris-con-2.rhai-wx.com:6443"
    return client.ApiClient(configuration)


def test_apply_env_proxy_populates_configuration(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    monkeypatch.setenv("NO_PROXY", "*.rhai-wx.com, example.internal")
    configuration = _blank_configuration()

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert configuration.proxy == "http://squid.corp.redhat.com:3128"
    entries = configuration.no_proxy.split(",")
    assert ".rhai-wx.com" in entries
    assert "example.internal" in entries
    assert "localhost" in entries


def test_apply_env_proxy_adds_scheme_to_bare_host_port(monkeypatch) -> None:
    """Jenkins may export ``https_proxy=squid.corp.redhat.com:3128``."""
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("https_proxy", "squid.corp.redhat.com:3128")
    configuration = _blank_configuration()

    assert k8s_utils.apply_env_proxy(configuration) is True
    assert configuration.proxy == "http://squid.corp.redhat.com:3128"

    api_client = _api_client(configuration)
    assert isinstance(api_client.rest_client.pool_manager, urllib3.ProxyManager)


def test_apply_env_proxy_is_noop_without_proxy(monkeypatch) -> None:
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    configuration = _blank_configuration()

    assert k8s_utils.apply_env_proxy(configuration) is False
    assert configuration.proxy is None


def test_normalized_no_proxy_matches_wildcard_subdomains() -> None:
    normalized = k8s_utils.normalize_no_proxy("*.rhai-wx.com")
    assert should_bypass_proxies(
        "https://api.atris-con-2.rhai-wx.com:6443", no_proxy=normalized
    )


def test_configured_client_uses_proxy_manager(monkeypatch) -> None:
    from kubernetes import client

    monkeypatch.setenv("HTTPS_PROXY", "http://squid.corp.redhat.com:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    configuration = client.Configuration()
    configuration.host = "https://api.atris-con-2.rhai-wx.com:6443"

    assert k8s_utils.apply_env_proxy(configuration) is True
    api_client = client.ApiClient(configuration)

    assert isinstance(api_client.rest_client.pool_manager, urllib3.ProxyManager)
