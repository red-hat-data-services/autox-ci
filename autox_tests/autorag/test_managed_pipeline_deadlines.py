"""Managed pipeline deadlines cover generated KFP HTTP calls and retry backoff."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import kfp
import kfp_server_api
import pytest
import urllib3

from autox_tests.lib import managed_pipelines
from autox_tests.lib.kfp_retry import install_kfp_api_retries, kfp_api_deadline

pytestmark = pytest.mark.config


@pytest.fixture
def bounded_client(monkeypatch, request):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(delay):
        clock.sleeps.append(delay)
        clock.now += delay

    monkeypatch.setattr(managed_pipelines.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(managed_pipelines.time, "sleep", sleep)
    monkeypatch.setenv("RHOAI_MANAGED_PIPELINE_WAIT_TIMEOUT", "2")
    api_client = kfp_server_api.ApiClient()
    transport = Mock()
    monkeypatch.setattr(api_client.rest_client.pool_manager, "request", transport)
    client = object.__new__(kfp.Client)
    client._pipelines_api = kfp_server_api.PipelineServiceApi(api_client)
    install_kfp_api_retries(client, attempts=getattr(request, "param", 10), sleep=sleep)
    yield client, clock, transport
    api_client.close()


def response(value):
    return urllib3.HTTPResponse(status=200, body=json.dumps(value).encode())


def test_slow_version_request_uses_remaining_budget(bounded_client):
    client, clock, transport = bounded_client

    def request(method, url, **kwargs):
        assert kwargs["retries"] is False
        assert kwargs["timeout"].total == 2.0
        clock.now += kwargs["timeout"].total
        raise urllib3.exceptions.ReadTimeoutError(None, url, "timed out")

    transport.side_effect = request
    assert managed_pipelines.wait_for_managed_pipeline_version(client, "pipeline") is None
    assert clock.now == 2.0
    assert transport.call_count == 1
    assert clock.sleeps == []


def test_retries_recalculate_request_budget(bounded_client):
    client, clock, transport = bounded_client
    budgets = []

    def request(method, url, **kwargs):
        remaining = kwargs["timeout"].total
        budgets.append(remaining)
        clock.now += 0.75 if len(budgets) == 1 else remaining
        raise ConnectionError("transient")

    transport.side_effect = request
    assert managed_pipelines.wait_for_managed_pipeline_version(client, "pipeline") is None
    assert budgets == [2.0, 0.25]
    assert clock.sleeps == [1.0]
    assert clock.now == 2.0


def test_backoff_stops_at_deadline_without_another_request(bounded_client):
    client, clock, transport = bounded_client
    transport.side_effect = ConnectionError("transient")
    assert managed_pipelines.wait_for_managed_pipeline_version(client, "pipeline") is None
    assert clock.now == 2.0
    assert clock.sleeps == [1.0, 1.0]
    assert transport.call_count == 2


def test_version_appearing_after_poll_is_returned(bounded_client, monkeypatch):
    client, clock, transport = bounded_client
    monkeypatch.setenv("RHOAI_MANAGED_PIPELINE_WAIT_TIMEOUT", "12")
    transport.side_effect = [
        response({"pipeline_versions": []}),
        response({"pipeline_versions": [{"pipeline_version_id": "version"}]}),
    ]
    assert managed_pipelines.wait_for_managed_pipeline_version(client, "pipeline") == "version"
    assert clock.sleeps == [10.0]
    assert [call.kwargs["timeout"].total for call in transport.call_args_list] == [12.0, 2.0]


def test_pipeline_lookup_and_version_share_deadline(bounded_client):
    client, clock, transport = bounded_client
    budgets = []

    def request(method, url, **kwargs):
        budgets.append(kwargs["timeout"].total)
        if len(budgets) == 1:
            clock.now += 1.75
            return response({"pipelines": [{"pipeline_id": "pipeline"}]})
        clock.now += kwargs["timeout"].total
        raise ConnectionError("transient")

    transport.side_effect = request
    assert managed_pipelines.wait_for_managed_pipeline(
        client, "name", timeout_seconds=2
    ) == ("pipeline", None)
    assert budgets == [2.0, 0.25]
    assert clock.now == 2.0


def test_missing_pipeline_poll_sleep_is_bounded(bounded_client):
    client, clock, transport = bounded_client
    transport.side_effect = lambda *args, **kwargs: response({"pipelines": []})
    with pytest.raises(TimeoutError, match="not found"):
        managed_pipelines.wait_for_managed_pipeline(client, "name", timeout_seconds=2)
    assert clock.now == 2.0
    assert clock.sleeps == [2.0]


def test_expired_deadline_does_not_send_request(bounded_client):
    client, clock, transport = bounded_client
    assert managed_pipelines._resolve_latest_pipeline_version_id(
        client, "pipeline", deadline=clock.now
    ) is None
    transport.assert_not_called()


def test_deadline_is_restored_after_failure(bounded_client):
    client, clock, transport = bounded_client
    with pytest.raises(RuntimeError):
        with kfp_api_deadline(client, 0.5):
            raise RuntimeError("failure")
    clock.now = 1.0
    transport.return_value = response({"pipeline_versions": []})
    client.list_pipeline_versions("pipeline")
    assert transport.call_args.kwargs["timeout"] is None
    assert "retries" not in transport.call_args.kwargs


def test_nested_deadline_does_not_extend_budget(bounded_client):
    client, clock, transport = bounded_client
    transport.return_value = response({"pipeline_versions": []})
    with kfp_api_deadline(client, 0.5):
        with kfp_api_deadline(client, 5.0):
            client.list_pipeline_versions("pipeline")
    assert transport.call_args.kwargs["timeout"].total == 0.5


@pytest.mark.parametrize("bounded_client", [1], indirect=True)
def test_deadline_applies_when_retries_are_disabled(bounded_client):
    client, clock, transport = bounded_client
    transport.return_value = response({"pipeline_versions": []})
    with kfp_api_deadline(client, 0.5):
        client.list_pipeline_versions("pipeline")
    assert transport.call_args.kwargs["timeout"].total == 0.5
    assert transport.call_args.kwargs["retries"] is False


@pytest.mark.parametrize("method,attempts", [("GET", 10), ("POST", 1)])
def test_existing_retry_policy_outside_deadline_is_preserved(bounded_client, method, attempts):
    client, clock, transport = bounded_client
    transport.side_effect = ConnectionError("transient")
    with pytest.raises(ConnectionError):
        client._pipelines_api.api_client.call_api("/test", method)
    assert transport.call_count == attempts
    assert len(clock.sleeps) == attempts - 1
