"""Retries for transient, read-only Kubeflow Pipelines API requests."""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_RETRYABLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_RETRYABLE_STATUS_CODES = frozenset({0, 408, 425, 429, 500, 502, 503, 504})
_DEFAULT_ATTEMPTS = 5
_DEFAULT_BACKOFF_SECONDS = 1.0
_DEFAULT_MAX_BACKOFF_SECONDS = 30.0

try:
    import urllib3

    _NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
        urllib3.exceptions.HTTPError,
    )
except ImportError:  # pragma: no cover - urllib3 is a KFP dependency
    _NETWORK_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
        return default


def _exception_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    if status is None:
        response = getattr(exc, "http_resp", None)
        status = getattr(response, "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_exception(exc: BaseException) -> bool:
    status = _exception_status(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    return isinstance(exc, _NETWORK_EXCEPTIONS)


def _request_method(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    method = kwargs.get("method")
    if method is None and len(args) > 1:
        method = args[1]
    return str(method or "").upper()


def _resource_path(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    resource_path = kwargs.get("resource_path")
    if resource_path is None and args:
        resource_path = args[0]
    return str(resource_path or "<unknown>")


def install_kfp_api_retries(
    client: Any,
    *,
    attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Install bounded retries for transient read-only calls on a KFP client.

    KFP exposes generated API objects that all share one ``ApiClient``. Wrapping its
    ``call_api`` method keeps retry handling in one place and covers new KFP read
    methods automatically. Only GET/HEAD/OPTIONS requests are retried because a
    repeated write could create a duplicate pipeline run or other resource.

    The number of attempts can be tuned with ``KFP_API_RETRY_ATTEMPTS``. Backoff
    starts at 1 s, doubles after each failure, and is capped internally at 30 s.
    """
    total_attempts = attempts if attempts is not None else _env_int(
        "KFP_API_RETRY_ATTEMPTS", _DEFAULT_ATTEMPTS
    )
    total_attempts = max(1, int(total_attempts))
    api_client = _find_api_client(client)
    if api_client is None:
        logger.warning("Could not find the generated ApiClient; KFP retries disabled")
        return
    if getattr(api_client, "_autox_kfp_retry_installed", False):
        return

    original_call_api = api_client.call_api

    @functools.wraps(original_call_api)
    def call_api_with_retry(*args: Any, **kwargs: Any) -> Any:
        method = _request_method(args, kwargs)
        if method not in _RETRYABLE_METHODS or total_attempts <= 1:
            return original_call_api(*args, **kwargs)

        path = _resource_path(args, kwargs)
        for attempt in range(1, total_attempts + 1):
            try:
                return original_call_api(*args, **kwargs)
            except Exception as exc:
                if not _is_retryable_exception(exc) or attempt >= total_attempts:
                    raise
                delay = min(
                    _DEFAULT_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    _DEFAULT_MAX_BACKOFF_SECONDS,
                )
                status = _exception_status(exc)
                logger.warning(
                    "Transient KFP API %s %s failed (attempt %d/%d, status=%s): %s; "
                    "retrying in %.1fs",
                    method,
                    path,
                    attempt,
                    total_attempts,
                    status if status is not None else "network",
                    exc,
                    delay,
                )
                sleep(delay)

        raise AssertionError("unreachable")

    api_client.call_api = call_api_with_retry
    api_client._autox_kfp_retry_installed = True


def _find_api_client(client: Any) -> Any | None:
    """Return the generated ApiClient shared by a ``kfp.Client`` instance."""
    for attr_name in (
        "_run_api",
        "_experiment_api",
        "_pipelines_api",
        "_recurring_run_api",
        "_upload_api",
        "_healthz_api",
    ):
        api = getattr(client, attr_name, None)
        api_client = getattr(api, "api_client", None)
        if api_client is not None and hasattr(api_client, "call_api"):
            return api_client
    return None
