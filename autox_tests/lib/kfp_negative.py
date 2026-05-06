"""Helpers for negative OpenShift AI / KFP pipeline integration tests."""

from __future__ import annotations

from typing import Any

import pytest

from autox_tests.lib.kfp_progress import run_failed_terminal, run_succeeded, wait_for_run_with_progress


def assert_pipeline_does_not_succeed(
    client: Any,
    package_path: str,
    arguments: dict[str, Any],
    run_name: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    pipeline_display_name: str | None = None,
    enable_caching: bool | None = None,
) -> None:
    """Assert run creation fails *or* the run finishes without ``SUCCEEDED``.

    If ``create_run_from_pipeline_package`` raises (e.g. API validation), the assertion passes.
    If a run is created, poll until a terminal state and require that it did not succeed.

    If the run ends in ``SUCCEEDED``, the test fails with an explicit message (e.g. unknown
    parameters ignored by the backend).
    """
    kw: dict[str, Any] = {
        "arguments": arguments,
        "run_name": run_name,
    }
    if enable_caching is not None:
        kw["enable_caching"] = enable_caching
    try:
        created = client.create_run_from_pipeline_package(package_path, **kw)
    except Exception:
        return

    run_id = created.run_id
    final = wait_for_run_with_progress(
        client,
        run_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        pipeline_display_name=pipeline_display_name,
    )
    if run_succeeded(final):
        pytest.fail(
            "Pipeline run succeeded unexpectedly (negative test). If the scenario was an unknown "
            f"parameter, the backend may ignore extra keys. run_id={run_id} "
            f"state={getattr(final, 'state', None)!r}"
        )
    if not run_failed_terminal(final):
        pytest.fail(
            "Negative test expected submit failure, FAILED, or ERROR terminal state; got "
            f"state={getattr(final, 'state', None)!r} run_id={run_id}"
        )
