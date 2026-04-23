"""Fail fast with explicit messages when OpenShift AI integration preconditions are not met."""

from __future__ import annotations

import pytest

from tests.lib.settings import (
    describe_autorag_integration_failure,
    describe_rhoai_automl_config_failure,
)


def require_rhoai_automl_env() -> None:
    """Raise a failed test if AutoML integration environment is incomplete."""
    msg = describe_rhoai_automl_config_failure()
    if msg is not None:
        pytest.fail(f"OpenShift AI AutoML integration is not configured.\n\n{msg}")


def require_autorag_env() -> None:
    """Raise a failed test if AutoRAG integration preconditions are not met."""
    msg = describe_autorag_integration_failure()
    if msg is not None:
        pytest.fail(f"OpenShift AI AutoRAG integration is not configured.\n\n{msg}")
