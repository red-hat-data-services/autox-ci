"""Run the AutoRAG optimization pipeline on OpenShift AI (KFP v2) with progress polling."""

from __future__ import annotations

from typing import Any

import pytest

from autox_tests.lib.config_loaders import AutoragOptimizationTestConfig, get_autorag_configs_for_run
from autox_tests.lib.data_resolution import resolve_autorag_s3_locations
from autox_tests.lib.integration_failures import require_autorag_env
from autox_tests.lib.kfp_progress import run_succeeded, wait_for_run_with_progress
from autox_tests.lib.settings import (
    autorag_pipeline_arguments,
    get_autorag_connection_config,
    get_test_data_source_defaults,
)

CONFIGS_FOR_RUN = get_autorag_configs_for_run()
PIPELINE_DISPLAY_NAME = "documents-rag-optimization-pipeline"


@pytest.mark.integration
@pytest.mark.openshift_ai
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestAutoragRhoaiKfp:
    """Submit AutoRAG optimization runs using a precompiled ``pipeline.yaml`` (local path or downloaded)."""

    def test_pipeline_run_succeeds(
        self,
        test_config: AutoragOptimizationTestConfig,
        rhoai_project_and_s3_secret: str | None,
        kfp_client_autorag: Any,
        autorag_pipeline_package: str,
        uploaded_autorag_by_config_id: dict[str, dict[str, str]],
        autorag_run_name: str,
        pipeline_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        """Submit the pipeline and assert the run ends in ``SUCCEEDED``."""
        require_autorag_env()
        if not kfp_client_autorag:
            pytest.fail(
                "AutoRAG session setup did not produce a KFP client.\n"
                "Check RHOAI_KFP_URL or RHOAI_CREATE_DSPA=true, and that the pipeline API route "
                "can be discovered."
            )
        conn = get_autorag_connection_config()
        assert conn is not None
        try:
            locations = resolve_autorag_s3_locations(
                test_config,
                uploaded_autorag_by_config_id,
                conn,
                get_test_data_source_defaults(),
            )
        except (KeyError, ValueError) as e:
            pytest.fail(f"Could not resolve document/benchmark S3 locations for this scenario: {e}")
        merged = {**conn, **locations}
        arguments = autorag_pipeline_arguments(merged)
        arguments.update(test_config.argument_overrides)

        run = kfp_client_autorag.create_run_from_pipeline_package(
            autorag_pipeline_package,
            arguments=arguments,
            run_name=autorag_run_name,
            enable_caching=False,
        )
        run_id = run.run_id
        final = wait_for_run_with_progress(
            kfp_client_autorag,
            run_id,
            timeout_seconds=pipeline_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=PIPELINE_DISPLAY_NAME,
        )
        assert run_succeeded(final), f"Run {run_id} did not succeed; state={getattr(final, 'state', None)}"
