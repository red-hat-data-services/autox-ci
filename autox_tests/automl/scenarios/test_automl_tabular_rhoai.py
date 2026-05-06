"""Run the tabular AutoML training pipeline on OpenShift AI (KFP v2) with progress polling."""

from __future__ import annotations

from typing import Any

import pytest

from autox_tests.lib.config_loaders import AutomlTabularTestConfig, get_automl_tabular_configs_for_run
from autox_tests.lib.data_resolution import resolve_automl_dataset_s3_location
from autox_tests.lib.integration_failures import require_rhoai_automl_env
from autox_tests.lib.kfp_progress import run_succeeded, wait_for_run_with_progress
from autox_tests.lib.settings import get_rhoai_automl_config, get_test_data_source_defaults

CONFIGS_FOR_RUN = get_automl_tabular_configs_for_run()
# Must match @dsl.pipeline name= in autogluon_tabular_training_pipeline (root DAG task filter).
PIPELINE_DISPLAY_NAME = "autogluon-tabular-training-pipeline"


@pytest.mark.integration
@pytest.mark.openshift_ai
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestAutomlTabularRhoaiKfp:
    """Submit tabular AutoML runs using a precompiled ``pipeline.yaml`` (local path or downloaded)."""

    def test_pipeline_run_succeeds(
        self,
        test_config: AutomlTabularTestConfig,
        rhoai_automl_project: str | None,
        uploaded_automl_tabular_datasets: dict[str, dict[str, str]],
        kfp_client_automl: Any,
        automl_tabular_pipeline_package: str,
        automl_run_name: str,
        pipeline_run_timeout: int,
        pipeline_poll_interval_seconds: int,
    ) -> None:
        """Submit the pipeline and assert the run ends in ``SUCCEEDED``."""
        require_rhoai_automl_env()
        if not rhoai_automl_project or not kfp_client_automl:
            pytest.fail(
                "AutoML session setup did not complete: missing project namespace and/or KFP client.\n"
                "Check OpenShift project + S3 secret creation (RHOAI_URL, token), and pipeline API "
                "access (RHOAI_KFP_URL or RHOAI_CREATE_DSPA=true with a resolvable route)."
            )
        cfg = get_rhoai_automl_config()
        assert cfg is not None
        try:
            bucket, key = resolve_automl_dataset_s3_location(
                test_config,
                uploaded_automl_tabular_datasets,
                cfg,
                get_test_data_source_defaults(),
            )
        except (KeyError, ValueError) as e:
            pytest.fail(f"Could not resolve train data location for this scenario: {e}")
        secret_name = cfg["s3_secret_name"]
        arguments = test_config.get_pipeline_arguments(bucket, key, secret_name)

        run = kfp_client_automl.create_run_from_pipeline_package(
            automl_tabular_pipeline_package,
            arguments=arguments,
            run_name=automl_run_name,
        )
        run_id = run.run_id
        final = wait_for_run_with_progress(
            kfp_client_automl,
            run_id,
            timeout_seconds=pipeline_run_timeout,
            poll_interval_seconds=pipeline_poll_interval_seconds,
            pipeline_display_name=PIPELINE_DISPLAY_NAME,
        )
        assert run_succeeded(final), f"Run {run_id} did not succeed; state={getattr(final, 'state', None)}"
