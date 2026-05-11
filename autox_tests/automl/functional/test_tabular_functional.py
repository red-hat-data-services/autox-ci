"""Functional tests for AutoGluon tabular training pipeline on RHOAI.

Test scenarios are defined in configs/tabular_test_configs.json. Each scenario
uploads a dataset CSV, submits a pipeline run, and validates the result.
Filter by tags with AUTOML_FUNCTIONAL_TESTS_TAGS (e.g. smoke, regression).

Passing criteria:
- Pipeline run finishes with SUCCEEDED status
- Model artifacts exist in S3 (.pkl files)
- Leaderboard artifacts exist in S3
"""

import logging

import pytest

from .configs.configs import AutoMLTabularFunctionalConfig, get_tabular_configs_for_run
from .conftest import AUTOML_FUNCTIONAL_CONFIG
from .utils import (
    collect_failure_details,
    get_run_state,
    run_pipeline_and_wait,
    run_succeeded,
    validate_artifacts_in_s3,
)

logger = logging.getLogger(__name__)

TABULAR_CONFIGS = get_tabular_configs_for_run()
PIPELINE_DISPLAY_NAME = "autogluon-tabular-training-pipeline"


@pytest.mark.tabular
@pytest.mark.skipif(
    AUTOML_FUNCTIONAL_CONFIG is None,
    reason="AutoML functional test env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, AUTOML_TRAIN_DATA_*; see .env)",
)
class TestAutoMLTabularFunctional:
    """Functional tests for AutoGluon tabular training pipeline."""

    @pytest.mark.parametrize(
        "test_config", TABULAR_CONFIGS, ids=[c.id for c in TABULAR_CONFIGS]
    )
    def test_tabular_pipeline_run_succeeds(
        self,
        test_config: AutoMLTabularFunctionalConfig,
        automl_functional_config,
        kfp_client_automl_functional,
        compiled_tabular_pipeline_path,
        uploaded_tabular_datasets,
        pipeline_run_timeout,
        s3_client_automl_functional,
    ):
        if not kfp_client_automl_functional:
            pytest.fail("AutoML functional test prerequisites not available")

        arguments = test_config.get_pipeline_arguments(automl_functional_config)

        run_id, detail = run_pipeline_and_wait(
            kfp_client_automl_functional,
            compiled_tabular_pipeline_path,
            arguments,
            pipeline_run_timeout,
        )

        if not run_succeeded(detail):
            failure_info = collect_failure_details(
                kfp_client_automl_functional, run_id, config=automl_functional_config
            )
            pytest.fail(
                f"[{test_config.id}] Pipeline run {run_id} expected SUCCEEDED but got "
                f"{get_run_state(detail)}{failure_info}"
            )

        if not s3_client_automl_functional or not automl_functional_config.get("s3_bucket_artifacts"):
            return

        bucket = automl_functional_config["s3_bucket_artifacts"]
        prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
        artifacts = validate_artifacts_in_s3(s3_client_automl_functional, bucket, prefix)

        assert len(artifacts["model_keys"]) >= 1, (
            f"[{test_config.id}] Expected at least 1 model artifact under {prefix}; "
            f"found {len(artifacts['model_keys'])}"
        )
