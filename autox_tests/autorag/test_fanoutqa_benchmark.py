"""FanOutQA AutoRAG benchmark run on real KFP and external S3.

Runs one full RAG optimization over ``s3://ai-eng-rag/datasets/rag/fanoutqa/txt/10_gold/``
through the benchmark orchestrator from ``autox_benchmarks/``. With four patterns and the
``balanced`` preset this takes roughly an hour, so the module is tagged ``integration``
and can be skipped with ``"not integration"``.

Credentials come from the environment loaded by ``autox_tests/autorag/conftest.py``
(``autox_tests/.env.rag`` locally, or the file passed to ``run_tests.sh --env-file``).

Run:
  ./run_tests.sh --suite autorag --env-file autox_tests/.env.rag "integration"
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from autox_tests.lib.settings import describe_autorag_connection_config_failure

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "autox_benchmarks"
if str(BENCHMARKS_ROOT) not in sys.path:
    # Append (not prepend) so ``autox_benchmarks`` cannot shadow top-level modules
    # such as ``config`` or ``templates`` for the rest of the test suite.
    sys.path.append(str(BENCHMARKS_ROOT))

from autorag_benchmark.config_loader import load_merged_benchmark_config  # noqa: E402
from autorag_benchmark.orchestrator import BenchmarkOrchestrator  # noqa: E402
from benchmark_common.kfp_client import create_kfp_client, resolve_kfp_token  # noqa: E402
from benchmark_common.manifest import load_dataset_entries  # noqa: E402
from benchmark_common.run_state import is_success_state  # noqa: E402
from benchmark_common.s3_client import make_s3_client, s3_cfg_usable  # noqa: E402

DATASET_ID = "fanoutqa-txt-10-gold"
INPUT_DATA_KEY = "datasets/rag/fanoutqa/txt/10_gold/knowledge_base"
TEST_DATA_KEY = "datasets/rag/fanoutqa/txt/10_gold/benchmark_data.json"
MANIFEST_FILE = "dataset_manifest.yaml"

BENCHMARK_CONFIG = {
    "pipeline": {"mode": "managed"},
    "run": {
        "optimization_metric": "answer_correctness",
        "optimization_max_rag_patterns": 4,
        "preset": "balanced",
        "poll_interval_seconds": 30,
        "timeout_seconds": 7200,
        "enable_caching": False,
        "run_name_prefix": "pytest-autorag-fanoutqa",
    },
    "storage": {"upload_benchmark_results": False},
    "dataset_manifest_path": MANIFEST_FILE,
}

DATASET_MANIFEST = {
    "datasets": [
        {
            "id": DATASET_ID,
            "name": "FanOutQA TXT (10 gold questions)",
            "input_data_key": INPUT_DATA_KEY,
            "test_data_key": TEST_DATA_KEY,
            "optimization_metric": "answer_correctness",
        }
    ]
}

_ENV_PROBLEM = describe_autorag_connection_config_failure()

pytestmark = [
    pytest.mark.autorag,
    pytest.mark.integration,
    pytest.mark.skipif(_ENV_PROBLEM is not None, reason=_ENV_PROBLEM or ""),
]


@dataclass(frozen=True)
class BenchmarkContext:
    """Merged benchmark config plus KFP and S3 clients validated against the cluster."""

    merged_config: dict[str, Any]
    config_dir: Path
    kfp_client: Any
    s3_client: Any


MODEL_ENV_ALIASES = {
    "BENCHMARK_EMBEDDING_MODELS": (
        "AUTORAG_EMBEDDING_MODELS",
        "FUNC_TEST_EMBEDDING_MODELS",
    ),
    "BENCHMARK_GENERATION_MODELS": (
        "AUTORAG_GENERATION_MODELS",
        "FUNC_TEST_GENERATION_MODELS",
    ),
}


def _seed_model_env() -> None:
    """Feed the benchmark loader from the suite's model env vars when needed."""
    for target, aliases in MODEL_ENV_ALIASES.items():
        if os.environ.get(target):
            continue
        for alias in aliases:
            if os.environ.get(alias):
                os.environ[target] = os.environ[alias]
                break


_seed_model_env()


def _write_benchmark_config(config_dir: Path) -> Path:
    """Write the FanOutQA benchmark config and dataset manifest into ``config_dir``."""
    (config_dir / MANIFEST_FILE).write_text(
        yaml.safe_dump(DATASET_MANIFEST, sort_keys=False), encoding="utf-8"
    )
    config_path = config_dir / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(BENCHMARK_CONFIG, sort_keys=False), encoding="utf-8"
    )
    return config_path


def _dataset_entry(cfg: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Return the single dataset entry declared by the FanOutQA manifest."""
    datasets = load_dataset_entries(cfg, config_dir)
    if len(datasets) != 1:
        raise ValueError(
            f"FanOutQA benchmark manifest must contain exactly one dataset, got {len(datasets)}"
        )
    return datasets[0]


def _validate_dataset_objects(
    dataset: dict[str, Any],
    cfg: dict[str, Any],
    s3_client: Any,
) -> list[str]:
    """Verify the dataset exists in S3 and its benchmark references real documents."""
    errors: list[str] = []
    storage = cfg.get("storage") or {}
    input_bucket = str(storage.get("input_data_bucket_name", "")).strip()
    test_bucket = str(storage.get("test_data_bucket_name", "")).strip()
    input_key = str(dataset.get("input_data_key", "")).strip().rstrip("/")
    test_key = str(dataset.get("test_data_key", "")).strip().lstrip("/")
    if not input_key:
        return ["Dataset manifest entry missing input_data_key"]
    if not test_key:
        return ["Dataset manifest entry missing test_data_key"]

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        document_keys = {
            obj["Key"]
            for page in paginator.paginate(Bucket=input_bucket, Prefix=f"{input_key}/")
            for obj in page.get("Contents") or []
        }
    except Exception as exc:
        return [f"Cannot list s3://{input_bucket}/{input_key}/: {exc}"]
    if not document_keys:
        errors.append(f"No input documents under s3://{input_bucket}/{input_key}/")

    try:
        payload = s3_client.get_object(Bucket=test_bucket, Key=test_key)["Body"].read()
        benchmark = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        return [*errors, f"Cannot read s3://{test_bucket}/{test_key}: {exc}"]
    if not isinstance(benchmark, list) or not benchmark:
        return [*errors, f"Benchmark data at s3://{test_bucket}/{test_key} is empty or not a list"]

    non_object_records = [
        index for index, record in enumerate(benchmark) if not isinstance(record, dict)
    ]
    if non_object_records:
        return [
            *errors,
            f"Benchmark data at s3://{test_bucket}/{test_key} contains "
            f"{len(non_object_records)} non-object record(s) (e.g. index {non_object_records[0]})",
        ]

    referenced = {
        key
        for record in benchmark
        for key in (record.get("correct_answer_document_keys") or [])
    }
    missing = sorted(referenced - document_keys)
    if missing:
        errors.append(
            f"Benchmark references {len(missing)} document(s) missing under "
            f"s3://{input_bucket}/{input_key}/ (e.g. {missing[:3]})"
        )
    return errors


def _collect_prerequisite_errors(
    cfg: dict[str, Any],
    config_dir: Path,
    s3_client: Any,
) -> list[str]:
    """Return FanOutQA benchmark prerequisite problems for the configured dataset."""
    try:
        dataset = _dataset_entry(cfg, config_dir)
    except Exception as exc:
        return [f"Invalid FanOutQA dataset manifest: {exc}"]

    errors = _validate_dataset_objects(dataset, cfg, s3_client)
    kfp = cfg.get("kfp") or {}
    storage = cfg.get("storage") or {}
    pipeline = cfg.get("pipeline") or {}
    if not str(kfp.get("host", "")).startswith("http"):
        errors.append("KFP host is required (RHOAI_KFP_URL or BENCHMARK_KFP_HOST)")
    if not str(kfp.get("namespace", "")).strip():
        errors.append("KFP namespace is required (RHOAI_PROJECT_NAME or BENCHMARK_KFP_NAMESPACE)")
    if not resolve_kfp_token(kfp):
        errors.append("KFP token is required (RHOAI_TOKEN or BENCHMARK_KFP_TOKEN)")
    for name in ("input_data_bucket_name", "test_data_bucket_name"):
        if not str(storage.get(name, "")).strip():
            errors.append(f"storage.{name} is required")
    secret_names = (
        "input_data_secret_name",
        "test_data_secret_name",
        "maas_secret_name",
        "vector_db_secret_name",
    )
    for name in secret_names:
        if not str(pipeline.get(name, "")).strip():
            errors.append(f"pipeline.{name} is required")
    return errors


def _build_context(config_path: Path) -> BenchmarkContext:
    """Build the benchmark context, failing fast on missing prerequisites."""
    try:
        cfg, config_dir = load_merged_benchmark_config(config_path)
    except Exception as exc:
        pytest.fail(f"Failed to load FanOutQA benchmark config: {exc}", pytrace=False)

    s3_cfg = cfg.get("s3")
    if not s3_cfg_usable(s3_cfg if isinstance(s3_cfg, dict) else None):
        pytest.fail(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required in the test environment",
            pytrace=False,
        )
    s3_client = make_s3_client(s3_cfg)

    errors = _collect_prerequisite_errors(cfg, config_dir, s3_client)
    if errors:
        pytest.fail(
            "FanOutQA benchmark prerequisites not met:\n  - " + "\n  - ".join(errors),
            pytrace=False,
        )

    try:
        kfp_client = create_kfp_client(cfg)
        kfp_client.list_experiments(page_size=1)
    except Exception as exc:
        pytest.fail(f"KFP connectivity check failed: {exc}", pytrace=False)

    return BenchmarkContext(
        merged_config=cfg,
        config_dir=config_dir,
        kfp_client=kfp_client,
        s3_client=s3_client,
    )


def _read_results_csv(path: Path) -> list[dict[str, str]]:
    """Read the orchestrator results CSV as a list of row dicts."""
    assert path.is_file(), f"Expected CSV at {path}"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dry_run_arguments(row: dict[str, str]) -> dict[str, Any]:
    """Decode the pipeline arguments captured in a dry-run row."""
    blob = row.get("metrics_blob")
    assert blob and str(blob).strip(), "metrics_blob should hold dry-run arguments"
    return json.loads(str(blob))


@pytest.fixture(scope="session")
def benchmark_config_path(tmp_path_factory) -> Path:
    """Benchmark config YAML and manifest written to a temporary directory."""
    return _write_benchmark_config(tmp_path_factory.mktemp("fanoutqa_benchmark"))


@pytest.fixture(scope="session")
def benchmark_context(benchmark_config_path: Path) -> BenchmarkContext:
    """Prerequisites and clients for the FanOutQA benchmark run."""
    return _build_context(benchmark_config_path)


@pytest.fixture(scope="session")
def benchmark_dataset(benchmark_context: BenchmarkContext) -> dict[str, Any]:
    """Single dataset entry declared by the FanOutQA manifest."""
    return _dataset_entry(benchmark_context.merged_config, benchmark_context.config_dir)


@pytest.fixture
def benchmark_output_csv(tmp_path_factory) -> Path:
    """Temporary output CSV path for the FanOutQA benchmark run."""
    return tmp_path_factory.mktemp("fanoutqa_output") / "fanoutqa_benchmark_runs.csv"


class TestFanoutqaBenchmarkPrerequisites:
    """Explicit first checks; session setup validates the same prerequisites."""

    def test_credentials_kfp_and_s3_ready(self, benchmark_context: BenchmarkContext) -> None:
        """KFP, S3 and pipeline secrets resolve from the test environment."""
        cfg = benchmark_context.merged_config
        kfp = cfg.get("kfp") or {}
        storage = cfg.get("storage") or {}
        pipeline = cfg.get("pipeline") or {}
        assert str(kfp.get("host", "")).startswith("http")
        assert str(kfp.get("namespace", "")).strip()
        assert benchmark_context.kfp_client is not None
        assert storage.get("input_data_bucket_name")
        assert storage.get("test_data_bucket_name")
        assert pipeline.get("maas_secret_name")
        assert pipeline.get("vector_db_secret_name")

    def test_fanoutqa_dataset_present_in_s3(self, benchmark_dataset: dict[str, Any]) -> None:
        """The manifest points at the expected FanOutQA S3 locations."""
        assert benchmark_dataset["id"] == DATASET_ID
        assert benchmark_dataset["input_data_key"] == INPUT_DATA_KEY
        assert benchmark_dataset["test_data_key"] == TEST_DATA_KEY


class TestFanoutqaBenchmarkRun:
    """End-to-end: one FanOutQA optimization run through the benchmark orchestrator."""

    @pytest.fixture
    def orchestrator(self, benchmark_config_path: Path) -> BenchmarkOrchestrator:
        """Orchestrator bound to the FanOutQA benchmark config."""
        return BenchmarkOrchestrator(benchmark_config_path)

    def test_orchestrator_fanoutqa_run_succeeds(
        self,
        orchestrator: BenchmarkOrchestrator,
        benchmark_output_csv: Path,
    ) -> None:
        """A real FanOutQA run finishes successfully and yields pattern scores."""
        code = orchestrator.execute(
            output_csv=benchmark_output_csv,
            dry_run=False,
            fail_fast=True,
            dataset_filter="all",
        )
        assert code == 0, "orchestrator returned non-zero exit code"

        rows = _read_results_csv(benchmark_output_csv)
        assert rows, "expected at least one result row"
        assert rows[0]["state"].upper() != "SUBMIT_ERROR", (
            f"run submission failed: {rows[0].get('error', '')}"
        )
        assert {row["dataset_id"] for row in rows} == {DATASET_ID}
        run_ids = {row["run_id"].strip() for row in rows}
        assert len(run_ids) == 1 and next(iter(run_ids)), "expected one KFP run_id"

        states = {row["state"].upper() for row in rows}
        assert len(states) == 1, f"expected one terminal state, got {states}"
        state = next(iter(states))
        assert is_success_state(state), (
            f"FanOutQA run failed: state={state}, error={rows[0].get('error', '')}"
        )

        assert "final_score" in rows[0], "pattern scores were not extracted from S3"
        scores = [float(row["final_score"]) for row in rows if row["final_score"].strip()]
        assert scores, "no pattern produced a final_score"
        assert all(score >= 0 for score in scores), f"negative pattern score: {scores}"

    def test_cli_dry_run_targets_fanoutqa_dataset(
        self,
        benchmark_config_path: Path,
        benchmark_output_csv: Path,
    ) -> None:
        """The CLI dry run targets the FanOutQA dataset and expected arguments."""
        out = benchmark_output_csv.parent / "cli_dry_run.csv"
        cmd = [
            sys.executable,
            str(BENCHMARKS_ROOT / "scripts" / "autorag_benchmark_orchestrator.py"),
            "--config",
            str(benchmark_config_path),
            "--output",
            str(out),
            "--dry-run",
        ]
        result = subprocess.run(
            cmd,
            cwd=BENCHMARKS_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
            timeout=300,
        )
        assert result.returncode == 0, (result.stderr or "") + (result.stdout or "")

        rows = _read_results_csv(out)
        assert len(rows) == 1
        assert rows[0]["dataset_id"] == DATASET_ID
        assert rows[0]["state"].upper() == "DRY_RUN"
        arguments = _dry_run_arguments(rows[0])
        assert arguments["db_secret_name"], "pipeline db_secret_name is missing"
        assert "vector_db_secret_name" not in arguments, (
            "benchmark must not send the unknown vector_db_secret_name parameter"
        )
        assert arguments["input_data_keys"] == [INPUT_DATA_KEY]
        assert arguments["test_data_key"] == TEST_DATA_KEY
        assert arguments["optimization_metric"] == "answer_correctness"
