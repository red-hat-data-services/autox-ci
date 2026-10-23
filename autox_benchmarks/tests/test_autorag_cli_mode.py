"""Check that a compiled package selects the matching submission mode."""

from __future__ import annotations

import os
from pathlib import Path

from autorag_benchmark import cli


def test_package_path_overrides_managed_config_mode(monkeypatch) -> None:
    observed: dict[str, str] = {}

    class FakeOrchestrator:
        def __init__(self, config_path: Path, env_file: Path | None = None) -> None:
            pass

        def execute(self, **kwargs: object) -> int:
            observed["mode"] = os.environ["BENCHMARK_USE_MANAGED_PIPELINES"]
            observed["package"] = str(kwargs["package_path_cli"])
            return 0

    monkeypatch.setattr(cli, "BenchmarkOrchestrator", FakeOrchestrator)
    monkeypatch.setenv("BENCHMARK_USE_MANAGED_PIPELINES", "true")
    config = Path(__file__).resolve().parents[1] / "config" / "benchmark_autorag.yaml"

    assert cli.main(["--config", str(config), "--package-path", "gpu.yaml"]) == 0
    assert observed == {"mode": "false", "package": "gpu.yaml"}
