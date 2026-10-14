"""Extract best HPO pattern settings and build documents-indexing-pipeline arguments."""

from __future__ import annotations

import logging
from typing import Any

from autorag_benchmark.pattern_scores import extract_pattern_scores
from autorag_benchmark.pipeline_names import INDEXING_PIPELINE_NAME, OPTIMIZATION_PIPELINE_NAME

logger = logging.getLogger(__name__)


def extract_best_pattern_settings(
    run_id: str,
    config: dict[str, Any],
    bucket: str | None = None,
    *,
    pattern_name_override: str | None = None,
    optimization_pipeline_name: str = OPTIMIZATION_PIPELINE_NAME,
) -> dict[str, Any]:
    """Return the settings dict of the best (or overridden) pattern from an HPO run.

    The returned dict has the shape::

        {
            "pattern_name": "PatternX",
            "final_score": 0.87,
            "embedding": {"model_id": "maas-embedding/bge-m3"},
            "chunking": {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 0},
            "retrieval": {...},
            "generation": {...},
        }

    Raises ``ValueError`` when no usable pattern is found.
    """
    result = extract_pattern_scores(
        run_id=run_id,
        config=config,
        bucket=bucket,
        pipeline_name=optimization_pipeline_name,
    )
    if "error" in result:
        raise ValueError(f"Failed to extract patterns from HPO run {run_id}: {result['error']}")

    patterns = result.get("patterns", [])
    if not patterns:
        raise ValueError(f"No patterns found in HPO run {run_id}")

    if pattern_name_override:
        matched = [p for p in patterns if p.get("pattern_name") == pattern_name_override]
        if not matched:
            available = [p.get("pattern_name") for p in patterns]
            raise ValueError(
                f"Pattern {pattern_name_override!r} not found in HPO run {run_id}. "
                f"Available: {available}"
            )
        best = matched[0]
    else:
        best = patterns[0]

    settings = best.get("settings", {})
    if not settings:
        raise ValueError(
            f"Pattern {best.get('pattern_name')!r} in HPO run {run_id} has no settings"
        )

    out: dict[str, Any] = {
        "pattern_name": best.get("pattern_name", ""),
        "final_score": best.get("final_score"),
    }
    for section in ("embedding", "chunking", "retrieval", "generation"):
        if section in settings:
            out[section] = dict(settings[section])

    logger.info(
        "Selected pattern %r (score=%.4f) from HPO run %s",
        out["pattern_name"],
        out.get("final_score") or 0,
        run_id,
    )
    return out


def build_indexing_arguments(
    pattern_settings: dict[str, Any],
    settings: Any,
    *,
    full_input_data_key: str | None = None,
) -> dict[str, Any]:
    """Map HPO pattern settings + benchmark config into indexing pipeline arguments.

    ``settings`` is a :class:`BenchmarkSettings` (or duck-typed equivalent) providing
    secrets, buckets, and fallback values.  ``full_input_data_key`` overrides the
    subsample ``input_data_key`` used during HPO.
    """
    embedding = pattern_settings.get("embedding", {})
    chunking = pattern_settings.get("chunking", {})

    embedding_model_id = embedding.get("model_id", "")
    if not embedding_model_id:
        raise ValueError("Pattern has no embedding.model_id -- cannot build indexing args")

    args: dict[str, Any] = {
        "embedding_model_id": embedding_model_id,
        "input_data_bucket_name": settings.input_data_bucket_name,
        "input_data_secret_name": settings.input_data_secret_name,
        "maas_secret_name": settings.maas_secret_name,
        "vector_db_secret_name": settings.vector_db_secret_name,
    }

    # The managed indexing pipeline takes a single input_data_key (string).
    if full_input_data_key:
        args["input_data_key"] = full_input_data_key

    if chunking.get("method"):
        args["chunking_method"] = chunking["method"]
    if chunking.get("chunk_size") is not None:
        args["chunk_size"] = int(chunking["chunk_size"])
    if chunking.get("chunk_overlap") is not None:
        args["chunk_overlap"] = int(chunking["chunk_overlap"])

    logger.info(
        "Built indexing args from pattern %r: embedding=%s, chunking=%s/%s/%s",
        pattern_settings.get("pattern_name"),
        embedding_model_id,
        chunking.get("method"),
        chunking.get("chunk_size"),
        chunking.get("chunk_overlap"),
    )
    return args


def resolve_indexing_pipeline_target(
    cfg: dict[str, Any],
    client: Any,
    *,
    kfp_pipeline_name: str = INDEXING_PIPELINE_NAME,
) -> Any:
    """Resolve a ``PipelineRunTarget`` for the documents-indexing-pipeline.

    Always uses managed mode (the indexing pipeline is DSPA-managed).
    """
    from benchmark_common.managed_pipelines import (
        PipelineRunTarget,
        get_managed_pipeline_wait_timeout,
        wait_for_managed_pipeline,
    )

    if client is None:
        return PipelineRunTarget(
            mode="managed",
            artifact_prefix=kfp_pipeline_name,
            kfp_pipeline_name=kfp_pipeline_name,
        )

    timeout = get_managed_pipeline_wait_timeout(cfg)
    pipeline_id, version_id = wait_for_managed_pipeline(
        client, kfp_pipeline_name, timeout_seconds=timeout,
    )
    return PipelineRunTarget(
        mode="managed",
        artifact_prefix=kfp_pipeline_name,
        pipeline_id=pipeline_id,
        pipeline_version_id=version_id,
        kfp_pipeline_name=kfp_pipeline_name,
    )
