"""Discover and download AutoML leaderboard artifacts from KFP run prefixes on S3.

Current pipelines emit the HTML leaderboard from the training component
(``autogluon-models-training`` / timeseries equivalent). Metrics that feed that
HTML are also persisted per model as ``models_artifact/<Model>/metrics/metrics.json``.

Legacy runs may still use a separate ``leaderboard-evaluation`` step.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from benchmark_common.s3_client import make_s3_client, s3_cfg_usable

logger = logging.getLogger(__name__)

# Preferred first: training component (current). Legacy evaluation step last.
TABULAR_ARTIFACT_FOLDERS: tuple[str, ...] = (
    "autogluon-models-training",
    "autogluon-models-training-2",
    "leaderboard-evaluation",
)
TIMESERIES_ARTIFACT_FOLDERS: tuple[str, ...] = (
    "autogluon-timeseries-models-training",
    "autogluon-timeseries-models-training-2",
    "timeseries-leaderboard-evaluation",
)

# Back-compat aliases used by older call sites / docs.
TABULAR_LEADERBOARD_FOLDER = TABULAR_ARTIFACT_FOLDERS[0]
TIMESERIES_LEADERBOARD_FOLDER = TIMESERIES_ARTIFACT_FOLDERS[0]


def _folders_for(is_timeseries: bool) -> tuple[str, ...]:
    return TIMESERIES_ARTIFACT_FOLDERS if is_timeseries else TABULAR_ARTIFACT_FOLDERS


def _list_prefix_for_run(artifact_root: str, run_id: str, folder: str) -> str:
    rid = run_id.strip()
    root = (artifact_root or "").strip().strip("/")
    if not root:
        return f"{rid}/{folder}/"
    return f"{root}/{rid}/{folder}/"


def _key_is_html_artifact(key: str, run_id: str, folder: str, artifact_root: str) -> bool:
    """
    Match keys shaped like::

        <run_id>/<folder>/<exec_id>/html_artifact
        <artifact_root>/<run_id>/<folder>/<exec_id>/html_artifact
        ... and nested keys under html_artifact/
    """
    rid = run_id.strip()
    root = (artifact_root or "").strip().strip("/")
    if not rid or not key:
        return False
    parts = key.split("/")
    if not root:
        if len(parts) < 4:
            return False
        if parts[0] != rid or parts[1] != folder:
            return False
        return parts[3] == "html_artifact" or parts[3].startswith("html_artifact")
    if len(parts) < 5:
        return False
    if parts[0] != root or parts[1] != rid or parts[2] != folder:
        return False
    return parts[4] == "html_artifact" or parts[4].startswith("html_artifact")


# Old name kept for imports/tests.
_key_is_leaderboard_html_artifact = _key_is_html_artifact


def _list_html_keys_for_folder(
    client: Any,
    bucket: str,
    run_id: str,
    folder: str,
    artifact_root: str,
) -> list[str]:
    prefix = _list_prefix_for_run(artifact_root, run_id, folder)
    found: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj.get("Key") or ""
            if _key_is_html_artifact(k, run_id, folder, artifact_root):
                found.append(k)
    return sorted(set(found))


def _list_metrics_json_keys_for_folder(
    client: Any,
    bucket: str,
    run_id: str,
    folder: str,
    artifact_root: str,
) -> list[tuple[str, str]]:
    """Return ``(s3_key, model_name)`` for each ``metrics.json`` under models_artifact."""
    prefix = _list_prefix_for_run(artifact_root, run_id, folder)
    found: list[tuple[str, str]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj.get("Key") or ""
            if "/models_artifact/" not in k or not k.endswith("/metrics/metrics.json"):
                continue
            model = k.split("/models_artifact/", 1)[1].split("/", 1)[0]
            if model:
                found.append((k, model))
    return sorted(set(found), key=lambda t: (t[1], t[0]))


def _to_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def discover_leaderboard_html_s3_uri(
    *,
    bucket: str,
    s3_cfg: dict[str, Any] | None,
    run_id: str,
    is_timeseries: bool,
    artifact_root_prefix: str = "",
    attempts: int = 6,
    delay_seconds: float = 3.0,
) -> str:
    """
    Return ``s3://bucket/<key>`` for the first matching HTML leaderboard artifact.

    Search order (tabular)::

        autogluon-models-training → autogluon-models-training-2 → leaderboard-evaluation
    """
    if not run_id.strip() or not bucket.strip():
        return ""
    if not s3_cfg or not s3_cfg_usable(s3_cfg):
        return ""

    folders = _folders_for(is_timeseries)
    root = (artifact_root_prefix or "").strip().strip("/")
    for attempt in range(attempts):
        try:
            client = make_s3_client(s3_cfg)
            for folder in folders:
                keys = _list_html_keys_for_folder(client, bucket.strip(), run_id.strip(), folder, root)
                if keys:
                    uri = _to_s3_uri(bucket.strip(), keys[0])
                    if attempt > 0:
                        logger.info(
                            "Leaderboard HTML found on S3 retry %d for run_id=%s folder=%s",
                            attempt + 1,
                            run_id,
                            folder,
                        )
                    return uri
        except Exception as e:
            logger.warning(
                "S3 leaderboard HTML lookup attempt %d/%d failed for run_id=%s: %s",
                attempt + 1,
                attempts,
                run_id,
                e,
            )
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)

    logger.info(
        "No leaderboard HTML S3 key found for run_id=%s under folders=%s root=%r",
        run_id,
        folders,
        root,
    )
    return ""


def parse_s3_uri(uri: str) -> tuple[str, str] | None:
    u = (uri or "").strip()
    if not u.startswith("s3://"):
        return None
    rest = u[5:]
    if "/" not in rest:
        return None
    bucket, _, key = rest.partition("/")
    key = key.lstrip("/")
    if not bucket or not key:
        return None
    return bucket, key


def _is_missing_key_error(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    err = exc.response.get("Error") or {}
    code = err.get("Code", "")
    if code in ("404", "NoSuchKey", "NotFound"):
        return True
    return bool(err.get("Message", "").lower().startswith("not found"))


def _list_html_keys_under_prefix(client: Any, bucket: str, prefix: str) -> list[str]:
    found: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj.get("Key") or ""
            if k.endswith((".html", ".htm")):
                found.append(k)
    return sorted(set(found))


def download_leaderboard_html_to_dir(
    s3_cfg: dict[str, Any],
    s3_uri: str,
    output_csv_parent: Path,
    *,
    run_id: str,
) -> str:
    """
    Download leaderboard HTML next to the results CSV under ``leaderboards/<run_id>.html``.

    Returns a path relative to ``output_csv_parent`` (POSIX), or ``""`` on failure.
    """
    parsed = parse_s3_uri(s3_uri)
    if not parsed or not s3_cfg_usable(s3_cfg):
        return ""
    bucket, key = parsed
    rid = run_id.strip()
    if not rid:
        return ""

    dest_dir = output_csv_parent / "leaderboards"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{rid}.html"

    try:
        from botocore.exceptions import ClientError

        client = make_s3_client(s3_cfg)
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
            dest_file.write_bytes(resp["Body"].read())
        except ClientError as e:
            if not _is_missing_key_error(e):
                raise
            prefix = key if key.endswith("/") else f"{key}/"
            candidates = _list_html_keys_under_prefix(client, bucket, prefix)
            if not candidates:
                logger.warning(
                    "Leaderboard download: no object at key %r and no .html under prefix %r",
                    key,
                    prefix,
                )
                return ""
            resp = client.get_object(Bucket=bucket, Key=candidates[0])
            dest_file.write_bytes(resp["Body"].read())
            if candidates[0] != key:
                logger.info(
                    "Leaderboard download: used nested key %r instead of URI key %r",
                    candidates[0],
                    key,
                )

        return (Path("leaderboards") / dest_file.name).as_posix()
    except Exception as e:
        logger.warning("Leaderboard download failed for %s: %s", s3_uri, e)
        return ""


def build_leaderboard_rows_from_metrics_jsons(
    metrics_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ranked leaderboard rows from per-model ``metrics.json`` payloads."""
    rows: list[dict[str, Any]] = []
    for model, metrics in metrics_by_model.items():
        if not isinstance(metrics, dict):
            continue
        scalar = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        rows.append({"model": model, **scalar})
    if not rows:
        return []
    # Prefer common ranking keys (AutoGluon higher-is-better convention, including negated errors).
    rank_keys = (
        "accuracy",
        "r2",
        "roc_auc",
        "f1",
        "score_test",
        "score_val",
        "MASE",
        "WQL",
        "root_mean_squared_error",
        "mean_absolute_error",
    )
    sort_key = next((k for k in rank_keys if k in rows[0]), None)
    if sort_key is None:
        # first numeric-looking column besides model
        for k, v in rows[0].items():
            if k == "model":
                continue
            if isinstance(v, (int, float)):
                sort_key = k
                break
    if sort_key:
        rows.sort(
            key=lambda r: (
                float("-inf")
                if r.get(sort_key) is None
                else float(r[sort_key])  # type: ignore[arg-type]
            ),
            reverse=True,
        )
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def download_leaderboard_scores_from_models_artifact(
    *,
    bucket: str,
    s3_cfg: dict[str, Any] | None,
    run_id: str,
    is_timeseries: bool,
    output_csv_parent: Path,
    artifact_root_prefix: str = "",
    attempts: int = 3,
    delay_seconds: float = 2.0,
) -> str:
    """
    Collect ``models_artifact/*/metrics/metrics.json`` for a run and write
    ``leaderboards/<run_id>.scores.json`` (the structured source behind the HTML).

    Returns a path relative to ``output_csv_parent``, or ``""`` on failure/miss.
    """
    if not run_id.strip() or not bucket.strip():
        return ""
    if not s3_cfg or not s3_cfg_usable(s3_cfg):
        return ""

    rid = run_id.strip()
    folders = _folders_for(is_timeseries)
    root = (artifact_root_prefix or "").strip().strip("/")
    dest_dir = output_csv_parent / "leaderboards"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{rid}.scores.json"

    for attempt in range(attempts):
        try:
            client = make_s3_client(s3_cfg)
            metrics_by_model: dict[str, dict[str, Any]] = {}
            source_keys: list[str] = []
            for folder in folders:
                pairs = _list_metrics_json_keys_for_folder(
                    client, bucket.strip(), rid, folder, root
                )
                if not pairs:
                    continue
                for key, model in pairs:
                    try:
                        body = client.get_object(Bucket=bucket.strip(), Key=key)["Body"].read()
                        payload = json.loads(body.decode("utf-8"))
                    except Exception as e:
                        logger.warning("Failed to read metrics.json %s: %s", key, e)
                        continue
                    if isinstance(payload, dict):
                        metrics_by_model[model] = payload
                        source_keys.append(key)
                if metrics_by_model:
                    break

            if not metrics_by_model:
                if attempt + 1 < attempts:
                    time.sleep(delay_seconds)
                    continue
                logger.info(
                    "No models_artifact metrics.json found for run_id=%s folders=%s root=%r",
                    rid,
                    folders,
                    root,
                )
                return ""

            rows = build_leaderboard_rows_from_metrics_jsons(metrics_by_model)
            doc = {
                "run_id": rid,
                "source": "models_artifact/*/metrics/metrics.json",
                "source_keys": source_keys,
                "leaderboard": rows,
            }
            dest_file.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            rel = (Path("leaderboards") / dest_file.name).as_posix()
            logger.info(
                "Wrote leaderboard scores JSON for run_id=%s models=%d -> %s",
                rid,
                len(rows),
                rel,
            )
            return rel
        except Exception as e:
            logger.warning(
                "S3 metrics.json collection attempt %d/%d failed for run_id=%s: %s",
                attempt + 1,
                attempts,
                rid,
                e,
            )
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)

    return ""
