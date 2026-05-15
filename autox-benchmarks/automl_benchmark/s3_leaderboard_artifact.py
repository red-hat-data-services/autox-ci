"""List S3 keys under a KFP run prefix to find and download the leaderboard HTML artifact."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from benchmark_common.s3_client import make_s3_client, s3_cfg_usable

logger = logging.getLogger(__name__)

TABULAR_LEADERBOARD_FOLDER = "leaderboard-evaluation"
TIMESERIES_LEADERBOARD_FOLDER = "timeseries-leaderboard-evaluation"


def _list_prefix_for_run(artifact_root: str, run_id: str, leaderboard_folder: str) -> str:
    rid = run_id.strip()
    root = (artifact_root or "").strip().strip("/")
    if not root:
        return f"{rid}/{leaderboard_folder}/"
    return f"{root}/{rid}/{leaderboard_folder}/"


def _key_is_leaderboard_html_artifact(
    key: str,
    run_id: str,
    leaderboard_folder: str,
    artifact_root: str,
) -> bool:
    """
    Match keys shaped like::

        <run_id>/<leaderboard_folder>/<exec_id>/html_artifact
        <artifact_root>/<run_id>/<leaderboard_folder>/<exec_id>/html_artifact
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
        if parts[0] != rid or parts[1] != leaderboard_folder:
            return False
        return parts[3] == "html_artifact" or parts[3].startswith("html_artifact")
    if len(parts) < 5:
        return False
    if parts[0] != root or parts[1] != rid or parts[2] != leaderboard_folder:
        return False
    return parts[4] == "html_artifact" or parts[4].startswith("html_artifact")


def _list_matching_keys(
    client: Any,
    bucket: str,
    run_id: str,
    leaderboard_folder: str,
    artifact_root: str,
) -> list[str]:
    prefix = _list_prefix_for_run(artifact_root, run_id, leaderboard_folder)
    found: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            k = obj.get("Key") or ""
            if _key_is_leaderboard_html_artifact(k, run_id, leaderboard_folder, artifact_root):
                found.append(k)
    return sorted(set(found))


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
    Return ``s3://bucket/<key>`` for the first matching object under the leaderboard step.

    With a non-empty ``artifact_root_prefix`` (e.g. ``autogluon-tabular-training-pipeline``),
    keys look like::
        <artifact_root_prefix>/<run_id>/leaderboard-evaluation/<exec_id>/html_artifact

    If ``artifact_root_prefix`` is empty, the legacy layout is used::
        <run_id>/leaderboard-evaluation/...
    """
    if not run_id.strip() or not bucket.strip():
        return ""
    if not s3_cfg or not s3_cfg_usable(s3_cfg):
        return ""

    folder = TIMESERIES_LEADERBOARD_FOLDER if is_timeseries else TABULAR_LEADERBOARD_FOLDER
    root = (artifact_root_prefix or "").strip().strip("/")
    for attempt in range(attempts):
        try:
            client = make_s3_client(s3_cfg)
            keys = _list_matching_keys(client, bucket.strip(), run_id.strip(), folder, root)
            if keys:
                uri = _to_s3_uri(bucket.strip(), keys[0])
                if attempt > 0:
                    logger.info("Leaderboard HTML found on S3 retry %d for run_id=%s", attempt + 1, run_id)
                return uri
        except Exception as e:
            logger.warning(
                "S3 leaderboard lookup attempt %d/%d failed for run_id=%s: %s",
                attempt + 1,
                attempts,
                run_id,
                e,
            )
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)

    logger.info(
        "No leaderboard HTML S3 key found for run_id=%s under prefix=%r folder=%s",
        run_id,
        _list_prefix_for_run(root, run_id, folder),
        folder,
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

    Returns a path relative to ``output_csv_parent`` (POSIX, e.g. ``leaderboards/<uuid>.html``),
    or ``""`` on failure. Tries the exact S3 key from ``s3_uri`` first; if that object does not
    exist, lists ``<key>/`` and uses the first ``.html`` / ``.htm`` object.
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

        rel = Path("leaderboards") / dest_file.name
        return rel.as_posix()
    except Exception as e:
        logger.warning("Leaderboard download failed for %s: %s", s3_uri, e)
        return ""
