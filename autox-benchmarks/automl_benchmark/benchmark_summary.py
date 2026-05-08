"""Build a summary CSV from benchmark_runs.csv + optional KFP re-fetch."""

from __future__ import annotations

import json
from typing import Any

# Columns treated as experiment configuration (exclude blob + derived summary cols)
_EXCLUDE_FROM_EXPERIMENT = frozenset(
    {
        "metrics_blob",  # huge / redundant with other columns
        "experiment_config_json",
        "best_model",
        "score_name",
        "score",
        "metrics_source",
    }
)


def experiment_config_json(row: dict[str, Any]) -> str:
    payload = {}
    for k, v in row.items():
        if k in _EXCLUDE_FROM_EXPERIMENT:
            continue
        if v is None or (isinstance(v, float) and str(v) == "nan"):
            continue
        if hasattr(v, "item"):  # numpy scalar
            try:
                v = v.item()
            except Exception:
                pass
        payload[k] = v
    return json.dumps(payload, default=str, sort_keys=True)


def _parse_json_loose(s: str) -> dict[str, Any] | None:
    if not s or not isinstance(s, str):
        return None
    t = s.strip()
    if not t.startswith("{"):
        return None
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _deep_find_key(obj: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(obj, dict):
        if key in obj:
            found.append(obj[key])
        for v in obj.values():
            found.extend(_deep_find_key(v, key))
    elif isinstance(obj, list):
        for x in obj:
            found.extend(_deep_find_key(x, key))
    return found


def _first_str(values: list[Any]) -> str | None:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _collect_numeric_metrics(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Collect (name, value_str) from nested dicts/lists (protobuf-style metrics, etc.)."""
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        if "name" in obj and ("number_value" in obj or "double_value" in obj):
            name = str(obj.get("name", "")).strip()
            val = obj.get("number_value", obj.get("double_value"))
            if name and val is not None:
                pairs.append((name, str(val)))
        for k, v in obj.items():
            nk = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k.lower() not in ("top_n", "timeout_seconds", "poll_interval_seconds"):
                    pairs.append((nk, str(v)))
            else:
                pairs.extend(_collect_numeric_metrics(v, nk))
    elif isinstance(obj, list):
        for x in obj:
            pairs.extend(_collect_numeric_metrics(x, prefix))
    return pairs


def _try_parse_embedded_json_strings(obj: Any) -> list[dict[str, Any]]:
    """Find JSON object strings inside the tree (some backends stringify outputs)."""
    found: list[dict[str, Any]] = []
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) > 2 and s[0] == "{":
            try:
                d = json.loads(s)
                if isinstance(d, dict):
                    found.append(d)
            except json.JSONDecodeError:
                pass
        return found
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(_try_parse_embedded_json_strings(v))
    elif isinstance(obj, list):
        for x in obj:
            found.extend(_try_parse_embedded_json_strings(x))
    return found


def _leaderboard_like_rows(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        keys = set().union(*(x.keys() for x in obj))
        model_keys = keys & {"model", "model_name", "name"}
        metric_keys = keys & {
            "score_val",
            "score_test",
            "accuracy",
            "r2",
            "rmse",
            "root_mean_squared_error",
            "f1",
            "auc",
        }
        if model_keys and metric_keys:
            return list(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            rows.extend(_leaderboard_like_rows(v))
    elif isinstance(obj, list):
        for x in obj:
            rows.extend(_leaderboard_like_rows(x))
    return rows


def extract_best_model_and_scores(parsed: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    """
    Return (best_model, [(score_name, score_value_str), ...]).
    Heuristic across KFP task outputs / metrics blobs.
    """
    best = _first_str([x for x in _deep_find_key(parsed, "best_model") if isinstance(x, str)])
    if not best:
        for x in _deep_find_key(parsed, "best_model"):
            if x is not None:
                s = str(x).strip()
                if s:
                    best = s
                    break

    score_pairs: list[tuple[str, str]] = []

    for emb in _try_parse_embedded_json_strings(parsed):
        score_pairs.extend(_collect_numeric_metrics(emb))
        if not best:
            best = _first_str([x for x in _deep_find_key(emb, "best_model") if isinstance(x, str)]) or best

    score_pairs.extend(_collect_numeric_metrics(parsed))

    eval_metric = _first_str([str(x) for x in _deep_find_key(parsed, "eval_metric") if x is not None])

    for r in _leaderboard_like_rows(parsed):
        model = r.get("model") or r.get("model_name") or r.get("name")
        if best and model is not None and str(model).strip() and str(model).strip() != str(best).strip():
            continue
        for metric_key in (
            "score_val",
            "score_test",
            "accuracy",
            "r2",
            "rmse",
            "root_mean_squared_error",
            "f1",
            "auc",
        ):
            if metric_key in r and r[metric_key] is not None:
                label = metric_key
                if eval_metric:
                    label = f"{eval_metric}.{metric_key}"
                score_pairs.append((label, str(r[metric_key])))

    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for p in score_pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return best or "", deduped


def rows_for_summary_csv(
    row: dict[str, Any],
    parsed_blob: dict[str, Any] | None,
    *,
    metrics_source: str,
) -> list[dict[str, Any]]:
    """One or more output rows (multiple if several scores found)."""
    exp = experiment_config_json(row)
    best, pairs = extract_best_model_and_scores(parsed_blob or {})

    base = {k: v for k, v in row.items() if k != "metrics_blob"}
    base["experiment_config_json"] = exp
    base["best_model"] = best
    base["metrics_source"] = metrics_source

    if not pairs:
        out = dict(base)
        out["score_name"] = ""
        out["score"] = ""
        return [out]

    return [
        {
            **base,
            "score_name": name,
            "score": val,
        }
        for name, val in pairs
    ]


def records_to_summary_rows(
    records: list[dict[str, Any]],
    enrich_client: Any | None = None,
    *,
    force_refetch: bool = False,
) -> list[dict[str, Any]]:
    """Expand benchmark run records (dict rows) to summary rows."""
    from benchmark_common.metrics_extraction import run_to_metrics_dict
    from benchmark_common.run_state import unwrap_run_from_get_run

    out: list[dict[str, Any]] = []
    for r in records:
        blob_raw = r.get("metrics_blob", "")
        parsed = _parse_json_loose(str(blob_raw) if blob_raw is not None else "")
        source = "metrics_blob"

        need_refetch = enrich_client is not None and (
            force_refetch
            or parsed is None
            or not _blob_has_useful_signals(parsed)
        )
        if need_refetch:
            rid = r.get("run_id")
            if rid and str(rid).strip():
                try:
                    detail = enrich_client.get_run(str(rid).strip())
                    run = unwrap_run_from_get_run(detail) or detail
                    parsed = run_to_metrics_dict(run)
                    source = "kfp_refetch"
                except Exception:
                    if parsed is None:
                        parsed = {}
                        source = "empty_invalid_blob"
            elif parsed is None:
                parsed = {}
                source = "empty_invalid_blob"

        if parsed is None:
            parsed = {}
            source = "empty_invalid_blob"

        out.extend(rows_for_summary_csv(r, parsed, metrics_source=source))
    return out

def dataframe_to_summary_rows(
    df: Any,
    enrich_client: Any | None = None,
    *,
    force_refetch: bool = False,
) -> list[dict[str, Any]]:
    """Same as records_to_summary_rows but accepts a pandas DataFrame."""
    records = df.to_dict("records")
    return records_to_summary_rows(records, enrich_client, force_refetch=force_refetch)


def _blob_has_useful_signals(parsed: dict[str, Any]) -> bool:
    if _deep_find_key(parsed, "best_model"):
        return True
    if parsed.get("task_details"):
        return True
    if parsed.get("metrics"):
        return True
    ps = parsed.get("pipeline_spec")
    if isinstance(ps, str) and "omitted" in ps.lower():
        return True
    if ps is not None and not isinstance(ps, str):
        return False
    return bool(parsed.get("runtime_context"))

