"""Small self-contained HTML report for the indexing stage and full-corpus QA."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

# Inlined so the report is a single portable file with no CDN/dependency.
_CSS = """
body { font: 15px system-ui, sans-serif; margin: 32px; color: #172033; background: #fafcff }
h1, h2 { color: #12243b }
section { background: white; border: 1px solid #dce5ef; border-radius: 10px; padding: 20px; margin: 20px 0 }
table { border-collapse: collapse; width: 100%; margin: 10px 0 }
th, td { padding: 8px; border-bottom: 1px solid #e6edf4; text-align: left }
th { background: #f2f6fa }
.cards { display: flex; flex-wrap: wrap; gap: 12px }
.card { padding: 12px 16px; border-radius: 8px; background: #f1f7ff; min-width: 160px }
.label { color: #58708b; font-size: 12px }
.value { font-size: 20px; font-weight: 650 }
.bar { height: 9px; background: #e6edf4; border-radius: 5px; overflow: hidden }
.bar i { display: block; height: 100%; background: #2f80ed }
"""

_PROFILE_TOKENS = ("per_second", "duration", "storage", "success_rate", "size", "chunks_per")


def generate_indexing_report(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Write a portable, dependency-free HTML report; returns the written path."""
    indexed = [row for row in rows if row.get("indexing_run_id")]
    body = "".join(_run_section(row) for row in indexed) or "<p>No indexing runs were recorded.</p>"
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>AutoRAG indexing report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>AutoRAG indexing &amp; quality report</h1>"
        f"{body}</body></html>"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _run_section(row: dict[str, Any]) -> str:
    title = str(row.get("dataset_name") or row.get("dataset_id") or "Indexing run")
    summary = [
        ("Dataset", row.get("dataset_name") or row.get("dataset_id")),
        ("HPO run", row.get("hpo_run_id")),
        ("Indexing run", row.get("indexing_run_id")),
        ("State", row.get("indexing_state")),
        ("Documents", row.get("indexing_total_documents")),
        ("Chunks", row.get("indexing_total_chunks")),
        ("Wall time", _seconds(row.get("indexing_duration_seconds"))),
        ("Exact-match accuracy", _percent(row.get("e2e_answer_exact_match_mean"))),
        ("Answer token F1", _percent(row.get("e2e_answer_token_f1_mean"))),
    ]
    cards = "".join(
        f"<div class=card><div class=label>{escape(label)}</div>"
        f"<div class=value>{escape(str(value if value not in (None, '') else '—'))}</div></div>"
        for label, value in summary
    )
    sections = [
        _quality_bars(row),
        _table(row, lambda k: k.startswith("e2e_") and k.endswith("_mean"), "Full-corpus answer and judge metrics"),
        _table(row, lambda k: k.startswith("hpo_") and k.endswith("_mean"), "HPO-subsample baseline metrics"),
        _table(row, lambda k: k.startswith("scale_"), "Scale drift: full corpus vs HPO subsample"),
        _table(row, _is_profile_key, "Indexing performance and capacity"),
    ]
    return f"<section><h2>{escape(title)}</h2><div class=cards>{cards}</div>{''.join(sections)}</section>"


def _is_profile_key(key: str) -> bool:
    return key.startswith("indexing_") and any(token in key for token in _PROFILE_TOKENS)


def _table(row: dict[str, Any], select: Any, heading: str) -> str:
    values = [(key, value) for key, value in sorted(row.items()) if select(key) and value not in (None, "")]
    if not values:
        return ""
    lines = "".join(f"<tr><td>{escape(key)}</td><td>{escape(str(value))}</td></tr>" for key, value in values)
    return f"<h3>{escape(heading)}</h3><table><tr><th>Metric</th><th>Value</th></tr>{lines}</table>"


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _seconds(value: Any) -> str:
    return f"{float(value):.1f} s" if isinstance(value, (int, float, str)) and str(value) else "—"


def _quality_bars(row: dict[str, Any]) -> str:
    candidates = [
        (key.removeprefix("e2e_").removesuffix("_mean"), value)
        for key, value in row.items()
        if key.startswith("e2e_") and key.endswith("_mean") and isinstance(value, (int, float)) and 0 <= value <= 1
    ]
    if not candidates:
        return ""
    bars = "".join(
        f"<tr><td>{escape(name)}</td>"
        f"<td><div class=bar><i style=\"width:{value * 100:.1f}%\"></i></div></td>"
        f"<td>{value:.1%}</td></tr>"
        for name, value in sorted(candidates)
    )
    return f"<h3>Full-corpus quality dashboard</h3><table><tr><th>Metric</th><th>Score</th><th>Value</th></tr>{bars}</table>"
