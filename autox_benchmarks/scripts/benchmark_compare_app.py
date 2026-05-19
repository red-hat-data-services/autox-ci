#!/usr/bin/env python3
"""
Streamlit UI to compare benchmark merged leaderboards: baseline vs S3 batch runs.

Install: pip install -e ".[compare]"
Run: streamlit run scripts/benchmark_compare_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DOCS_S3_SCHEMA = "docs/s3-storage-schema.md"
JOINED_RESULTS_LABEL = "joined_results.csv (rolling)"


def _default_credentials_path() -> Path:
    env = os.environ.get("BENCHMARK_CREDENTIALS_PATH")
    if env:
        return Path(env)
    return _REPO_ROOT / "config" / "credentials.ini"


def _score_range(z_values: Any) -> tuple[float, float]:
    import pandas as pd

    flat = pd.Series(z_values.ravel()).dropna()
    if flat.empty:
        return 0.0, 1.0
    zmin, zmax = float(flat.min()), float(flat.max())
    if zmin == zmax:
        zmax = zmin + 1e-6
    return zmin, zmax


def _render_score_heatmap(pivot: Any, title: str) -> Any:
    """Plotly heatmap: datasets × models, colored by score."""
    import plotly.graph_objects as go

    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)")
        return fig

    z = pivot.values.astype(float)
    zmin, zmax = _score_range(z)
    n_rows, n_cols = pivot.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Score"),
            hoverongaps=False,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Dataset",
        height=min(1000, 140 + 24 * max(n_rows, 1)),
        width=None,
        margin=dict(l=120, r=40, t=60, b=80),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    )
    return fig


def main() -> None:
    import pandas as pd
    import streamlit as st
    from benchmark_common.ini_credentials import load_credentials_ini
    from autorag_compare_ui import render_autorag_tabs

    st.set_page_config(page_title="Benchmark compare", layout="wide")

    # Benchmark type selector
    benchmark_type = st.sidebar.radio(
        "Benchmark Type",
        ["AutoML", "AutoRAG"],
        horizontal=True,
        help="Select which benchmark type to compare"
    )

    # Dynamic import based on type
    if benchmark_type == "AutoML":
        from automl_benchmark.compare_logic import (
            align_score_matrix,
            baseline_only_keys,
            collapse_baseline_latest_per_dataset,
            compare_only_keys,
            compare_to_baseline,
            coverage_stats,
            detect_score_column,
            list_available_score_columns,
            load_merged_csv,
            score_is_lower_better,
            score_matrix_for_heatmap,
        )
        from automl_benchmark.compare_s3 import (
            DEFAULT_CACHE_DIR,
            fetch_joined_results,
            fetch_merged_leaderboards,
            list_batch_ids,
            storage_from_credentials,
        )
        entity_label = "Model"
        entity_column = "model"
    else:  # AutoRAG
        from autorag_benchmark.compare_logic import (
            align_score_matrix,
            baseline_only_keys,
            collapse_baseline_latest_per_dataset,
            compare_only_keys,
            compare_to_baseline,
            coverage_stats,
            detect_score_column,
            list_available_score_columns,
            load_merged_csv,
            score_is_lower_better,
            score_matrix_for_heatmap,
        )
        from autorag_benchmark.compare_s3 import (
            DEFAULT_CACHE_DIR,
            fetch_joined_results,
            fetch_merged_leaderboards,
            list_batch_ids,
            storage_from_credentials,
        )
        entity_label = "Pattern"
        entity_column = "pattern_name"

    st.title(f"{benchmark_type} Benchmark Compare")

    baseline_label = JOINED_RESULTS_LABEL
    collapse_baseline = True

    with st.sidebar:
        st.header("Data source")
        use_s3 = st.radio("Source", ["S3", "Local files"], horizontal=True) == "S3"
        cache_dir = Path(st.text_input("Cache directory", str(DEFAULT_CACHE_DIR)))
        force_refresh = st.checkbox("Force refresh from S3", value=False)
        cred_path = Path(st.text_input("credentials.ini", str(_default_credentials_path())))

        if use_s3:
            if not cred_path.is_file():
                st.error(f"Missing credentials: {cred_path}")
                st.stop()
            try:
                ini_cfg = load_credentials_ini(cred_path)
                bucket, bench_prefix, s3_cfg = storage_from_credentials(ini_cfg)
            except (ValueError, FileNotFoundError) as e:
                st.error(str(e))
                st.stop()
            st.caption(f"Bucket: `{bucket}` · prefix: `{bench_prefix}/`")
            if st.button("Refresh batch list"):
                st.session_state.pop("batch_ids", None)
            batch_ids: list[str] = st.session_state.get("batch_ids") or []
            if not batch_ids:
                try:
                    batch_ids = list_batch_ids(s3_cfg=s3_cfg, bucket=bucket, benchmark_prefix=bench_prefix)
                    st.session_state["batch_ids"] = batch_ids
                except Exception as e:
                    st.error(f"Could not list batches: {e}")
                    st.stop()

            # Simplified batch selection for AutoRAG, traditional for AutoML
            if benchmark_type == "AutoRAG":
                # AutoRAG: Just select batches to view (no baseline/compare distinction)
                st.subheader("Select Batches")
                selected_batches = st.multiselect(
                    "Batches to analyze",
                    batch_ids,
                    default=batch_ids[:3] if len(batch_ids) >= 3 else batch_ids,
                    help="Select one or more benchmark batches to view best configurations"
                )
                if not selected_batches:
                    st.info("Select at least one batch to analyze.")
                    st.stop()
                # No baseline needed for AutoRAG minimal UI
                baseline_batch_id = None
                baseline_label = "all_batches"
                collapse_baseline = False
            else:
                # AutoML: Traditional baseline vs compare selection
                st.subheader("Baseline")
                baseline_mode = st.radio(
                    "Baseline source",
                    [JOINED_RESULTS_LABEL, "S3 batch (merged_leaderboards.csv)"],
                    label_visibility="collapsed",
                )
                baseline_batch_id: str | None = None
                if baseline_mode == JOINED_RESULTS_LABEL:
                    collapse_baseline = st.checkbox(
                        "Collapse to latest finished_at per dataset",
                        value=True,
                        help="Recommended for rolling joined_results.csv",
                    )
                    baseline_label = JOINED_RESULTS_LABEL
                else:
                    collapse_baseline = False
                    baseline_batch_id = st.selectbox(
                        "Baseline batch",
                        batch_ids,
                        index=0 if batch_ids else None,
                    )
                    baseline_label = f"batch {baseline_batch_id}" if baseline_batch_id else "batch"

                compare_candidates = [b for b in batch_ids if b != baseline_batch_id]
                selected_batches = st.multiselect(
                    "Compare batches",
                    compare_candidates,
                    default=compare_candidates[:1] if compare_candidates else [],
                )
                if not selected_batches:
                    st.info("Select at least one compare batch.")
                    st.stop()
        else:
            # Local files mode
            if benchmark_type == "AutoRAG":
                # AutoRAG: Simple file upload (no baseline/compare distinction)
                st.subheader("Upload Benchmark Results")
                uploaded_files = st.file_uploader(
                    "Benchmark CSV files",
                    type=["csv"],
                    accept_multiple_files=True,
                    help="Upload one or more merged_leaderboards.csv files"
                )
                if not uploaded_files:
                    st.info("Upload at least one benchmark CSV file.")
                    st.stop()
                baseline_batch_id = None
                baseline_label = "uploaded_files"
                collapse_baseline = False
                selected_batches = [f.name for f in uploaded_files]
            else:
                # AutoML: Traditional baseline + compare upload
                baseline_upload = st.file_uploader("Baseline CSV", type=["csv"])
                compare_upload = st.file_uploader("Compare CSV (merged_leaderboards)", type=["csv"])
                baseline_path_str = st.text_input("Or baseline path", "")
                compare_path_str = st.text_input("Or compare path", "")
                collapse_baseline = st.checkbox(
                    "Collapse baseline to latest finished_at per dataset",
                    value=False,
                    help="Enable if baseline is joined_results-style rolling CSV",
                )
                baseline_batch_id = None

            bucket = bench_prefix = s3_cfg = None  # type: ignore
            batch_ids = []

    @st.cache_data(show_spinner="Loading joined_results…")
    def load_joined_s3(_bucket: str, _prefix: str, _s3_cfg: dict, _cache: str, _refresh: bool) -> bytes:
        return fetch_joined_results(
            s3_cfg=_s3_cfg,
            bucket=_bucket,
            benchmark_prefix=_prefix,
            cache_dir=Path(_cache),
            force_refresh=_refresh,
        )

    @st.cache_data(show_spinner="Loading batch CSV…")
    def load_batch_s3(
        _batch: str, _bucket: str, _prefix: str, _s3_cfg: dict, _cache: str, _refresh: bool
    ) -> bytes:
        return fetch_merged_leaderboards(
            s3_cfg=_s3_cfg,
            bucket=_bucket,
            benchmark_prefix=_prefix,
            batch_id=_batch,
            cache_dir=Path(_cache),
            force_refresh=_refresh,
        )

    try:
        if use_s3:
            # AutoRAG: Skip baseline, just load selected batches
            if benchmark_type == "AutoRAG":
                baseline_df = pd.DataFrame()  # Empty baseline for AutoRAG
                compare_dfs: dict[str, pd.DataFrame] = {}
                for bid in selected_batches:
                    compare_dfs[bid] = load_merged_csv(
                        load_batch_s3(bid, bucket, bench_prefix, s3_cfg, str(cache_dir), force_refresh)
                    )
            else:
                # AutoML: Traditional baseline + compare loading
                if baseline_batch_id:
                    baseline_df = load_merged_csv(
                        load_batch_s3(
                            baseline_batch_id, bucket, bench_prefix, s3_cfg, str(cache_dir), force_refresh
                        )
                    )
                else:
                    baseline_df = load_merged_csv(
                        load_joined_s3(bucket, bench_prefix, s3_cfg, str(cache_dir), force_refresh)
                    )
                compare_dfs: dict[str, pd.DataFrame] = {}
                for bid in selected_batches:
                    compare_dfs[bid] = load_merged_csv(
                        load_batch_s3(bid, bucket, bench_prefix, s3_cfg, str(cache_dir), force_refresh)
                    )
        else:
            # Local files loading
            if benchmark_type == "AutoRAG":
                # AutoRAG: Load multiple uploaded files (no baseline)
                baseline_df = pd.DataFrame()
                compare_dfs = {}
                for uploaded_file in uploaded_files:
                    compare_dfs[uploaded_file.name] = load_merged_csv(uploaded_file.getvalue())
                selected_batches = list(compare_dfs.keys())
            else:
                # AutoML: Traditional baseline + compare loading
                if baseline_upload is not None:
                    baseline_df = load_merged_csv(baseline_upload.getvalue())
                    baseline_label = baseline_upload.name or "uploaded baseline"
                elif baseline_path_str.strip():
                    baseline_df = load_merged_csv(Path(baseline_path_str.strip()))
                    baseline_label = Path(baseline_path_str.strip()).name
                else:
                    st.warning("Provide a baseline CSV.")
                    st.stop()
                compare_dfs = {}
                if compare_upload is not None:
                    compare_dfs["local"] = load_merged_csv(compare_upload.getvalue())
                elif compare_path_str.strip():
                    compare_dfs["local"] = load_merged_csv(Path(compare_path_str.strip()))
                else:
                    st.warning("Provide a compare CSV.")
                    st.stop()
                selected_batches = list(compare_dfs.keys())
    except FileNotFoundError as e:
        st.error(str(e))
        st.markdown(f"See [{DOCS_S3_SCHEMA}]({DOCS_S3_SCHEMA}) for S3 layout.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Skip baseline processing for AutoRAG (not needed for minimal UI)
    if benchmark_type == "AutoRAG":
        baseline_for_compare = baseline_df  # Empty df, not used
    else:
        baseline_for_compare = (
            collapse_baseline_latest_per_dataset(baseline_df) if collapse_baseline else baseline_df
        )

    # Score column selection (type-specific)
    if benchmark_type == "AutoRAG":
        # Hardcode to final_score for AutoRAG (minimal UI)
        score_col = "final_score"
        lower_better = False
        only_changed = False
    else:
        # AutoML: show score column selector
        score_options = list_available_score_columns(baseline_for_compare, *compare_dfs.values())
        try:
            default_score = detect_score_column(baseline_for_compare)
        except ValueError:
            default_score = score_options[0] if score_options else ""
        score_col = st.selectbox(
            "Score column",
            score_options or [default_score],
            index=score_options.index(default_score) if default_score in score_options else 0,
        )
        lower_better = st.checkbox(
            "Lower score is better (auto-detected from metric name)",
            value=score_is_lower_better(score_col),
        )
        only_changed = st.checkbox("Show only rows where delta ≠ 0", value=False)

    # Type-specific comparison logic (AutoML only - AutoRAG uses simpler direct display)
    if benchmark_type == "AutoML":
        # AutoML filters and comparison
        task_filter = st.multiselect(
            "Filter task_type",
            sorted(baseline_for_compare["task_type"].dropna().unique().tolist())
            if "task_type" in baseline_for_compare.columns
            else [],
        )
        task_types_arg = task_filter if task_filter else None
        optimization_metrics_arg = None

        all_matched: list[pd.DataFrame] = []
        all_warnings: list[str] = []
        coverage_rows: list[dict] = []

        for batch_id, cmp_df in compare_dfs.items():
            matched, warnings = compare_to_baseline(
                baseline_df,
                cmp_df,
                score_column=score_col,
                compare_batch_id=batch_id,
                collapse_baseline=collapse_baseline,
            )
            all_warnings.extend(warnings)

            # Apply task type filters
            if task_filter and "baseline_task_type" in matched.columns:
                matched = matched[matched["baseline_task_type"].isin(task_filter)]
            elif task_filter and "task_type" in matched.columns:
                matched = matched[matched["task_type"].isin(task_filter)]

            if only_changed and not matched.empty:
                matched = matched[matched["delta"].fillna(0) != 0]
            all_matched.append(matched)
            coverage_rows.append(
                coverage_stats(matched, baseline_for_compare, cmp_df, batch_id=batch_id)
            )

        if all_warnings:
            with st.expander("Warnings", expanded=False):
                for w in sorted(set(all_warnings)):
                    st.warning(w)

        matched_combined = pd.concat(all_matched, ignore_index=True) if all_matched else pd.DataFrame()
    else:
        # AutoRAG: Skip comparison logic, handled by minimal UI
        task_types_arg = None
        optimization_metrics_arg = None
        matched_combined = pd.DataFrame()  # Not used by AutoRAG tabs
        all_warnings = []
        coverage_rows = []

    # Use specialized UI for AutoRAG, generic UI for AutoML
    if benchmark_type == "AutoRAG":
        render_autorag_tabs(
            baseline_df=baseline_df,
            compare_dfs=list(compare_dfs.values()),
            matched_combined=matched_combined,
            score_column=score_col,
            baseline_label=baseline_label,
            compare_batch_ids=list(compare_dfs.keys()),
        )
    else:
        # AutoML tabs
        tab_overview, tab_heatmap, tab_coverage, tab_raw = st.tabs(
            ["Overview", "Heatmaps", "Coverage", "Raw data"]
        )

        with tab_overview:
            st.caption(f"Baseline: **{baseline_label}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matched pairs", len(matched_combined))
            c2.metric("Compare batches", len(compare_dfs))
            c3.metric("Baseline datasets", baseline_for_compare["dataset_name"].nunique())
            if "leaderboard_parse_ok" in baseline_df.columns:
                failed = int((baseline_df["leaderboard_parse_ok"] == False).sum())  # noqa: E712
                c4.metric("Baseline parse failures", failed)

            if matched_combined.empty:
                st.info(f"No matching dataset_name + {entity_column} rows between baseline and compare.")
            else:
                # Build display columns dynamically based on benchmark type
                if benchmark_type == "AutoML":
                    base_display_cols = [
                        "dataset_name",
                        entity_column,
                        "compare_batch_id",
                        "baseline_score",
                        "compare_score",
                        "delta",
                        "pct_change",
                        "baseline_task_type",
                        "compare_finished_at",
                    ]
                else:  # AutoRAG - show pattern parameters
                    # Core columns
                    core_cols = [
                        "dataset_name",
                        "baseline_score",
                        "compare_score",
                        "delta",
                        "pct_change",
                    ]

                    # Pattern parameter columns (may have _baseline/_compare suffix after merge)
                    param_cols = [
                        "chunking_method",
                        "chunking_chunk_size",
                        "chunking_chunk_overlap",
                        "embeddings_model_id",
                        "retrieval_method",
                        "retrieval_number_of_chunks",
                        "retrieval_search_mode",
                        "retrieval_ranker_strategy",
                        "generation_model_id",
                    ]

                    # Metric columns (may have _baseline/_compare suffix)
                    metric_cols = [
                        "mean_faithfulness",
                        "mean_answer_correctness",
                        "mean_context_correctness",
                        "mean_answer_relevance",
                    ]

                    # Build final column list checking for existence
                    base_display_cols = core_cols.copy()

                    # Add compare parameters (prefer _compare suffix, then no suffix)
                    for param in param_cols:
                        if f"{param}_compare" in matched_combined.columns:
                            base_display_cols.append(f"{param}_compare")
                        elif param in matched_combined.columns:
                            base_display_cols.append(param)

                    # Add metric columns
                    for metric in metric_cols:
                        if f"{metric}_compare" in matched_combined.columns:
                            base_display_cols.append(f"{metric}_compare")
                        elif metric in matched_combined.columns:
                            base_display_cols.append(metric)

                    # Add metadata
                    base_display_cols.extend([
                        "compare_batch_id",
                        "baseline_optimization_metric",
                        "compare_finished_at",
                    ])

                display_cols = [c for c in base_display_cols if c in matched_combined.columns]

                # Rename columns for better readability (remove _compare suffix for display)
                display_df = matched_combined[display_cols].copy()
                rename_map = {col: col.replace("_compare", "") for col in display_cols if col.endswith("_compare")}
                if rename_map:
                    display_df.rename(columns=rename_map, inplace=True)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with tab_heatmap:
            if len(compare_dfs) > 1:
                heatmap_batch = st.selectbox(
                    "Batch for compare heatmap",
                    list(compare_dfs.keys()),
                    key="heatmap_compare_batch",
                )
            else:
                heatmap_batch = next(iter(compare_dfs))

            cmp_df_hm = compare_dfs[heatmap_batch]
            try:
                if benchmark_type == "AutoML":
                    baseline_matrix = score_matrix_for_heatmap(
                        baseline_for_compare,
                        score_column=score_col,
                        collapse_latest=False,
                        task_types=task_types_arg,
                    )
                    compare_matrix = score_matrix_for_heatmap(
                        cmp_df_hm,
                        score_column=score_col,
                        collapse_latest=False,
                        task_types=task_types_arg,
                    )
                else:  # AutoRAG
                    baseline_matrix = score_matrix_for_heatmap(
                        baseline_for_compare,
                        score_column=score_col,
                        collapse_latest=False,
                        optimization_metrics=optimization_metrics_arg,
                    )
                    compare_matrix = score_matrix_for_heatmap(
                        cmp_df_hm,
                        score_column=score_col,
                        collapse_latest=False,
                        optimization_metrics=optimization_metrics_arg,
                    )
                compare_matrix = align_score_matrix(baseline_matrix, compare_matrix)
            except ValueError as e:
                st.error(str(e))
                baseline_matrix = pd.DataFrame()
                compare_matrix = pd.DataFrame()

            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(
                    _render_score_heatmap(
                        baseline_matrix,
                        f"Baseline · {baseline_label} · {score_col}",
                    ),
                    use_container_width=True,
                )
            with col_r:
                st.plotly_chart(
                    _render_score_heatmap(
                        compare_matrix,
                        f"Compare · {heatmap_batch} · {score_col}",
                    ),
                    use_container_width=True,
                )
            if lower_better:
                st.caption("Darker cells are higher scores; for RMSE/loss-style metrics, lower values are better.")
            st.caption(f"Rows = datasets, columns = {entity_label.lower()}s. Compare heatmap uses the same row/column order as baseline.")

        with tab_coverage:
            st.subheader("Per-batch coverage")
            st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
            if len(compare_dfs) == 1:
                bid = next(iter(compare_dfs))
                cmp_df = compare_dfs[bid]
                b_only = baseline_only_keys(
                    baseline_df, cmp_df, collapse_baseline=collapse_baseline
                )
                c_only = compare_only_keys(
                    baseline_df, cmp_df, collapse_baseline=collapse_baseline
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Baseline only** (dataset + {entity_label.lower()})")
                    st.dataframe(b_only, use_container_width=True, hide_index=True)
                with col_b:
                    st.markdown("**Compare only**")
                    st.dataframe(c_only, use_container_width=True, hide_index=True)

            if coverage_rows:
                import plotly.graph_objects as go

                fig_cov = go.Figure(
                    data=go.Bar(
                        x=[r["batch_id"] for r in coverage_rows],
                        y=[r["dataset_coverage_pct"] for r in coverage_rows],
                        text=[f"{r['matched_datasets']}/{r['baseline_datasets']}" for r in coverage_rows],
                        textposition="auto",
                    )
                )
                fig_cov.update_layout(
                    title=f"Dataset coverage (% baseline datasets with ≥1 matched {entity_label.lower()})",
                    yaxis_title="Percent",
                )
                st.plotly_chart(fig_cov, use_container_width=True)

        with tab_raw:
            st.download_button(
                "Download matched comparison CSV",
                matched_combined.to_csv(index=False).encode("utf-8"),
                file_name="benchmark_compare_matched.csv",
                mime="text/csv",
                disabled=matched_combined.empty,
            )
            with st.expander("Baseline table used for comparison"):
                st.dataframe(baseline_for_compare, use_container_width=True)


if __name__ == "__main__":
    main()
