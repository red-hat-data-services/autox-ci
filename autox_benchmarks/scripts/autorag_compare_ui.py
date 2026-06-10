"""Minimal Streamlit UI for AutoRAG benchmark comparison.

Focuses on finding best configurations per run with clean, actionable insights.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_best_configs_tab(df: pd.DataFrame, score_column: str = "final_score"):
    """Render best score per unique configuration, dropping incomplete rows."""
    st.subheader("Best Configurations")

    # Configuration parameter columns
    CONFIG_COLS = [
        "chunking_method",
        "chunking_chunk_size",
        "chunking_chunk_overlap",
        "embeddings_model_id",
        "retrieval_method",
        "retrieval_number_of_chunks",
        "retrieval_search_mode",
        "generation_model_id",
    ]

    METRIC_COLS = [
        "mean_faithfulness",
        "mean_answer_correctness",
        "mean_context_correctness",
        "mean_answer_relevance",
    ]

    if df.empty:
        st.info("No data available")
        return

    # Drop rows missing score or any configuration parameter
    required = [score_column] + [c for c in CONFIG_COLS if c in df.columns]
    clean = df.dropna(subset=[c for c in required if c in df.columns])

    if clean.empty:
        st.info("No complete configuration rows (rows with missing parameters were removed)")
        return

    st.caption(f"{len(clean)} of {len(df)} rows have complete configurations")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Show top", 5, 50, 10, 5, key="best_top_n_slider")
    with col2:
        datasets = sorted(clean["dataset_name"].dropna().unique().tolist()) if "dataset_name" in clean.columns else []
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="best_datasets_filter")

    # Apply filters
    filtered = clean.copy()
    if selected_datasets and "dataset_name" in filtered.columns:
        filtered = filtered[filtered["dataset_name"].isin(selected_datasets)]

    if filtered.empty:
        st.info("No results match filters")
        return

    # Best score per unique configuration (group by config params + dataset)
    group_cols = ["dataset_name"] + [c for c in CONFIG_COLS if c in filtered.columns]
    best = filtered.loc[filtered.groupby(group_cols, dropna=False)[score_column].idxmax()]
    best = best.nlargest(top_n, score_column)

    if best.empty:
        st.info("No results match filters")
        return

    # Best overall metrics
    best_row = best.iloc[0]
    col1, col2 = st.columns(2)
    col1.metric("Best Score", f"{best_row[score_column]:.4f}")
    col2.metric("Dataset", best_row.get("dataset_name", "N/A"))

    # Build display columns
    display_cols = [score_column, "dataset_name"]
    display_cols += [c for c in CONFIG_COLS if c in best.columns]
    display_cols += [c for c in METRIC_COLS if c in best.columns]

    # Display with color coding for scores
    def color_score(val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ""
        if val >= 0.8:
            return "background-color: #d4edda"
        elif val >= 0.6:
            return "background-color: #fff3cd"
        else:
            return "background-color: #f8d7da"

    styled = best[display_cols].style.applymap(
        color_score, subset=[score_column] if score_column in display_cols else []
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Export
    st.download_button(
        "Download Best Configurations CSV",
        best[display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"best_{top_n}_configs.csv",
        mime="text/csv",
    )


def render_all_results_tab(df: pd.DataFrame):
    """Render all results in a searchable, sortable table."""
    st.subheader("All Benchmark Results")

    if df.empty:
        st.info("No data available")
        return

    st.caption(f"Total results: {len(df)}")

    # Dataset filter only (batch selection is done in sidebar)
    if "dataset_name" in df.columns:
        datasets = sorted(df["dataset_name"].dropna().unique().tolist())
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="all_datasets_filter")
    else:
        selected_datasets = []

    filtered = df.copy()
    if selected_datasets and "dataset_name" in filtered.columns:
        filtered = filtered[filtered["dataset_name"].isin(selected_datasets)]

    st.caption(f"Showing: {len(filtered)} results")

    # Display full table (sortable/searchable by default in streamlit)
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Export
    st.download_button(
        "Download All Results CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="all_benchmark_results.csv",
        mime="text/csv",
    )


def _config_label(row: pd.Series, config_cols: list[str]) -> str:
    """Short human-readable label for a configuration."""
    parts = []
    for c in config_cols:
        val = row.get(c)
        if pd.notna(val):
            short = str(val).rsplit("/", 1)[-1]
            parts.append(short)
    return " | ".join(parts) if parts else "unknown"


def render_score_history_tab(df: pd.DataFrame, score_column: str = "final_score"):
    """Per-configuration score over time to track regressions."""
    st.subheader("Score History")

    CONFIG_COLS = [
        "chunking_method",
        "chunking_chunk_size",
        "chunking_chunk_overlap",
        "embeddings_model_id",
        "retrieval_method",
        "retrieval_number_of_chunks",
        "retrieval_search_mode",
        "generation_model_id",
    ]

    METRIC_COLS = [
        "mean_faithfulness",
        "mean_answer_correctness",
        "mean_context_correctness",
        "mean_answer_relevance",
    ]

    if df.empty:
        st.info("No data available")
        return

    # Resolve timestamp: prefer finished_at, fall back to parsing compare_batch_id
    df = df.copy()
    df["_timestamp"] = pd.to_datetime(df.get("finished_at"), errors="coerce")
    mask = df["_timestamp"].isna()
    if mask.any() and "compare_batch_id" in df.columns:
        df.loc[mask, "_timestamp"] = pd.to_datetime(
            df.loc[mask, "compare_batch_id"], format="%Y%m%dT%H%M%SZ", errors="coerce"
        )

    # Keep only rows with complete config + score + timestamp
    present_config = [c for c in CONFIG_COLS if c in df.columns]
    required = [score_column, "_timestamp"] + present_config
    clean = df.dropna(subset=[c for c in required if c in df.columns])

    if clean.empty:
        st.info("No complete rows with timestamps and configuration parameters")
        return

    # Build config fingerprint
    clean["_config"] = clean.apply(lambda r: _config_label(r, present_config), axis=1)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        datasets = sorted(clean["dataset_name"].dropna().unique().tolist()) if "dataset_name" in clean.columns else []
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="history_datasets_filter")
    with col2:
        models = sorted(clean["generation_model_id"].dropna().unique().tolist()) if "generation_model_id" in clean.columns else []
        selected_models = st.multiselect("Generation Model", models, default=models, key="history_model_filter")
    with col3:
        score_options = [score_column] + [c for c in METRIC_COLS if c in clean.columns]
        selected_metric = st.selectbox("Metric", score_options, key="history_metric")

    if selected_datasets and "dataset_name" in clean.columns:
        clean = clean[clean["dataset_name"].isin(selected_datasets)]
    if selected_models and "generation_model_id" in clean.columns:
        clean = clean[clean["generation_model_id"].isin(selected_models)]

    if clean.empty:
        st.info("No results match filters")
        return

    # Chart: one line per config, x=timestamp, y=score
    configs = sorted(clean["_config"].unique().tolist())
    selected_configs = st.multiselect(
        "Configurations",
        configs,
        default=configs[:10],
        key="history_config_filter",
    )

    if not selected_configs:
        st.info("Select at least one configuration")
        return

    chart_data = clean[clean["_config"].isin(selected_configs)].sort_values("_timestamp")
    chart_data = chart_data.reset_index(drop=True)

    # Assign run index per config+dataset: keep last 10 runs, "Newest" on the right
    MAX_RUNS = 10
    group_cols = ["_config"]
    has_dataset = "dataset_name" in chart_data.columns
    if has_dataset:
        group_cols.append("dataset_name")
    indexed_parts = []
    for _key, grp in chart_data.groupby(group_cols, sort=False):
        grp = grp.sort_values("_timestamp").tail(MAX_RUNS).copy()
        n = len(grp)
        grp["_run_index"] = list(range(MAX_RUNS - n + 1, MAX_RUNS + 1))
        indexed_parts.append(grp)
    chart_data = pd.concat(indexed_parts, ignore_index=True)

    # Unique line identity: config + dataset so lines don't cross datasets
    if has_dataset:
        chart_data["_series"] = chart_data["_config"] + " | " + chart_data["dataset_name"].fillna("")
    else:
        chart_data["_series"] = chart_data["_config"]

    run_labels = {MAX_RUNS - i: f"-{i}" if i > 0 else "Newest" for i in range(MAX_RUNS)}

    detail_cols = ["compare_batch_id", "_config", "dataset_name", selected_metric]
    detail_cols += [c for c in METRIC_COLS if c in chart_data.columns and c != selected_metric]
    detail_cols += present_config
    detail_cols = [c for c in detail_cols if c in chart_data.columns]
    table_data = chart_data[detail_cols].copy()
    table_data = table_data.sort_values(["_config", "compare_batch_id"]).reset_index(drop=True)

    # Plotly line chart with selection support
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        fig = px.line(
            chart_data,
            x="_run_index",
            y=selected_metric,
            color="_series",
            markers=True,
            hover_data=["compare_batch_id", "_config", "dataset_name", "_timestamp"] + present_config,
            labels={
                "_run_index": "Run",
                selected_metric: selected_metric.replace("_", " ").title(),
                "_series": "Config / Dataset",
                "compare_batch_id": "Batch",
                "_timestamp": "Run Time",
            },
        )
        fig.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="top", y=-0.2),
            xaxis=dict(
                title="Run",
                tickmode="array",
                tickvals=list(range(1, MAX_RUNS + 1)),
                ticktext=[run_labels[i] for i in range(1, MAX_RUNS + 1)],
                range=[0.5, MAX_RUNS + 0.5],
            ),
            yaxis_title=selected_metric.replace("_", " ").title(),
        )

        # Highlight point from table selection
        table_sel_rows = st.session_state.get("_history_table_sel_rows")
        if table_sel_rows:
            for row_idx in table_sel_rows:
                if row_idx < len(table_data):
                    row = table_data.iloc[row_idx]
                    ts_match = chart_data[
                        (chart_data["_config"] == row["_config"])
                        & (chart_data["compare_batch_id"] == row["compare_batch_id"])
                    ]
                    if not ts_match.empty:
                        pt = ts_match.iloc[0]
                        fig.add_trace(go.Scatter(
                            x=[pt["_run_index"]],
                            y=[pt[selected_metric]],
                            mode="markers",
                            marker=dict(size=16, color="red", symbol="star", line=dict(width=2, color="black")),
                            name="Selected",
                            showlegend=False,
                        ))

        chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="history_chart")

        # Map chart selection to table rows
        highlighted_rows: list[int] = []
        if chart_event and chart_event.selection and chart_event.selection.points:
            for pt in chart_event.selection.points:
                pt_idx = pt.get("point_index")
                curve_idx = pt.get("curve_number", 0)
                if pt_idx is not None:
                    series_in_order = chart_data["_series"].unique().tolist()
                    if curve_idx < len(series_in_order):
                        cfg = series_in_order[curve_idx]
                        cfg_data = chart_data[chart_data["_series"] == cfg]
                        if pt_idx < len(cfg_data):
                            pt_row = cfg_data.iloc[pt_idx]
                            match = table_data[
                                (table_data["_config"] == pt_row["_config"])
                                & (table_data["compare_batch_id"] == pt_row["compare_batch_id"])
                            ]
                            highlighted_rows.extend(match.index.tolist())

        # Style highlighted rows in table
        def highlight_selected(row):
            if row.name in highlighted_rows:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        styled = table_data.style.apply(highlight_selected, axis=1)
        table_event = st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="history_table",
        )

        # Store table selection for chart highlighting on next rerun
        if table_event and table_event.selection and table_event.selection.rows:
            st.session_state["_history_table_sel_rows"] = table_event.selection.rows
        else:
            st.session_state["_history_table_sel_rows"] = []

    except ImportError:
        st.line_chart(
            chart_data.pivot_table(index="_timestamp", columns="_config", values=selected_metric)
        )
        st.dataframe(table_data, use_container_width=True, hide_index=True)


def render_autorag_tabs(
    baseline_df: pd.DataFrame,
    compare_dfs: list[pd.DataFrame],
    matched_combined: pd.DataFrame,
    score_column: str,
    baseline_label: str,
    compare_batch_ids: list[str],
):
    """Render AutoRAG UI with best configs, score history, and all results."""

    # Combine all data (baseline + compare batches)
    all_data = []

    # Add baseline data (skip if empty to avoid NaN rows)
    if not baseline_df.empty:
        baseline_copy = baseline_df.copy()
        if "compare_batch_id" not in baseline_copy.columns:
            baseline_copy["compare_batch_id"] = baseline_label
        all_data.append(baseline_copy)

    # Add compare batches
    for idx, compare_df in enumerate(compare_dfs):
        if compare_df.empty:
            continue
        compare_copy = compare_df.copy()
        if "compare_batch_id" not in compare_copy.columns:
            batch_id = compare_batch_ids[idx] if idx < len(compare_batch_ids) else f"batch_{idx}"
            compare_copy["compare_batch_id"] = batch_id
        all_data.append(compare_copy)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    tab_best, tab_history, tab_all = st.tabs(["Best Configurations", "Score History", "All Results"])

    with tab_best:
        render_best_configs_tab(combined_df, score_column)

    with tab_history:
        render_score_history_tab(combined_df, score_column)

    with tab_all:
        render_all_results_tab(combined_df)
