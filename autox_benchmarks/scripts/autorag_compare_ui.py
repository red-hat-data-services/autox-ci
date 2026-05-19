"""Minimal Streamlit UI for AutoRAG benchmark comparison.

Focuses on finding best configurations per run with clean, actionable insights.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_best_configs_tab(df: pd.DataFrame, score_column: str = "final_score"):
    """Render top N best configurations with inline parameters."""
    st.subheader("Top Performing Configurations")

    if df.empty:
        st.info("No data available")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Show top", 5, 50, 10, 5, key="best_top_n_slider")
    with col2:
        datasets = df["dataset_name"].unique().tolist() if "dataset_name" in df.columns else []
        selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="best_datasets_filter")
    with col3:
        batches = df["compare_batch_id"].unique().tolist() if "compare_batch_id" in df.columns else []
        selected_batches = st.multiselect("Batches", batches, default=batches, key="best_batches_filter")

    # Filter data
    filtered = df.copy()
    if selected_datasets and "dataset_name" in filtered.columns:
        filtered = filtered[filtered["dataset_name"].isin(selected_datasets)]
    if selected_batches and "compare_batch_id" in filtered.columns:
        filtered = filtered[filtered["compare_batch_id"].isin(selected_batches)]

    # Sort by score and take top N
    if score_column in filtered.columns:
        top_configs = filtered.nlargest(top_n, score_column)
    else:
        top_configs = filtered.head(top_n)

    if top_configs.empty:
        st.info("No results match filters")
        return

    # Best overall metrics
    if score_column in top_configs.columns:
        best_row = top_configs.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Score", f"{best_row[score_column]:.4f}")
        col2.metric("Dataset", best_row.get("dataset_name", "N/A"))
        col3.metric("Batch", best_row.get("compare_batch_id", "N/A"))

    # Display columns - configuration inline, no Pattern* names
    display_cols = [
        score_column,
        "dataset_name",
        "compare_batch_id",
        # Configuration parameters (inline, portable across runs)
        "chunking_method",
        "chunking_chunk_size",
        "chunking_chunk_overlap",
        "embeddings_model_id",
        "retrieval_method",
        "retrieval_number_of_chunks",
        "retrieval_search_mode",
        "generation_model_id",
        # Evaluation metrics
        "mean_faithfulness",
        "mean_answer_correctness",
        "mean_context_correctness",
        "mean_answer_relevance",
    ]

    # Filter to existing columns
    display_cols = [c for c in display_cols if c in top_configs.columns]

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

    styled = top_configs[display_cols].style.applymap(
        color_score, subset=[score_column] if score_column in display_cols else []
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Export
    st.download_button(
        "Download Top Configurations CSV",
        top_configs[display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"top_{top_n}_configs.csv",
        mime="text/csv",
    )


def render_all_results_tab(df: pd.DataFrame):
    """Render all results in a searchable, sortable table."""
    st.subheader("All Benchmark Results")

    if df.empty:
        st.info("No data available")
        return

    st.caption(f"Total results: {len(df)}")

    # Simple filters
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if "dataset_name" in df.columns:
                datasets = df["dataset_name"].unique().tolist()
                selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="all_datasets_filter")
            else:
                selected_datasets = []

        with col2:
            if "compare_batch_id" in df.columns:
                batches = df["compare_batch_id"].unique().tolist()
                selected_batches = st.multiselect("Batches", batches, default=batches, key="all_batches_filter")
            else:
                selected_batches = []

    # Apply filters
    filtered = df.copy()
    if selected_datasets and "dataset_name" in filtered.columns:
        filtered = filtered[filtered["dataset_name"].isin(selected_datasets)]
    if selected_batches and "compare_batch_id" in filtered.columns:
        filtered = filtered[filtered["compare_batch_id"].isin(selected_batches)]

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


def render_autorag_tabs(
    baseline_df: pd.DataFrame,
    compare_dfs: list[pd.DataFrame],
    matched_combined: pd.DataFrame,
    score_column: str,
    baseline_label: str,
    compare_batch_ids: list[str],
):
    """Render minimal AutoRAG UI with 2 tabs focused on best configurations."""

    # Combine all data (baseline + compare batches)
    all_data = []

    # Add baseline data
    baseline_copy = baseline_df.copy()
    if "compare_batch_id" not in baseline_copy.columns:
        baseline_copy["compare_batch_id"] = baseline_label
    all_data.append(baseline_copy)

    # Add compare batches
    for idx, compare_df in enumerate(compare_dfs):
        compare_copy = compare_df.copy()
        if "compare_batch_id" not in compare_copy.columns:
            batch_id = compare_batch_ids[idx] if idx < len(compare_batch_ids) else f"batch_{idx}"
            compare_copy["compare_batch_id"] = batch_id
        all_data.append(compare_copy)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    # Create 2 minimal tabs
    tab_best, tab_all = st.tabs(["Best Configurations", "All Results"])

    with tab_best:
        render_best_configs_tab(combined_df, score_column)

    with tab_all:
        render_all_results_tab(combined_df)
