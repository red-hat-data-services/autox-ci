#!/usr/bin/env python3
"""Generate the Parquet variants of the AutoML CSV test datasets.

Each Parquet file is a schema-faithful conversion of the corresponding CSV: same
columns, same order, same values — only the file format and the Parquet compression
codec differ. Columns keep the types pandas infers from the CSV (numeric columns
become typed, date-like strings stay strings), so the data loader sees exactly what
the CSV scenarios see and the input format is the only variable under test.

The compression codec is part of the filename (``*_snappy.parquet``, ``*_gzip.parquet``,
...) so a failing scenario can be traced to its codec from the S3 key alone. The suite
covers snappy, gzip, zstd, brotli and uncompressed Parquet — see the "Parquet datasets"
section in autox_tests/README.md. Source CSVs stay in ``automl/data/`` even when no
scenario uploads them anymore: they are the generation inputs for these Parquet files.

Run from the repository root after changing any source CSV:

    uv run --extra benchmarks python autox_tests/scripts/generate_parquet_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[1] / "automl" / "data"

# (source CSV, output Parquet, compression codec or None for uncompressed)
CONVERSIONS: list[tuple[str, str, str | None]] = [
    # Tabular regression — TC-A-4_energy_regression (switched from CSV)
    (
        "regression/energy_efficiency_regression.csv",
        "regression/energy_efficiency_regression_zstd.parquet",
        "zstd",
    ),
    # Tabular binary classification — TC-A-5_credit_default_binary (switched from CSV)
    (
        "classification_binary/credit_default_binary_light.csv",
        "classification_binary/credit_default_binary_light_gzip.parquet",
        "gzip",
    ),
    # Tabular multiclass classification — TC-A-6_wine_multiclass (switched from CSV)
    (
        "classification_multi/wine_quality_multiclass_limited_classes_light.csv",
        "classification_multi/wine_quality_multiclass_limited_classes_light_snappy.parquet",
        "snappy",
    ),
    # Tabular binary, user test data: Parquet train + CSV test —
    # TC-A-8_user_test_data_parquet_train_csv_test
    (
        "classification_binary/german_credit_data_biased_train.csv",
        "classification_binary/german_credit_data_biased_train_uncompressed.parquet",
        None,
    ),
    # Time series — TC-B-2_timeseries_m4_hourly (switched from CSV)
    (
        "timeseries/m4_hourly_subset_train_ultra_light.csv",
        "timeseries/m4_hourly_subset_train_ultra_light_brotli.parquet",
        "brotli",
    ),
    # Time series, user test data: CSV train + Parquet test —
    # TC-B-5_user_test_data_csv_train_parquet_test
    (
        "timeseries/poland_daily_cases_03_04-28_2021.csv",
        "timeseries/poland_daily_cases_03_04-28_2021_snappy.parquet",
        "snappy",
    ),
]


def main() -> None:
    for csv_rel, parquet_rel, codec in CONVERSIONS:
        csv_path = DATA_DIR / csv_rel
        parquet_path = DATA_DIR / parquet_rel
        if not csv_path.is_file():
            raise FileNotFoundError(f"source CSV not found: {csv_path}")

        # index=False keeps the column list identical to the CSV (no index column).
        frame = pd.read_csv(csv_path)
        frame.to_parquet(parquet_path, engine="pyarrow", compression=codec, index=False)

        # Round-trip check: the Parquet must deserialize to the same data.
        pd.testing.assert_frame_equal(frame, pd.read_parquet(parquet_path))

        # Codec check: every column chunk must carry the requested compression.
        metadata = pq.ParquetFile(parquet_path).metadata
        codecs = {
            metadata.row_group(g).column(c).compression
            for g in range(metadata.num_row_groups)
            for c in range(metadata.num_columns)
        }
        expected = (codec or "UNCOMPRESSED").upper()
        if codecs != {expected}:
            raise RuntimeError(
                f"{parquet_path}: expected {expected} compression on all column "
                f"chunks, found {sorted(codecs)}"
            )

        print(
            f"{csv_rel} -> {parquet_rel}: {len(frame)} rows x {len(frame.columns)} "
            f"columns, compression={expected}"
        )

    print(f"\n{len(CONVERSIONS)} Parquet dataset(s) written to {DATA_DIR}")


if __name__ == "__main__":
    main()
