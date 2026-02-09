#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantAQ Data Processor
======================

This script processes raw and final data files from QuantAQ MODULAIR-PM
sensors, expanding nested dictionary columns and merging calibrated values
into a unified analysis-ready format.

Key Functions:
    - parse_dict_string: Safely parse dictionary strings from CSV
    - expand_dict_column: Expand nested dicts into individual columns
    - process_raw_file: Extract and organize raw sensor data
    - process_final_file: Extract calibrated PM values
    - merge_raw_and_final: Combine datasets on timestamp

Processing Features:
    - Expands geo, met, neph, opc dictionary columns into flat structure
    - Renames pm25 to pm2.5 for consistency
    - Reorders PM columns: pm1, pm2.5, pm10
    - Prefixes calibrated values with final_ (final_pm1, final_pm2.5, final_pm10)
    - Outer join preserves all timestamps from both datasets

Methodology:
    1. Find matching raw/final file pairs in data directory
    2. Parse raw file and expand dictionary columns
    3. Extract calibrated PM values from final file
    4. Merge on timestamp_local with outer join
    5. Reorder columns and save processed output

Input Files:
    - {date}-quantaq-outside-raw.csv
    - {date}-quantaq-outside-final.csv
    - {date}-quantaq-inside-raw.csv
    - {date}-quantaq-inside-final.csv

Output Files:
    - {date}-quantaq-outside-processed.csv: Merged outside sensor data
    - {date}-quantaq-inside-processed.csv: Merged inside sensor data

Column Organization:
    - Base: timestamp_local, timestamp, sn, operating_state, flag
    - Geo: geo_lat, geo_lon
    - Met: met_rh, met_temp
    - Neph: neph_bin0-5, neph_pm1, neph_pm2.5, neph_pm10, neph_rh, neph_temp
    - OPC: opc_bin0-23, opc_pm1, opc_pm2.5, opc_pm10, opc_rh, opc_temp
    - Final: final_pm1, final_pm2.5, final_pm10

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import ast
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_instrument_path


def get_quantaq_data_path() -> Path:
    """
    Get the QuantAQ data storage path from configuration.

    Returns:
        Path: Directory path for QuantAQ data files.
    """
    return get_instrument_path("QuantAQ_MODULAIR_PM")


# Device/location mapping (must match download script)
DEVICE_NAMES = ["quantaq-outside", "quantaq-inside"]
CHUNK_DATA_TYPES = ["raw", "final"]


def get_chunks_path() -> Path:
    """
    Get the path to the chunks subdirectory.

    Returns:
        Path: Directory path for chunk files (QuantAQ/chunks/).
    """
    return get_quantaq_data_path() / "chunks"


def find_chunk_files(
    chunks_dir: Path,
    device_name: str,
    data_type: str,
) -> List[Path]:
    """
    Find all chunk files for a given device and data type.

    Args:
        chunks_dir: Path to the chunks/ directory.
        device_name: Device name (e.g., 'quantaq-outside').
        data_type: Data type ('raw' or 'final').

    Returns:
        List of chunk file paths sorted by filename (chronological).
    """
    if not chunks_dir.exists():
        return []

    pattern = f"{device_name}-{data_type}-*.csv"
    return sorted(chunks_dir.glob(pattern))


def combine_chunks(
    chunk_files: List[Path],
    output_path: Path,
) -> pd.DataFrame:
    """
    Concatenate chunk files, deduplicate by timestamp_local, and sort.

    Args:
        chunk_files: List of chunk CSV file paths to combine.
        output_path: Path to save the combined CSV file.

    Returns:
        pd.DataFrame: The combined, deduplicated, sorted DataFrame.
    """
    if not chunk_files:
        return pd.DataFrame()

    print(f"    Combining {len(chunk_files)} chunk files...")

    dfs = []
    for chunk_path in chunk_files:
        df = pd.read_csv(chunk_path)
        print(f"      {chunk_path.name}: {len(df)} records")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"    Total records before dedup: {len(combined)}")

    # Deduplicate by timestamp_local (keep first occurrence)
    if "timestamp_local" in combined.columns:
        combined = combined.drop_duplicates(subset=["timestamp_local"], keep="first")
        combined = combined.sort_values("timestamp_local", ascending=True)
        combined = combined.reset_index(drop=True)

    print(f"    Total records after dedup: {len(combined)}")

    # Save combined file
    combined.to_csv(output_path, index=False)
    print(f"    Saved combined file: {output_path.name}")

    return combined


def combine_all_chunks(data_path: Path) -> List[Tuple[Path, Path, str]]:
    """
    Combine all chunk files into per-device, per-type combined files.

    For each device/type combination, concatenates all chunks into a single
    file named {YYYYMMDD}-{device_name}-{data_type}.csv in the main
    data directory.

    Args:
        data_path: Path to the QuantAQ data directory.

    Returns:
        List of (raw_path, final_path, base_name) tuples for downstream
        processing, same format as find_matching_files returns.
    """
    chunks_dir = data_path / "chunks"

    if not chunks_dir.exists():
        print("No chunks/ directory found. Skipping chunk combination.")
        return []

    today_str = datetime.now().strftime("%Y%m%d")
    combined_pairs = {}  # base_name -> {"raw": Path, "final": Path}

    for device_name in DEVICE_NAMES:
        base_name = f"{today_str}-{device_name}"
        combined_pairs[base_name] = {}

        for data_type in CHUNK_DATA_TYPES:
            chunk_files = find_chunk_files(chunks_dir, device_name, data_type)

            if not chunk_files:
                print(f"  No chunks found for {device_name} {data_type}. Skipping.")
                continue

            output_filename = f"{today_str}-{device_name}-{data_type}.csv"
            output_path = data_path / output_filename

            print(f"\n  Combining {device_name} {data_type}:")
            combine_chunks(chunk_files, output_path)

            combined_pairs[base_name][data_type] = output_path

    # Build matching pairs list for downstream processing
    matching_pairs = []
    for base_name, paths in combined_pairs.items():
        if "raw" in paths and "final" in paths:
            matching_pairs.append((paths["raw"], paths["final"], base_name))
        elif "raw" in paths:
            print(f"  Warning: {base_name} has raw but no final chunks.")
        elif "final" in paths:
            print(f"  Warning: {base_name} has final but no raw chunks.")

    return matching_pairs


def parse_dict_string(dict_str: str) -> dict:
    """
    Parse a dictionary string from CSV into a Python dictionary.

    Args:
        dict_str: String representation of a dictionary (e.g., "{'lat': 39.144, 'lon': -77.2016}")

    Returns:
        dict: Parsed dictionary, or empty dict if parsing fails.
    """
    if pd.isna(dict_str) or dict_str == "":
        return {}

    try:
        # Use ast.literal_eval for safe parsing
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return {}


def expand_dict_column(df: pd.DataFrame, col_name: str, prefix: str) -> pd.DataFrame:
    """
    Expand a column containing dictionary strings into multiple columns.

    Args:
        df: DataFrame containing the column to expand.
        col_name: Name of the column containing dictionary strings.
        prefix: Prefix to add to new column names.

    Returns:
        pd.DataFrame: DataFrame with expanded columns (original column removed).
    """
    if col_name not in df.columns:
        return df

    # Parse dictionaries
    parsed_dicts = df[col_name].apply(parse_dict_string)

    # Convert to DataFrame
    expanded_df = pd.DataFrame(parsed_dicts.tolist(), index=df.index)

    # Add prefix to column names
    expanded_df.columns = [f"{prefix}_{col}" for col in expanded_df.columns]

    # Drop original column and join expanded columns
    df = df.drop(columns=[col_name])
    df = pd.concat([df, expanded_df], axis=1)

    return df


def reorder_pm_columns(columns: List[str], prefix: str) -> Tuple[List[str], List[str]]:
    """
    Reorder pm columns within a list to be pm1, pm2.5, pm10.
    Also renames pm25 to pm2.5.

    Args:
        columns: List of column names.
        prefix: The prefix for pm columns (e.g., 'neph', 'opc', 'final').

    Returns:
        Tuple of (non-pm columns, reordered pm column names).
    """
    pm1_col = f"{prefix}_pm1"
    pm25_col = f"{prefix}_pm25"
    pm25_new = f"{prefix}_pm2.5"
    pm10_col = f"{prefix}_pm10"

    result = []
    pm_cols_found = []

    for col in columns:
        if col == pm25_col:
            pm_cols_found.append((1, pm25_new))  # Will be renamed and placed second
        elif col == pm1_col:
            pm_cols_found.append((0, pm1_col))  # First
        elif col == pm10_col:
            pm_cols_found.append((2, pm10_col))  # Third
        else:
            result.append(col)

    # Sort pm columns by order and insert them
    pm_cols_found.sort(key=lambda x: x[0])
    pm_ordered = [col for _, col in pm_cols_found]

    return result, pm_ordered


def organize_expanded_columns(
    df: pd.DataFrame, prefix: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Organize expanded columns: separate bin columns from pm columns,
    rename pm25 to pm2.5, and order pm columns as pm1, pm2.5, pm10.

    Args:
        df: DataFrame with expanded columns.
        prefix: The prefix used for the columns (e.g., 'neph', 'opc').

    Returns:
        Tuple of (DataFrame with renamed columns, list of ordered column names for this prefix).
    """
    prefix_cols = [col for col in df.columns if col.startswith(f"{prefix}_")]

    # Separate into categories
    bin_cols = sorted(
        [col for col in prefix_cols if "_bin" in col],
        key=lambda x: int(m.group(1)) if (m := re.search(r"bin(\d+)", x)) else 0,
    )
    pm_cols = [col for col in prefix_cols if "_pm" in col]
    other_cols = [
        col for col in prefix_cols if col not in bin_cols and col not in pm_cols
    ]

    # Sort other cols (like rh, temp)
    other_cols = sorted(other_cols)

    # Rename pm25 to pm2.5
    rename_map = {}
    pm25_col = f"{prefix}_pm25"
    if pm25_col in df.columns:
        rename_map[pm25_col] = f"{prefix}_pm2.5"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Order pm columns: pm1, pm2.5, pm10
    ordered_pm = []
    for pm_name in ["pm1", "pm2.5", "pm10"]:
        col_name = f"{prefix}_{pm_name}"
        if col_name in df.columns:
            ordered_pm.append(col_name)

    # Final order for this prefix: bins first, then pm (ordered), then other (rh, temp)
    ordered_cols = bin_cols + ordered_pm + other_cols

    return df, ordered_cols


def process_raw_file(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data file.

    Steps:
    1. Drop url column
    2. Expand geo, met, neph, opc dictionary columns
    3. Reorder and rename columns appropriately

    Args:
        raw_df: Raw data DataFrame.

    Returns:
        pd.DataFrame: Processed raw data.
    """
    df = raw_df.copy()

    # Drop url column
    if "url" in df.columns:
        df = df.drop(columns=["url"])

    # Expand dictionary columns
    df = expand_dict_column(df, "geo", "geo")
    df = expand_dict_column(df, "met", "met")
    df = expand_dict_column(df, "neph", "neph")
    df = expand_dict_column(df, "opc", "opc")

    # Organize expanded columns and get ordered column lists
    df, neph_cols = organize_expanded_columns(df, "neph")
    df, opc_cols = organize_expanded_columns(df, "opc")

    # Get geo and met columns (sorted)
    geo_cols = sorted([col for col in df.columns if col.startswith("geo_")])
    met_cols = sorted([col for col in df.columns if col.startswith("met_")])

    # Build final column order:
    # timestamp_local, timestamp, sn, operating_state, flag, geo_*, met_*, neph_*, opc_*
    base_cols = ["timestamp_local", "timestamp", "sn", "operating_state", "flag"]

    # Ensure all base columns exist
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    final_col_order = base_cols + geo_cols + met_cols + neph_cols + opc_cols

    # Only include columns that exist in the DataFrame
    final_col_order = [col for col in final_col_order if col in df.columns]

    # Add any remaining columns not yet included (safety check)
    remaining = [col for col in df.columns if col not in final_col_order]
    final_col_order.extend(remaining)

    df = df[final_col_order]

    return df


def process_final_file(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process final data file to extract calibrated pm values.

    Args:
        final_df: Final data DataFrame.

    Returns:
        pd.DataFrame: Processed final data with timestamp_local and final_pm columns.
    """
    df = final_df.copy()

    # Keep only needed columns
    keep_cols = ["timestamp_local", "pm1", "pm10", "pm25"]
    df = df[[col for col in keep_cols if col in df.columns]]

    # Rename pm columns with final_ prefix and pm25 to pm2.5
    rename_map = {"pm1": "final_pm1", "pm25": "final_pm2.5", "pm10": "final_pm10"}
    df = df.rename(columns=rename_map)

    # Reorder to put pm2.5 between pm1 and pm10
    col_order = ["timestamp_local"]
    for col in ["final_pm1", "final_pm2.5", "final_pm10"]:
        if col in df.columns:
            col_order.append(col)

    df = df[col_order]

    return df


def merge_raw_and_final(raw_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge processed raw and final DataFrames on timestamp_local (outer join).

    Args:
        raw_df: Processed raw DataFrame.
        final_df: Processed final DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Ensure timestamp_local is string type for consistent merging
    raw_df["timestamp_local"] = raw_df["timestamp_local"].astype(str)
    final_df["timestamp_local"] = final_df["timestamp_local"].astype(str)

    # Outer join on timestamp_local
    merged = pd.merge(raw_df, final_df, on="timestamp_local", how="outer")

    # Sort by timestamp_local
    merged = merged.sort_values("timestamp_local").reset_index(drop=True)

    # Reorder columns to put final_pm columns at the end, in correct order
    final_pm_cols = ["final_pm1", "final_pm2.5", "final_pm10"]
    other_cols = [col for col in merged.columns if col not in final_pm_cols]
    final_pm_cols_present = [col for col in final_pm_cols if col in merged.columns]

    merged = merged[other_cols + final_pm_cols_present]

    return merged


def find_matching_files(data_path: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find matching raw and final file pairs in the data directory.

    Args:
        data_path: Path to QuantAQ data directory.

    Returns:
        List of tuples: (raw_file_path, final_file_path, base_name_for_output)
    """
    matching_pairs = []

    # Find all raw files
    raw_files = list(data_path.glob("*-raw.csv"))

    for raw_file in raw_files:
        # Extract the base name (e.g., "20260116-quantaq-inside" from "20260116-quantaq-inside-raw.csv")
        base_name = raw_file.stem.replace("-raw", "")

        # Look for matching final file
        final_file = data_path / f"{base_name}-final.csv"

        if final_file.exists():
            matching_pairs.append((raw_file, final_file, base_name))
        else:
            print(f"Warning: No matching final file found for {raw_file.name}")

    return matching_pairs


def process_file_pair(raw_path: Path, final_path: Path, output_path: Path) -> bool:
    """
    Process a pair of raw and final files into a processed output file.

    Args:
        raw_path: Path to raw CSV file.
        final_path: Path to final CSV file.
        output_path: Path for output processed CSV file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print(f"\n  Reading raw file: {raw_path.name}")
        raw_df = pd.read_csv(raw_path)
        print(f"    Raw records: {len(raw_df)}")

        print(f"  Reading final file: {final_path.name}")
        final_df = pd.read_csv(final_path)
        print(f"    Final records: {len(final_df)}")

        print("  Processing raw data...")
        processed_raw = process_raw_file(raw_df)

        print("  Processing final data...")
        processed_final = process_final_file(final_df)

        print("  Merging raw and final data (outer join)...")
        merged = merge_raw_and_final(processed_raw, processed_final)
        print(f"    Merged records: {len(merged)}")

        print(f"  Saving to: {output_path.name}")
        merged.to_csv(output_path, index=False)

        return True

    except Exception as e:
        print(f"  Error processing files: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main function to combine chunk files and process all QuantAQ data.
    """
    print("=" * 60)
    print("QuantAQ Data Processor")
    print("=" * 60)
    print()

    # Get data path from configuration
    try:
        data_path = get_quantaq_data_path()
        print(f"Data path: {data_path}")
    except Exception as e:
        print(f"Error getting data path from config: {e}")
        print("Using current directory as fallback.")
        data_path = Path.cwd()

    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)

    print()

    # ---- STEP 1: Combine chunk files ----
    print("=" * 60)
    print("STEP 1: Combine chunk files")
    print("=" * 60)

    chunk_pairs = combine_all_chunks(data_path)

    if chunk_pairs:
        print(f"\nCombined {len(chunk_pairs)} device pair(s) from chunks.")
    else:
        print("\nNo chunk files combined.")

    print()

    # ---- STEP 2: Find and process matching file pairs ----
    print("=" * 60)
    print("STEP 2: Process raw/final file pairs")
    print("=" * 60)

    # Use chunk_pairs if available, otherwise fall back to file scan
    if chunk_pairs:
        file_pairs = chunk_pairs
        print(f"Using {len(file_pairs)} pair(s) from chunk combination.")
    else:
        print("Searching for raw/final file pairs...")
        file_pairs = find_matching_files(data_path)

    if not file_pairs:
        print("No matching raw/final file pairs found.")
        print("Looking for files matching pattern: *-raw.csv and *-final.csv")
        print(f"in directory: {data_path}")
        sys.exit(1)

    print(f"Found {len(file_pairs)} file pair(s) to process.")

    # Process each pair
    results = []

    for raw_path, final_path, base_name in file_pairs:
        print("-" * 60)
        print(f"Processing: {base_name}")
        print("-" * 60)

        output_path = data_path / f"{base_name}-processed.csv"

        success = process_file_pair(raw_path, final_path, output_path)

        results.append(
            {
                "base_name": base_name,
                "success": success,
                "output_file": output_path.name if success else None,
            }
        )

    # Summary
    print()
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    for r in results:
        status = f"SUCCESS -> {r['output_file']}" if r["success"] else "FAILED"
        print(f"  {r['base_name']}: {status}")

    successful = sum(1 for r in results if r["success"])
    print()
    print(f"Processed {successful}/{len(results)} file pairs successfully.")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
