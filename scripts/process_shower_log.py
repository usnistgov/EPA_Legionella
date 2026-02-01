#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shower Program Log Processor
============================

This script consolidates daily 1-second shower program log files into a single
state-change log. It reduces millions of redundant records to only the
timestamps when valve states actually change, enabling efficient event analysis.

Key Functions:
    - parse_mixed_delimiter_file: Handle mixed tab/comma delimited input
    - process_shower_logs: Main processing workflow

Processing Features:
    - Parses mixed delimiter format (tab after datetime, commas between values)
    - Detects state changes in shower valve and bathroom fan
    - Filters out redundant consecutive identical states
    - Renames columns for consistency (Shower -> shower, Fan -> bath_fan)
    - Excludes backup files created by fix_log_files.py

Methodology:
    1. Locate all MH_ShowerProgram_*.txt files in input directory
    2. Parse each file handling mixed delimiter format
    3. Combine and sort all records chronologically
    4. Detect state changes by comparing each row to previous
    5. Output only rows where shower or bath_fan changed

Input Files:
    - MH_ShowerProgram_YYYYMMDD.txt (daily 1-second logs)
    - Format: datetime_EDT<tab>Shower,Fan,channel2
    - Located in: data_root/Log - Shower/

Output Files:
    - shower_log_file.csv: State-change log with columns:
        - datetime_EDT: Timestamp of state change
        - shower: Shower valve state (0=off, 1=on)
        - bath_fan: Bathroom fan state (0=off, 1=on)

Data Reduction:
    - Typical reduction: ~86,400 records/day -> ~4-8 state changes/day
    - Enables efficient event matching with CO2 injection log

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_paths import get_common_file, get_common_file_config, get_data_root


def parse_mixed_delimiter_file(filepath):
    """
    Parse files with mixed delimiters (tab after datetime, commas between values).

    Format: datetime_EDT,Shower,Fan,channel2 (header)
            1/12/2026 12:00:00 AM\t0.000000,0.000000,0.000000 (data)
    """
    rows = []

    with open(filepath, "r") as f:
        # Read header line
        header_line = f.readline().strip()
        # Header is comma-separated
        headers = [h.strip() for h in header_line.split(",")]

        # Read data lines
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by tab first to separate datetime from values
            parts = line.split("\t")
            if len(parts) >= 2:
                datetime_str = parts[0]
                # Values are comma-separated
                values = parts[1].split(",")
                row = [datetime_str] + values
                rows.append(row)

    df = pd.DataFrame(rows, columns=headers[: len(rows[0])] if rows else headers)
    return df


def process_shower_logs(input_dir, output_file):
    """
    Process all shower program log files and create a state-change log.

    Args:
        input_dir: Directory containing MH_ShowerProgram_*.txt files
        output_file: Path for output CSV file
    """
    input_path = Path(input_dir)

    # Find all daily log files (exclude backup files created by fix_log_files.py)
    pattern = str(input_path / "MH_ShowerProgram_*.txt")
    files = sorted([f for f in glob.glob(pattern) if "_backup" not in os.path.basename(f)])

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} shower log files to process")

    all_data = []

    for filepath in files:
        print(f"  Processing: {os.path.basename(filepath)}")
        try:
            df = parse_mixed_delimiter_file(filepath)
            all_data.append(df)
        except Exception as e:
            print(f"    Error processing {filepath}: {e}")
            continue

    if not all_data:
        print("No data was successfully loaded")
        return

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Parse datetime
    combined["datetime_EDT"] = pd.to_datetime(
        combined["datetime_EDT"], format="%m/%d/%Y %I:%M:%S %p"
    )

    # Sort by datetime
    combined = combined.sort_values("datetime_EDT").reset_index(drop=True)

    # Convert value columns to numeric
    combined["Shower"] = pd.to_numeric(combined["Shower"], errors="coerce")
    combined["Fan"] = pd.to_numeric(combined["Fan"], errors="coerce")

    # Rename columns: Shower -> shower, Fan -> bath_fan, drop channel2
    combined = combined.rename(columns={"Shower": "shower", "Fan": "bath_fan"})
    combined = combined[["datetime_EDT", "shower", "bath_fan"]]

    # Detect state changes
    # A state change occurs when shower or bath_fan value differs from previous row
    combined["shower_changed"] = combined["shower"] != combined["shower"].shift(1)
    combined["fan_changed"] = combined["bath_fan"] != combined["bath_fan"].shift(1)
    combined["state_changed"] = combined["shower_changed"] | combined["fan_changed"]

    # Keep only rows where state changed (always keep first row)
    combined.loc[0, "state_changed"] = True
    state_changes = combined[combined["state_changed"]].copy()

    # Drop helper columns
    state_changes = state_changes[["datetime_EDT", "shower", "bath_fan"]]

    print(f"\nOriginal records: {len(combined):,}")
    print(f"State changes: {len(state_changes):,}")

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    state_changes.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")


def main():
    # Get paths from data_config.json via data_paths module
    shower_config = get_common_file_config("shower_log_file")
    data_root = get_data_root()

    input_dir = data_root / shower_config["path"]
    output_file = get_common_file("shower_log_file")

    print("Shower Program Log Processor")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    process_shower_logs(input_dir, output_file)


if __name__ == "__main__":
    main()
