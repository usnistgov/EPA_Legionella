"""
Process CO2 injection log files into a consolidated state-change log.

This script reads daily 1-second CO2 injection log files and produces a single
continuous file containing only the timestamps when valve states change.

Input files: MH_CO2andFanProgram_YYYYMMDD.txt
Output file: CO2_log_file.csv

Input columns: datetime_EDT, CO2, Fan, notAssigned
Output columns: datetime_EDT, CO2, mixing_fan

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
from src.data_paths import get_data_root, get_common_file, get_common_file_config


def parse_mixed_delimiter_file(filepath):
    """
    Parse files with mixed delimiters (tab after datetime, commas between values).

    Format: datetime_EDT,CO2,Fan,notAssigned (header)
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


def process_co2_logs(input_dir, output_file):
    """
    Process all CO2 injection log files and create a state-change log.

    Args:
        input_dir: Directory containing MH_CO2andFanProgram_*.txt files
        output_file: Path for output CSV file
    """
    input_path = Path(input_dir)

    # Find all daily log files
    pattern = str(input_path / "MH_CO2andFanProgram_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found {len(files)} CO2 log files to process")

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
    combined["CO2"] = pd.to_numeric(combined["CO2"], errors="coerce")
    combined["Fan"] = pd.to_numeric(combined["Fan"], errors="coerce")

    # Rename Fan to mixing_fan, drop notAssigned
    combined = combined.rename(columns={"Fan": "mixing_fan"})
    combined = combined[["datetime_EDT", "CO2", "mixing_fan"]]

    # Detect state changes
    # A state change occurs when CO2 or mixing_fan value differs from previous row
    combined["co2_changed"] = combined["CO2"] != combined["CO2"].shift(1)
    combined["fan_changed"] = combined["mixing_fan"] != combined["mixing_fan"].shift(1)
    combined["state_changed"] = combined["co2_changed"] | combined["fan_changed"]

    # Keep only rows where state changed (always keep first row)
    combined.loc[0, "state_changed"] = True
    state_changes = combined[combined["state_changed"]].copy()

    # Drop helper columns
    state_changes = state_changes[["datetime_EDT", "CO2", "mixing_fan"]]

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
    co2_config = get_common_file_config("co2_log_file")
    data_root = get_data_root()

    input_dir = data_root / co2_config["path"]
    output_file = get_common_file("co2_log_file")

    print("CO2 Injection Log Processor")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    process_co2_logs(input_dir, output_file)


if __name__ == "__main__":
    main()
