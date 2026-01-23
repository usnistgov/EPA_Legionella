#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Corrupted CO2 and Shower Log Files
=======================================

This utility script repairs corrupted or empty daily log files for the CO2 injection
and shower program systems. It uses a known good reference file as a template and
generates corrected versions of the target files with properly adjusted dates.

The script handles files with mixed delimiters where the header is comma-separated
but data rows use tab after the datetime and commas between values.

Key Functions:
    - Read and parse mixed-delimiter log files
    - Create backups of target files before modification
    - Adjust datetime values by the appropriate day offset
    - Write corrected files maintaining the original format

Processing Features:
    - Supports both CO2 log files (MH_CO2andFanProgram_YYYYMMDD.txt)
    - Supports shower log files (MH_ShowerProgram_YYYYMMDD.txt)
    - Interactive mode with defaults for current problem files
    - Automatic backup creation with _backup suffix
    - Date increment based on file date difference from reference

Methodology:
    1. Load configuration from data_config.json for file paths
    2. Prompt user for reference file and target files (with defaults)
    3. Parse reference file to extract header and data rows
    4. For each target file:
       a. Create backup of existing file
       b. Calculate day offset from reference date
       c. Adjust all datetime values by offset
       d. Write corrected file with same format

Output Files:
    - Backups: Original files renamed with _backup suffix (e.g., MH_CO2andFanProgram_20260118_backup.txt)
    - Fixed files: Corrected files replace the originals

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_paths import get_common_file_config, get_data_root

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default reference and target dates (January 2026)
DEFAULT_REFERENCE_DATE = "20260116"
DEFAULT_TARGET_DATES = [
    "20260117",
    "20260118",
    "20260119",
    "20260120",
    "20260121",
    "20260122",
]

# Log file configurations
LOG_CONFIGS = {
    "co2": {
        "name": "CO2 Injection",
        "config_key": "co2_log_file",
        "file_prefix": "MH_CO2andFanProgram_",
        "file_suffix": ".txt",
    },
    "shower": {
        "name": "Shower Program",
        "config_key": "shower_log_file",
        "file_prefix": "MH_ShowerProgram_",
        "file_suffix": ".txt",
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def parse_date_from_filename(filename, prefix, suffix):
    """
    Extract date string from filename.

    Parameters:
        filename (str): The filename to parse
        prefix (str): File prefix before date (e.g., 'MH_CO2andFanProgram_')
        suffix (str): File suffix after date (e.g., '.txt')

    Returns:
        str: Date string in YYYYMMDD format
    """
    basename = os.path.basename(filename)
    date_str = basename.replace(prefix, "").replace(suffix, "")
    return date_str


def parse_date_string(date_str):
    """
    Parse YYYYMMDD date string to datetime.date object.

    Parameters:
        date_str (str): Date in YYYYMMDD format

    Returns:
        datetime.date: Parsed date object
    """
    return datetime.strptime(date_str, "%Y%m%d").date()


def calculate_day_offset(reference_date_str, target_date_str):
    """
    Calculate the number of days between reference and target dates.

    Parameters:
        reference_date_str (str): Reference date in YYYYMMDD format
        target_date_str (str): Target date in YYYYMMDD format

    Returns:
        int: Number of days to add (target - reference)
    """
    ref_date = parse_date_string(reference_date_str)
    target_date = parse_date_string(target_date_str)
    return (target_date - ref_date).days


def adjust_datetime_in_line(datetime_str, day_offset):
    """
    Adjust a datetime string by the specified number of days.

    Parameters:
        datetime_str (str): Datetime in 'M/D/YYYY H:MM:SS AM/PM' format
        day_offset (int): Number of days to add

    Returns:
        str: Adjusted datetime string in same format
    """
    # Parse the datetime
    dt = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p")

    # Add the day offset
    adjusted_dt = dt + timedelta(days=day_offset)

    # Format back to original format (with leading zeros for single-digit month/day)
    # The original format uses no leading zeros: 1/12/2026 not 01/12/2026
    return adjusted_dt.strftime("%-m/%-d/%Y %I:%M:%S %p")


def adjust_datetime_in_line_windows(datetime_str, day_offset):
    """
    Adjust a datetime string by the specified number of days (Windows compatible).

    Windows uses # instead of - for removing leading zeros in strftime.

    Parameters:
        datetime_str (str): Datetime in 'M/D/YYYY H:MM:SS AM/PM' format
        day_offset (int): Number of days to add

    Returns:
        str: Adjusted datetime string in same format
    """
    # Parse the datetime
    dt = datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p")

    # Add the day offset
    adjusted_dt = dt + timedelta(days=day_offset)

    # Format back - Windows uses %#m and %#d to remove leading zeros
    # For cross-platform, we'll use regular format and strip zeros manually
    month = adjusted_dt.month
    day = adjusted_dt.day
    year = adjusted_dt.year
    time_str = adjusted_dt.strftime("%I:%M:%S %p")

    return f"{month}/{day}/{year} {time_str}"


def read_log_file(filepath):
    """
    Read a log file and return header and data lines.

    Parameters:
        filepath (Path): Path to the log file

    Returns:
        tuple: (header_line, data_lines) where data_lines is a list of strings
    """
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        data_lines = [line.strip() for line in f if line.strip()]
    return header, data_lines


def write_log_file(filepath, header, data_lines):
    """
    Write a log file with header and data lines.

    Parameters:
        filepath (Path): Path to write the file
        header (str): Header line
        data_lines (list): List of data line strings
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in data_lines:
            f.write(line + "\n")


def create_backup(filepath):
    """
    Create a backup of a file with _backup suffix.

    Parameters:
        filepath (Path): Path to the file to backup

    Returns:
        Path: Path to the backup file
    """
    backup_path = filepath.parent / (filepath.stem + "_backup" + filepath.suffix)
    if filepath.exists():
        shutil.copy2(filepath, backup_path)
    return backup_path


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================


def process_log_file(reference_path, target_path, reference_date_str, target_date_str):
    """
    Process a single target log file using the reference file.

    Parameters:
        reference_path (Path): Path to the reference (good) file
        target_path (Path): Path to the target (bad) file to fix
        reference_date_str (str): Reference date in YYYYMMDD format
        target_date_str (str): Target date in YYYYMMDD format

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Calculate day offset
        day_offset = calculate_day_offset(reference_date_str, target_date_str)
        print(f"    Day offset: +{day_offset} days")

        # Read reference file
        header, data_lines = read_log_file(reference_path)
        print(f"    Reference file has {len(data_lines):,} data rows")

        # Create backup of target file
        backup_path = create_backup(target_path)
        if target_path.exists():
            print(f"    Backup created: {backup_path.name}")

        # Adjust datetime in each data line
        adjusted_lines = []
        for line in data_lines:
            # Split by tab to get datetime and values
            parts = line.split("\t")
            if len(parts) >= 2:
                original_datetime = parts[0]
                values = parts[1]

                # Adjust the datetime
                adjusted_datetime = adjust_datetime_in_line_windows(
                    original_datetime, day_offset
                )

                # Reconstruct the line
                adjusted_line = adjusted_datetime + "\t" + values
                adjusted_lines.append(adjusted_line)
            else:
                # Keep line as-is if format is unexpected
                adjusted_lines.append(line)

        # Write the fixed file
        write_log_file(target_path, header, adjusted_lines)
        print(f"    Fixed file written: {target_path.name}")
        print(f"    Rows written: {len(adjusted_lines):,}")

        return True

    except Exception as e:
        print(f"    ERROR: {str(e)[:100]}")
        return False


def process_log_type(log_type, reference_date_str, target_date_strs):
    """
    Process all target files for a specific log type (CO2 or Shower).

    Parameters:
        log_type (str): Type of log ('co2' or 'shower')
        reference_date_str (str): Reference date in YYYYMMDD format
        target_date_strs (list): List of target dates in YYYYMMDD format

    Returns:
        tuple: (success_count, failure_count)
    """
    config = LOG_CONFIGS[log_type]
    print(f"\n{'=' * 60}")
    print(f"Processing {config['name']} Log Files")
    print(f"{'=' * 60}")

    # Get directory path from data_config.json
    try:
        file_config = get_common_file_config(config["config_key"])
        data_root = get_data_root()
        log_dir = data_root / file_config["path"]
    except Exception as e:
        print(f"ERROR: Could not load configuration: {e}")
        return 0, len(target_date_strs)

    print(f"Directory: {log_dir}")

    if not log_dir.exists():
        print(f"ERROR: Directory does not exist: {log_dir}")
        return 0, len(target_date_strs)

    # Build reference file path
    reference_filename = (
        f"{config['file_prefix']}{reference_date_str}{config['file_suffix']}"
    )
    reference_path = log_dir / reference_filename

    print(f"Reference file: {reference_filename}")

    if not reference_path.exists():
        print(f"ERROR: Reference file does not exist: {reference_path}")
        return 0, len(target_date_strs)

    # Check reference file is not empty
    if reference_path.stat().st_size == 0:
        print(f"ERROR: Reference file is empty: {reference_path}")
        return 0, len(target_date_strs)

    success_count = 0
    failure_count = 0

    # Process each target file
    for target_date_str in target_date_strs:
        target_filename = (
            f"{config['file_prefix']}{target_date_str}{config['file_suffix']}"
        )
        target_path = log_dir / target_filename

        print(f"\n  Processing: {target_filename}")

        if process_log_file(
            reference_path, target_path, reference_date_str, target_date_str
        ):
            success_count += 1
        else:
            failure_count += 1

    return success_count, failure_count


def get_user_input():
    """
    Prompt user for reference date and target dates.

    Returns:
        tuple: (reference_date_str, target_date_list)
    """
    print("Log File Repair Utility")
    print("=" * 60)
    print()
    print("This script fixes empty/corrupted log files by copying data from")
    print("a reference file and adjusting the dates appropriately.")
    print()
    print(f"Default reference date: {DEFAULT_REFERENCE_DATE}")
    print(f"Default target dates: {', '.join(DEFAULT_TARGET_DATES)}")
    print()

    # Get reference date
    reference_input = input(
        f"Enter reference date (YYYYMMDD) or press Enter for default [{DEFAULT_REFERENCE_DATE}]: "
    ).strip()
    reference_date = reference_input if reference_input else DEFAULT_REFERENCE_DATE

    # Validate reference date format
    try:
        parse_date_string(reference_date)
    except ValueError:
        print(f"ERROR: Invalid date format '{reference_date}'. Using default.")
        reference_date = DEFAULT_REFERENCE_DATE

    # Get target dates
    targets_input = input(
        "Enter target dates (comma-separated YYYYMMDD) or press Enter for defaults: "
    ).strip()

    if targets_input:
        # Parse comma-separated dates
        target_dates = [d.strip() for d in targets_input.split(",")]
        # Validate each date
        valid_targets = []
        for date_str in target_dates:
            try:
                parse_date_string(date_str)
                valid_targets.append(date_str)
            except ValueError:
                print(f"  WARNING: Skipping invalid date '{date_str}'")
        target_dates = valid_targets if valid_targets else DEFAULT_TARGET_DATES
    else:
        target_dates = DEFAULT_TARGET_DATES

    print()
    print(f"Reference date: {reference_date}")
    print(f"Target dates: {', '.join(target_dates)}")
    print()

    # Confirm
    confirm = input("Proceed with these settings? (y/n) [y]: ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted by user.")
        sys.exit(0)

    return reference_date, target_dates


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main entry point for the log file repair utility."""
    # Get user input
    reference_date, target_dates = get_user_input()

    total_success = 0
    total_failure = 0

    # Process both log types
    for log_type in ["co2", "shower"]:
        success, failure = process_log_type(log_type, reference_date, target_dates)
        total_success += success
        total_failure += failure

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_success + total_failure}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failure}")

    if total_failure > 0:
        print("\nSome files failed to process. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll files processed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
