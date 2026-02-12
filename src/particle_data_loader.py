#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Data Loading & Event Identification
=============================================

Data loading and event identification functions for the particle decay analysis.
Handles loading QuantAQ particle data, shower logs, CO2 lambda results, and
identifying shower events from log files and the unified event registry.

Key Functions:
    - load_quantaq_data: Load QuantAQ CSV files for a location
    - load_and_merge_quantaq_data: Merge inside/outside particle data
    - load_shower_log: Load shower state-change log
    - load_co2_lambda_results: Load CO2 analysis results for lambda values
    - identify_shower_events: Identify shower events from log file
    - get_events_from_registry: Load events from unified event registry

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.particle_calculations import (
    DEPOSITION_WINDOW_HOURS,
    PARTICLE_BINS,
    ROLLING_WINDOW_MIN,
)


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_quantaq_data(location: str) -> pd.DataFrame:
    """
    Load all QuantAQ processed CSV files for a location (inside or outside).

    Parameters:
        location (str): 'inside' or 'outside'

    Returns:
        pd.DataFrame: Combined DataFrame with datetime index and particle bins
    """
    from src.data_paths import get_instrument_path

    quantaq_path = get_instrument_path("QuantAQ_MODULAIR_PM")
    file_pattern = f"*-quantaq-{location}-processed.csv"

    files = sorted(quantaq_path.glob(file_pattern))

    if not files:
        raise FileNotFoundError(
            f"No QuantAQ {location} files found with pattern {file_pattern} in {quantaq_path}"
        )

    print(f"  Found {len(files)} file(s) for QuantAQ {location}")

    all_data = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            # Parse datetime
            if "timestamp_local" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp_local"])
            elif "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"])
            else:
                print(f"    Warning: No timestamp column in {filepath.name}")
                continue

            # Extract only particle bin columns
            bin_cols = ["datetime"] + [
                PARTICLE_BINS[i]["column"] for i in PARTICLE_BINS
            ]
            available_cols = [col for col in bin_cols if col in df.columns]
            df = df[available_cols]

            all_data.append(df)
            print(f"    Loaded: {filepath.name} ({len(df)} rows)")
        except Exception as e:
            print(f"    Error loading {filepath.name}: {str(e)[:100]}")

    if not all_data:
        raise ValueError(f"No valid QuantAQ {location} data could be loaded")

    # Combine and remove duplicates
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    combined = combined.reset_index(drop=True)

    return combined


def load_and_merge_quantaq_data() -> pd.DataFrame:
    """
    Load and merge QuantAQ inside and outside data into a single DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns for inside and outside particle bins
    """
    print("\nLoading QuantAQ particle data...")

    inside_data = load_quantaq_data("inside")
    outside_data = load_quantaq_data("outside")

    # Rename columns to distinguish inside vs outside
    for bin_num, bin_info in PARTICLE_BINS.items():
        col = bin_info["column"]
        if col in inside_data.columns:
            inside_data = inside_data.rename(columns={col: f"{col}_inside"})
        if col in outside_data.columns:
            outside_data = outside_data.rename(columns={col: f"{col}_outside"})

    # Merge on datetime
    merged = pd.merge(inside_data, outside_data, on="datetime", how="outer")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Set datetime as index for resampling
    merged = merged.set_index("datetime")

    # Resample to regular 1-minute intervals and interpolate short gaps
    merged = merged.resample("1min").mean()
    merged = merged.interpolate(method="linear", limit=5)

    # Apply rolling average if configured
    if ROLLING_WINDOW_MIN > 0:
        print(f"  Applying {ROLLING_WINDOW_MIN}-minute rolling average")
        for col in merged.columns:
            merged[col] = (
                merged[col]
                .rolling(window=ROLLING_WINDOW_MIN, center=True, min_periods=1)
                .mean()
            )

    merged = merged.reset_index()

    print(f"\nMerged data: {len(merged)} rows")
    print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

    return merged


def load_shower_log() -> pd.DataFrame:
    """
    Load the shower state-change log.

    Returns:
        pd.DataFrame: DataFrame with shower events
    """
    from src.data_paths import get_common_file

    log_path = get_common_file("shower_log_file")

    if not log_path.exists():
        raise FileNotFoundError(
            f"Shower log file not found: {log_path}\n"
            "Run scripts/process_shower_log.py first to generate this file."
        )

    df = pd.read_csv(log_path)
    df["datetime_EDT"] = pd.to_datetime(df["datetime_EDT"])

    return df


def load_co2_lambda_results() -> pd.DataFrame:
    """
    Load CO2 analysis results to get air change rates (lambda) for each event.

    Returns:
        pd.DataFrame: DataFrame with lambda values per event

    Raises:
        FileNotFoundError: If CO2 analysis results are not found
    """
    from src.data_paths import get_data_root

    output_dir = get_data_root() / "output"
    co2_results_path = output_dir / "co2_lambda_summary.csv"

    if not co2_results_path.exists():
        print("\n" + "=" * 80)
        print("DEPENDENCY REQUIRED")
        print("=" * 80)
        print(f"\n[ERROR] CO2 lambda results not found: {co2_results_path}")
        print("\nRequired Dependency:")
        print(
            "   The particle analysis requires air change rates (lambda) from CO2 analysis."
        )
        print("\nAction Required:")
        print("   Run the CO2 decay analysis first:")
        print("   python src/co2_decay_analysis.py")
        print("\nAnalysis Order:")
        print("   1. co2_decay_analysis.py     (provides lambda values)")
        print("   2. rh_temp_other_analysis.py  (optional)")
        print("   3. particle_decay_analysis.py (requires lambda from step 1)")
        print("\n" + "=" * 80 + "\n")
        raise FileNotFoundError(
            "CO2 lambda results not found. Please run co2_decay_analysis.py first."
        )

    df = pd.read_csv(co2_results_path)
    df["decay_start"] = pd.to_datetime(df["decay_start"])
    df["injection_start"] = pd.to_datetime(df["injection_start"])

    # Handle column names with units (rename back to internal names for processing)
    column_rename = {
        "decay_duration (h)": "decay_duration_hours",
        "c_bedroom_initial (ppm)": "c_bedroom_initial",
        "c_bedroom_final (ppm)": "c_bedroom_final",
        "c_source_mean (ppm)": "c_source_mean",
        "c_outside_mean (ppm)": "c_outside_mean",
        "c_entry_mean (ppm)": "c_entry_mean",
    }
    for mode in ["average", "outside", "entry"]:
        column_rename[f"lambda_{mode}_mean (h-1)"] = f"lambda_{mode}_mean"
        column_rename[f"lambda_{mode}_std (h-1)"] = f"lambda_{mode}_std"
    df = df.rename(columns=column_rename)

    print(f"\nLoaded CO2 lambda results: {len(df)} events")

    return df


# =============================================================================
# Event Identification Functions
# =============================================================================


def identify_shower_events(shower_log: pd.DataFrame) -> List[Dict]:
    """
    Identify shower events from the log file.

    A shower event is identified when the shower valve turns ON (value = 1.0).

    Parameters:
        shower_log (pd.DataFrame): DataFrame with shower log

    Returns:
        List[Dict]: List of dicts with shower event details
    """
    events = []
    shower_log = shower_log.sort_values("datetime_EDT").reset_index(drop=True)

    event_number = 0

    for i in range(len(shower_log) - 1):
        current_row = shower_log.iloc[i]
        next_row = shower_log.iloc[i + 1]

        # Check for shower turning ON
        if current_row["shower"] == 0 and next_row["shower"] > 0:
            event_number += 1
            shower_on = next_row["datetime_EDT"]

            # Find when shower turns OFF
            shower_off = None
            for j in range(i + 2, min(i + 30, len(shower_log))):
                row = shower_log.iloc[j]
                if row["shower"] == 0:
                    shower_off = row["datetime_EDT"]
                    break

            if shower_off is None:
                shower_off = shower_on + timedelta(minutes=10)

            # Calculate analysis windows
            deposition_start = shower_off
            deposition_end = shower_off + timedelta(hours=DEPOSITION_WINDOW_HOURS)

            events.append(
                {
                    "event_number": event_number,
                    "shower_on": shower_on,
                    "shower_off": shower_off,
                    "shower_duration_min": (shower_off - shower_on).total_seconds()
                    / 60,
                    "deposition_start": deposition_start,
                    "deposition_end": deposition_end,
                }
            )

    return events


def get_events_from_registry(output_dir: Path) -> tuple:
    """
    Try to load shower events from the unified event registry.

    If registry exists, use it for consistent event numbering across all scripts.
    The registry also provides lambda values from CO2 analysis.

    Parameters:
        output_dir: Output directory where registry is stored

    Returns:
        Tuple of (events_list, co2_results_df, used_registry: bool)
    """
    from scripts.event_registry import REGISTRY_FILENAME, load_event_registry

    registry_path = output_dir / REGISTRY_FILENAME

    if not registry_path.exists():
        return [], pd.DataFrame(), False

    try:
        print(f"\nLoading events from registry: {registry_path}")
        registry_df = load_event_registry(registry_path)

        events = []
        for _, row in registry_df.iterrows():
            # Include all events (excluded ones will be skipped in analysis)
            events.append(
                {
                    "event_number": int(row["event_number"]),
                    "test_name": row["test_name"],
                    "config_key": f"{row.get('water_temp', '')}_{row.get('door_position', 'Open')}_FanOff",
                    "water_temp": row.get("water_temp", ""),
                    "door_position": row.get("door_position", "Open"),
                    "time_of_day": row.get("time_of_day", ""),
                    "fan_during_test": row.get("fan_during_test", False),
                    "replicate_num": row.get("replicate_num", 0),
                    "shower_on": pd.to_datetime(row["shower_on"]),
                    "shower_off": pd.to_datetime(row["shower_off"]),
                    "shower_duration_min": row.get("shower_duration_min", 0),
                    "lambda_ach": row.get("lambda_average_mean", np.nan),
                    "co2_event_idx": None,  # Not needed when using registry
                    "deposition_start": pd.to_datetime(row.get("deposition_start"))
                    if pd.notna(row.get("deposition_start"))
                    else None,
                    "deposition_end": pd.to_datetime(row.get("deposition_end"))
                    if pd.notna(row.get("deposition_end"))
                    else None,
                    "is_excluded": row.get("is_excluded", False),
                    "exclusion_reason": row.get("exclusion_reason", ""),
                }
            )

        # Create CO2 results DataFrame from registry for compatibility
        co2_results_df = registry_df[
            ["event_number", "co2_injection_start", "lambda_average_mean"]
        ].copy()
        co2_results_df = co2_results_df.rename(
            columns={"co2_injection_start": "injection_start"}
        )

        print(f"  Loaded {len(events)} events from registry")
        n_with_lambda = sum(
            1 for e in events if not np.isnan(e.get("lambda_ach", np.nan))
        )
        print(f"  Events with lambda values: {n_with_lambda}")

        return events, co2_results_df, True

    except Exception as e:
        print(f"  Warning: Could not load registry: {e}")
        return [], pd.DataFrame(), False