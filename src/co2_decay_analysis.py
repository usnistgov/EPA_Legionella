#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2 Decay & Air-Change Rate (λ) Analysis
=========================================

This script analyzes CO2 decay data from Aranet4 sensors to calculate the
air-change rate (λ) for the EPA Legionella study manufactured home test
facility. The analysis uses an analytical approach with linear regression
to solve the mass balance equation for ventilation rate.

Key Metrics Calculated:
    - λ (air-change rate): Rate of air exchange with outdoors (h⁻¹)
    - λ_average: Using average source concentration (C_outside + C_entry)/2
    - λ_outside: Using only outdoor concentration as source
    - λ_entry: Using only entry zone concentration as source
    - R² values: Goodness of fit for each linear regression

Analysis Features:
    - Combines three Aranet4 data files: Bedroom, Entry, and Outside
    - 6-minute rolling average applied to reduce sensor noise
    - Fixed 2-hour decay analysis window starting 10 minutes before the hour
    - Three source concentration methods for uncertainty assessment
    - Linear regression to determine λ from transformed decay equation

Methodology:
    The mass balance equation for a well-mixed zone:
        V ∂C/∂t = Q_out·C_out + Q_Entry·C_Entry - Q·C

    Assuming 50% flow from outside and 50% from entry zone:
        ∂C/∂t = λ(C_average,Out&Entry - C_t)

    Analytical solution (integrated form):
        -ln[(C(t) - C_avg) / (C_0 - C_avg)] = λ·t

    Where C_avg is the mean of C_outside and C_entry over the decay window,
    and λ is determined by linear regression (slope of y vs t).

    1. Load and merge CO2 data from three Aranet4 sensors (Bedroom, Entry, Outside)
    2. Resample to 1-minute intervals and apply rolling average smoothing
    3. Identify CO2 injection events from the state-change log
    4. For each event, extract 2-hour decay window starting at :50
    5. Calculate y = -ln[(C(t) - C_avg) / (C_0 - C_avg)] for each timestep
    6. Perform linear regression of y vs t to obtain λ (slope)
    7. Repeat with three source concentration methods for uncertainty
    8. Generate diagnostic plots and summary statistics

Output Files:
    - co2_lambda_summary.csv: Per-event results with all λ calculations
    - co2_lambda_overall_summary.csv: Aggregated statistics across all events
    - plots/event_XX_co2_decay.png: Individual decay plots with fitted curves
    - plots/lambda_summary.png: Summary bar chart of λ values

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_manager import (  # noqa: E402
    EXPERIMENT_START_DATE,
    is_event_excluded,
)
from scripts.plot_utils import (  # noqa: E402
    plot_co2_decay_event_analytical,
    plot_lambda_summary,
)
from src.data_paths import (  # noqa: E402
    get_common_file,
    get_data_root,
    get_instrument_config,
    get_instrument_path,
)

# =============================================================================
# Configuration
# =============================================================================

# Default flow fraction parameters (α from outside, β from entry zone)
DEFAULT_ALPHA = 0.5  # Fraction from outside
DEFAULT_BETA = 0.5  # Fraction from entry zone (α + β = 1)

# Analysis timing parameters (minutes)
DECAY_START_OFFSET_MIN = -10  # Start 10 min before the hour (at :50)
DECAY_DURATION_HOURS = 2  # Default decay analysis duration (hours)

# Custom decay durations for specific events (event_number: duration_hours)
# Event numbers are 1-indexed (i.e., the first event is event 1)
CUSTOM_DECAY_DURATIONS = {
    5: 1.75,  # Event 5 uses 1.75 hours instead of default 2 hours
    # Add more custom durations as needed:
    # 7: 1.5,
    # 10: 2.5,
}

# Rolling average parameters
ROLLING_WINDOW_MIN = 6  # Rolling average window (minutes)

# Minimum concentration difference for valid analysis (ppm)
MIN_CONCENTRATION_DIFF = 50  # C_bedroom - C_source must exceed this


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_aranet_file(filepath: Path) -> pd.DataFrame:
    """
    Load an Aranet4 Excel file and parse the datetime column.

    Parameters:
        filepath (Path): Path to the Aranet4 Excel file

    Returns:
        pd.DataFrame: DataFrame with parsed datetime index
    """
    df = pd.read_excel(filepath)

    # Rename the datetime column
    datetime_col = "Time(DD/MM/YYYY h:mm:ss A)"
    if datetime_col in df.columns:
        df = df.rename(columns={datetime_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %I:%M:%S %p")
    else:
        # Try alternative formats
        for col in df.columns:
            if "time" in col.lower():
                df = df.rename(columns={col: "datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"])
                break

    return df


def load_all_aranet_files(location: str) -> pd.DataFrame:
    """
    Load all Aranet4 files for a specific location and combine them.

    Parameters:
        location (str): Sensor location ('Bedroom', 'Entry', or 'Mh Outside')

    Returns:
        pd.DataFrame: Combined DataFrame with all data for the location
    """
    config = get_instrument_config("Aranet4")
    base_path = get_instrument_path("Aranet4")

    # Get the file pattern for this location
    location_config = config["locations"].get(location, {})
    file_pattern = location_config.get("file_pattern", f"{location}_*_all.xlsx")

    # Find all matching files
    files = sorted(base_path.glob(file_pattern))

    if not files:
        print(f"  Warning: No files found for {location} with pattern {file_pattern}")
        return pd.DataFrame()

    print(f"  Found {len(files)} file(s) for {location}")

    all_data = []
    for filepath in files:
        try:
            df = load_aranet_file(filepath)
            all_data.append(df)
            print(f"    Loaded: {filepath.name} ({len(df)} rows)")
        except Exception as e:
            print(f"    Error loading {filepath.name}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Combine and remove duplicates
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    combined = combined.reset_index(drop=True)

    return combined


def load_and_merge_co2_data() -> pd.DataFrame:
    """
    Load CO2 data from all three Aranet4 sensors and merge into a single DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns: datetime, C_bedroom, C_entry, C_outside,
        T_bedroom, T_entry, T_outside, RH_bedroom, RH_entry, RH_outside
    """
    print("\nLoading Aranet4 CO2 data...")

    # Load data for each location
    locations = {
        "Bedroom": "bedroom",
        "Entry": "entry",
        "Mh Outside": "outside",
    }

    dfs = {}
    for loc_name, suffix in locations.items():
        df = load_all_aranet_files(loc_name)
        if not df.empty:
            # Rename columns with location suffix
            df = df.rename(
                columns={
                    "Carbon dioxide(ppm)": f"C_{suffix}",
                    "Temperature(°C)": f"T_{suffix}",
                    "Relative humidity(%)": f"RH_{suffix}",
                    "Atmospheric pressure(hPa)": f"P_{suffix}",
                }
            )
            dfs[suffix] = df[["datetime", f"C_{suffix}", f"T_{suffix}", f"RH_{suffix}"]]

    if not dfs:
        raise ValueError("No Aranet4 data could be loaded")

    # Merge all dataframes on datetime
    merged: Optional[pd.DataFrame] = None
    for suffix, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="datetime", how="outer")

    if merged is None:
        raise ValueError("No data could be merged")

    # Sort by datetime and interpolate missing values (linear interpolation)
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Set datetime as index for resampling
    merged = merged.set_index("datetime")

    # Resample to regular 1-minute intervals and interpolate
    merged = merged.resample("1min").mean()
    merged = merged.interpolate(method="linear", limit=5)

    # Apply rolling average to reduce noise
    co2_cols = ["C_bedroom", "C_entry", "C_outside"]
    for col in co2_cols:
        if col in merged.columns:
            merged[col] = (
                merged[col]
                .rolling(window=ROLLING_WINDOW_MIN, center=True, min_periods=1)
                .mean()
            )

    merged = merged.reset_index()

    print(f"\nMerged data: {len(merged)} rows")
    print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
    print(f"Applied {ROLLING_WINDOW_MIN}-minute rolling average to CO2 data")

    return merged


def load_co2_injection_log() -> pd.DataFrame:
    """
    Load the CO2 injection state-change log.

    Returns:
        pd.DataFrame: DataFrame with CO2 injection events
    """
    log_path = get_common_file("co2_log_file")

    if not log_path.exists():
        raise FileNotFoundError(
            f"CO2 log file not found: {log_path}\n"
            "Run scripts/process_co2_log.py first to generate this file."
        )

    df = pd.read_csv(log_path)
    df["datetime_EDT"] = pd.to_datetime(df["datetime_EDT"])

    return df


def identify_injection_events(co2_log: pd.DataFrame) -> List[Dict]:
    """
    Identify CO2 injection events from the log file.

    An injection event is identified when CO2 valve turns ON (value > 0).
    CO2 is injected from 40 to 44 minutes after the hour; the mixing fan
    turns off at 45 minutes after the hour.

    Parameters:
        co2_log (pd.DataFrame): DataFrame with CO2 injection log

    Returns:
        List[Dict]: List of dicts with injection event details:
        {
            'event_number': int (1-indexed),
            'injection_start': datetime when CO2 valve opened,
            'injection_end': datetime when CO2 valve closed,
            'fan_off': datetime when mixing fan turned off,
            'decay_start': datetime to begin decay analysis,
            'decay_end': datetime to end decay analysis,
            'decay_duration_hours': float (decay window duration)
        }
    """
    events = []

    # Find rows where CO2 transitions from 0 to non-zero (injection start)
    co2_log = co2_log.sort_values("datetime_EDT").reset_index(drop=True)

    event_number = 0  # Track event number (will be 1-indexed)

    for i in range(len(co2_log) - 1):
        current_row = co2_log.iloc[i]
        next_row = co2_log.iloc[i + 1]

        # Check for CO2 turning ON
        if current_row["CO2"] == 0 and next_row["CO2"] > 0:
            event_number += 1  # Increment event number (1-indexed)
            injection_start = next_row["datetime_EDT"]

            # Find when CO2 turns OFF (within the next hour)
            injection_end = None
            fan_off = None

            for j in range(i + 2, min(i + 20, len(co2_log))):
                row = co2_log.iloc[j]
                if row["CO2"] == 0 and injection_end is None:
                    injection_end = row["datetime_EDT"]
                if row["mixing_fan"] == 0 and fan_off is None:
                    fan_off = row["datetime_EDT"]
                if injection_end is not None and fan_off is not None:
                    break

            if injection_end is None:
                injection_end = injection_start + timedelta(minutes=4)
            if fan_off is None:
                fan_off = injection_start + timedelta(minutes=5)

            # Calculate decay analysis window
            # Start 10 min before the top of the next hour (at :50)
            hour_after_injection = injection_start.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
            decay_start = hour_after_injection + timedelta(
                minutes=DECAY_START_OFFSET_MIN
            )

            # Use custom decay duration if specified for this event, otherwise use default
            decay_duration = CUSTOM_DECAY_DURATIONS.get(
                event_number, DECAY_DURATION_HOURS
            )
            decay_end = decay_start + timedelta(hours=decay_duration)

            events.append(
                {
                    "event_number": event_number,
                    "injection_start": injection_start,
                    "injection_end": injection_end,
                    "fan_off": fan_off,
                    "decay_start": decay_start,
                    "decay_end": decay_end,
                    "decay_duration_hours": decay_duration,
                }
            )

    return events


# =============================================================================
# Air-Change Rate Calculation (Analytical Linear Regression Method)
# =============================================================================


def calculate_lambda_analytical(
    co2_data: pd.DataFrame,
    decay_start: datetime,
    decay_end: datetime,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    source_mode: str = "average",
) -> Dict:
    """
    Calculate air-change rate (λ) using analytical linear regression approach.

    The analytical solution of the mass balance equation:
        -ln[(C(t) - C_avg) / (C_0 - C_avg)] = λ·t

    λ is determined by linear regression of y vs t, where:
        y = -ln[(C(t) - C_avg) / (C_0 - C_avg)]
        t = time in hours since decay start

    Parameters:
        co2_data (pd.DataFrame): DataFrame with CO2 concentrations
        decay_start (datetime): Start of decay analysis window
        decay_end (datetime): End of decay analysis window
        alpha (float): Fraction of infiltration from outside
        beta (float): Fraction of infiltration from entry zone
        source_mode (str): 'average', 'outside', or 'entry'

    Returns:
        Dict: Dictionary with lambda statistics and regression results
    """
    # Check if decay window is within data range
    data_start = co2_data["datetime"].min()
    data_end = co2_data["datetime"].max()

    if decay_start > data_end:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": np.nan,
            "n_points": 0,
            "skip_reason": (
                f"Decay window starts after data ends "
                f"(decay_start={decay_start}, data_end={data_end})"
            ),
        }

    if decay_end < data_start:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": np.nan,
            "n_points": 0,
            "skip_reason": (
                f"Decay window ends before data starts "
                f"(decay_end={decay_end}, data_start={data_start})"
            ),
        }

    # Filter data to decay window
    mask = (co2_data["datetime"] >= decay_start) & (co2_data["datetime"] <= decay_end)
    decay_data = co2_data[mask].copy()

    if len(decay_data) < 10:
        reason = (
            "No data points in decay window"
            if len(decay_data) == 0
            else f"Only {len(decay_data)} data points (minimum 10 required)"
        )
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": np.nan,
            "n_points": 0,
            "skip_reason": reason,
        }

    # Get concentration arrays
    c_bedroom = np.asarray(decay_data["C_bedroom"].values, dtype=np.float64)
    c_outside = np.asarray(decay_data["C_outside"].values, dtype=np.float64)
    c_entry = np.asarray(decay_data["C_entry"].values, dtype=np.float64)

    # Calculate source concentration based on mode
    if source_mode == "average":
        c_source_array = alpha * c_outside + beta * c_entry
    elif source_mode == "outside":
        c_source_array = c_outside
    elif source_mode == "entry":
        c_source_array = c_entry
    else:
        raise ValueError(f"Unknown source_mode: {source_mode}")

    # Calculate mean source concentration over the decay window (constant)
    c_avg = float(np.nanmean(c_source_array))

    # Get initial bedroom concentration
    c_0 = float(c_bedroom[0])

    # Check for sufficient concentration difference
    if (c_0 - c_avg) < MIN_CONCENTRATION_DIFF:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": np.nan,
            "n_points": 0,
            "skip_reason": (
                f"Insufficient concentration difference: "
                f"C_0={c_0:.0f}, C_avg={c_avg:.0f}, diff={c_0 - c_avg:.0f} ppm "
                f"(minimum {MIN_CONCENTRATION_DIFF} ppm required)"
            ),
        }

    # Calculate time in hours since decay start
    t0 = decay_data["datetime"].iloc[0]
    t_hours = np.asarray(
        (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0,
        dtype=np.float64,
    )

    # Calculate y = -ln[(C(t) - C_avg) / (C_0 - C_avg)]
    numerator = c_bedroom - c_avg
    denominator = c_0 - c_avg

    # Filter out invalid values (numerator must be positive for log)
    valid_mask = numerator > 0
    if not np.any(valid_mask):
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": np.nan,
            "n_points": 0,
            "skip_reason": (
                "No valid points: C_bedroom already at or below C_avg "
                f"(C_bedroom range: {c_bedroom.min():.0f}-{c_bedroom.max():.0f}, "
                f"C_avg: {c_avg:.0f})"
            ),
        }

    t_valid = t_hours[valid_mask]
    numerator_valid = numerator[valid_mask]

    # Calculate y values
    y = -np.log(numerator_valid / denominator)

    # Perform linear regression: y = λ·t (forced through origin)
    # Using ordinary least squares: λ = Σ(t·y) / Σ(t²)
    lambda_value = np.sum(t_valid * y) / np.sum(t_valid**2)

    # Calculate R² for goodness of fit
    y_pred = lambda_value * t_valid
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Calculate standard error of the slope
    n = len(t_valid)
    if n > 2:
        mse = ss_res / (n - 1)  # 1 parameter (slope, forced through origin)
        se_lambda = np.sqrt(mse / np.sum(t_valid**2))
    else:
        se_lambda = np.nan

    # Filter unreasonable λ values
    if lambda_value <= 0 or lambda_value > 10:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "r_squared": r_squared,
            "n_points": n,
            "skip_reason": (
                f"Unreasonable λ value: {lambda_value:.4f} h⁻¹ "
                f"(expected 0 < λ < 10 h⁻¹)"
            ),
        }

    return {
        "lambda_mean": float(lambda_value),
        "lambda_std": float(se_lambda),
        "r_squared": float(r_squared),
        "n_points": n,
        "n_total": len(c_bedroom),
        "c_bedroom_initial": c_0,
        "c_bedroom_final": float(c_bedroom[-1]) if len(c_bedroom) > 0 else np.nan,
        "c_source_mean": c_avg,
        "c_outside_mean": float(np.nanmean(c_outside)),
        "c_entry_mean": float(np.nanmean(c_entry)),
        "y_values": y.tolist(),
        "t_values": t_valid.tolist(),
    }


def analyze_injection_event(
    co2_data: pd.DataFrame,
    event: Dict,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict:
    """
    Analyze a single CO2 injection event.

    Calculates λ using three methods for uncertainty assessment:
        1. C_average (α·C_outside + β·C_entry)
        2. C_outside only
        3. C_entry only

    Parameters:
        co2_data (pd.DataFrame): DataFrame with CO2 concentrations
        event (Dict): Dict with injection event timing
        alpha (float): Fraction of infiltration from outside
        beta (float): Fraction of infiltration from entry zone

    Returns:
        Dict: Dictionary with analysis results for all three methods
    """
    result = {
        "event_number": event.get("event_number", None),
        "injection_start": event["injection_start"],
        "decay_start": event["decay_start"],
        "decay_end": event["decay_end"],
        "decay_duration_hours": event.get("decay_duration_hours", DECAY_DURATION_HOURS),
    }

    # Calculate lambda using three methods
    for mode in ["average", "outside", "entry"]:
        lambda_result = calculate_lambda_analytical(
            co2_data,
            event["decay_start"],
            event["decay_end"],
            alpha=alpha,
            beta=beta,
            source_mode=mode,
        )

        result[f"lambda_{mode}_mean"] = lambda_result["lambda_mean"]
        result[f"lambda_{mode}_std"] = lambda_result["lambda_std"]
        result[f"lambda_{mode}_r_squared"] = lambda_result.get("r_squared", np.nan)
        result[f"lambda_{mode}_n_points"] = lambda_result["n_points"]

        if mode == "average":
            result["c_bedroom_initial"] = lambda_result.get("c_bedroom_initial", np.nan)
            result["c_bedroom_final"] = lambda_result.get("c_bedroom_final", np.nan)
            result["c_source_mean"] = lambda_result.get("c_source_mean", np.nan)
            result["c_outside_mean"] = lambda_result.get("c_outside_mean", np.nan)
            result["c_entry_mean"] = lambda_result.get("c_entry_mean", np.nan)
            result["skip_reason"] = lambda_result.get("skip_reason", None)
            # Store for plotting
            result["_y_values"] = lambda_result.get("y_values", [])
            result["_t_values"] = lambda_result.get("t_values", [])

    return result


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_co2_decay_analysis(
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
) -> pd.DataFrame:
    """
    Run the complete CO2 decay analysis.

    Parameters:
        alpha (float): Fraction of infiltration from outside
        beta (float): Fraction of infiltration from entry zone
        output_dir (Path): Optional output directory (defaults to data_root/output)
        generate_plots (bool): If True, generate plots for each event and summary

    Returns:
        pd.DataFrame: DataFrame with analysis results for all injection events
    """
    print("=" * 60)
    print("CO2 Decay & Air-Change Rate (λ) Analysis")
    print("Analytical Method (Linear Regression)")
    print("=" * 60)
    print(f"Parameters: α={alpha}, β={beta}")
    print(
        f"Decay window: {abs(DECAY_START_OFFSET_MIN)} min before hour to "
        f"{DECAY_DURATION_HOURS} hours after"
    )

    # Set output directory
    if output_dir is None:
        output_dir = get_data_root() / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CO2 concentration data
    co2_data = load_and_merge_co2_data()

    # Load CO2 injection log and identify events
    print("\nLoading CO2 injection log...")
    co2_log = load_co2_injection_log()
    raw_events = identify_injection_events(co2_log)
    print(f"Found {len(raw_events)} raw injection events")

    # Filter events by experiment start date
    print(f"\nFiltering events (keeping >= {EXPERIMENT_START_DATE.date()})...")
    events = [e for e in raw_events if e["injection_start"] >= EXPERIMENT_START_DATE]
    print(f"  {len(events)} events after date filtering")

    # Analyze each event
    print("\nAnalyzing injection events...")
    results = []
    plot_dir = output_dir / "plots"

    for i, event in enumerate(events):
        event_num = event.get("event_number", i + 1)
        test_name = event.get("test_name", f"Event_{event_num:02d}")
        injection_time = event["injection_start"]

        # Check if corresponding shower event is excluded
        # CO2 injection happens ~20 minutes before shower
        expected_shower_time = injection_time + timedelta(minutes=20)
        is_excluded_flag, exclusion_reason = is_event_excluded(expected_shower_time)

        if is_excluded_flag:
            print(
                f"  {test_name} "
                f"({injection_time.strftime('%Y-%m-%d %H:%M')}): "
                f"Skipped (excluded: {exclusion_reason})"
            )
            # Add to results as skipped
            result = {
                "event_number": event_num,
                "test_name": test_name,
                "injection_start": injection_time,
                "decay_start": event["decay_start"],
                "decay_end": event["decay_end"],
                "decay_duration_hours": event.get("decay_duration_hours", DECAY_DURATION_HOURS),
                "lambda_average_mean": np.nan,
                "lambda_average_std": np.nan,
                "lambda_average_r_squared": np.nan,
                "lambda_average_n_points": 0,
                "lambda_outside_mean": np.nan,
                "lambda_outside_std": np.nan,
                "lambda_outside_r_squared": np.nan,
                "lambda_outside_n_points": 0,
                "lambda_entry_mean": np.nan,
                "lambda_entry_std": np.nan,
                "lambda_entry_r_squared": np.nan,
                "lambda_entry_n_points": 0,
                "skip_reason": f"Excluded: {exclusion_reason}",
            }
            results.append(result)
            continue

        decay_duration = event.get("decay_duration_hours", DECAY_DURATION_HOURS)
        duration_info = (
            f" [custom: {decay_duration}h]"
            if event_num in CUSTOM_DECAY_DURATIONS
            else ""
        )
        print(
            f"  {test_name}: "
            f"{injection_time.strftime('%Y-%m-%d %H:%M')}"
            f"{duration_info}"
        )

        result = analyze_injection_event(co2_data, event, alpha, beta)
        result["test_name"] = test_name  # Add test_name to result
        results.append(result)

        # Print summary for this event
        if not np.isnan(result["lambda_average_mean"]):
            print(
                f"    λ (average): {result['lambda_average_mean']:.4f} ± "
                f"{result['lambda_average_std']:.4f} h⁻¹ "
                f"(R²={result['lambda_average_r_squared']:.4f})"
            )

            # Generate plot for this event if enabled
            if generate_plots:
                # Format filename: event_01-0114_hw_morning_co2_decay.png
                from scripts.plot_style import format_test_name_for_filename
                formatted_name = format_test_name_for_filename(test_name)
                plot_path = plot_dir / f"event_{event_num:02d}-{formatted_name}_co2_decay.png"
                plot_co2_decay_event_analytical(
                    co2_data=co2_data,
                    event=event,
                    result=result,
                    output_path=plot_path,
                    event_number=event_num,
                    test_name=test_name,
                    alpha=alpha,
                    beta=beta,
                )
        else:
            skip_reason = result.get("skip_reason", "Unknown reason")
            print(f"    Skipped: {skip_reason}")

    # Create results DataFrame (exclude internal plotting data)
    results_clean = []
    for r in results:
        r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
        results_clean.append(r_clean)
    results_df = pd.DataFrame(results_clean)

    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)

    # Count excluded and skipped events
    n_total = len(results_df)
    n_excluded = results_df["skip_reason"].str.contains("Excluded:", na=False).sum()
    n_other_skipped = results_df["lambda_average_mean"].isna().sum() - n_excluded

    print(f"\nEvent Summary:")
    print(f"  Total events analyzed: {n_total}")
    print(f"  Excluded events: {n_excluded}")
    print(f"  Other skipped events: {n_other_skipped}")

    for mode in ["average", "outside", "entry"]:
        col = f"lambda_{mode}_mean"
        r2_col = f"lambda_{mode}_r_squared"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            valid_r2 = results_df.loc[valid_values.index, r2_col]
            print(f"\nλ using C_{mode}:")
            print(f"  Mean:   {valid_values.mean():.4f} h⁻¹")
            print(f"  Std:    {valid_values.std():.4f} h⁻¹")
            print(f"  Median: {valid_values.median():.4f} h⁻¹")
            print(f"  Range:  {valid_values.min():.4f} - {valid_values.max():.4f} h⁻¹")
            print(f"  Mean R²: {valid_r2.mean():.4f}")
            print(f"  N events: {len(valid_values)}")

    # Save results
    output_file = output_dir / "co2_lambda_summary.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save detailed summary
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "method": "analytical_linear_regression",
        "alpha": alpha,
        "beta": beta,
        "decay_duration_hours": DECAY_DURATION_HOURS,
        "rolling_window_min": ROLLING_WINDOW_MIN,
        "n_events": len(events),
        "n_valid_events": int(results_df["lambda_average_mean"].notna().sum()),
    }

    valid_values = pd.Series(dtype=float)
    for mode in ["average", "outside", "entry"]:
        col = f"lambda_{mode}_mean"
        r2_col = f"lambda_{mode}_r_squared"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            valid_r2 = results_df.loc[valid_values.index, r2_col]
            summary[f"lambda_{mode}_overall_mean"] = float(valid_values.mean())
            summary[f"lambda_{mode}_overall_std"] = float(valid_values.std())
            summary[f"lambda_{mode}_mean_r_squared"] = float(valid_r2.mean())

    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / "co2_lambda_overall_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Generate summary plot if enabled
    if generate_plots and len(valid_values) > 0:
        summary_plot_path = plot_dir / "lambda_summary.png"
        plot_lambda_summary(results_df, output_path=summary_plot_path)
        print(f"Plots saved to: {plot_dir}")

    return results_df


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CO2 Decay & Air-Change Rate (λ) Analysis (Analytical Method)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Fraction of infiltration from outside (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"Fraction of infiltration from entry zone (default: {DEFAULT_BETA})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data_root/output)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )

    args = parser.parse_args()

    # Validate alpha + beta = 1
    if abs(args.alpha + args.beta - 1.0) > 0.001:
        print(f"Warning: α ({args.alpha}) + β ({args.beta}) != 1.0")
        print("Normalizing values...")
        total = args.alpha + args.beta
        args.alpha = args.alpha / total
        args.beta = args.beta / total
        print(f"New values: α={args.alpha:.3f}, β={args.beta:.3f}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_co2_decay_analysis(
        alpha=args.alpha,
        beta=args.beta,
        output_dir=output_dir,
        generate_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()
