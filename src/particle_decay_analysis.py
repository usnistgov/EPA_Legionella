#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Decay & Emission Analysis
===================================

This script analyzes particle concentration decay data from QuantAQ MODULAIR-PM
sensors to calculate particle penetration factors, deposition rates, and shower
emission rates for the EPA Legionella study. The analysis uses a numerical
approach to solve the mass balance equation for seven particle size bins.

Particle size bins analyzed (µm):
    - Bin 0: 0.35-0.46
    - Bin 1: 0.46-0.66
    - Bin 2: 0.66-1.0
    - Bin 3: 1.0-1.3
    - Bin 4: 1.3-1.7
    - Bin 5: 1.7-2.3
    - Bin 6: 2.3-3.0

Key Metrics Calculated:
    - p: Particle penetration factor (dimensionless, 0-1 range)
    - β_deposition: Effective deposition loss rate (h⁻¹)
    - E: Shower emission rate (particles/minute)
    - λ: Air change rate from CO2 analysis (h⁻¹)

Analysis Features:
    - Numerical solution of time-dependent mass balance equation
    - Integration with CO2-derived air change rates
    - Per-bin analysis for size-dependent behavior
    - Statistical summaries across all shower events
    - Comprehensive visualization of decay curves and emissions

Methodology:
    The mass balance equation for indoor particle concentration:
        V ∂C/∂t = pQC_out - QC - β_deposition CV + E
        ∂C/∂t = pλC_out - λC - β_deposition C + E/V

    1. Calculate penetration factor (p):
       - Use two averaging windows around each shower event (before and after)
       - For Night events:
           Before: 9pm (day before) to 2am (day of)
           After:  9am (day of) to 2pm (day of)
       - For Day events:
           Before: 9am (day of) to 2pm (day of)
           After:  9pm (day of) to 2am (next day)
       - p = C_inside / C_outside (averaged over each window, zeros excluded)
       - Final p = average of before and after window p values
       - Allowable range: 0-1 (values > 1 are capped at 1)

    2. Obtain air change rate (λ):
       - Load from CO2 decay analysis results
       - Units: h⁻¹

    3. Calculate deposition rate (β_deposition) when E=0:
       - Use 2-hour window after shower ends
       - Solve numerically for each time step:
         β_deposition = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t
       - Average β over the 2-hour window

    4. Calculate emission rate (E) during shower:
       - Use shower ON to OFF period from log
       - Solve numerically for each time step:
         E = pλVC_out,t + V(C_t - C_t(i+1))/Δt - λVC_t - β_deposition VC_t
       - Average E over the shower duration

Output Files:
    - particle_analysis_summary.xlsx: Multi-sheet workbook with:
        * p_penetration: Penetration factors per event and bin
        * beta_deposition: Deposition rates per event and bin
        * E_emission: Emission rates per event and bin
        * overall_summary: Aggregated statistics
    - plots/event_XX_bin_Y_decay.png: Individual decay curves
    - plots/penetration_summary.png: Summary of p values
    - plots/deposition_summary.png: Summary of β values
    - plots/emission_summary.png: Summary of E values

Applications:
    - Characterize particle transport and deposition in indoor environments
    - Quantify aerosol emissions from shower fixtures
    - Support exposure assessment for waterborne pathogens
    - Validate indoor air quality models

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
    is_event_excluded,
    process_events_with_management,
)
from scripts.event_registry import (  # noqa: E402
    REGISTRY_FILENAME,
    load_event_registry,
)
from src.data_paths import (  # noqa: E402
    get_common_file,
    get_data_root,
    get_instrument_path,
)

# =============================================================================
# Configuration Constants
# =============================================================================

# Particle size bin definitions (µm) - Alphasense OPC-N3
PARTICLE_BINS = {
    0: {"name": "0.35-0.46", "min": 0.35, "max": 0.46, "column": "opc_bin0"},
    1: {"name": "0.46-0.66", "min": 0.46, "max": 0.66, "column": "opc_bin1"},
    2: {"name": "0.66-1.0", "min": 0.66, "max": 1.0, "column": "opc_bin2"},
    3: {"name": "1.0-1.3", "min": 1.0, "max": 1.3, "column": "opc_bin3"},
    4: {"name": "1.3-1.7", "min": 1.3, "max": 1.7, "column": "opc_bin4"},
    5: {"name": "1.7-2.3", "min": 1.7, "max": 2.3, "column": "opc_bin5"},
    6: {"name": "2.3-3.0", "min": 2.3, "max": 3.0, "column": "opc_bin6"},
}

# Physical parameters
BEDROOM_VOLUME_M3 = 36.1  # Bedroom volume in cubic meters (36.10859771 m³ from CAD)
CM3_PER_M3 = 1e6  # Conversion factor: cubic centimeters per cubic meter

# Analysis timing parameters
DEPOSITION_WINDOW_HOURS = 2.0  # Hours after shower for β calculation
TIME_STEP_MINUTES = 1.0  # Time resolution for numerical calculations

# Smoothing parameters (set to 0 to disable)
ROLLING_WINDOW_MIN = 0  # Rolling average window in minutes (0 = no smoothing)

# Validation thresholds
MAX_DEPOSITION_RATE = 15.0  # Maximum reasonable β (h⁻¹) (increased from 10.0)
MIN_CONCENTRATION_RATIO = (
    1.05  # Minimum C_inside/C_outside during decay (reduced from 1.2)
)

# Minimum data point requirements
MIN_POINTS_PENETRATION = 10  # Minimum points for penetration calculation
MIN_POINTS_DEPOSITION = (
    10  # Minimum points for deposition calculation (reduced from 20)
)
MIN_POINTS_EMISSION = 3  # Minimum points for emission calculation (reduced from 5)
MIN_VALID_BETA = 5  # Minimum valid β values required (reduced from 10)


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
    Load CO2 analysis results to get air change rates (λ) for each event.

    Returns:
        pd.DataFrame: DataFrame with λ values per event

    Raises:
        FileNotFoundError: If CO2 analysis results are not found
    """
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


# =============================================================================
# Particle Analysis Functions
# =============================================================================


def get_penetration_windows(
    shower_on: datetime,
    time_of_day: str,
) -> List[tuple]:
    """
    Calculate penetration factor averaging windows based on shower time and time of day.

    For Night events:
        Before: 9pm (day before) to 2am (day of)
        After:  9am (day of) to 2pm (day of)

    For Day events:
        Before: 9am (day of) to 2pm (day of)
        After:  9pm (day of) to 2am (next day)

    Parameters:
        shower_on (datetime): Shower start time
        time_of_day (str): "Night" or "Day"

    Returns:
        List of (window_start, window_end) tuples for before and after windows
    """
    shower_date = shower_on.replace(hour=0, minute=0, second=0, microsecond=0)

    # Classify as night or day event
    is_night_event = time_of_day == "Night" or (
        time_of_day == "" and shower_on.hour < 12
    )

    if is_night_event:
        # 3am event: before = 9pm (day before) to 2am (day of)
        #             after  = 9am (day of) to 2pm (day of)
        before_start = shower_date - timedelta(hours=3)   # 9pm day before
        before_end = shower_date + timedelta(hours=2)     # 2am day of
        after_start = shower_date + timedelta(hours=9)    # 9am day of
        after_end = shower_date + timedelta(hours=14)     # 2pm day of
    else:
        # 3pm event: before = 9am (day of) to 2pm (day of)
        #             after  = 9pm (day of) to 2am (next day)
        before_start = shower_date + timedelta(hours=9)   # 9am day of
        before_end = shower_date + timedelta(hours=14)    # 2pm day of
        after_start = shower_date + timedelta(hours=21)   # 9pm day of
        after_end = shower_date + timedelta(hours=26)     # 2am next day

    return [(before_start, before_end), (after_start, after_end)]


def _calculate_p_for_window(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
) -> Dict:
    """
    Calculate penetration factor (p) for a single window.

    p = C_inside / C_outside averaged over the window, excluding zero values.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of analysis window
        window_end (datetime): End of analysis window
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict with p_mean, p_std, n_points, and optional skip_reason
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to window
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < MIN_POINTS_PENETRATION:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "n_points": len(window_data),
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum {MIN_POINTS_PENETRATION} required)",
        }

    c_inside = np.asarray(window_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(window_data[col_outside].values, dtype=np.float64)

    # Remove invalid points: exclude zeros and NaNs
    valid_mask = (
        (c_inside > 0)
        & (c_outside > 0)
        & (~np.isnan(c_inside))
        & (~np.isnan(c_outside))
    )

    if np.sum(valid_mask) < MIN_POINTS_PENETRATION:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "n_points": int(np.sum(valid_mask)),
            "skip_reason": f"Insufficient valid points: {np.sum(valid_mask)} (minimum {MIN_POINTS_PENETRATION} required)",
        }

    c_inside_valid = c_inside[valid_mask]
    c_outside_valid = c_outside[valid_mask]

    # Calculate p for each point
    p_values = c_inside_valid / c_outside_valid

    return {
        "p_mean": float(np.mean(p_values)),
        "p_std": float(np.std(p_values)),
        "n_points": len(p_values),
    }


def calculate_penetration_factor(
    particle_data: pd.DataFrame,
    shower_on: datetime,
    time_of_day: str,
    bin_num: int,
) -> Dict:
    """
    Calculate penetration factor (p) for a particle bin using before/after windows.

    p = average of C_inside / C_outside from the before and after windows.
    Zero concentration values are excluded. Values above 1 are capped at 1.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        time_of_day (str): "Day" or "Night" time classification
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict: Dictionary with p value and statistics
    """
    windows = get_penetration_windows(shower_on, time_of_day)

    window_p_values = []
    total_points = 0
    skip_reasons = []

    for i, (w_start, w_end) in enumerate(windows):
        label = "before" if i == 0 else "after"
        result = _calculate_p_for_window(particle_data, w_start, w_end, bin_num)

        if not np.isnan(result.get("p_mean", np.nan)):
            window_p_values.append(result["p_mean"])
            total_points += result["n_points"]
        else:
            skip_reasons.append(f"{label}: {result.get('skip_reason', 'Unknown')}")

    if not window_p_values:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "c_inside_mean": np.nan,
            "c_outside_mean": np.nan,
            "n_points": total_points,
            "skip_reason": "; ".join(skip_reasons),
        }

    # Average across available windows, then cap at 1
    p_avg = float(np.mean(window_p_values))
    p_capped = min(p_avg, 1.0)

    return {
        "p_mean": p_capped,
        "p_std": float(np.std(window_p_values)) if len(window_p_values) > 1 else 0.0,
        "c_inside_mean": np.nan,
        "c_outside_mean": np.nan,
        "n_points": total_points,
        "n_windows": len(window_p_values),
    }


def _calculate_linearized_fit(
    c_inside: np.ndarray,
    c_outside: np.ndarray,
    datetimes: np.ndarray,
    p: float,
    lambda_ach: float,
    beta_numerical: float,
) -> Dict:
    """
    Calculate linearized regression fit for particle decay.

    Uses the linearized form: y = -ln[(C(t) - C_ss) / (C_0 - C_ss)] = (λ + β) * t
    where C_ss = pλC_out_avg / (λ + β) is the steady-state concentration.

    Parameters:
        c_inside: Inside concentration array
        c_outside: Outside concentration array
        datetimes: Datetime array
        p: Penetration factor
        lambda_ach: Air change rate (h⁻¹)
        beta_numerical: Numerically calculated β (h⁻¹)

    Returns:
        Dict with fit parameters: beta_fit, r_squared, _t_values, _y_values, c_steady_state
    """
    from scipy import stats

    # Calculate steady-state concentration using numerical β
    c_outside_mean = float(np.nanmean(c_outside))
    total_loss_rate = lambda_ach + beta_numerical

    if total_loss_rate <= 0:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
        }

    c_steady_state = (p * lambda_ach * c_outside_mean) / total_loss_rate

    # Get initial concentration
    c_0 = c_inside[0]
    delta_c_0 = c_0 - c_steady_state

    if delta_c_0 <= 0:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": c_steady_state,
        }

    # Calculate time in hours from start
    # Convert numpy datetime64 to hours since start
    t0 = datetimes[0]
    # numpy timedelta64 division gives hours directly
    t_hours = (datetimes - t0).astype("timedelta64[s]").astype(float) / 3600.0

    # Calculate linearized y values: y = -ln[(C(t) - C_ss) / (C_0 - C_ss)]
    t_values = []
    y_values = []

    for i in range(len(c_inside)):
        c_t = c_inside[i]
        if np.isnan(c_t) or c_t <= c_steady_state:
            continue

        delta_c_t = c_t - c_steady_state
        ratio = delta_c_t / delta_c_0

        if ratio > 0 and ratio <= 1:
            y = -np.log(ratio)
            t_values.append(t_hours[i])
            y_values.append(y)

    if len(t_values) < 5:
        return {
            "beta_fit": np.nan,
            "r_squared": np.nan,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": c_steady_state,
        }

    # Linear regression: y = slope * t (forced through origin would be ideal, but we allow intercept)
    t_arr = np.array(t_values)
    y_arr = np.array(y_values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(t_arr, y_arr)
    r_squared = r_value**2

    # slope = λ + β, so β_fit = slope - λ
    beta_fit = slope - lambda_ach

    return {
        "beta_fit": float(beta_fit) if beta_fit >= 0 else np.nan,
        "r_squared": float(r_squared),
        "_t_values": t_values,
        "_y_values": y_values,
        "c_steady_state": float(c_steady_state),
        "_slope": float(slope),
        "_intercept": float(intercept),
    }


def calculate_deposition_rate(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
) -> Dict:
    """
    Calculate deposition rate (β_deposition) using numerical approach and linearized fit.

    The decay calculation starts from the peak concentration within the deposition
    window (not from window_start), as the peak may occur after shower_off.

    Numerical method solves: β_deposition = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t

    Also performs linearized regression fit for plotting:
    y = -ln[(C(t) - C_ss) / (C_0 - C_ss)] = (λ + β) * t

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of deposition window (shower_off)
        window_end (datetime): End of deposition window
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)

    Returns:
        Dict: Dictionary with β statistics, R², fit data for plotting, and peak_time
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to full deposition window first
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < MIN_POINTS_DEPOSITION:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(window_data),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": None,
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum {MIN_POINTS_DEPOSITION} required)",
        }

    # Find peak concentration within the deposition window for this bin
    c_inside_full = np.asarray(window_data[col_inside].values, dtype=np.float64)

    # Find index of maximum concentration (ignoring NaN values)
    valid_mask = ~np.isnan(c_inside_full)
    if not np.any(valid_mask):
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": 0,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": None,
            "skip_reason": "No valid concentration data in window",
        }

    # Get peak index within the full window
    peak_idx = np.nanargmax(c_inside_full)
    peak_time = pd.Timestamp(window_data["datetime"].iloc[peak_idx])

    # Now filter data from peak to end of window for decay calculation
    decay_data = window_data.iloc[peak_idx:].copy()

    if len(decay_data) < MIN_POINTS_DEPOSITION:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(decay_data),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": f"Insufficient data after peak: {len(decay_data)} points (minimum {MIN_POINTS_DEPOSITION} required)",
        }

    c_inside = np.asarray(decay_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(decay_data[col_outside].values, dtype=np.float64)
    datetimes = decay_data["datetime"].values

    # Check for sufficient concentration difference (now using peak concentration)
    c_outside_mean = np.mean(c_outside)
    c_ratio = c_inside[0] / c_outside_mean if c_outside_mean > 0 else 0
    if c_ratio < MIN_CONCENTRATION_RATIO:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": 0,
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient concentration ratio at peak: {c_ratio:.3f} "
                f"(minimum {MIN_CONCENTRATION_RATIO}). "
                f"C_peak={c_inside[0]:.1f}, C_outside_mean={c_outside_mean:.1f}"
            ),
        }

    # Calculate β for each time step (numerical method) starting from peak
    dt_hours = TIME_STEP_MINUTES / 60.0  # Convert to hours
    beta_values = []

    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid points
        if c_t <= 0 or np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t):
            continue

        # Calculate β_deposition
        # β = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t
        term1 = 1.0 / dt_hours
        term2 = -lambda_ach
        term3 = -c_t_next / (c_t * dt_hours)
        term4 = (p * lambda_ach * c_out_t) / c_t

        beta = term1 + term2 + term3 + term4

        # Filter unreasonable values
        if 0 <= beta <= MAX_DEPOSITION_RATE:
            beta_values.append(beta)

    if len(beta_values) < MIN_VALID_BETA:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "beta_r_squared": np.nan,
            "n_points": len(beta_values),
            "_t_values": [],
            "_y_values": [],
            "c_steady_state": np.nan,
            "peak_time": peak_time,
            "skip_reason": (
                f"Insufficient valid β values: {len(beta_values)} "
                f"(minimum {MIN_VALID_BETA} required)"
            ),
        }

    beta_mean = float(np.mean(beta_values))

    # Calculate linearized fit for plotting and R² (using data from peak onward)
    fit_result = _calculate_linearized_fit(
        c_inside, c_outside, datetimes, p, lambda_ach, beta_mean
    )

    return {
        "beta_mean": beta_mean,
        "beta_std": float(np.std(beta_values)),
        "beta_median": float(np.median(beta_values)),
        "beta_r_squared": fit_result.get("r_squared", np.nan),
        "beta_fit": fit_result.get("beta_fit", np.nan),  # β from linearized regression
        "_fit_slope": fit_result.get("_slope", np.nan),  # Actual slope from regression
        "_fit_intercept": fit_result.get(
            "_intercept", 0.0
        ),  # Intercept from regression
        "n_points": len(beta_values),
        "_t_values": fit_result.get("_t_values", []),
        "_y_values": fit_result.get("_y_values", []),
        "c_steady_state": fit_result.get("c_steady_state", np.nan),
        "peak_time": peak_time,
    }


def calculate_emission_rate(
    particle_data: pd.DataFrame,
    shower_on: datetime,
    shower_off: datetime,
    bin_num: int,
    p: float,
    lambda_ach: float,
    beta: float,
) -> Dict:
    """
    Calculate emission rate (E) during shower using numerical approach.

    Solves: E = pλVC_out,t + V(C_t - C_t(i+1))/Δt - λVC_t - β_deposition VC_t

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        shower_on (datetime): Shower start time
        shower_off (datetime): Shower end time
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)
        beta (float): Deposition rate (h⁻¹)

    Returns:
        Dict: Dictionary with E statistics (particles/minute)
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to shower window
    mask = (particle_data["datetime"] >= shower_on) & (
        particle_data["datetime"] <= shower_off
    )
    shower_data = particle_data[mask].copy()

    if len(shower_data) < MIN_POINTS_EMISSION:
        return {
            "E_mean": np.nan,
            "E_std": np.nan,
            "E_median": np.nan,
            "E_total": np.nan,
            "n_points": len(shower_data),
            "skip_reason": (
                f"Insufficient data: {len(shower_data)} points "
                f"(minimum {MIN_POINTS_EMISSION} required)"
            ),
        }

    c_inside = np.asarray(shower_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(shower_data[col_outside].values, dtype=np.float64)

    V = (
        BEDROOM_VOLUME_M3 * CM3_PER_M3
    )  # Convert m³ to cm³ for concentration units (#/cm³)
    dt_minutes = TIME_STEP_MINUTES  # minutes

    # Calculate E for each time step
    E_values = []

    for i in range(len(c_inside) - 1):
        c_t = c_inside[i]
        c_t_next = c_inside[i + 1]
        c_out_t = c_outside[i]

        # Skip invalid points
        if np.isnan(c_t) or np.isnan(c_t_next) or np.isnan(c_out_t):
            continue

        # Calculate E (particles/minute)
        # E = pλVC_out,t + V(C_t - C_t(i+1))/Δt - λVC_t - β_deposition VC_t
        # Convert λ and β from h⁻¹ to min⁻¹
        lambda_per_min = lambda_ach / 60.0
        beta_per_min = beta / 60.0

        term1 = p * lambda_per_min * V * c_out_t
        term2 = V * (c_t - c_t_next) / dt_minutes
        term3 = -lambda_per_min * V * c_t
        term4 = -beta_per_min * V * c_t

        E = term1 + term2 + term3 + term4

        # Only keep positive emission rates
        if E > 0:
            E_values.append(E)

    if len(E_values) == 0:
        return {
            "E_mean": np.nan,
            "E_std": np.nan,
            "E_median": np.nan,
            "E_total": np.nan,
            "n_points": 0,
            "skip_reason": "No positive emission values calculated",
        }

    # Calculate total emission over shower duration
    E_total = np.sum(E_values) * dt_minutes  # Total particles emitted

    return {
        "E_mean": float(np.mean(E_values)),
        "E_std": float(np.std(E_values)),
        "E_median": float(np.median(E_values)),
        "E_total": float(E_total),
        "n_points": len(E_values),
    }


def analyze_event_all_bins(
    particle_data: pd.DataFrame,
    event: Dict,
    lambda_ach: float,
) -> Dict:
    """
    Analyze all particle bins for a single shower event.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        event (Dict): Event timing information
        lambda_ach (float): Air change rate (h⁻¹)

    Returns:
        Dict: Results for all bins
    """
    results = {
        "event_number": event.get("event_number", 0),
        "test_name": event.get("test_name", ""),
        "config_key": event.get("config_key", ""),
        "water_temp": event.get("water_temp", ""),
        "door_position": event.get("door_position", ""),
        "planned_fan": event.get("planned_fan", ""),
        "time_of_day": event.get("time_of_day", ""),
        "fan_during_test": event.get("fan_during_test", False),
        "replicate_num": event.get("replicate_num", 0),
        "shower_on": event["shower_on"],
        "shower_off": event["shower_off"],
        "shower_duration_min": event.get(
            "shower_duration_min", event.get("duration_min", 0)
        ),
        "lambda_ach": lambda_ach,
        "co2_event_idx": event.get("co2_event_idx", None),
    }

    time_of_day = event.get("time_of_day", "")

    for bin_num in PARTICLE_BINS.keys():
        # Calculate penetration factor using before/after windows
        p_result = calculate_penetration_factor(
            particle_data,
            event["shower_on"],
            time_of_day,
            bin_num,
        )

        results[f"bin{bin_num}_p_mean"] = p_result.get("p_mean", np.nan)
        results[f"bin{bin_num}_p_std"] = p_result.get("p_std", np.nan)

        # Skip further calculations if p is invalid
        if np.isnan(p_result.get("p_mean", np.nan)):
            results[f"bin{bin_num}_beta_mean"] = np.nan
            results[f"bin{bin_num}_beta_std"] = np.nan
            results[f"bin{bin_num}_beta_r_squared"] = np.nan
            results[f"bin{bin_num}_beta_fit"] = np.nan
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = p_result.get(
                "skip_reason", "Unknown"
            )
            # Store empty fit data for plotting
            results[f"bin{bin_num}_fit_t_values"] = []
            results[f"bin{bin_num}_fit_y_values"] = []
            results[f"bin{bin_num}_fit_slope"] = np.nan
            results[f"bin{bin_num}_fit_intercept"] = 0.0
            results[f"bin{bin_num}_c_steady_state"] = np.nan
            results[f"bin{bin_num}_peak_time"] = None
            continue

        p_mean = p_result["p_mean"]

        # Calculate deposition rate
        beta_result = calculate_deposition_rate(
            particle_data,
            event["deposition_start"],
            event["deposition_end"],
            bin_num,
            p_mean,
            lambda_ach,
        )

        results[f"bin{bin_num}_beta_mean"] = beta_result.get("beta_mean", np.nan)
        results[f"bin{bin_num}_beta_std"] = beta_result.get("beta_std", np.nan)
        results[f"bin{bin_num}_beta_r_squared"] = beta_result.get(
            "beta_r_squared", np.nan
        )
        results[f"bin{bin_num}_beta_fit"] = beta_result.get(
            "beta_fit", np.nan
        )  # From linearized regression

        # Store fit data for plotting (even if beta is valid, we want the data)
        results[f"bin{bin_num}_fit_t_values"] = beta_result.get("_t_values", [])
        results[f"bin{bin_num}_fit_y_values"] = beta_result.get("_y_values", [])
        results[f"bin{bin_num}_fit_slope"] = beta_result.get(
            "_fit_slope", np.nan
        )  # Actual regression slope
        results[f"bin{bin_num}_fit_intercept"] = beta_result.get(
            "_fit_intercept", 0.0
        )  # Regression intercept
        results[f"bin{bin_num}_c_steady_state"] = beta_result.get(
            "c_steady_state", np.nan
        )
        results[f"bin{bin_num}_peak_time"] = beta_result.get("peak_time", None)

        # Skip emission calculation if beta is invalid
        if np.isnan(beta_result.get("beta_mean", np.nan)):
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = beta_result.get(
                "skip_reason", "Unknown"
            )
            continue

        beta_mean = beta_result["beta_mean"]

        # Calculate emission rate
        E_result = calculate_emission_rate(
            particle_data,
            event["shower_on"],
            event["shower_off"],
            bin_num,
            p_mean,
            lambda_ach,
            beta_mean,
        )

        results[f"bin{bin_num}_E_mean"] = E_result.get("E_mean", np.nan)
        results[f"bin{bin_num}_E_std"] = E_result.get("E_std", np.nan)
        results[f"bin{bin_num}_E_total"] = E_result.get("E_total", np.nan)
        results[f"bin{bin_num}_skip_reason"] = E_result.get("skip_reason", None)

    return results


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_particle_analysis(
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
) -> pd.DataFrame:
    """
    Run the complete particle decay and emission analysis.

    Parameters:
        output_dir (Path): Optional output directory (defaults to data_root/output)
        generate_plots (bool): If True, generate plots for each event and summary

    Returns:
        pd.DataFrame: DataFrame with analysis results for all events and bins
    """
    print("=" * 80)
    print("Particle Decay & Emission Analysis")
    print("Numerical Approach - Seven Particle Size Bins")
    print("=" * 80)
    print(f"Bedroom volume: {BEDROOM_VOLUME_M3} m^3")
    print(f"Time step: {TIME_STEP_MINUTES} minute(s)")
    print("Penetration factor: averaged before/after windows (p capped at 1)")
    print(f"Deposition window: {DEPOSITION_WINDOW_HOURS} hour(s) after shower")
    print("\nValidation thresholds:")
    print(f"  Max deposition rate (beta): {MAX_DEPOSITION_RATE} h^-1")
    print(f"  Min concentration ratio: {MIN_CONCENTRATION_RATIO}")
    print(
        f"  Min data points: p={MIN_POINTS_PENETRATION}, beta={MIN_POINTS_DEPOSITION}, E={MIN_POINTS_EMISSION}"
    )

    # Set output directory
    if output_dir is None:
        output_dir = get_data_root() / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load particle data
    particle_data = load_and_merge_quantaq_data()

    # Try to load events from unified registry first (for consistent numbering)
    events, co2_results, used_registry = get_events_from_registry(output_dir)

    if used_registry:
        print("  Using unified event registry for consistent event numbering")
    else:
        # Fall back to existing event management system
        print("\nNote: Registry not found. Using process_events_with_management().")
        print("  Run 'python scripts/event_registry.py' for unified numbering.\n")

        # Load shower log and identify events
        print("Loading shower log...")
        shower_log = load_shower_log()
        raw_events = identify_shower_events(shower_log)
        print(f"Found {len(raw_events)} raw shower events")

        # Load CO2 lambda results
        co2_results = load_co2_lambda_results()

        # Process events using the enhanced event management system
        print("\nProcessing events with event management system...")
        events, co2_events_processed, event_log = process_events_with_management(
            raw_events,
            [],  # CO2 events (will be loaded from co2_results)
            shower_log,
            co2_results,
            output_dir,
            create_synthetic=False,
        )

    # Match events with CO2 lambda values
    # (Already done in process_events_with_management, but we print summary here)
    print("\nEvent Matching Summary:")
    matched_count = 0
    excluded_count = 0
    missing_lambda_count = 0

    for event in events:
        shower_time = event["shower_on"]

        # Check if excluded
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if is_excluded_flag:
            excluded_count += 1
            print(
                f"  Event {event.get('event_number', '?')} "
                f"({shower_time.strftime('%Y-%m-%d %H:%M')}): "
                f"EXCLUDED - {exclusion_reason}"
            )
            continue

        # Check if has lambda value
        lambda_val = event.get("lambda_ach", np.nan)
        if not np.isnan(lambda_val):
            matched_count += 1
            co2_idx = event.get("co2_event_idx")
            if co2_idx is not None and co2_idx < len(co2_results):
                co2_time = co2_results.iloc[co2_idx]["injection_start"]
                print(
                    f"  {event.get('test_name', 'Event ' + str(event.get('event_number', '?')))} "
                    f"({shower_time.strftime('%m/%d %H:%M')}) "
                    f"-> CO2 {co2_idx + 1} ({co2_time.strftime('%H:%M')}), "
                    f"lambda={lambda_val:.4f} h^-1"
                )
        else:
            missing_lambda_count += 1
            print(
                f"  {event.get('test_name', 'Event ' + str(event.get('event_number', '?')))} "
                f"({shower_time.strftime('%m/%d %H:%M')}): "
                f"No lambda value available"
            )

    print(
        f"\nTotal: {len(events)} events | Matched: {matched_count} | "
        f"Excluded: {excluded_count} | Missing lambda: {missing_lambda_count}"
    )

    # Analyze each event
    print("\nAnalyzing shower events...")
    results = []

    # Setup plot directory
    plot_dir = output_dir / "plots"
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)

    for event in events:
        event_num = event.get("event_number", 0)
        test_name = event.get("test_name", f"Event_{event_num}")
        shower_time = event["shower_on"]
        lambda_ach = event.get("lambda_ach", np.nan)

        # Skip excluded events
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if is_excluded_flag:
            print(f"  {test_name}: Skipped (excluded: {exclusion_reason})")
            continue

        # Skip events without lambda
        if np.isnan(lambda_ach):
            print(f"  {test_name}: Skipped (no lambda from CO2 analysis)")
            continue

        print(
            f"  {test_name} ({shower_time.strftime('%m/%d %H:%M')}): "
            f"lambda={lambda_ach:.4f} h^-1"
        )

        result = analyze_event_all_bins(particle_data, event, lambda_ach)
        results.append(result)

        # Print summary for this event with detailed skip reasons
        valid_bins = 0
        skipped_bins = []
        for bin_num in PARTICLE_BINS.keys():
            if not np.isnan(result.get(f"bin{bin_num}_E_mean", np.nan)):
                valid_bins += 1
            else:
                skip_reason = result.get(f"bin{bin_num}_skip_reason", "Unknown")
                skipped_bins.append((bin_num, skip_reason))

        print(f"    Successfully analyzed {valid_bins}/{len(PARTICLE_BINS)} bins")

        # Print skip reasons for failed bins (up to 3 for brevity)
        if skipped_bins and valid_bins < len(PARTICLE_BINS):
            for bin_num, reason in skipped_bins[:3]:
                bin_name = PARTICLE_BINS[bin_num]["name"]
                # Truncate long reasons
                if len(reason) > 80:
                    reason = reason[:77] + "..."
                print(f"      Bin {bin_num} ({bin_name} µm): {reason}")
            if len(skipped_bins) > 3:
                print(f"      ... and {len(skipped_bins) - 3} more bins skipped")

        # Generate individual event plot if enabled (all bins on one plot)
        if generate_plots and valid_bins > 0:
            try:
                from scripts.plot_particle import plot_particle_decay_event
                from scripts.plot_style import format_test_name_for_filename

                # Format filename: event_01-0114_hw_morning_pm_decay.png
                formatted_name = format_test_name_for_filename(test_name)
                plot_path = (
                    plot_dir / f"event_{event_num:02d}-{formatted_name}_pm_decay.png"
                )
                plot_particle_decay_event(
                    particle_data=particle_data,
                    event=event,
                    particle_bins=PARTICLE_BINS,
                    result=result,
                    output_path=plot_path,
                    event_number=event_num,
                    test_name=test_name,
                )
            except ImportError:
                pass  # Already warned about missing plot module
            except Exception as e:
                print(f"    Warning: Failed to generate plot for {test_name}: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print overall statistics
    print("\n" + "=" * 80)
    print("Overall Results Summary")
    print("=" * 80)

    if results_df.empty:
        print(
            "\nNo events were analyzed (all skipped due to missing lambda or exclusions)."
        )
    else:
        for bin_num, bin_info in PARTICLE_BINS.items():
            bin_name = bin_info["name"]
            p_col = f"bin{bin_num}_p_mean"
            beta_col = f"bin{bin_num}_beta_mean"
            E_col = f"bin{bin_num}_E_mean"

            valid_p = results_df[p_col].dropna()
            valid_beta = results_df[beta_col].dropna()
            valid_E = results_df[E_col].dropna()

            print(f"\nBin {bin_num} ({bin_name} µm):")
            if len(valid_p) > 0:
                print(
                    f"  p (penetration):     {valid_p.mean():.3f} +/- {valid_p.std():.3f}"
                )
            if len(valid_beta) > 0:
                print(
                    f"  beta (deposition):   {valid_beta.mean():.3f} +/- {valid_beta.std():.3f} h^-1"
                )
            if len(valid_E) > 0:
                print(
                    f"  E (emission):        {valid_E.mean():.2e} +/- {valid_E.std():.2e} #/min"
                )
            print(f"  Valid events:        {len(valid_E)}/{len(results)}")

    # Save results with units in column names
    output_file = output_dir / "particle_analysis_summary.xlsx"

    if results_df.empty:
        print(f"\nNo results to save - skipping {output_file}")
    else:
        # Create column rename mapping for units
        column_rename = {
            "shower_duration_min": "shower_duration (min)",
            "lambda_ach": "lambda_ach (h-1)",
        }
        for bin_num in PARTICLE_BINS.keys():
            column_rename[f"bin{bin_num}_p_mean"] = f"bin{bin_num}_p_mean (-)"
            column_rename[f"bin{bin_num}_p_std"] = f"bin{bin_num}_p_std (-)"
            column_rename[f"bin{bin_num}_beta_mean"] = f"bin{bin_num}_beta_mean (h-1)"
            column_rename[f"bin{bin_num}_beta_std"] = f"bin{bin_num}_beta_std (h-1)"
            column_rename[f"bin{bin_num}_beta_fit"] = f"bin{bin_num}_beta_fit (h-1)"
            column_rename[f"bin{bin_num}_E_mean"] = f"bin{bin_num}_E_mean (#/min)"
            column_rename[f"bin{bin_num}_E_std"] = f"bin{bin_num}_E_std (#/min)"
            column_rename[f"bin{bin_num}_E_total"] = f"bin{bin_num}_E_total (#)"

        results_df_export = results_df.rename(columns=column_rename)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Main results
            results_df_export.to_excel(writer, sheet_name="all_results", index=False)

            # Separate sheets for each metric
            p_cols = ["event_number", "shower_on"] + [
                f"bin{i}_p_mean (-)" for i in PARTICLE_BINS.keys()
            ]
            beta_cols = ["event_number", "shower_on"] + [
                f"bin{i}_beta_mean (h-1)" for i in PARTICLE_BINS.keys()
            ]
            beta_r2_cols = ["event_number", "shower_on"] + [
                f"bin{i}_beta_r_squared" for i in PARTICLE_BINS.keys()
            ]
            E_cols = ["event_number", "shower_on"] + [
                f"bin{i}_E_mean (#/min)" for i in PARTICLE_BINS.keys()
            ]

            results_df_export[p_cols].to_excel(
                writer, sheet_name="p_penetration", index=False
            )
            results_df_export[beta_cols].to_excel(
                writer, sheet_name="beta_deposition", index=False
            )
            # Add R² sheet for deposition fits
            results_df_export[beta_r2_cols].to_excel(
                writer, sheet_name="beta_r_squared", index=False
            )
            results_df_export[E_cols].to_excel(
                writer, sheet_name="E_emission", index=False
            )

        print(f"\nResults saved to: {output_file}")

    # Generate plots if enabled
    if generate_plots and not results_df.empty:
        print("\nGenerating plots...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Import plot_particle functions
        try:
            from scripts.plot_particle import (
                plot_deposition_summary,
                plot_emission_summary,
                plot_penetration_summary,
            )
        except ImportError:
            print("  Warning: plot_particle module not found. Skipping plots.")
        else:
            # Generate summary plots individually to identify which one fails
            try:
                plot_penetration_summary(
                    results_df, PARTICLE_BINS, plot_dir / "penetration_summary.png"
                )
                print("  Generated: penetration_summary.png")
            except Exception as e:
                print(f"  Error generating penetration_summary.png: {e}")

            try:
                plot_deposition_summary(
                    results_df, PARTICLE_BINS, plot_dir / "deposition_summary.png"
                )
                print("  Generated: deposition_summary.png")
            except Exception as e:
                print(f"  Error generating deposition_summary.png: {e}")

            try:
                plot_emission_summary(
                    results_df, PARTICLE_BINS, plot_dir / "emission_summary.png"
                )
                print("  Generated: emission_summary.png")
            except Exception as e:
                print(f"  Error generating emission_summary.png: {e}")

            print(f"  Plots saved to: {plot_dir}")
    elif generate_plots and results_df.empty:
        print("\nSkipping plot generation - no results to plot.")

    return results_df


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Particle Decay & Emission Analysis")
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

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_particle_analysis(
        output_dir=output_dir,
        generate_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()
