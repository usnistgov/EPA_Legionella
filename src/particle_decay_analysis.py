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
    - p: Particle penetration factor (dimensionless, 0.7-0.9 expected)
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
       - Use 1-hour window before shower starts
       - p = C_inside / C_outside (averaged over window)
       - Expected range: 0.7-0.9

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
BEDROOM_VOLUME_M3 = 36.0  # Bedroom volume in cubic meters

# Analysis timing parameters
PENETRATION_WINDOW_HOURS = 1.0  # Hours before shower for p calculation
DEPOSITION_WINDOW_HOURS = 2.0  # Hours after shower for β calculation
TIME_STEP_MINUTES = 1.0  # Time resolution for numerical calculations

# Smoothing parameters (set to 0 to disable)
ROLLING_WINDOW_MIN = 0  # Rolling average window in minutes (0 = no smoothing)

# Validation thresholds
MIN_PENETRATION = 0.5  # Minimum reasonable p value
MAX_PENETRATION = 1.0  # Maximum reasonable p value
MAX_DEPOSITION_RATE = 10.0  # Maximum reasonable β (h⁻¹)
MIN_CONCENTRATION_RATIO = 1.2  # Minimum C_inside/C_outside during decay


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
    """
    output_dir = get_data_root() / "output"
    co2_results_path = output_dir / "co2_lambda_summary.csv"

    if not co2_results_path.exists():
        raise FileNotFoundError(
            f"CO2 lambda results not found: {co2_results_path}\n"
            "Run src/co2_decay_analysis.py first to generate this file."
        )

    df = pd.read_csv(co2_results_path)
    df["decay_start"] = pd.to_datetime(df["decay_start"])

    print(f"\nLoaded CO2 λ results: {len(df)} events")

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
            penetration_start = shower_on - timedelta(hours=PENETRATION_WINDOW_HOURS)
            deposition_start = shower_off
            deposition_end = shower_off + timedelta(hours=DEPOSITION_WINDOW_HOURS)

            events.append(
                {
                    "event_number": event_number,
                    "shower_on": shower_on,
                    "shower_off": shower_off,
                    "shower_duration_min": (shower_off - shower_on).total_seconds()
                    / 60,
                    "penetration_start": penetration_start,
                    "penetration_end": shower_on,
                    "deposition_start": deposition_start,
                    "deposition_end": deposition_end,
                }
            )

    return events


# =============================================================================
# Particle Analysis Functions
# =============================================================================


def calculate_penetration_factor(
    particle_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    bin_num: int,
) -> Dict:
    """
    Calculate penetration factor (p) for a particle bin.

    p = C_inside / C_outside averaged over the pre-shower window.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of analysis window
        window_end (datetime): End of analysis window
        bin_num (int): Particle bin number (0-6)

    Returns:
        Dict: Dictionary with p value and statistics
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to window
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < 10:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "c_inside_mean": np.nan,
            "c_outside_mean": np.nan,
            "n_points": len(window_data),
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum 10 required)",
        }

    c_inside = np.asarray(window_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(window_data[col_outside].values, dtype=np.float64)

    # Remove invalid points
    valid_mask = (
        (c_inside > 0)
        & (c_outside > 0)
        & (~np.isnan(c_inside))
        & (~np.isnan(c_outside))
    )

    if np.sum(valid_mask) < 10:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "c_inside_mean": np.nan,
            "c_outside_mean": np.nan,
            "n_points": np.sum(valid_mask),
            "skip_reason": f"Insufficient valid points: {np.sum(valid_mask)}",
        }

    c_inside_valid = c_inside[valid_mask]
    c_outside_valid = c_outside[valid_mask]

    # Calculate p for each point
    p_values = c_inside_valid / c_outside_valid

    # Filter unreasonable p values
    reasonable_mask = (p_values >= MIN_PENETRATION) & (p_values <= MAX_PENETRATION)

    if np.sum(reasonable_mask) < 10:
        return {
            "p_mean": np.nan,
            "p_std": np.nan,
            "c_inside_mean": float(np.mean(c_inside_valid)),
            "c_outside_mean": float(np.mean(c_outside_valid)),
            "n_points": np.sum(reasonable_mask),
            "skip_reason": f"Insufficient reasonable p values: {np.sum(reasonable_mask)}",
        }

    p_reasonable = p_values[reasonable_mask]

    return {
        "p_mean": float(np.mean(p_reasonable)),
        "p_std": float(np.std(p_reasonable)),
        "c_inside_mean": float(np.mean(c_inside_valid)),
        "c_outside_mean": float(np.mean(c_outside_valid)),
        "n_points": len(p_reasonable),
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
    Calculate deposition rate (β_deposition) using numerical approach.

    Solves: β_deposition = 1/Δt - λ - C_t(i+1)/(C_t Δt) + (pλC_out,t)/C_t

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        window_start (datetime): Start of decay window
        window_end (datetime): End of decay window
        bin_num (int): Particle bin number (0-6)
        p (float): Penetration factor
        lambda_ach (float): Air change rate (h⁻¹)

    Returns:
        Dict: Dictionary with β statistics
    """
    bin_info = PARTICLE_BINS[bin_num]
    col_inside = f"{bin_info['column']}_inside"
    col_outside = f"{bin_info['column']}_outside"

    # Filter to window
    mask = (particle_data["datetime"] >= window_start) & (
        particle_data["datetime"] <= window_end
    )
    window_data = particle_data[mask].copy()

    if len(window_data) < 20:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "n_points": len(window_data),
            "skip_reason": f"Insufficient data: {len(window_data)} points (minimum 20 required)",
        }

    c_inside = np.asarray(window_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(window_data[col_outside].values, dtype=np.float64)

    # Check for sufficient concentration difference
    c_ratio = c_inside[0] / np.mean(c_outside)
    if c_ratio < MIN_CONCENTRATION_RATIO:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "n_points": 0,
            "skip_reason": f"Insufficient concentration ratio: {c_ratio:.2f} (minimum {MIN_CONCENTRATION_RATIO})",
        }

    # Calculate β for each time step
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

    if len(beta_values) < 10:
        return {
            "beta_mean": np.nan,
            "beta_std": np.nan,
            "beta_median": np.nan,
            "n_points": len(beta_values),
            "skip_reason": f"Insufficient valid β values: {len(beta_values)}",
        }

    return {
        "beta_mean": float(np.mean(beta_values)),
        "beta_std": float(np.std(beta_values)),
        "beta_median": float(np.median(beta_values)),
        "n_points": len(beta_values),
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

    if len(shower_data) < 5:
        return {
            "E_mean": np.nan,
            "E_std": np.nan,
            "E_median": np.nan,
            "E_total": np.nan,
            "n_points": len(shower_data),
            "skip_reason": f"Insufficient data: {len(shower_data)} points (minimum 5 required)",
        }

    c_inside = np.asarray(shower_data[col_inside].values, dtype=np.float64)
    c_outside = np.asarray(shower_data[col_outside].values, dtype=np.float64)

    V = BEDROOM_VOLUME_M3  # m³
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
        "event_number": event["event_number"],
        "shower_on": event["shower_on"],
        "shower_off": event["shower_off"],
        "shower_duration_min": event["shower_duration_min"],
        "lambda_ach": lambda_ach,
    }

    for bin_num in PARTICLE_BINS.keys():
        # Calculate penetration factor
        p_result = calculate_penetration_factor(
            particle_data,
            event["penetration_start"],
            event["penetration_end"],
            bin_num,
        )

        results[f"bin{bin_num}_p_mean"] = p_result.get("p_mean", np.nan)
        results[f"bin{bin_num}_p_std"] = p_result.get("p_std", np.nan)

        # Skip further calculations if p is invalid
        if np.isnan(p_result.get("p_mean", np.nan)):
            results[f"bin{bin_num}_beta_mean"] = np.nan
            results[f"bin{bin_num}_beta_std"] = np.nan
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = p_result.get(
                "skip_reason", "Unknown"
            )
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
    print(f"Bedroom volume: {BEDROOM_VOLUME_M3} m³")
    print(f"Time step: {TIME_STEP_MINUTES} minute(s)")
    print(f"Penetration window: {PENETRATION_WINDOW_HOURS} hour(s) before shower")
    print(f"Deposition window: {DEPOSITION_WINDOW_HOURS} hour(s) after shower")

    # Set output directory
    if output_dir is None:
        output_dir = get_data_root() / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load particle data
    particle_data = load_and_merge_quantaq_data()

    # Load shower log and identify events
    print("\nLoading shower log...")
    shower_log = load_shower_log()
    events = identify_shower_events(shower_log)
    print(f"Found {len(events)} shower events")

    # Load CO2 lambda results
    co2_results = load_co2_lambda_results()

    # Match events with CO2 lambda values (by closest decay_start time)
    for event in events:
        # Find matching CO2 event (within ±30 minutes of shower_off)
        time_diffs = abs(
            (co2_results["decay_start"] - event["shower_off"]).dt.total_seconds() / 60
        )
        closest_idx = time_diffs.idxmin()

        if time_diffs[closest_idx] < 30:
            event["lambda_ach"] = co2_results.loc[closest_idx, "lambda_average_mean"]
        else:
            event["lambda_ach"] = np.nan
            print(
                f"  Warning: No matching CO2 event for shower {event['event_number']} "
                f"({event['shower_on'].strftime('%Y-%m-%d %H:%M')})"
            )

    # Analyze each event
    print("\nAnalyzing shower events...")
    results = []

    # Setup plot directory
    plot_dir = output_dir / "plots"
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)

    for event in events:
        event_num = event["event_number"]
        lambda_ach = event.get("lambda_ach", np.nan)

        if np.isnan(lambda_ach):
            print(f"  Event {event_num}: Skipped (no λ from CO2 analysis)")
            continue

        print(
            f"  Event {event_num}/{len(events)}: "
            f"{event['shower_on'].strftime('%Y-%m-%d %H:%M')} "
            f"(λ={lambda_ach:.4f} h⁻¹)"
        )

        result = analyze_event_all_bins(particle_data, event, lambda_ach)
        results.append(result)

        # Print summary for this event
        valid_bins = 0
        for bin_num in PARTICLE_BINS.keys():
            if not np.isnan(result.get(f"bin{bin_num}_E_mean", np.nan)):
                valid_bins += 1

        print(f"    Successfully analyzed {valid_bins}/{len(PARTICLE_BINS)} bins")

        # Generate individual event plot if enabled (all bins on one plot)
        if generate_plots and valid_bins > 0:
            try:
                from scripts.plot_particle import plot_particle_decay_event

                plot_path = plot_dir / f"event_{event_num:02d}_pm_decay.png"
                plot_particle_decay_event(
                    particle_data=particle_data,
                    event=event,
                    particle_bins=PARTICLE_BINS,
                    result=result,
                    output_path=plot_path,
                    event_number=event_num,
                )
            except ImportError:
                pass  # Already warned about missing plot module

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print overall statistics
    print("\n" + "=" * 80)
    print("Overall Results Summary")
    print("=" * 80)

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
            print(f"  p (penetration):     {valid_p.mean():.3f} ± {valid_p.std():.3f}")
        if len(valid_beta) > 0:
            print(
                f"  β (deposition):      {valid_beta.mean():.3f} ± {valid_beta.std():.3f} h⁻¹"
            )
        if len(valid_E) > 0:
            print(
                f"  E (emission):        {valid_E.mean():.2e} ± {valid_E.std():.2e} #/min"
            )
        print(f"  Valid events:        {len(valid_E)}/{len(results)}")

    # Save results
    output_file = output_dir / "particle_analysis_summary.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Main results
        results_df.to_excel(writer, sheet_name="all_results", index=False)

        # Separate sheets for each metric
        p_cols = ["event_number", "shower_on"] + [
            f"bin{i}_p_mean" for i in PARTICLE_BINS.keys()
        ]
        beta_cols = ["event_number", "shower_on"] + [
            f"bin{i}_beta_mean" for i in PARTICLE_BINS.keys()
        ]
        E_cols = ["event_number", "shower_on"] + [
            f"bin{i}_E_mean" for i in PARTICLE_BINS.keys()
        ]

        results_df[p_cols].to_excel(writer, sheet_name="p_penetration", index=False)
        results_df[beta_cols].to_excel(
            writer, sheet_name="beta_deposition", index=False
        )
        results_df[E_cols].to_excel(writer, sheet_name="E_emission", index=False)

    print(f"\nResults saved to: {output_file}")

    # Generate plots if enabled
    if generate_plots:
        print("\nGenerating plots...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Import plot_particle functions
        try:
            from scripts.plot_particle import (
                plot_emission_summary,
                plot_penetration_summary,
            )

            # Generate summary plots
            plot_penetration_summary(
                results_df, PARTICLE_BINS, plot_dir / "penetration_summary.png"
            )
            plot_emission_summary(
                results_df, PARTICLE_BINS, plot_dir / "emission_summary.png"
            )

            print(f"  Plots saved to: {plot_dir}")
        except ImportError:
            print("  Warning: plot_particle module not found. Skipping plots.")

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
