"""
CO2 Decay & Air-Change Rate (λ) Analysis for EPA Legionella Project.

This script analyzes CO2 decay data from Aranet4 sensors to calculate
the air-change rate (λ) using a numerical approach that accounts for
time-varying indoor and outdoor concentrations.

The air-change rate equation is solved at each timestep:
    dC_bedroom/dt = λ(α·C_outside + β·C_entry - C_bedroom)

Where:
    C_bedroom = CO2 concentration in bedroom (ppm)
    C_outside = CO2 concentration outside (ppm)
    C_entry   = CO2 concentration at entry (ppm)
    α = fraction of infiltration from outside (default 0.5)
    β = fraction of infiltration from entry zone (default 0.5, where α + β = 1)
    λ = air-change rate (h⁻¹)

Analysis periods:
    - CO2 injection: minutes 40-44 of each hour
    - Mixing fan off: minute 45 of each hour
    - Decay analysis: starts 10 min before the hour, continues 2 hours post-injection

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.plot_utils import plot_co2_decay_event, plot_lambda_summary
from src.data_paths import (
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
DECAY_DURATION_HOURS = 2  # Analyze 2 hours of decay

# =============================================================================
# Data Loading Functions
# =============================================================================


def load_aranet_file(filepath: Path) -> pd.DataFrame:
    """
    Load an Aranet4 Excel file and parse the datetime column.

    Args:
        filepath: Path to the Aranet4 Excel file

    Returns:
        DataFrame with parsed datetime index
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

    Args:
        location: Sensor location ('Bedroom', 'Entry', or 'Mh Outside')

    Returns:
        Combined DataFrame with all data for the location
    """
    config = get_instrument_config("Aranet4")
    base_path = get_instrument_path("Aranet4")

    # Get the file pattern for this location
    location_config = config["locations"].get(location, {})
    file_pattern = location_config.get("file_pattern", f"{location}_*_week.xlsx")

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
        DataFrame with columns: datetime, C_bedroom, C_entry, C_outside,
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
    merged: pd.DataFrame | None = None
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

    merged = merged.reset_index()

    print(f"\nMerged data: {len(merged)} rows")
    print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

    return merged


def load_co2_injection_log() -> pd.DataFrame:
    """
    Load the CO2 injection state-change log.

    Returns:
        DataFrame with CO2 injection events
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


def identify_injection_events(co2_log: pd.DataFrame) -> list[dict]:
    """
    Identify CO2 injection events from the log file.

    An injection event is identified when CO2 valve turns ON (value > 0).

    Args:
        co2_log: DataFrame with CO2 injection log

    Returns:
        List of dicts with injection event details:
        {
            'injection_start': datetime when CO2 valve opened,
            'injection_end': datetime when CO2 valve closed,
            'fan_off': datetime when mixing fan turned off,
            'decay_start': datetime to begin decay analysis,
            'decay_end': datetime to end decay analysis
        }
    """
    events = []

    # Find rows where CO2 transitions from 0 to non-zero (injection start)
    co2_log = co2_log.sort_values("datetime_EDT").reset_index(drop=True)

    for i in range(len(co2_log) - 1):
        current_row = co2_log.iloc[i]
        next_row = co2_log.iloc[i + 1]

        # Check for CO2 turning ON
        if current_row["CO2"] == 0 and next_row["CO2"] > 0:
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
            # Start 10 min before the top of the next hour
            hour_after_injection = injection_start.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
            decay_start = hour_after_injection + timedelta(
                minutes=DECAY_START_OFFSET_MIN
            )
            decay_end = hour_after_injection + timedelta(hours=DECAY_DURATION_HOURS)

            events.append(
                {
                    "injection_start": injection_start,
                    "injection_end": injection_end,
                    "fan_off": fan_off,
                    "decay_start": decay_start,
                    "decay_end": decay_end,
                }
            )

    return events


# =============================================================================
# Air-Change Rate Calculation
# =============================================================================


def calculate_lambda_numerical(
    co2_data: pd.DataFrame,
    decay_start: datetime,
    decay_end: datetime,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    source_mode: str = "average",
) -> dict:
    """
    Calculate air-change rate (λ) using numerical approach.

    The equation solved at each timestep:
        λ = (dC_bedroom/dt) / (C_source - C_bedroom)

    Where C_source depends on source_mode:
        - 'average': α·C_outside + β·C_entry
        - 'outside': C_outside only
        - 'entry': C_entry only

    Args:
        co2_data: DataFrame with CO2 concentrations
        decay_start: Start of decay analysis window
        decay_end: End of decay analysis window
        alpha: Fraction of infiltration from outside
        beta: Fraction of infiltration from entry zone
        source_mode: 'average', 'outside', or 'entry'

    Returns:
        Dict with lambda statistics and time series
    """
    # Check if decay window is within data range
    data_start = co2_data["datetime"].min()
    data_end = co2_data["datetime"].max()

    if decay_start > data_end:
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "lambda_series": [],
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
            "lambda_series": [],
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
        # Provide detailed reason for insufficient data
        if len(decay_data) == 0:
            reason = "No data points in decay window"
        else:
            reason = f"Only {len(decay_data)} data points (minimum 10 required)"
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "lambda_series": [],
            "n_points": 0,
            "skip_reason": reason,
        }

    # Get concentration columns as numpy arrays
    c_bedroom = np.asarray(decay_data["C_bedroom"].values, dtype=np.float64)
    c_outside = np.asarray(decay_data["C_outside"].values, dtype=np.float64)
    c_entry = np.asarray(decay_data["C_entry"].values, dtype=np.float64)
    timestamps = decay_data["datetime"].values

    # Calculate dC/dt using central difference
    # Time step in hours (data is at 1-minute intervals)
    dt_hours = 1.0 / 60.0

    # Calculate concentration derivative
    dc_dt = np.gradient(c_bedroom, dt_hours)

    # Calculate source concentration based on mode
    if source_mode == "average":
        c_source = alpha * c_outside + beta * c_entry
    elif source_mode == "outside":
        c_source = c_outside
    elif source_mode == "entry":
        c_source = c_entry
    else:
        raise ValueError(f"Unknown source_mode: {source_mode}")

    # Calculate lambda at each timestep
    # λ = (dC/dt) / (C_source - C_bedroom)
    # Note: During decay, C_bedroom > C_source, so denominator is negative
    # and dC/dt is negative, giving positive lambda
    denominator = c_source - c_bedroom

    # Avoid division by zero or very small denominators
    min_denominator = 10  # ppm - minimum concentration difference
    valid_mask = np.abs(denominator) > min_denominator

    lambda_values = np.full_like(dc_dt, np.nan)
    lambda_values[valid_mask] = -dc_dt[valid_mask] / denominator[valid_mask]

    # Filter out negative or unreasonably high lambda values
    reasonable_mask = (lambda_values > 0) & (lambda_values < 10)
    valid_lambdas = lambda_values[reasonable_mask]

    if len(valid_lambdas) == 0:
        # Build detailed reason
        n_negative = int(np.sum(lambda_values < 0))
        n_too_high = int(np.sum(lambda_values >= 10))
        n_nan = int(np.sum(np.isnan(lambda_values)))
        n_total = len(lambda_values)
        reason_parts = []
        if n_nan > 0:
            reason_parts.append(f"{n_nan}/{n_total} NaN (small concentration gradient)")
        if n_negative > 0:
            reason_parts.append(f"{n_negative}/{n_total} negative (CO2 increasing)")
        if n_too_high > 0:
            reason_parts.append(f"{n_too_high}/{n_total} unreasonably high (≥10 h⁻¹)")
        reason = "; ".join(reason_parts) if reason_parts else "Unknown"
        return {
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "lambda_series": lambda_values.tolist(),
            "n_points": 0,
            "skip_reason": f"No valid λ values: {reason}",
        }

    return {
        "lambda_mean": float(np.nanmean(valid_lambdas)),
        "lambda_std": float(np.nanstd(valid_lambdas)),
        "lambda_median": float(np.nanmedian(valid_lambdas)),
        "lambda_series": lambda_values.tolist(),
        "timestamps": [pd.Timestamp(t).isoformat() for t in timestamps],
        "n_points": len(valid_lambdas),
        "n_total": len(lambda_values),
        "c_bedroom_initial": float(c_bedroom[0]) if len(c_bedroom) > 0 else np.nan,
        "c_bedroom_final": float(c_bedroom[-1]) if len(c_bedroom) > 0 else np.nan,
        "c_outside_mean": float(np.nanmean(c_outside)),
        "c_entry_mean": float(np.nanmean(c_entry)),
    }


def analyze_injection_event(
    co2_data: pd.DataFrame,
    event: dict,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict:
    """
    Analyze a single CO2 injection event.

    Calculates λ using three methods for uncertainty assessment:
        1. C_average (α·C_outside + β·C_entry)
        2. C_outside only
        3. C_entry only

    Args:
        co2_data: DataFrame with CO2 concentrations
        event: Dict with injection event timing
        alpha: Fraction of infiltration from outside
        beta: Fraction of infiltration from entry zone

    Returns:
        Dict with analysis results for all three methods
    """
    result = {
        "injection_start": event["injection_start"],
        "decay_start": event["decay_start"],
        "decay_end": event["decay_end"],
    }

    # Calculate lambda using three methods
    for mode in ["average", "outside", "entry"]:
        lambda_result = calculate_lambda_numerical(
            co2_data,
            event["decay_start"],
            event["decay_end"],
            alpha=alpha,
            beta=beta,
            source_mode=mode,
        )

        result[f"lambda_{mode}_mean"] = lambda_result["lambda_mean"]
        result[f"lambda_{mode}_std"] = lambda_result["lambda_std"]
        result[f"lambda_{mode}_n_points"] = lambda_result["n_points"]

        if mode == "average":
            result["c_bedroom_initial"] = lambda_result.get("c_bedroom_initial", np.nan)
            result["c_bedroom_final"] = lambda_result.get("c_bedroom_final", np.nan)
            result["c_outside_mean"] = lambda_result.get("c_outside_mean", np.nan)
            result["c_entry_mean"] = lambda_result.get("c_entry_mean", np.nan)
            # Capture skip reason if present
            result["skip_reason"] = lambda_result.get("skip_reason", None)

    return result


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_co2_decay_analysis(
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    output_dir: Optional[Path] = None,
    generate_plots: bool = False,
) -> pd.DataFrame:
    """
    Run the complete CO2 decay analysis.

    Args:
        alpha: Fraction of infiltration from outside
        beta: Fraction of infiltration from entry zone
        output_dir: Optional output directory (defaults to data_root/output)
        generate_plots: If True, generate plots for each event and summary

    Returns:
        DataFrame with analysis results for all injection events
    """
    print("=" * 60)
    print("CO2 Decay & Air-Change Rate (λ) Analysis")
    print("=" * 60)
    print(f"Parameters: α={alpha}, β={beta}")

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
    events = identify_injection_events(co2_log)
    print(f"Found {len(events)} injection events")

    # Analyze each event
    print("\nAnalyzing injection events...")
    results = []

    for i, event in enumerate(events):
        print(
            f"  Event {i + 1}/{len(events)}: "
            f"{event['injection_start'].strftime('%Y-%m-%d %H:%M')}"
        )

        result = analyze_injection_event(co2_data, event, alpha, beta)
        results.append(result)

        # Print summary for this event
        if not np.isnan(result["lambda_average_mean"]):
            print(
                f"    λ (average): {result['lambda_average_mean']:.3f} ± "
                f"{result['lambda_average_std']:.3f} h⁻¹"
            )

            # Generate plot for this event if enabled
            if generate_plots:
                plot_dir = output_dir / "plots"
                plot_path = plot_dir / f"event_{i + 1:02d}_decay.png"
                plot_co2_decay_event(
                    co2_data=co2_data,
                    injection_time=event["injection_start"],
                    lambda_value=result["lambda_average_mean"],
                    lambda_std=result["lambda_average_std"],
                    output_path=plot_path,
                    event_number=i + 1,
                )
        else:
            skip_reason = result.get("skip_reason", "Unknown reason")
            print(f"    ⚠ Skipped: {skip_reason}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)

    for mode in ["average", "outside", "entry"]:
        col = f"lambda_{mode}_mean"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            print(f"\nλ using C_{mode}:")
            print(f"  Mean:   {valid_values.mean():.4f} h⁻¹")
            print(f"  Std:    {valid_values.std():.4f} h⁻¹")
            print(f"  Median: {valid_values.median():.4f} h⁻¹")
            print(f"  Range:  {valid_values.min():.4f} - {valid_values.max():.4f} h⁻¹")
            print(f"  N events: {len(valid_values)}")

    # Save results
    output_file = output_dir / "co2_lambda_summary.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save detailed summary
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "alpha": alpha,
        "beta": beta,
        "n_events": len(events),
        "n_valid_events": int(results_df["lambda_average_mean"].notna().sum()),
    }

    for mode in ["average", "outside", "entry"]:
        col = f"lambda_{mode}_mean"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            summary[f"lambda_{mode}_overall_mean"] = float(valid_values.mean())
            summary[f"lambda_{mode}_overall_std"] = float(valid_values.std())

    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / "co2_lambda_overall_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Generate summary plot if enabled
    if generate_plots:
        plot_dir = output_dir / "plots"
        summary_plot_path = plot_dir / "lambda_summary.png"
        plot_lambda_summary(results_df, output_path=summary_plot_path)
        print(f"Plots saved to: {plot_dir}")

    return results_df


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CO2 Decay & Air-Change Rate (λ) Analysis"
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
        "--plot",
        action="store_true",
        help="Generate plots for each event and summary",
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
        generate_plots=args.plot,
    )


if __name__ == "__main__":
    main()
