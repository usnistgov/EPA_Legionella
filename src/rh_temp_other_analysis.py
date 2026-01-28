#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relative Humidity, Temperature & Wind Analysis
===============================================

This script analyzes relative humidity, temperature, and wind data from multiple
instruments around shower events for the EPA Legionella study manufactured home
test facility. The analysis compares pre-shower baseline conditions to post-shower
environmental responses across indoor and outdoor monitoring locations.

Environmental conditions are critical for understanding aerosol behavior, transport,
and potential exposure risks from shower-generated Legionella-containing particles.
This analysis characterizes the indoor climate before and after shower operations.

Key Metrics Calculated:
    - Pre-shower (30 min): Mean, standard deviation of RH, Temperature, Wind
    - Post-shower (2 hr): Mean, standard deviation, min-max range for each variable
    - Delta values: Change between pre and post shower conditions

Analysis Features:
    - Multi-instrument data integration (Aranet4, QuantAQ, Vaisala, AIO2)
    - Automatic shower event detection from state-change log
    - Statistical summary tables for each sensor location
    - Time series visualization around shower events
    - Box plot comparisons of pre vs post shower distributions

Methodology:
    1. Load shower event log to identify shower ON/OFF times
    2. Load environmental data from all configured instruments
    3. For each shower event:
       a. Extract 30-minute pre-shower window (before shower ON)
       b. Extract 2-hour post-shower window (after shower OFF)
    4. Calculate descriptive statistics for each window
    5. Generate summary tables and visualizations
    6. Export results to Excel workbook

Output Files:
    - rh_temp_wind_summary.xlsx: Multi-sheet workbook with all statistics
    - plots/event_XX_rh_timeseries.png: RH time series per event
    - plots/event_XX_temp_timeseries.png: Temperature time series per event
    - plots/event_XX_wind_timeseries.png: Wind data time series per event
    - plots/rh_pre_post_boxplot.png: Pre/post RH comparison
    - plots/temp_pre_post_boxplot.png: Pre/post temperature comparison
    - plots/wind_pre_post_boxplot.png: Pre/post wind comparison

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_manager import (  # noqa: E402
    process_events_with_management,
    filter_events_by_date,
    is_event_excluded,
)
from scripts.plot_utils import (  # noqa: E402
    plot_environmental_time_series,
    plot_pre_post_comparison,
    plot_sensor_summary_bars,
)
from src.data_paths import get_data_root  # noqa: E402
from src.env_data_loader import (  # noqa: E402
    SENSOR_CONFIG,
    identify_shower_events,
    load_sensor_data,
    load_shower_log,
)

# =============================================================================
# Configuration
# =============================================================================

# Analysis window parameters
PRE_SHOWER_MINUTES = 30  # Minutes before shower ON for baseline
POST_SHOWER_HOURS = 2  # Hours after shower OFF for analysis

# Custom display order for RH and Temperature sensors (indoor to outdoor)
# This order is used for bar charts and boxplots
SENSOR_DISPLAY_ORDER = [
    "Vaisala MBa",
    "Vaisala Bed1",
    "Aranet4 Bedroom",
    "QuantAQ Inside",
    "Vaisala Liv",
    "Aranet4 Entry",
    "Aranet4 Outside",
    "QuantAQ Outside",
]


def get_sensor_sort_key(sensor_name: str) -> int:
    """
    Get sort key for a sensor based on the display order.

    Args:
        sensor_name: Full sensor name (e.g., "Vaisala MBa RH")

    Returns:
        Integer sort key (lower = displayed first)
    """
    for i, prefix in enumerate(SENSOR_DISPLAY_ORDER):
        if sensor_name.startswith(prefix):
            return i
    return len(SENSOR_DISPLAY_ORDER)  # Unknown sensors go last


# =============================================================================
# Statistical Analysis Functions
# =============================================================================


def calculate_window_stats(
    data: pd.DataFrame, window_start: datetime, window_end: datetime
) -> Dict:
    """
    Calculate descriptive statistics for a time window.

    Args:
        data: DataFrame with 'datetime' and 'value' columns
        window_start: Start of analysis window
        window_end: End of analysis window

    Returns:
        Dict with mean, std, min, max, range, and n_points
    """
    if data is None or data.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "n_points": 0,
        }

    mask = (data["datetime"] >= window_start) & (data["datetime"] <= window_end)
    window_data = data.loc[mask, "value"].dropna()

    if len(window_data) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "n_points": 0,
        }

    return {
        "mean": float(window_data.mean()),
        "std": float(window_data.std()),
        "min": float(window_data.min()),
        "max": float(window_data.max()),
        "range": float(window_data.max() - window_data.min()),
        "n_points": len(window_data),
    }


def analyze_shower_event(
    event: Dict, sensor_data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Analyze all sensors for a single shower event.

    Args:
        event: Shower event dict with timing information
        sensor_data: Dict of {sensor_name: DataFrame} with loaded data

    Returns:
        Dict of {sensor_name: {pre_stats, post_stats}}
    """
    results = {}

    for sensor_name, data in sensor_data.items():
        pre_stats = calculate_window_stats(data, event["pre_start"], event["shower_on"])
        post_stats = calculate_window_stats(
            data, event["shower_off"], event["post_end"]
        )
        results[sensor_name] = {"pre": pre_stats, "post": post_stats}

    return results


# =============================================================================
# Output Generation Functions
# =============================================================================


def create_summary_dataframe(
    all_results: List[Dict], variable_type: str
) -> pd.DataFrame:
    """
    Create summary DataFrame for a specific variable type across all events.

    Args:
        all_results: List of per-event result dicts
        variable_type: 'rh', 'temperature', 'wind_speed', or 'wind_direction'

    Returns:
        DataFrame with summary statistics per sensor
    """
    sensors = [
        name
        for name, cfg in SENSOR_CONFIG.items()
        if cfg["variable_type"] == variable_type
    ]
    # Sort sensors by display order for RH and temperature
    if variable_type in ("rh", "temperature"):
        sensors = sorted(sensors, key=get_sensor_sort_key)

    rows = []
    for sensor_name in sensors:
        pre_means, pre_stds, pre_maxs = [], [], []
        post_means, post_stds, post_maxs, post_ranges = [], [], [], []

        for event_results in all_results:
            if sensor_name in event_results:
                stats = event_results[sensor_name]
                if not np.isnan(stats["pre"]["mean"]):
                    pre_means.append(stats["pre"]["mean"])
                    pre_stds.append(stats["pre"]["std"])
                    pre_maxs.append(stats["pre"]["max"])
                if not np.isnan(stats["post"]["mean"]):
                    post_means.append(stats["post"]["mean"])
                    post_stds.append(stats["post"]["std"])
                    post_maxs.append(stats["post"]["max"])
                    post_ranges.append(stats["post"]["range"])

        row = {
            "Sensor": sensor_name,
            "Pre_Mean": np.mean(pre_means) if pre_means else np.nan,
            "Pre_Std": np.mean(pre_stds) if pre_stds else np.nan,
            "Pre_Max": np.mean(pre_maxs) if pre_maxs else np.nan,
            "Post_Mean": np.mean(post_means) if post_means else np.nan,
            "Post_Std": np.mean(post_stds) if post_stds else np.nan,
            "Post_Max": np.mean(post_maxs) if post_maxs else np.nan,
            "Post_Range": np.mean(post_ranges) if post_ranges else np.nan,
            "N_Events": len(pre_means),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_event_details_dataframe(
    all_results: List[Dict], events: List[Dict], variable_type: str
) -> pd.DataFrame:
    """
    Create detailed per-event DataFrame for a specific variable type.

    Args:
        all_results: List of per-event result dicts
        events: List of shower event dicts
        variable_type: 'rh', 'temperature', 'wind_speed', or 'wind_direction'

    Returns:
        DataFrame with detailed per-event statistics
    """
    sensors = [
        name
        for name, cfg in SENSOR_CONFIG.items()
        if cfg["variable_type"] == variable_type
    ]
    # Sort sensors by display order for RH and temperature
    if variable_type in ("rh", "temperature"):
        sensors = sorted(sensors, key=get_sensor_sort_key)

    rows = []
    for i, (event, results) in enumerate(zip(events, all_results)):
        for sensor_name in sensors:
            if sensor_name in results:
                stats = results[sensor_name]
                duration_min = event.get("duration_min",
                                        (event["shower_off"] - event["shower_on"]).total_seconds() / 60)
                row = {
                    "Event": event.get("event_number", i + 1),
                    "Test_Name": event.get("test_name", f"Event_{i+1}"),
                    "Date": event["shower_on"].strftime("%Y-%m-%d"),
                    "Shower_ON": event["shower_on"].strftime("%H:%M"),
                    "Shower_OFF": event["shower_off"].strftime("%H:%M"),
                    "Duration_min": duration_min,
                    "Water_Temp": event.get("water_temp", ""),
                    "Time_of_Day": event.get("time_of_day", ""),
                    "Sensor": sensor_name,
                    "Pre_Mean": stats["pre"]["mean"],
                    "Pre_Std": stats["pre"]["std"],
                    "Pre_N": stats["pre"]["n_points"],
                    "Post_Mean": stats["post"]["mean"],
                    "Post_Std": stats["post"]["std"],
                    "Post_Min": stats["post"]["min"],
                    "Post_Max": stats["post"]["max"],
                    "Post_Range": stats["post"]["range"],
                    "Post_N": stats["post"]["n_points"],
                }
                rows.append(row)

    return pd.DataFrame(rows)


def save_results_to_excel(
    all_results: List[Dict], events: List[Dict], output_path: Path
):
    """
    Save all analysis results to an Excel workbook.

    Args:
        all_results: List of per-event result dicts
        events: List of shower event dicts
        output_path: Path for output Excel file
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheets
        for var_type, sheet_name in [
            ("rh", "RH_Summary"),
            ("temperature", "Temp_Summary"),
            ("wind_speed", "WindSpeed_Summary"),
            ("wind_direction", "WindDir_Summary"),
        ]:
            summary_df = create_summary_dataframe(all_results, var_type)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Detail sheets
        for var_type, sheet_name in [
            ("rh", "RH_Details"),
            ("temperature", "Temp_Details"),
            ("wind_speed", "WindSpeed_Details"),
            ("wind_direction", "WindDir_Details"),
        ]:
            details_df = create_event_details_dataframe(all_results, events, var_type)
            if not details_df.empty:
                details_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Event log
        event_df = pd.DataFrame(
            [
                {
                    "Event": e.get("event_number", i + 1),
                    "Test_Name": e.get("test_name", f"Event_{i+1}"),
                    "Shower_ON": e["shower_on"],
                    "Shower_OFF": e["shower_off"],
                    "Duration_min": e.get("duration_min",
                                         (e["shower_off"] - e["shower_on"]).total_seconds() / 60),
                    "Water_Temp": e.get("water_temp", ""),
                    "Time_of_Day": e.get("time_of_day", ""),
                    "Fan_During_Test": e.get("fan_during_test", False),
                    "Pre_Start": e["pre_start"],
                    "Post_End": e["post_end"],
                }
                for i, e in enumerate(events)
            ]
        )
        event_df.to_excel(writer, sheet_name="Event_Log", index=False)

    print(f"Results saved to: {output_path}")


# =============================================================================
# Plot Generation Functions
# =============================================================================


def generate_time_series_plots(
    events: List[Dict],
    output_dir: Path,
    max_events: Optional[int] = None,
    specific_events: Optional[List[int]] = None,
):
    """
    Generate time series plots for each shower event.

    Args:
        events: List of shower event dicts
        output_dir: Directory for output plots
        max_events: Maximum events to plot (None=all, ignored if specific_events set)
        specific_events: List of specific event numbers to plot (1-indexed)
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Determine which events to plot
    if specific_events:
        # Convert 1-indexed to 0-indexed and filter valid indices
        event_indices = [i - 1 for i in specific_events if 1 <= i <= len(events)]
        if not event_indices:
            print(f"\nWarning: No valid event numbers in {specific_events}")
            print(f"  Valid range: 1 to {len(events)}")
            return
        print(f"\nGenerating time series plots for events: {specific_events}")
    elif max_events is not None:
        event_indices = list(range(min(max_events, len(events))))
        print(
            f"\nGenerating time series plots for first {len(event_indices)} events..."
        )
    else:
        # Plot all events by default
        event_indices = list(range(len(events)))
        print(f"\nGenerating time series plots for all {len(event_indices)} events...")

    for idx in event_indices:
        event = events[idx]
        event_num = idx + 1  # 1-indexed for display
        print(f"  Event {event_num}: {event['shower_on'].strftime('%Y-%m-%d %H:%M')}")

        start_date = event["pre_start"] - timedelta(hours=1)
        end_date = event["post_end"] + timedelta(hours=1)

        for var_type in ["rh", "temperature", "wind"]:
            sensors = {
                name: cfg
                for name, cfg in SENSOR_CONFIG.items()
                if cfg["variable_type"].startswith(var_type)
            }

            data_dict = {}
            for sensor_name, cfg in sensors.items():
                data = load_sensor_data(sensor_name, cfg, start_date, end_date)
                if data is not None and not data.empty:
                    data = data.rename(columns={"value": cfg["column"]})
                    data_dict[sensor_name] = data

            if data_dict:
                output_path = (
                    plot_dir / f"event_{event_num:02d}_{var_type}_timeseries.png"
                )
                plot_environmental_time_series(
                    data_dict=data_dict,
                    shower_on=event["shower_on"],
                    shower_off=event["shower_off"],
                    variable_type=var_type,
                    output_path=output_path,
                    event_number=event_num,
                )


def generate_comparison_plots(all_results: List[Dict], output_dir: Path):
    """
    Generate box plots comparing pre vs post shower conditions.

    Args:
        all_results: List of per-event result dicts
        output_dir: Directory for output plots
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison plots...")

    for var_type in ["rh", "temperature", "wind_speed", "wind_direction"]:
        sensors = [
            name
            for name, cfg in SENSOR_CONFIG.items()
            if cfg["variable_type"] == var_type
        ]
        # Sort sensors by display order for RH and temperature
        if var_type in ("rh", "temperature"):
            sensors = sorted(sensors, key=get_sensor_sort_key)

        pre_data = {s: [] for s in sensors}
        post_data = {s: [] for s in sensors}

        for event_results in all_results:
            for sensor_name in sensors:
                if sensor_name in event_results:
                    stats = event_results[sensor_name]
                    if not np.isnan(stats["pre"]["mean"]):
                        pre_data[sensor_name].append(stats["pre"]["mean"])
                    if not np.isnan(stats["post"]["mean"]):
                        post_data[sensor_name].append(stats["post"]["mean"])

        pre_data = {k: v for k, v in pre_data.items() if v}
        post_data = {k: v for k, v in post_data.items() if v}

        if pre_data and post_data:
            output_path = plot_dir / f"{var_type}_pre_post_boxplot.png"
            plot_pre_post_comparison(
                pre_data=pre_data,
                post_data=post_data,
                variable_type=var_type,
                output_path=output_path,
            )
            print(f"  Saved: {output_path.name}")


def generate_summary_bar_charts(all_results: List[Dict], output_dir: Path):
    """
    Generate bar charts showing mean values across sensors.

    Args:
        all_results: List of per-event result dicts
        output_dir: Directory for output plots
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating summary bar charts...")

    var_labels = {
        "rh": "Relative Humidity",
        "temperature": "Temperature",
        "wind_speed": "Wind Speed",
        "wind_direction": "Wind Direction",
    }

    for var_type, var_label in var_labels.items():
        summary_df = create_summary_dataframe(all_results, var_type)
        if summary_df.empty or summary_df["Pre_Mean"].isna().all():
            continue

        # Filter to sensors with data
        summary_df = summary_df.dropna(subset=["Pre_Mean"])
        if summary_df.empty:
            continue

        summary_df = summary_df.set_index("Sensor")

        # Pre-shower summary bar chart
        pre_path = plot_dir / f"{var_type}_pre_shower_summary.png"
        plot_sensor_summary_bars(
            summary_data=summary_df,
            metric_col="Pre_Mean",
            error_col="Pre_Std",
            variable_type=var_type,
            output_path=pre_path,
            title=f"{var_label} - Pre-Shower Mean (30 min baseline)",
        )
        print(f"  Saved: {pre_path.name}")

        # Post-shower summary bar chart
        post_path = plot_dir / f"{var_type}_post_shower_summary.png"
        plot_sensor_summary_bars(
            summary_data=summary_df,
            metric_col="Post_Mean",
            error_col="Post_Std",
            variable_type=var_type,
            output_path=post_path,
            title=f"{var_label} - Post-Shower Mean (2 hr after)",
        )
        print(f"  Saved: {post_path.name}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_rh_temp_analysis(
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
    max_plot_events: Optional[int] = None,
    specific_events: Optional[List[int]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the complete RH, Temperature, and Wind analysis.

    Args:
        output_dir: Output directory (defaults to data_root/output)
        generate_plots: Whether to generate plots
        max_plot_events: Maximum events to plot (None=all, ignored if specific_events set)
        specific_events: List of specific event numbers to plot (1-indexed)

    Returns:
        Tuple of (events list, results list)
    """
    print("=" * 60)
    print("Relative Humidity, Temperature & Wind Analysis")
    print("=" * 60)

    if output_dir is None:
        output_dir = get_data_root() / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shower log and identify raw events
    print("\nLoading shower event log...")
    shower_log = load_shower_log()
    raw_events = identify_shower_events(shower_log, PRE_SHOWER_MINUTES, POST_SHOWER_HOURS)
    print(f"Found {len(raw_events)} raw shower events")

    # Process events using the enhanced event management system
    # This handles:
    # - Date filtering (>= 2026-01-14)
    # - Missing event detection
    # - Test condition naming (e.g., 0114_HW_Morning_R01)
    # - Event exclusions (e.g., 2026-01-22 15:00 tour)
    # - Comprehensive logging to event_log.csv
    print("\nProcessing events with event management system...")

    # Create empty CO2 results DataFrame (not needed for RH/temp analysis)
    import pandas as pd
    co2_results = pd.DataFrame()

    events, co2_events_processed, event_log = process_events_with_management(
        raw_events,
        [],  # CO2 events (not needed for this analysis)
        shower_log,
        co2_results,
        output_dir,
        create_synthetic=False  # No synthetic events needed for RH/temp
    )

    print(f"Processed {len(events)} events for analysis")

    if not events:
        print("No shower events found. Exiting.")
        return [], []

    # Analyze each event
    print(f"\nAnalyzing {len(events)} shower events...")
    all_results = []

    for i, event in enumerate(events):
        event_num = event.get("event_number", i + 1)
        test_name = event.get("test_name", f"Event_{event_num}")
        shower_time = event["shower_on"]

        # Check if excluded
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if is_excluded_flag:
            print(
                f"\n  {test_name}: Skipped (excluded: {exclusion_reason})"
            )
            continue

        duration_min = event.get("duration_min",
                                 (event["shower_off"] - event["shower_on"]).total_seconds() / 60)

        print(
            f"\n  {test_name} ({shower_time.strftime('%Y-%m-%d %H:%M')}): "
            f"duration: {duration_min:.1f} min"
        )

        start_date = event["pre_start"]
        end_date = event["post_end"]

        sensor_data = {}
        for sensor_name, cfg in SENSOR_CONFIG.items():
            data = load_sensor_data(sensor_name, cfg, start_date, end_date)
            if data is not None and not data.empty:
                sensor_data[sensor_name] = data

        if not sensor_data:
            print("    Warning: No sensor data available for this event")
            all_results.append({})
            continue

        print(f"    Loaded data from {len(sensor_data)} sensors")
        event_results = analyze_shower_event(event, sensor_data)
        all_results.append(event_results)

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    excel_path = output_dir / "rh_temp_wind_summary.xlsx"
    save_results_to_excel(all_results, events, excel_path)

    # Generate plots
    if generate_plots:
        generate_time_series_plots(events, output_dir, max_plot_events, specific_events)
        generate_comparison_plots(all_results, output_dir)
        print(f"\nPlots saved to: {output_dir / 'plots'}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for var_type, var_label in [
        ("rh", "Relative Humidity (%)"),
        ("temperature", "Temperature (°C)"),
        ("wind_speed", "Wind Speed (m/s)"),
        ("wind_direction", "Wind Direction (°)"),
    ]:
        summary_df = create_summary_dataframe(all_results, var_type)
        if not summary_df.empty:
            print(f"\n{var_label}:")
            print("-" * 40)
            for _, row in summary_df.iterrows():
                if not np.isnan(row["Pre_Mean"]):
                    print(f"  {row['Sensor']}:")
                    print(
                        f"    Pre:  {row['Pre_Mean']:.2f} ± {row['Pre_Std']:.2f} "
                        f"(max: {row['Pre_Max']:.2f})"
                    )
                    print(
                        f"    Post: {row['Post_Mean']:.2f} ± {row['Post_Std']:.2f} "
                        f"(max: {row['Post_Max']:.2f}, range: {row['Post_Range']:.2f})"
                    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return events, all_results


def parse_event_list(event_str: str) -> List[int]:
    """
    Parse a comma-separated list of event numbers.

    Args:
        event_str: Comma-separated event numbers (e.g., "1,3,5" or "2")

    Returns:
        List of event numbers (1-indexed)
    """
    if not event_str:
        return []
    try:
        return [int(x.strip()) for x in event_str.split(",") if x.strip()]
    except ValueError:
        return []


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Relative Humidity, Temperature & Wind Analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data_root/output)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--max-plot-events",
        type=int,
        default=None,
        help="Limit time series plots to first N events (default: all events)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Specific event numbers to plot, comma-separated (e.g., '1,3,5'). "
        "Overrides --max-plot-events. Events are 1-indexed.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Parse specific events if provided
    specific_events = parse_event_list(args.events) if args.events else None

    run_rh_temp_analysis(
        output_dir=output_dir,
        generate_plots=not args.no_plots,
        max_plot_events=args.max_plot_events,
        specific_events=specific_events,
    )


if __name__ == "__main__":
    main()
