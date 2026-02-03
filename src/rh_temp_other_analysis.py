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
    is_event_excluded,
    process_events_with_management,
)
from scripts.event_registry import (  # noqa: E402
    load_event_registry,
    REGISTRY_FILENAME,
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

# Global cache for pre-loaded sensor data (populated once at start)
_SENSOR_DATA_CACHE: Dict[str, pd.DataFrame] = {}

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


def preload_all_sensor_data(events: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Pre-load all sensor data for the entire date range of events.

    This dramatically improves performance by loading data once instead of
    repeatedly for each event. The data is cached globally for reuse.

    Args:
        events: List of shower event dicts with timing information

    Returns:
        Dict mapping sensor names to their full DataFrames
    """
    global _SENSOR_DATA_CACHE

    if _SENSOR_DATA_CACHE:
        return _SENSOR_DATA_CACHE

    if not events:
        return {}

    # Find the overall date range needed (with padding for analysis windows)
    all_starts = [e["pre_start"] for e in events]
    all_ends = [e["post_end"] for e in events]
    global_start = min(all_starts) - timedelta(hours=2)
    global_end = max(all_ends) + timedelta(hours=2)

    print(
        f"\nPre-loading sensor data for {global_start.date()} to {global_end.date()}..."
    )

    for sensor_name, cfg in SENSOR_CONFIG.items():
        data = load_sensor_data(sensor_name, cfg, global_start, global_end)
        if data is not None and not data.empty:
            _SENSOR_DATA_CACHE[sensor_name] = data

    print(f"  Loaded {len(_SENSOR_DATA_CACHE)} sensors into cache")
    return _SENSOR_DATA_CACHE


def get_cached_sensor_data(
    sensor_name: str, start_date: datetime, end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Get sensor data from cache, filtered to the requested date range.

    Args:
        sensor_name: Name of the sensor
        start_date: Start of analysis window
        end_date: End of analysis window

    Returns:
        DataFrame filtered to the date range, or None if not in cache
    """
    if sensor_name not in _SENSOR_DATA_CACHE:
        return None

    data = _SENSOR_DATA_CACHE[sensor_name]
    mask = (data["datetime"] >= start_date) & (data["datetime"] <= end_date)
    filtered = data.loc[mask].copy()

    if filtered.empty:
        return None
    return filtered


def clear_sensor_cache():
    """Clear the global sensor data cache."""
    global _SENSOR_DATA_CACHE
    _SENSOR_DATA_CACHE = {}


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

    Includes config_key column for grouping and adds summary rows per configuration.

    Args:
        all_results: List of per-event result dicts
        events: List of shower event dicts
        variable_type: 'rh', 'temperature', 'wind_speed', or 'wind_direction'

    Returns:
        DataFrame with detailed per-event statistics and configuration summary rows
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
                duration_min = event.get(
                    "duration_min",
                    (event["shower_off"] - event["shower_on"]).total_seconds() / 60,
                )
                row = {
                    "Event": event.get("event_number", i + 1),
                    "Test_Name": event.get("test_name", f"Event_{i + 1}"),
                    "Config_Key": event.get("config_key", ""),
                    "Date": event["shower_on"].strftime("%Y-%m-%d"),
                    "Shower_ON": event["shower_on"].strftime("%H:%M"),
                    "Shower_OFF": event["shower_off"].strftime("%H:%M"),
                    "Duration_min": duration_min,
                    "Water_Temp": event.get("water_temp", ""),
                    "Door_Position": event.get("door_position", ""),
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

    df = pd.DataFrame(rows)

    # Add summary rows per configuration if config_key exists
    if len(df) > 0 and "Config_Key" in df.columns:
        config_keys = df["Config_Key"].dropna().unique()
        if len(config_keys) > 0:
            summary_rows = []
            for config_key in config_keys:
                config_df = df[df["Config_Key"] == config_key]
                for sensor_name in sensors:
                    sensor_df = config_df[config_df["Sensor"] == sensor_name]
                    if len(sensor_df) > 0:
                        summary_row = {
                            "Event": "SUMMARY",
                            "Test_Name": f"Config: {config_key}",
                            "Config_Key": config_key,
                            "Date": "",
                            "Shower_ON": "",
                            "Shower_OFF": "",
                            "Duration_min": "",
                            "Water_Temp": config_key.split("_")[0]
                            if "_" in config_key
                            else "",
                            "Door_Position": "",
                            "Time_of_Day": "",
                            "Sensor": sensor_name,
                            "Pre_Mean": sensor_df["Pre_Mean"].mean(),
                            "Pre_Std": sensor_df["Pre_Mean"].std(),
                            "Pre_N": len(sensor_df),
                            "Post_Mean": sensor_df["Post_Mean"].mean(),
                            "Post_Std": sensor_df["Post_Mean"].std(),
                            "Post_Min": sensor_df["Post_Min"].min(),
                            "Post_Max": sensor_df["Post_Max"].max(),
                            "Post_Range": sensor_df["Post_Max"].max()
                            - sensor_df["Post_Min"].min(),
                            "Post_N": len(sensor_df),
                        }
                        summary_rows.append(summary_row)

            # Append summary rows at the end
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                df = pd.concat([df, summary_df], ignore_index=True)

    return df


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
    # Units for each variable type
    unit_map = {
        "rh": "%",
        "temperature": "degC",
        "wind_speed": "m/s",
        "wind_direction": "deg",
    }

    # Columns that need units (excluding Sensor and N_Events)
    stat_cols = [
        "Pre_Mean",
        "Pre_Std",
        "Pre_Max",
        "Post_Mean",
        "Post_Std",
        "Post_Max",
        "Post_Range",
    ]
    detail_stat_cols = [
        "Pre_Mean",
        "Pre_Std",
        "Post_Mean",
        "Post_Std",
        "Post_Min",
        "Post_Max",
        "Post_Range",
    ]

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
                # Rename columns with units
                unit = unit_map.get(var_type, "")
                rename_map = {
                    col: f"{col} ({unit})"
                    for col in stat_cols
                    if col in summary_df.columns
                }
                summary_df = summary_df.rename(columns=rename_map)
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
                # Rename columns with units
                unit = unit_map.get(var_type, "")
                rename_map = {
                    col: f"{col} ({unit})"
                    for col in detail_stat_cols
                    if col in details_df.columns
                }
                # Also rename Duration_min
                rename_map["Duration_min"] = "Duration (min)"
                details_df = details_df.rename(columns=rename_map)
                details_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Event log with full configuration details
        event_df = pd.DataFrame(
            [
                {
                    "Event": e.get("event_number", i + 1),
                    "Test_Name": e.get("test_name", f"Event_{i + 1}"),
                    "Config_Key": e.get("config_key", ""),
                    "Shower_ON": e["shower_on"],
                    "Shower_OFF": e["shower_off"],
                    "Duration (min)": e.get(
                        "duration_min",
                        (e["shower_off"] - e["shower_on"]).total_seconds() / 60,
                    ),
                    "Water_Temp": e.get("water_temp", ""),
                    "Door_Position": e.get("door_position", ""),
                    "Planned_Fan": e.get("planned_fan", ""),
                    "Fan_During_Test": e.get("fan_during_test", False),
                    "Time_of_Day": e.get("time_of_day", ""),
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
        event_num = event.get(
            "event_number", idx + 1
        )  # Use event_number from event dict
        test_name = event.get("test_name", f"Event_{event_num:02d}")
        print(f"  {test_name}: {event['shower_on'].strftime('%Y-%m-%d %H:%M')}")

        start_date = event["pre_start"] - timedelta(hours=1)
        end_date = event["post_end"] + timedelta(hours=1)

        for var_type in ["rh", "temperature", "wind"]:
            sensors = {
                name: cfg
                for name, cfg in SENSOR_CONFIG.items()
                if cfg["variable_type"].startswith(var_type)
            }

            # Use cached data instead of reloading for each plot
            data_dict = {}
            for sensor_name, cfg in sensors.items():
                data = get_cached_sensor_data(sensor_name, start_date, end_date)
                if data is not None:
                    data = data.rename(columns={"value": cfg["column"]})
                    data_dict[sensor_name] = data

            if data_dict:
                # Format filename: event_01-0114_hw_morning_rh_timeseries.png
                from scripts.plot_style import format_test_name_for_filename

                formatted_name = format_test_name_for_filename(test_name)
                output_path = (
                    plot_dir
                    / f"event_{event_num:02d}-{formatted_name}_{var_type}_timeseries.png"
                )
                plot_environmental_time_series(
                    data_dict=data_dict,
                    shower_on=event["shower_on"],
                    shower_off=event["shower_off"],
                    variable_type=var_type,
                    output_path=output_path,
                    event_number=event_num,
                    test_name=test_name,
                )


def generate_comparison_plots(
    all_results: List[Dict], events: List[Dict], output_dir: Path
):
    """
    Generate box plots comparing pre vs post shower conditions.

    If multiple configurations exist, creates subplots (one per configuration).

    Args:
        all_results: List of per-event result dicts
        events: List of shower event dicts (for configuration info)
        output_dir: Directory for output plots
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison plots...")

    # Check if we have configuration data
    has_config = len(events) > 0 and "config_key" in events[0]
    config_keys = []
    if has_config:
        config_keys = list(
            set(e.get("config_key", "") for e in events if e.get("config_key"))
        )
        config_keys = [k for k in config_keys if k]  # Remove empty strings

    for var_type in ["rh", "temperature", "wind_speed", "wind_direction"]:
        sensors = [
            name
            for name, cfg in SENSOR_CONFIG.items()
            if cfg["variable_type"] == var_type
        ]
        # Sort sensors by display order for RH and temperature
        if var_type in ("rh", "temperature"):
            sensors = sorted(sensors, key=get_sensor_sort_key)

        # Collect data for all events (for backward compatibility)
        pre_data = {s: [] for s in sensors}
        post_data = {s: [] for s in sensors}

        # Also collect data grouped by configuration
        config_grouped_data = {}
        if has_config and len(config_keys) > 1:
            for config_key in config_keys:
                config_grouped_data[config_key] = {
                    "pre": {s: [] for s in sensors},
                    "post": {s: [] for s in sensors},
                }

        for i, event_results in enumerate(all_results):
            event = events[i] if i < len(events) else {}
            event_config = event.get("config_key", "") if has_config else ""

            for sensor_name in sensors:
                if sensor_name in event_results:
                    stats = event_results[sensor_name]
                    if not np.isnan(stats["pre"]["mean"]):
                        pre_data[sensor_name].append(stats["pre"]["mean"])
                        if event_config and event_config in config_grouped_data:
                            config_grouped_data[event_config]["pre"][
                                sensor_name
                            ].append(stats["pre"]["mean"])
                    if not np.isnan(stats["post"]["mean"]):
                        post_data[sensor_name].append(stats["post"]["mean"])
                        if event_config and event_config in config_grouped_data:
                            config_grouped_data[event_config]["post"][
                                sensor_name
                            ].append(stats["post"]["mean"])

        pre_data = {k: v for k, v in pre_data.items() if v}
        post_data = {k: v for k, v in post_data.items() if v}

        # Clean up config_grouped_data (remove empty sensors)
        if config_grouped_data:
            for config_key in config_grouped_data:
                config_grouped_data[config_key]["pre"] = {
                    k: v for k, v in config_grouped_data[config_key]["pre"].items() if v
                }
                config_grouped_data[config_key]["post"] = {
                    k: v
                    for k, v in config_grouped_data[config_key]["post"].items()
                    if v
                }

        if pre_data and post_data:
            output_path = plot_dir / f"{var_type}_pre_post_boxplot.png"
            plot_pre_post_comparison(
                pre_data=pre_data,
                post_data=post_data,
                variable_type=var_type,
                output_path=output_path,
                config_grouped_data=config_grouped_data
                if len(config_keys) > 1
                else None,
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
# Registry Integration
# =============================================================================


def get_events_from_registry(output_dir: Path) -> tuple:
    """
    Try to load shower events from the unified event registry.

    If registry exists, use it for consistent event numbering across all scripts.

    Parameters:
        output_dir: Output directory where registry is stored

    Returns:
        Tuple of (events_list, used_registry: bool)
    """
    registry_path = output_dir / REGISTRY_FILENAME

    if not registry_path.exists():
        return [], False

    try:
        print(f"\nLoading events from registry: {registry_path}")
        registry_df = load_event_registry(registry_path)

        events = []
        for _, row in registry_df.iterrows():
            # Calculate analysis windows if not in registry
            shower_on = pd.to_datetime(row["shower_on"])
            shower_off = pd.to_datetime(row["shower_off"])
            pre_start = shower_on - timedelta(minutes=PRE_SHOWER_MINUTES)
            post_end = shower_off + timedelta(hours=POST_SHOWER_HOURS)

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
                    "shower_on": shower_on,
                    "shower_off": shower_off,
                    "duration_min": row.get("shower_duration_min", 0),
                    "pre_start": pre_start,
                    "post_end": post_end,
                    "is_excluded": row.get("is_excluded", False),
                    "exclusion_reason": row.get("exclusion_reason", ""),
                }
            )

        print(f"  Loaded {len(events)} events from registry")
        return events, True

    except Exception as e:
        print(f"  Warning: Could not load registry: {e}")
        return [], False


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

    # Try to load events from unified registry first (for consistent numbering)
    events, used_registry = get_events_from_registry(output_dir)

    if used_registry:
        print("  Using unified event registry for consistent event numbering")
    else:
        # Fall back to existing event management system
        print("\nNote: Registry not found. Using process_events_with_management().")
        print("  Run 'python scripts/event_registry.py' for unified numbering.\n")

        # Load shower log and identify raw events
        print("Loading shower event log...")
        shower_log = load_shower_log()
        raw_events = identify_shower_events(
            shower_log, PRE_SHOWER_MINUTES, POST_SHOWER_HOURS
        )
        print(f"Found {len(raw_events)} raw shower events")

        # Process events using the enhanced event management system
        print("\nProcessing events with event management system...")

        # Create empty CO2 results DataFrame (not needed for RH/temp analysis)
        co2_results = pd.DataFrame()

        events, co2_events_processed, event_log = process_events_with_management(
            raw_events,
            [],  # CO2 events (not needed for this analysis)
            shower_log,
            co2_results,
            output_dir,
            create_synthetic=False,
        )

    print(f"Processed {len(events)} events for analysis")

    if not events:
        print("No shower events found. Exiting.")
        return [], []

    # Pre-load all sensor data once (major performance optimization)
    preload_all_sensor_data(events)

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
            print(f"\n  {test_name}: Skipped (excluded: {exclusion_reason})")
            continue

        duration_min = event.get(
            "duration_min",
            (event["shower_off"] - event["shower_on"]).total_seconds() / 60,
        )

        print(
            f"\n  {test_name} ({shower_time.strftime('%Y-%m-%d %H:%M')}): "
            f"duration: {duration_min:.1f} min"
        )

        start_date = event["pre_start"]
        end_date = event["post_end"]

        # Use cached data (filtered to event window) instead of reloading
        sensor_data = {}
        for sensor_name in SENSOR_CONFIG.keys():
            data = get_cached_sensor_data(sensor_name, start_date, end_date)
            if data is not None:
                sensor_data[sensor_name] = data

        if not sensor_data:
            print("    Warning: No sensor data available for this event")
            all_results.append({})
            continue

        print(f"    Using data from {len(sensor_data)} sensors")
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
        generate_comparison_plots(all_results, events, output_dir)
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
