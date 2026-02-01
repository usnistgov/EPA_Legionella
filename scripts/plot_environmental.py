#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Data Plotting Functions
======================================

This module provides plotting functions for environmental data (RH, Temperature,
Wind) analysis around shower events in the EPA Legionella project. Supports
multi-sensor visualization and pre/post shower comparisons.

Key Functions:
    - add_shower_markers: Add shower ON/OFF event markers with consistent styling
    - add_analysis_windows: Add shaded pre/post analysis windows
    - plot_environmental_time_series: Time series plots for RH/Temp/Wind
    - plot_pre_post_comparison: Box plots comparing pre vs post shower
    - plot_sensor_summary_bars: Bar charts of sensor summary statistics
    - simplify_sensor_name: Convert internal names to display names

Plot Features:
    - Multi-sensor overlay with distinct colors per sensor
    - Dual y-axis support for wind speed + direction
    - Shaded pre-shower (30 min) and post-shower (2 hr) windows
    - Box plot distributions for pre/post comparison
    - Configuration-based subplot grouping
    - Sensor display name mapping for cleaner legends

Methodology:
    1. Load time series data for all sensors of given variable type
    2. Filter to event window (±hours before/after shower)
    3. Plot each sensor with distinct color from SENSOR_COLORS
    4. Add shower ON/OFF vertical line markers
    5. Shade analysis windows for baseline and response periods

Output Files:
    - Time series: {test_name}_rh.png, {test_name}_temperature.png, {test_name}_wind.png
    - Comparisons: rh_pre_post_comparison.png, temperature_pre_post_comparison.png
    - Summaries: rh_sensor_summary.png, temperature_sensor_summary.png

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from scripts.plot_style import (
    COLORS,
    CONFIG_KEY_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_ANNOTATION,
    LINE_WIDTH_DATA,
    SENSOR_COLORS,
    SHOWER_OFF_STYLE,
    SHOWER_ON_STYLE,
    TITLE_FONTWEIGHT,
    add_shower_off_marker,
    add_shower_on_marker,
    apply_style,
    create_figure,
    format_datetime_axis,
    format_test_name_for_title,
    get_config_color,
    save_figure,
)


# Suffixes to remove from sensor names (since figure title contains this info)
_SENSOR_NAME_SUFFIXES = [" RH", " Temp", " Speed", " Direction"]

# Mapping from internal sensor names to display names for plots
SENSOR_DISPLAY_NAMES = {
    "Vaisala MBa": "Vaisala Bathroom",
    "Vaisala Bed1": "Vaisala Bedroom",
    "QuantAQ Inside": "QuantAQ Bedroom",
    "Vaisala Liv": "Vaisala Livingroom",
    "Aranet4 Entry": "Aranet4 Livingroom",
    "Aranet4 Outside": "Aranet4 OutsideN",
    "QuantAQ Outside": "QuantAQ OutsideS",
    # Aranet4 Bedroom stays the same (no mapping needed)
}


def simplify_sensor_name(sensor_name: str) -> str:
    """
    Simplify sensor name by removing variable type suffix and applying display mapping.

    The figure title already indicates the variable type, so we remove
    redundant suffixes like "RH", "Temp", "Speed", "Direction".
    Then applies the display name mapping for more descriptive plot labels.

    Parameters:
        sensor_name: Full sensor name (e.g., "Vaisala MBa RH")

    Returns:
        Display name (e.g., "Vaisala Bathroom")
    """
    result = sensor_name
    for suffix in _SENSOR_NAME_SUFFIXES:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break
    # Apply display name mapping if available
    return SENSOR_DISPLAY_NAMES.get(result, result)


def add_shower_markers(
    ax,
    shower_on: datetime,
    shower_off: datetime,
    label_on: str = "Shower ON",
    label_off: str = "Shower OFF",
) -> None:
    """
    Add vertical lines marking shower start and end times using centralized styles.

    Parameters:
        ax: Matplotlib axes object
        shower_on: Datetime when shower turned on
        shower_off: Datetime when shower turned off
        label_on: Label for shower start
        label_off: Label for shower end
    """
    add_shower_on_marker(ax, shower_on, label=label_on)
    add_shower_off_marker(ax, shower_off, label=label_off)


def add_analysis_windows(
    ax,
    pre_start: datetime,
    pre_end: datetime,
    post_start: datetime,
    post_end: datetime,
    alpha: float = 0.1,
) -> None:
    """
    Add shaded regions indicating pre and post shower analysis windows.

    Parameters:
        ax: Matplotlib axes object
        pre_start: Start of pre-shower window
        pre_end: End of pre-shower window (shower ON time)
        post_start: Start of post-shower window (shower OFF time)
        post_end: End of post-shower window
        alpha: Transparency for shaded regions
    """
    ax.axvspan(
        pre_start,
        pre_end,
        alpha=alpha,
        color=COLORS["pre_shower"],
        label="Pre-shower (30 min)",
    )
    ax.axvspan(
        post_start,
        post_end,
        alpha=alpha,
        color=COLORS["post_shower"],
        label="Post-shower (2 hr)",
    )


# Variable-specific settings for environmental plots
_VAR_SETTINGS = {
    "rh": {
        "ylabel": "Relative Humidity (%)",
        "title_base": "Relative Humidity",
        "value_col_pattern": ["rh", "humidity", "RH"],
    },
    "temperature": {
        "ylabel": "Temperature (\u00b0C)",
        "title_base": "Temperature",
        "value_col_pattern": ["temp", "temperature", "T_"],
    },
    "wind": {
        "ylabel": "Wind Speed (m/s)",
        "ylabel2": "Wind Direction (\u00b0)",
        "title_base": "Wind Data",
        "value_col_pattern": ["wind", "Wind"],
    },
    "wind_speed": {
        "ylabel": "Wind Speed (m/s)",
        "title_base": "Wind Speed",
    },
    "wind_direction": {
        "ylabel": "Wind Direction (\u00b0)",
        "title_base": "Wind Direction",
    },
}


def _find_value_column(df: pd.DataFrame, patterns: list) -> Optional[str]:
    """Find the first column matching any of the patterns."""
    for col in df.columns:
        if col == "datetime":
            continue
        for pattern in patterns:
            if pattern.lower() in col.lower():
                return col
    # Fallback to first non-datetime column
    non_dt_cols = [c for c in df.columns if c != "datetime"]
    return non_dt_cols[0] if non_dt_cols else None


def _plot_wind_data(
    ax, data_dict: dict, window_start, window_end, settings: dict
) -> bool:
    """Plot wind data with dual y-axes. Returns True if data was plotted."""
    ax2 = ax.twinx()
    speed_plotted = False
    direction_plotted = False

    for sensor_name, df in data_dict.items():
        if df is None or df.empty or "datetime" not in df.columns:
            continue

        mask = (df["datetime"] >= window_start) & (df["datetime"] <= window_end)
        plot_data = df[mask].copy()

        if len(plot_data) == 0:
            continue

        value_col = None
        for col in plot_data.columns:
            if col != "datetime":
                value_col = col
                break

        if value_col is None:
            continue

        is_direction = (
            "direction" in sensor_name.lower() or "direction" in value_col.lower()
        )

        if is_direction:
            ax2.plot(
                plot_data["datetime"],
                plot_data[value_col],
                color=COLORS["wind_direction"],
                linewidth=LINE_WIDTH_DATA,
                label=simplify_sensor_name(sensor_name),
                alpha=0.8,
                linestyle="--",
            )
            direction_plotted = True
        else:
            ax.plot(
                plot_data["datetime"],
                plot_data[value_col],
                color=COLORS["wind_speed"],
                linewidth=LINE_WIDTH_DATA,
                label=simplify_sensor_name(sensor_name),
                alpha=0.8,
            )
            speed_plotted = True

    if not speed_plotted and not direction_plotted:
        return False

    ax.set_ylabel(settings["ylabel"], color=COLORS["wind_speed"])
    ax.tick_params(axis="y", labelcolor=COLORS["wind_speed"])
    ax2.set_ylabel(settings.get("ylabel2", ""), color=COLORS["wind_direction"])
    ax2.tick_params(axis="y", labelcolor=COLORS["wind_direction"])

    if direction_plotted:
        ax2.set_ylim(0, 360)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=FONT_SIZE_LEGEND - 1,
    )

    return True


def _plot_standard_data(
    ax, data_dict: dict, window_start, window_end, settings: dict
) -> bool:
    """Plot standard (non-wind) environmental data. Returns True if data was plotted."""
    plotted_any = False

    for i, (sensor_name, df) in enumerate(data_dict.items()):
        if df is None or df.empty or "datetime" not in df.columns:
            continue

        mask = (df["datetime"] >= window_start) & (df["datetime"] <= window_end)
        plot_data = df[mask].copy()

        if len(plot_data) == 0:
            continue

        value_col = _find_value_column(plot_data, settings.get("value_col_pattern", []))
        if value_col is None:
            continue

        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        ax.plot(
            plot_data["datetime"],
            plot_data[value_col],
            color=color,
            linewidth=LINE_WIDTH_DATA,
            label=simplify_sensor_name(sensor_name),
            alpha=0.8,
        )
        plotted_any = True

    if plotted_any:
        ax.set_ylabel(settings["ylabel"])
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 1, ncol=2)

    return plotted_any


def plot_environmental_time_series(
    data_dict: dict,
    shower_on: datetime,
    shower_off: datetime,
    variable_type: str,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    test_name: Optional[str] = None,
    hours_before: float = 1.0,
    hours_after: float = 3.0,
    show_windows: bool = True,
) -> Optional[Figure]:
    """
    Plot time series of environmental data around a shower event.

    Parameters:
        data_dict: Dictionary of {sensor_name: DataFrame} with 'datetime' column
        shower_on: Datetime when shower turned on
        shower_off: Datetime when shower turned off
        variable_type: Type of variable ('rh', 'temperature', or 'wind')
        output_path: Path to save figure (optional)
        event_number: Event number for title
        test_name: Test name for title (e.g., "0114_HW_Morning_R01")
        hours_before: Hours before shower ON to include
        hours_after: Hours after shower OFF to include
        show_windows: If True, shade pre/post analysis windows

    Returns:
        Matplotlib figure object or None if no data
    """
    window_start = shower_on - pd.Timedelta(hours=hours_before)
    window_end = shower_off + pd.Timedelta(hours=hours_after)

    settings = _VAR_SETTINGS.get(variable_type, _VAR_SETTINGS["rh"])

    fig, ax = create_figure(figsize=(12, 6))

    # Plot data based on variable type
    if variable_type == "wind":
        plotted = _plot_wind_data(ax, data_dict, window_start, window_end, settings)
    else:
        plotted = _plot_standard_data(ax, data_dict, window_start, window_end, settings)

    if not plotted:
        plt.close(fig)
        return None

    # Add shower markers
    add_shower_markers(ax, shower_on, shower_off)

    # Add analysis window shading
    if show_windows:
        pre_start = shower_on - pd.Timedelta(minutes=30)
        post_end = shower_off + pd.Timedelta(hours=2)
        add_analysis_windows(ax, pre_start, shower_on, shower_off, post_end)

    ax.set_xlabel("Time")

    # Use consistent title formatting: "Event 01 - 0114 HW Morning: Relative Humidity"
    if test_name:
        formatted_name = format_test_name_for_title(test_name)
        title = f"Event {event_number:02d} - {formatted_name}: {settings['title_base']}"
    else:
        title = settings["title_base"]
        if event_number is not None:
            title = f"Event {event_number}: {title}"
        title += f"\n{shower_on.strftime('%Y-%m-%d')}"
    ax.set_title(title)

    format_datetime_axis(ax, interval_minutes=30)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def plot_pre_post_comparison(
    pre_data: dict,
    post_data: dict,
    variable_type: str,
    output_path: Optional[Path] = None,
    title_suffix: str = "",
    config_grouped_data: Optional[dict] = None,
) -> Figure:
    """
    Create box plots comparing pre-shower vs post-shower distributions.

    If config_grouped_data is provided, creates subplots (one per configuration).

    Parameters:
        pre_data: Dictionary of {sensor_name: array of pre-shower values}
        post_data: Dictionary of {sensor_name: array of post-shower values}
        variable_type: Type of variable ('rh', 'temperature', 'wind_speed', etc.)
        output_path: Path to save figure (optional)
        title_suffix: Additional text for title
        config_grouped_data: Optional dict of {config_key: {'pre': {sensor: values}, 'post': {sensor: values}}}

    Returns:
        Matplotlib figure object
    """
    apply_style()
    settings = _VAR_SETTINGS.get(
        variable_type, {"ylabel": "Value", "title_base": variable_type}
    )

    # If config_grouped_data is provided, use subplots
    if config_grouped_data and len(config_grouped_data) > 1:
        config_keys = list(config_grouped_data.keys())
        n_configs = len(config_keys)

        # Determine sensors from all configurations
        all_sensors = set()
        for config_key in config_keys:
            all_sensors.update(config_grouped_data[config_key].get("pre", {}).keys())
            all_sensors.update(config_grouped_data[config_key].get("post", {}).keys())
        sensors = sorted(list(all_sensors))

        if not sensors:
            fig, ax = create_figure(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return fig

        n_sensors = len(sensors)
        fig, axes = plt.subplots(
            n_configs, 1,
            figsize=(max(12, n_sensors * 0.8), 5 * n_configs),
            squeeze=False
        )
        axes = axes.flatten()

        for idx, config_key in enumerate(config_keys):
            ax = axes[idx]
            config_pre = config_grouped_data[config_key].get("pre", {})
            config_post = config_grouped_data[config_key].get("post", {})

            positions_pre = np.arange(n_sensors) * 2
            positions_post = positions_pre + 0.7

            pre_values = [np.array(config_pre.get(s, [])) for s in sensors]
            post_values = [np.array(config_post.get(s, [])) for s in sensors]

            valid_pre = [(i, v) for i, v in enumerate(pre_values) if len(v) > 0]
            valid_post = [(i, v) for i, v in enumerate(post_values) if len(v) > 0]

            # Get config-specific color for boxes
            config_color = get_config_color(config_key, idx)

            if valid_pre:
                bp_pre = ax.boxplot(
                    [v for _, v in valid_pre],
                    positions=[positions_pre[i] for i, _ in valid_pre],
                    widths=0.6,
                    patch_artist=True,
                )
                for patch in bp_pre["boxes"]:
                    patch.set_facecolor(COLORS["pre_shower"])
                    patch.set_alpha(0.7)

            if valid_post:
                bp_post = ax.boxplot(
                    [v for _, v in valid_post],
                    positions=[positions_post[i] for i, _ in valid_post],
                    widths=0.6,
                    patch_artist=True,
                )
                for patch in bp_post["boxes"]:
                    patch.set_facecolor(COLORS["post_shower"])
                    patch.set_alpha(0.7)

            ax.set_xticks((positions_pre + 0.35).tolist())
            simplified_names = [simplify_sensor_name(s) for s in sensors]
            ax.set_xticklabels(simplified_names, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1)
            ax.set_ylabel(settings["ylabel"], fontsize=FONT_SIZE_LABEL)

            n_events = len(list(config_pre.values())[0]) if config_pre else 0
            ax.set_title(
                f"Configuration: {config_key} (n={n_events})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )

            legend_elements = [
                Patch(facecolor=COLORS["pre_shower"], alpha=0.7, label="Pre-shower (30 min)"),
                Patch(facecolor=COLORS["post_shower"], alpha=0.7, label="Post-shower (2 hr)"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=FONT_SIZE_LEGEND)

        fig.suptitle(
            f"{settings['title_base']} - Pre vs Post Shower Comparison{title_suffix}",
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
            y=1.02,
        )

        plt.tight_layout()

        if output_path is not None:
            save_figure(fig, output_path, close=False)

        return fig

    # Original single-plot behavior
    sensors = [s for s in pre_data.keys() if s in post_data]
    if not sensors:
        fig, ax = create_figure(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    n_sensors = len(sensors)
    fig, ax = create_figure(figsize=(max(10, n_sensors * 0.8), 6))

    positions_pre = np.arange(n_sensors) * 2
    positions_post = positions_pre + 0.7

    pre_values = [np.array(pre_data[s]) for s in sensors]
    post_values = [np.array(post_data[s]) for s in sensors]

    valid_pre = [(i, v) for i, v in enumerate(pre_values) if len(v) > 0]
    valid_post = [(i, v) for i, v in enumerate(post_values) if len(v) > 0]

    if valid_pre:
        bp_pre = ax.boxplot(
            [v for _, v in valid_pre],
            positions=[positions_pre[i] for i, _ in valid_pre],
            widths=0.6,
            patch_artist=True,
        )
        for patch in bp_pre["boxes"]:
            patch.set_facecolor(COLORS["pre_shower"])
            patch.set_alpha(0.7)

    if valid_post:
        bp_post = ax.boxplot(
            [v for _, v in valid_post],
            positions=[positions_post[i] for i, _ in valid_post],
            widths=0.6,
            patch_artist=True,
        )
        for patch in bp_post["boxes"]:
            patch.set_facecolor(COLORS["post_shower"])
            patch.set_alpha(0.7)

    ax.set_xticks((positions_pre + 0.35).tolist())
    # Simplify sensor names by removing variable type suffix (title has this info)
    simplified_names = [simplify_sensor_name(s) for s in sensors]
    ax.set_xticklabels(simplified_names, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1)
    ax.set_ylabel(settings["ylabel"])
    ax.set_title(
        f"{settings['title_base']} - Pre vs Post Shower Comparison{title_suffix}"
    )

    legend_elements = [
        Patch(facecolor=COLORS["pre_shower"], alpha=0.7, label="Pre-shower (30 min)"),
        Patch(facecolor=COLORS["post_shower"], alpha=0.7, label="Post-shower (2 hr)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def plot_sensor_summary_bars(
    summary_data: pd.DataFrame,
    metric_col: str,
    error_col: Optional[str] = None,
    variable_type: str = "rh",
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Create bar chart of summary statistics across sensors.

    Parameters:
        summary_data: DataFrame with sensor names as index and metric columns
        metric_col: Column name for the main metric to plot
        error_col: Column name for error bars (optional)
        variable_type: Type of variable for y-axis label
        output_path: Path to save figure (optional)
        title: Custom title (optional)

    Returns:
        Matplotlib figure object
    """
    settings = _VAR_SETTINGS.get(variable_type, {"ylabel": "Value"})

    n_sensors = len(summary_data)
    fig, ax = create_figure(figsize=(max(10, n_sensors * 0.6), 6))

    x = np.arange(n_sensors)
    values = np.asarray(summary_data[metric_col].astype(float).values, dtype=float)
    errors = (
        np.asarray(summary_data[error_col].astype(float).values, dtype=float)
        if error_col and error_col in summary_data
        else None
    )

    colors = [SENSOR_COLORS[i % len(SENSOR_COLORS)] for i in range(n_sensors)]

    ax.bar(x, values, yerr=errors, capsize=3, color=colors, alpha=0.8)

    ax.set_xticks(x)
    # Simplify sensor names by removing variable type suffix (title has this info)
    simplified_names = [simplify_sensor_name(s) for s in summary_data.index]
    ax.set_xticklabels(
        simplified_names, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1
    )
    ax.set_ylabel(settings["ylabel"])

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig
