#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Utilities for EPA Legionella Project
==============================================

This module provides consistent styling and helper functions for generating
publication-quality figures across the EPA Legionella project. All plots
produced by this module follow a unified visual style suitable for scientific
publications, presentations, and reports.

The utilities ensure that figures maintain consistent formatting regardless
of which analysis script generates them, facilitating professional-quality
output for peer-reviewed publications and EPA reporting requirements.

Key Functions:
    - create_figure: Create consistently styled matplotlib figures
    - save_figure: Save figures with proper DPI and formatting
    - format_datetime_axis: Apply standard datetime axis formatting
    - add_injection_marker: Add CO2 injection event markers to plots
    - add_shower_markers: Add shower ON/OFF event markers to plots
    - plot_co2_decay_event: Generate CO2 decay analysis plots with fitted curves
    - plot_lambda_summary: Generate summary bar charts of air-change rates
    - plot_environmental_time_series: Time series plots for RH/Temp/Wind data
    - plot_pre_post_comparison: Box plots comparing pre vs post shower periods
    - plot_sensor_comparison_bars: Bar charts comparing sensor readings

Processing Features:
    - 300 DPI output for publication-quality resolution
    - Clean scientific style with serif fonts and minimal grid
    - Colorblind-friendly color palette
    - Automatic datetime axis formatting with rotation
    - Exponential decay curve fitting and overlay
    - Shower event annotation support

Methodology:
    1. Apply consistent matplotlib rcParams for all figures
    2. Create figure with specified dimensions and layout
    3. Plot data with standardized colors, line widths, and markers
    4. Add annotations, legends, and axis labels
    5. Save to PNG format with tight bounding box

Output Files:
    - event_XX_decay.png: Individual CO2 decay event plots
    - lambda_summary.png: Bar chart summary of air-change rates across events
    - event_XX_rh_timeseries.png: RH time series around shower events
    - event_XX_temp_timeseries.png: Temperature time series around shower events
    - event_XX_wind_timeseries.png: Wind data time series around shower events
    - rh_pre_post_boxplot.png: Box plot comparing pre/post shower RH
    - temp_pre_post_boxplot.png: Box plot comparing pre/post shower temperature

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

# =============================================================================
# Style Configuration
# =============================================================================

# Figure defaults
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Font settings (clean scientific style)
FONT_FAMILY = "serif"
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOTATION = 8

# Color palette (colorblind-friendly)
COLORS = {
    "bedroom": "#1f77b4",  # Blue
    "entry": "#ff7f0e",  # Orange
    "outside": "#2ca02c",  # Green
    "lambda": "#d62728",  # Red
    "fit": "#9467bd",  # Purple
    "injection": "#e377c2",  # Pink
    "grid": "#cccccc",
    # Extended palette for environmental sensors
    "inside": "#17becf",  # Cyan
    "living": "#bcbd22",  # Yellow-green
    "family": "#8c564b",  # Brown
    "bathroom": "#e377c2",  # Pink
    "shower_on": "#d62728",  # Red
    "shower_off": "#2ca02c",  # Green
    "pre_shower": "#1f77b4",  # Blue
    "post_shower": "#ff7f0e",  # Orange
    "wind_speed": "#9467bd",  # Purple
    "wind_direction": "#17becf",  # Cyan
}

# Extended color list for multi-sensor plots
SENSOR_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Cyan
]

# Line styles
LINE_WIDTH_DATA = 1.5
LINE_WIDTH_FIT = 2.0
LINE_WIDTH_ANNOTATION = 1.0

# Marker settings
MARKER_SIZE = 4


def apply_style():
    """Apply consistent matplotlib style settings for the project."""
    plt.rcParams.update(
        {
            # Font settings
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_TICK,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            # Figure settings
            "figure.dpi": FIGURE_DPI,
            "figure.max_open_warning": 100,  # Suppress warning for many open figures
            "savefig.dpi": FIGURE_DPI,
            "savefig.format": FIGURE_FORMAT,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Axes settings
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid settings
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Legend settings
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            # Line settings
            "lines.linewidth": LINE_WIDTH_DATA,
        }
    )


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[tuple] = None,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple:
    """
    Create a figure with consistent styling.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size in inches (width, height). Defaults based on layout.
        sharex: Share x-axis among subplots
        sharey: Share y-axis among subplots

    Returns:
        Tuple of (figure, axes)
    """
    apply_style()

    if figsize is None:
        width = 8 if ncols == 1 else 6 * ncols
        height = 4 if nrows == 1 else 3 * nrows
        figsize = (width, height)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )

    return fig, axes


def format_datetime_axis(ax, interval_minutes: int = 30):
    """
    Format datetime x-axis with appropriate tick locators and formatters.

    Args:
        ax: Matplotlib axes object
        interval_minutes: Interval between major ticks in minutes
    """
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def add_injection_marker(
    ax,
    injection_time: datetime,
    label: str = "CO2 Injection",
    color: Optional[str] = None,
):
    """
    Add a vertical line marking CO2 injection time.

    Args:
        ax: Matplotlib axes object
        injection_time: Datetime of injection
        label: Label for legend
        color: Line color (defaults to COLORS['injection'])
    """
    if color is None:
        color = COLORS["injection"]
    ax.axvline(
        injection_time,
        color=color,
        linestyle="--",
        linewidth=LINE_WIDTH_ANNOTATION,
        label=label,
        alpha=0.7,
    )


def save_figure(fig, filepath: Path, close: bool = True):
    """
    Save figure to file with consistent settings.

    Args:
        fig: Matplotlib figure object
        filepath: Output file path (extension determines format)
        close: Close figure after saving to free memory
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)


# =============================================================================
# CO2 Decay Analysis Plots
# =============================================================================


def plot_co2_decay_event(
    co2_data: pd.DataFrame,
    injection_time: datetime,
    decay_start: datetime,
    decay_end: datetime,
    lambda_value: float,
    lambda_std: float,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    hours_before: float = 1.0,
    hours_after: float = 3.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[Figure]:
    """
    Plot CO2 decay data for a single injection event with fitted exponential decay.

    Shows CO2 concentrations from all sensors with the fitted exponential decay
    curve overlaid on the bedroom data during the decay analysis window.

    Args:
        co2_data: DataFrame with columns: datetime, C_bedroom, C_entry, C_outside
        injection_time: Datetime of CO2 injection
        decay_start: Start of decay analysis window
        decay_end: End of decay analysis window
        lambda_value: Calculated mean air-change rate (h^-1)
        lambda_std: Standard deviation of lambda
        output_path: Path to save figure (optional)
        event_number: Event number for title
        hours_before: Hours before injection to include
        hours_after: Hours after injection to include
        alpha: Fraction of infiltration from outside
        beta: Fraction of infiltration from entry zone

    Returns:
        Matplotlib figure object or None if no data
    """
    # Define time window
    window_start = injection_time - pd.Timedelta(hours=hours_before)
    window_end = injection_time + pd.Timedelta(hours=hours_after)

    # Filter data to window
    mask = (co2_data["datetime"] >= window_start) & (co2_data["datetime"] <= window_end)
    plot_data = co2_data[mask].copy()

    if len(plot_data) == 0:
        return None

    # Create single-panel figure
    fig, ax = create_figure(figsize=(10, 5))

    # Plot CO2 concentrations
    ax.plot(
        plot_data["datetime"],
        plot_data["C_bedroom"],
        color=COLORS["bedroom"],
        linewidth=LINE_WIDTH_DATA,
        label="Bedroom",
    )
    ax.plot(
        plot_data["datetime"],
        plot_data["C_entry"],
        color=COLORS["entry"],
        linewidth=LINE_WIDTH_DATA,
        label="Entry",
    )
    ax.plot(
        plot_data["datetime"],
        plot_data["C_outside"],
        color=COLORS["outside"],
        linewidth=LINE_WIDTH_DATA,
        label="Outside",
    )

    # Add fitted exponential decay curve with uncertainty band if lambda is valid
    if not np.isnan(lambda_value):
        fit_curve = _calculate_exponential_decay(
            co2_data, decay_start, decay_end, lambda_value, lambda_std, alpha, beta
        )
        if fit_curve is not None and len(fit_curve) > 0:
            # Plot uncertainty band based on calculated λ standard deviation
            if not np.isnan(lambda_std) and lambda_std > 0:
                ax.fill_between(
                    fit_curve["datetime"],
                    fit_curve["C_fit_lower"],
                    fit_curve["C_fit_upper"],
                    color=COLORS["fit"],
                    alpha=0.2,
                    label=f"Fit uncertainty (±{lambda_std:.3f} h⁻¹)",
                )
            # Plot the mean fit line
            ax.plot(
                fit_curve["datetime"],
                fit_curve["C_fit"],
                color=COLORS["fit"],
                linewidth=LINE_WIDTH_FIT,
                linestyle="--",
                label=f"Fit (λ={lambda_value:.3f}±{lambda_std:.3f} h⁻¹)",
            )

    # Add markers for injection and decay window
    add_injection_marker(ax, injection_time)
    ax.axvline(
        decay_start,
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
        label="Decay window",
    )
    ax.axvline(
        decay_end,
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
    )

    ax.set_ylabel("CO$_2$ Concentration (ppm)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # Title
    title = "CO$_2$ Decay Event"
    if event_number is not None:
        title = f"Event {event_number}: {title}"
    title += f"\n{injection_time.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title)

    format_datetime_axis(ax, interval_minutes=30)

    # Save if path provided
    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def _calculate_exponential_decay(
    co2_data: pd.DataFrame,
    decay_start: datetime,
    decay_end: datetime,
    lambda_value: float,
    lambda_std: float = 0.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Calculate fitted exponential decay curve with uncertainty bounds for plotting.

    Uses the equation: C(t) = C_source + (C_0 - C_source) * exp(-λ*t)

    Uncertainty bounds are calculated using λ ± σ:
        - Upper bound (slower decay): λ - σ
        - Lower bound (faster decay): λ + σ

    Args:
        co2_data: DataFrame with CO2 concentrations
        decay_start: Start of decay window
        decay_end: End of decay window
        lambda_value: Calculated air-change rate (h^-1)
        lambda_std: Standard deviation of lambda for uncertainty bounds
        alpha: Fraction from outside
        beta: Fraction from entry

    Returns:
        DataFrame with datetime, C_fit, C_fit_upper, C_fit_lower columns,
        or None if calculation fails
    """
    # Filter to decay window
    mask = (co2_data["datetime"] >= decay_start) & (co2_data["datetime"] <= decay_end)
    decay_data = co2_data[mask].copy()

    if len(decay_data) < 2:
        return None

    # Get initial conditions
    c_bedroom_0 = float(decay_data["C_bedroom"].iloc[0])
    c_outside: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_outside"].values, dtype=np.float64
    )
    c_entry: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_entry"].values, dtype=np.float64
    )

    # Calculate source concentration (time-varying)
    c_source: npt.NDArray[np.float64] = alpha * c_outside + beta * c_entry

    # Calculate time since decay start in hours
    t0 = decay_data["datetime"].iloc[0]
    t_hours: npt.NDArray[np.float64] = np.asarray(
        (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0,
        dtype=np.float64,
    )

    # Use mean source concentration for simplified exponential fit
    c_source_mean = float(np.mean(c_source))

    # Exponential decay: C(t) = C_source + (C_0 - C_source) * exp(-λ*t)
    delta_c = c_bedroom_0 - c_source_mean
    c_fit: npt.NDArray[np.float64] = c_source_mean + delta_c * np.exp(
        -lambda_value * t_hours
    )

    # Calculate uncertainty bounds using λ ± σ
    # Ensure lambda bounds stay positive
    lambda_upper = max(lambda_value - lambda_std, 0.001)  # Slower decay → upper bound
    lambda_lower = lambda_value + lambda_std  # Faster decay → lower bound

    c_fit_upper: npt.NDArray[np.float64] = c_source_mean + delta_c * np.exp(
        -lambda_upper * t_hours
    )
    c_fit_lower: npt.NDArray[np.float64] = c_source_mean + delta_c * np.exp(
        -lambda_lower * t_hours
    )

    return pd.DataFrame(
        {
            "datetime": decay_data["datetime"].values,
            "C_fit": c_fit,
            "C_fit_upper": c_fit_upper,
            "C_fit_lower": c_fit_lower,
        }
    )


def plot_lambda_summary(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Figure:
    """
    Plot summary of lambda values across all events.

    Args:
        results_df: DataFrame with lambda results for all events
        output_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    fig, ax = create_figure(figsize=(10, 5))

    # Get event indices and lambda values
    events = range(1, len(results_df) + 1)
    lambda_avg: npt.NDArray[np.float64] = np.asarray(
        results_df["lambda_average_mean"].values, dtype=np.float64
    )
    lambda_std: npt.NDArray[np.float64] = np.asarray(
        results_df["lambda_average_std"].values, dtype=np.float64
    )

    # Plot bars with error bars
    valid_mask = ~np.isnan(lambda_avg)
    x_valid = [e for e, v in zip(events, valid_mask) if v]
    y_valid = lambda_avg[valid_mask]
    yerr_valid = lambda_std[valid_mask]

    ax.bar(
        x_valid,
        y_valid,
        yerr=yerr_valid,
        color=COLORS["bedroom"],
        alpha=0.7,
        capsize=3,
        label="λ (average method)",
    )

    # Mark skipped events
    x_skipped = [e for e, v in zip(events, valid_mask) if not v]
    if x_skipped:
        ax.scatter(
            x_skipped,
            [0] * len(x_skipped),
            marker="x",
            color=COLORS["lambda"],
            s=50,
            label="Skipped (insufficient data)",
            zorder=5,
        )

    # Add mean line
    overall_mean = np.nanmean(lambda_avg)
    if not np.isnan(overall_mean):
        ax.axhline(
            overall_mean,
            color=COLORS["fit"],
            linestyle="--",
            linewidth=LINE_WIDTH_FIT,
            label=f"Overall mean: {overall_mean:.3f} h⁻¹",
        )

    ax.set_xlabel("Event Number")
    ax.set_ylabel("Air-Change Rate λ (h⁻¹)")
    ax.set_title("Air-Change Rate Summary Across All Events")
    ax.set_xticks(list(events))
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def plot_co2_decay_event_analytical(
    co2_data: pd.DataFrame,
    event: dict,
    result: dict,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[Figure]:
    """
    Plot CO2 decay data with analytical linear regression fit for a single event.

    Creates a two-panel figure:
        1. Top panel: CO2 concentrations vs time with exponential fit curve
        2. Bottom panel: Linearized plot (y vs t) with regression line

    The analytical method uses: -ln[(C(t) - C_avg) / (C_0 - C_avg)] = λ·t
    where λ is determined by linear regression (slope).

    Parameters:
        co2_data (pd.DataFrame): DataFrame with columns: datetime, C_bedroom, C_entry, C_outside
        event (dict): Dict with injection event timing containing:
            - injection_start: datetime of CO2 injection
            - decay_start: datetime to begin decay analysis
            - decay_end: datetime to end decay analysis
        result (dict): Dict with analysis results containing:
            - lambda_average_mean: calculated λ value (h⁻¹)
            - lambda_average_std: standard error of λ
            - lambda_average_r_squared: R² of linear fit
            - c_source_mean: mean source concentration
            - _y_values: list of y values for regression plot
            - _t_values: list of t values for regression plot
        output_path (Path): Path to save figure (optional)
        event_number (int): Event number for title
        alpha (float): Fraction of infiltration from outside
        beta (float): Fraction of infiltration from entry zone

    Returns:
        Matplotlib figure object or None if no data
    """
    apply_style()

    injection_time = event["injection_start"]
    decay_start = event["decay_start"]
    decay_end = event["decay_end"]
    lambda_value = result.get("lambda_average_mean", np.nan)
    lambda_std = result.get("lambda_average_std", np.nan)

    # Define time window for plotting
    hours_before = 1.0
    hours_after = 0.5  # After decay_end
    window_start = injection_time - pd.Timedelta(hours=hours_before)
    window_end = decay_end + pd.Timedelta(hours=hours_after)

    # Filter data to window
    mask = (co2_data["datetime"] >= window_start) & (co2_data["datetime"] <= window_end)
    plot_data = co2_data[mask].copy()

    if len(plot_data) == 0:
        return None

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # -------------------------------------------------------------------------
    # Top panel: CO2 concentrations vs time
    # -------------------------------------------------------------------------
    ax1.plot(
        plot_data["datetime"],
        plot_data["C_bedroom"],
        color=COLORS["bedroom"],
        linewidth=LINE_WIDTH_DATA,
        label="Bedroom",
    )
    ax1.plot(
        plot_data["datetime"],
        plot_data["C_entry"],
        color=COLORS["entry"],
        linewidth=LINE_WIDTH_DATA,
        label="Entry",
    )
    ax1.plot(
        plot_data["datetime"],
        plot_data["C_outside"],
        color=COLORS["outside"],
        linewidth=LINE_WIDTH_DATA,
        label="Outside",
    )

    # Add fitted exponential decay curve if lambda is valid
    if not np.isnan(lambda_value):
        # Get decay window data
        decay_mask = (co2_data["datetime"] >= decay_start) & (
            co2_data["datetime"] <= decay_end
        )
        decay_data = co2_data[decay_mask].copy()

        if len(decay_data) > 0:
            c_0 = float(decay_data["C_bedroom"].iloc[0])
            c_avg = result.get("c_source_mean", result.get("c_outside_mean", 400))

            # Calculate time in hours
            t0 = decay_data["datetime"].iloc[0]
            t_hours = (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0

            # Calculate fitted curve: C(t) = C_avg + (C_0 - C_avg) * exp(-λ*t)
            c_fit = c_avg + (c_0 - c_avg) * np.exp(-lambda_value * t_hours)

            # Format label with uncertainty
            if not np.isnan(lambda_std):
                fit_label = f"Fit (λ={lambda_value:.3f}±{lambda_std:.3f} h⁻¹)"
            else:
                fit_label = f"Fit (λ={lambda_value:.3f} h⁻¹)"

            ax1.plot(
                decay_data["datetime"],
                c_fit,
                color=COLORS["fit"],
                linewidth=LINE_WIDTH_FIT,
                linestyle="--",
                label=fit_label,
            )

    # Add markers for injection and decay window
    add_injection_marker(ax1, injection_time)
    ax1.axvline(
        decay_start,
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
        label="Decay window",
    )
    ax1.axvline(
        decay_end,
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
    )

    ax1.set_ylabel("CO$_2$ Concentration (ppm)")
    ax1.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    ax1.set_ylim(bottom=0)

    # Title
    title = "CO$_2$ Decay Analysis (Analytical Method)"
    if event_number is not None:
        title = f"Event {event_number}: {title}"
    title += f"\n{injection_time.strftime('%Y-%m-%d %H:%M')}"
    ax1.set_title(title)

    format_datetime_axis(ax1, interval_minutes=30)

    # -------------------------------------------------------------------------
    # Bottom panel: Linearized plot (y vs t) with regression line
    # -------------------------------------------------------------------------
    y_values = result.get("_y_values", [])
    t_values = result.get("_t_values", [])

    if y_values and t_values and not np.isnan(lambda_value):
        y_arr = np.array(y_values)
        t_arr = np.array(t_values)

        ax2.scatter(
            t_arr, y_arr, color=COLORS["bedroom"], alpha=0.5, s=10, label="Data"
        )

        # Plot regression line (forced through origin)
        t_line = np.array([0, t_arr.max()])
        y_line = lambda_value * t_line
        r_squared = result.get("lambda_average_r_squared", np.nan)

        if not np.isnan(r_squared):
            reg_label = f"λ = {lambda_value:.4f} h⁻¹ (R² = {r_squared:.4f})"
        else:
            reg_label = f"λ = {lambda_value:.4f} h⁻¹"

        ax2.plot(
            t_line,
            y_line,
            color=COLORS["fit"],
            linewidth=LINE_WIDTH_FIT,
            label=reg_label,
        )

        ax2.set_xlabel("Time since decay start (hours)")
        ax2.set_ylabel("$-\\ln[(C(t) - C_{avg}) / (C_0 - C_{avg})]$")
        ax2.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient data for linear regression",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=FONT_SIZE_LABEL,
        )
        ax2.set_xlabel("Time since decay start (hours)")
        ax2.set_ylabel("$-\\ln[(C(t) - C_{avg}) / (C_0 - C_{avg})]$")

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


# =============================================================================
# Environmental Data Plots (RH, Temperature, Wind)
# =============================================================================


def add_shower_markers(
    ax,
    shower_on: datetime,
    shower_off: datetime,
    label_on: str = "Shower ON",
    label_off: str = "Shower OFF",
):
    """
    Add vertical lines marking shower start and end times.

    Args:
        ax: Matplotlib axes object
        shower_on: Datetime when shower turned on
        shower_off: Datetime when shower turned off
        label_on: Label for shower start
        label_off: Label for shower end
    """
    ax.axvline(
        shower_on,
        color=COLORS["shower_on"],
        linestyle="--",
        linewidth=LINE_WIDTH_ANNOTATION,
        label=label_on,
        alpha=0.8,
    )
    ax.axvline(
        shower_off,
        color=COLORS["shower_off"],
        linestyle="--",
        linewidth=LINE_WIDTH_ANNOTATION,
        label=label_off,
        alpha=0.8,
    )


def add_analysis_windows(
    ax,
    pre_start: datetime,
    pre_end: datetime,
    post_start: datetime,
    post_end: datetime,
    alpha: float = 0.1,
):
    """
    Add shaded regions indicating pre and post shower analysis windows.

    Args:
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


def plot_environmental_time_series(
    data_dict: dict,
    shower_on: datetime,
    shower_off: datetime,
    variable_type: str,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    hours_before: float = 1.0,
    hours_after: float = 3.0,
    show_windows: bool = True,
) -> Optional[Figure]:
    """
    Plot time series of environmental data (RH, Temperature, or Wind) around a shower event.

    Args:
        data_dict: Dictionary of {sensor_name: DataFrame} with 'datetime' and value columns
        shower_on: Datetime when shower turned on
        shower_off: Datetime when shower turned off
        variable_type: Type of variable ('rh', 'temperature', or 'wind')
        output_path: Path to save figure (optional)
        event_number: Event number for title
        hours_before: Hours before shower ON to include
        hours_after: Hours after shower OFF to include
        show_windows: If True, shade pre/post analysis windows

    Returns:
        Matplotlib figure object or None if no data
    """
    # Define time window
    window_start = shower_on - pd.Timedelta(hours=hours_before)
    window_end = shower_off + pd.Timedelta(hours=hours_after)

    # Variable-specific settings
    var_settings = {
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
    }

    settings = var_settings.get(variable_type, var_settings["rh"])

    # Create figure
    fig, ax = create_figure(figsize=(12, 6))

    # For wind data, use dual y-axes
    if variable_type == "wind":
        ax2 = ax.twinx()
        speed_plotted = False
        direction_plotted = False

        for sensor_name, df in data_dict.items():
            if df is None or df.empty:
                continue

            if "datetime" not in df.columns:
                continue

            mask = (df["datetime"] >= window_start) & (df["datetime"] <= window_end)
            plot_data = df[mask].copy()

            if len(plot_data) == 0:
                continue

            # Find value column
            value_col = None
            for col in plot_data.columns:
                if col != "datetime":
                    value_col = col
                    break

            if value_col is None:
                continue

            # Determine if this is speed or direction based on sensor name or column
            is_direction = (
                "direction" in sensor_name.lower() or "direction" in value_col.lower()
            )

            if is_direction:
                ax2.plot(
                    plot_data["datetime"],
                    plot_data[value_col],
                    color=COLORS["wind_direction"],
                    linewidth=LINE_WIDTH_DATA,
                    label=sensor_name,
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
                    label=sensor_name,
                    alpha=0.8,
                )
                speed_plotted = True

        if not speed_plotted and not direction_plotted:
            plt.close(fig)
            return None

        # Set y-axis labels with units
        ax.set_ylabel(settings["ylabel"], color=COLORS["wind_speed"])
        ax.tick_params(axis="y", labelcolor=COLORS["wind_speed"])
        ax2.set_ylabel(settings["ylabel2"], color=COLORS["wind_direction"])
        ax2.tick_params(axis="y", labelcolor=COLORS["wind_direction"])

        # Set direction axis limits
        if direction_plotted:
            ax2.set_ylim(0, 360)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Add shower markers
        add_shower_markers(ax, shower_on, shower_off)

        # Get shower marker handles
        lines_shower, labels_shower = ax.get_legend_handles_labels()
        # Filter out previously added lines
        shower_lines = lines_shower[len(lines1) :]
        shower_labels = labels_shower[len(labels1) :]

        ax.legend(
            lines1 + lines2 + shower_lines,
            labels1 + labels2 + shower_labels,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND - 1,
        )

    else:
        # Standard single-axis plotting for RH and temperature
        plotted_any = False
        for i, (sensor_name, df) in enumerate(data_dict.items()):
            if df is None or df.empty:
                continue

            if "datetime" not in df.columns:
                continue

            mask = (df["datetime"] >= window_start) & (df["datetime"] <= window_end)
            plot_data = df[mask].copy()

            if len(plot_data) == 0:
                continue

            # Find value column
            value_col = None
            for col in plot_data.columns:
                if col != "datetime":
                    for pattern in settings["value_col_pattern"]:
                        if pattern.lower() in col.lower():
                            value_col = col
                            break
                    if value_col:
                        break
            if value_col is None:
                non_dt_cols = [c for c in plot_data.columns if c != "datetime"]
                if non_dt_cols:
                    value_col = non_dt_cols[0]

            if value_col is None:
                continue

            color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            ax.plot(
                plot_data["datetime"],
                plot_data[value_col],
                color=color,
                linewidth=LINE_WIDTH_DATA,
                label=sensor_name,
                alpha=0.8,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            return None

        # Add shower markers
        add_shower_markers(ax, shower_on, shower_off)

        # Formatting
        ax.set_ylabel(settings["ylabel"])
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 1, ncol=2)

    # Add analysis window shading
    if show_windows:
        pre_start = shower_on - pd.Timedelta(minutes=30)
        post_end = shower_off + pd.Timedelta(hours=2)
        add_analysis_windows(ax, pre_start, shower_on, shower_off, post_end)

    ax.set_xlabel("Time")

    # Title
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
) -> Figure:
    """
    Create box plots comparing pre-shower vs post-shower distributions for each sensor.

    Args:
        pre_data: Dictionary of {sensor_name: array of pre-shower values}
        post_data: Dictionary of {sensor_name: array of post-shower values}
        variable_type: Type of variable ('rh', 'temperature', or 'wind')
        output_path: Path to save figure (optional)
        title_suffix: Additional text for title

    Returns:
        Matplotlib figure object
    """
    var_settings = {
        "rh": {"ylabel": "Relative Humidity (%)", "title_base": "Relative Humidity"},
        "temperature": {"ylabel": "Temperature (\u00b0C)", "title_base": "Temperature"},
        "wind_speed": {"ylabel": "Wind Speed (m/s)", "title_base": "Wind Speed"},
        "wind_direction": {
            "ylabel": "Wind Direction (\u00b0)",
            "title_base": "Wind Direction",
        },
    }
    settings = var_settings.get(
        variable_type, {"ylabel": "Value", "title_base": variable_type}
    )

    # Get sensors that have both pre and post data
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

    # Prepare data for boxplots
    pre_values = [np.array(pre_data[s]) for s in sensors]
    post_values = [np.array(post_data[s]) for s in sensors]

    # Filter out empty arrays
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

    # Labels and formatting
    ax.set_xticks(positions_pre + 0.35)
    ax.set_xticklabels(sensors, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1)
    ax.set_ylabel(settings["ylabel"])
    ax.set_title(
        f"{settings['title_base']} - Pre vs Post Shower Comparison{title_suffix}"
    )

    # Legend
    from matplotlib.patches import Patch

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

    Args:
        summary_data: DataFrame with sensor names as index and metric columns
        metric_col: Column name for the main metric to plot
        error_col: Column name for error bars (optional)
        variable_type: Type of variable for y-axis label
        output_path: Path to save figure (optional)
        title: Custom title (optional)

    Returns:
        Matplotlib figure object
    """
    var_settings = {
        "rh": {"ylabel": "Relative Humidity (%)"},
        "temperature": {"ylabel": "Temperature (\u00b0C)"},
        "wind_speed": {"ylabel": "Wind Speed (m/s)"},
        "wind_direction": {"ylabel": "Wind Direction (\u00b0)"},
    }
    settings = var_settings.get(variable_type, {"ylabel": "Value"})

    n_sensors = len(summary_data)
    fig, ax = create_figure(figsize=(max(10, n_sensors * 0.6), 6))

    x = np.arange(n_sensors)
    values = summary_data[metric_col].values
    errors = (
        summary_data[error_col].values
        if error_col and error_col in summary_data
        else None
    )

    colors = [SENSOR_COLORS[i % len(SENSOR_COLORS)] for i in range(n_sensors)]

    ax.bar(x, values, yerr=errors, capsize=3, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        summary_data.index, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1
    )
    ax.set_ylabel(settings["ylabel"])

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig
