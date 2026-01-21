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
    - plot_co2_decay_event: Generate CO2 decay analysis plots with fitted curves
    - plot_lambda_summary: Generate summary bar charts of air-change rates

Processing Features:
    - 300 DPI output for publication-quality resolution
    - Clean scientific style with serif fonts and minimal grid
    - Colorblind-friendly color palette
    - Automatic datetime axis formatting with rotation
    - Exponential decay curve fitting and overlay

Methodology:
    1. Apply consistent matplotlib rcParams for all figures
    2. Create figure with specified dimensions and layout
    3. Plot data with standardized colors, line widths, and markers
    4. Add annotations, legends, and axis labels
    5. Save to PNG format with tight bounding box

Output Files:
    - event_XX_decay.png: Individual CO2 decay event plots
    - lambda_summary.png: Bar chart summary of air-change rates across events

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
}

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

    # Add fitted exponential decay curve if lambda is valid
    if not np.isnan(lambda_value):
        fit_curve = _calculate_exponential_decay(
            co2_data, decay_start, decay_end, lambda_value, alpha, beta
        )
        if fit_curve is not None and len(fit_curve) > 0:
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
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Calculate fitted exponential decay curve for plotting.

    Uses the equation: C(t) = C_source + (C_0 - C_source) * exp(-λ*t)

    Args:
        co2_data: DataFrame with CO2 concentrations
        decay_start: Start of decay window
        decay_end: End of decay window
        lambda_value: Calculated air-change rate (h^-1)
        alpha: Fraction from outside
        beta: Fraction from entry

    Returns:
        DataFrame with datetime and C_fit columns, or None if calculation fails
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
    c_fit: npt.NDArray[np.float64] = c_source_mean + (
        c_bedroom_0 - c_source_mean
    ) * np.exp(-lambda_value * t_hours)

    return pd.DataFrame(
        {
            "datetime": decay_data["datetime"].values,
            "C_fit": c_fit,
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
