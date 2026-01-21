"""
Plotting utilities for EPA Legionella Project.

This module provides consistent styling and helper functions for all plots
in the project, ensuring publication-quality figures with uniform appearance.

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
    lambda_value: float,
    lambda_std: float,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    hours_before: float = 1.0,
    hours_after: float = 3.0,
) -> Optional[Figure]:
    """
    Plot CO2 decay data for a single injection event with calculated lambda.

    Creates a two-panel figure:
    - Top: CO2 concentrations from all sensors
    - Bottom: Calculated instantaneous lambda values with mean overlay

    Args:
        co2_data: DataFrame with columns: datetime, C_bedroom, C_entry, C_outside
        injection_time: Datetime of CO2 injection
        lambda_value: Calculated mean air-change rate (h^-1)
        lambda_std: Standard deviation of lambda
        output_path: Path to save figure (optional)
        event_number: Event number for title
        hours_before: Hours before injection to include
        hours_after: Hours after injection to include

    Returns:
        Matplotlib figure object
    """
    # Define time window
    window_start = injection_time - pd.Timedelta(hours=hours_before)
    window_end = injection_time + pd.Timedelta(hours=hours_after)

    # Filter data to window
    mask = (co2_data["datetime"] >= window_start) & (co2_data["datetime"] <= window_end)
    plot_data = co2_data[mask].copy()

    if len(plot_data) == 0:
        return None

    # Create figure with two subplots
    fig, (ax1, ax2) = create_figure(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

    # === Top panel: CO2 concentrations ===
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

    add_injection_marker(ax1, injection_time)

    ax1.set_ylabel("CO$_2$ Concentration (ppm)")
    ax1.legend(loc="upper right")
    ax1.set_ylim(bottom=0)

    # Title
    title = "CO$_2$ Decay Event"
    if event_number is not None:
        title = f"Event {event_number}: {title}"
    title += f"\n{injection_time.strftime('%Y-%m-%d %H:%M')}"
    ax1.set_title(title)

    # === Bottom panel: Lambda calculation verification ===
    # Calculate instantaneous lambda for verification
    lambda_series = _calculate_instantaneous_lambda(plot_data, injection_time)

    if lambda_series is not None and len(lambda_series) > 0:
        ax2.plot(
            lambda_series["datetime"],
            lambda_series["lambda"],
            color=COLORS["lambda"],
            linewidth=LINE_WIDTH_DATA,
            alpha=0.6,
            label="Instantaneous λ",
        )

        # Add mean lambda as horizontal line
        if not np.isnan(lambda_value):
            ax2.axhline(
                lambda_value,
                color=COLORS["fit"],
                linewidth=LINE_WIDTH_FIT,
                linestyle="-",
                label=f"Mean λ = {lambda_value:.3f} ± {lambda_std:.3f} h⁻¹",
            )
            # Add uncertainty band
            ax2.axhspan(
                lambda_value - lambda_std,
                lambda_value + lambda_std,
                color=COLORS["fit"],
                alpha=0.2,
            )

    add_injection_marker(ax2, injection_time)

    ax2.set_ylabel("Air-Change Rate λ (h⁻¹)")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, 5)  # Reasonable range for ACH
    ax2.legend(loc="upper right")

    format_datetime_axis(ax2, interval_minutes=30)

    # Save if path provided
    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def _calculate_instantaneous_lambda(
    co2_data: pd.DataFrame,
    injection_time: datetime,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[pd.DataFrame]:
    """
    Calculate instantaneous lambda values for plotting verification.

    Args:
        co2_data: DataFrame with CO2 concentrations
        injection_time: Time of CO2 injection
        alpha: Fraction from outside
        beta: Fraction from entry

    Returns:
        DataFrame with datetime and lambda columns, or None if calculation fails
    """
    # Only calculate for decay period (after injection + mixing)
    decay_start = injection_time + pd.Timedelta(minutes=20)
    mask = co2_data["datetime"] >= decay_start
    decay_data = co2_data[mask].copy()

    if len(decay_data) < 5:
        return None

    c_bedroom: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_bedroom"].values, dtype=np.float64
    )
    c_outside: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_outside"].values, dtype=np.float64
    )
    c_entry: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_entry"].values, dtype=np.float64
    )

    # Calculate source concentration
    c_source: npt.NDArray[np.float64] = alpha * c_outside + beta * c_entry

    # Calculate dC/dt using gradient (1-minute intervals = 1/60 hour)
    dt_hours = 1.0 / 60.0
    dc_dt: npt.NDArray[np.float64] = np.gradient(c_bedroom, dt_hours)

    # Calculate lambda: λ = -dC/dt / (C_source - C_bedroom)
    denominator: npt.NDArray[np.float64] = c_source - c_bedroom
    lambda_values = np.where(
        np.abs(denominator) > 10,
        -dc_dt / denominator,
        np.nan,
    )

    # Filter unreasonable values
    lambda_values = np.where(
        (lambda_values > 0) & (lambda_values < 10),
        lambda_values,
        np.nan,
    )

    result = pd.DataFrame(
        {
            "datetime": decay_data["datetime"].values,
            "lambda": lambda_values,
        }
    )

    return result


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
