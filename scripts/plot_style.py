#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Styling Configuration and Core Utilities
==============================================

This module provides consistent styling constants and core helper functions
for generating publication-quality figures across the EPA Legionella project.

All plots use these shared settings to maintain visual consistency suitable
for scientific publications, presentations, and reports.

Key Components:
    - Style constants: Colors, fonts, line widths, DPI settings
    - apply_style(): Apply matplotlib rcParams for consistent styling
    - create_figure(): Create figures with standard settings
    - save_figure(): Save figures with proper DPI and formatting
    - format_datetime_axis(): Standard datetime axis formatting
    - format_title(): Standard title formatting for consistency

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =============================================================================
# Style Constants
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

# Title formatting - CONSISTENT ACROSS ALL PLOTS
TITLE_FONTWEIGHT = "normal"  # Changed from 'bold' for consistency

# Color palette (colorblind-friendly)
COLORS = {
    # CO2 analysis colors
    "bedroom": "#1f77b4",  # Blue
    "entry": "#ff7f0e",  # Orange
    "outside": "#2ca02c",  # Green
    "lambda": "#d62728",  # Red
    "fit": "#9467bd",  # Purple
    "injection": "#e377c2",  # Pink
    "grid": "#cccccc",
    # Environmental sensor colors
    "inside": "#17becf",  # Cyan
    "living": "#bcbd22",  # Yellow-green
    "family": "#8c564b",  # Brown
    "bathroom": "#e377c2",  # Pink
    # Shower event colors
    "shower_on": "#d62728",  # Red
    "shower_off": "#2ca02c",  # Green
    "pre_shower": "#1f77b4",  # Blue
    "post_shower": "#ff7f0e",  # Orange
    # Wind data colors
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

# Shower/Activation event marker styles (centralized for consistency)
SHOWER_ON_STYLE = {
    "color": "#2ca02c",  # Green
    "linestyle": "--",
    "linewidth": 2.0,
    "alpha": 0.8,
}

SHOWER_OFF_STYLE = {
    "color": "#d62728",  # Red
    "linestyle": "--",
    "linewidth": 2.0,
    "alpha": 0.8,
}

# Analysis window shaded region styles
WINDOW_ALPHA = 0.15  # Transparency for shaded analysis windows


# =============================================================================
# Core Utility Functions
# =============================================================================


def apply_style() -> None:
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
            "figure.max_open_warning": 100,
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
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Optional[list] = None,
) -> Tuple[Figure, Union[Axes, list]]:
    """
    Create a figure with consistent styling.

    Parameters:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size in inches (width, height). Defaults based on layout.
        sharex: Share x-axis among subplots
        sharey: Share y-axis among subplots
        height_ratios: Height ratios for subplots (for gridspec)

    Returns:
        Tuple of (figure, axes)
    """
    apply_style()

    if figsize is None:
        width = 8 if ncols == 1 else 6 * ncols
        height = 4 if nrows == 1 else 3 * nrows
        figsize = (width, height)

    gridspec_kw = {}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw if gridspec_kw else None,
        constrained_layout=True,
    )

    return fig, axes


def save_figure(fig: Figure, filepath: Path, close: bool = True) -> None:
    """
    Save figure to file with consistent settings.

    Parameters:
        fig: Matplotlib figure object
        filepath: Output file path (extension determines format)
        close: Close figure after saving to free memory
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)


def format_datetime_axis(ax: Axes, interval_minutes: int = 30) -> None:
    """
    Format datetime x-axis with appropriate tick locators and formatters.

    Parameters:
        ax: Matplotlib axes object
        interval_minutes: Interval between major ticks in minutes
    """
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval_minutes))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def format_title(
    base_title: str,
    event_number: Optional[int] = None,
    event_datetime: Optional[datetime] = None,
) -> str:
    """
    Format title consistently across all plot types.

    Parameters:
        base_title: Base title string (e.g., "CO2 Decay Analysis")
        event_number: Event number to prepend (optional)
        event_datetime: Datetime to append (optional)

    Returns:
        Formatted title string
    """
    title = base_title
    if event_number is not None:
        title = f"Event {event_number}: {title}"
    if event_datetime is not None:
        title += f"\n{event_datetime.strftime('%Y-%m-%d %H:%M')}"
    return title


def format_test_name_for_filename(test_name: str) -> str:
    """
    Format test name for use in filenames.

    Removes replicate number and converts to lowercase with underscores.
    Example: "0114_HW_Morning_R01" -> "0114_hw_morning"

    Parameters:
        test_name: Original test name (e.g., "0114_HW_Morning_R01")

    Returns:
        Formatted filename string (lowercase, underscores, no replicate)
    """
    # Remove replicate number (_R01, _R02, etc.)
    import re
    name = re.sub(r'_R\d+$', '', test_name)
    # Convert to lowercase
    return name.lower()


def format_test_name_for_title(test_name: str) -> str:
    """
    Format test name for use in figure titles.

    Removes replicate number and converts to proper case with spaces.
    Example: "0114_HW_Morning_R01" -> "0114 HW Morning"

    Parameters:
        test_name: Original test name (e.g., "0114_HW_Morning_R01")

    Returns:
        Formatted title string (proper case, spaces, no replicate)
    """
    # Remove replicate number (_R01, _R02, etc.)
    import re
    name = re.sub(r'_R\d+$', '', test_name)
    # Replace underscores with spaces
    return name.replace('_', ' ')


def add_vertical_marker(
    ax: Axes,
    time: datetime,
    color: str,
    linestyle: str = "--",
    label: Optional[str] = None,
    alpha: float = 0.7,
) -> None:
    """
    Add a vertical line marker at a specific time.

    Parameters:
        ax: Matplotlib axes object
        time: Datetime for the vertical line
        color: Line color
        linestyle: Line style (default '--')
        label: Label for legend (optional)
        alpha: Line transparency
    """
    ax.axvline(
        float(mdates.date2num(time)),
        color=color,
        linestyle=linestyle,
        linewidth=LINE_WIDTH_ANNOTATION,
        label=label,
        alpha=alpha,
    )


def add_shower_on_marker(
    ax: Axes,
    time: datetime,
    label: str = "Shower ON",
) -> None:
    """
    Add shower ON marker with consistent styling.

    Parameters:
        ax: Matplotlib axes object
        time: Datetime when shower turned on
        label: Label for legend
    """
    ax.axvline(
        float(mdates.date2num(time)),
        label=label,
        **SHOWER_ON_STYLE,
    )


def add_shower_off_marker(
    ax: Axes,
    time: datetime,
    label: str = "Shower OFF",
) -> None:
    """
    Add shower OFF marker with consistent styling.

    Parameters:
        ax: Matplotlib axes object
        time: Datetime when shower turned off
        label: Label for legend
    """
    ax.axvline(
        float(mdates.date2num(time)),
        label=label,
        **SHOWER_OFF_STYLE,
    )


def add_shaded_window(
    ax: Axes,
    start_time: datetime,
    end_time: datetime,
    color: str,
    label: Optional[str] = None,
    alpha: Optional[float] = None,
) -> None:
    """
    Add a shaded time window to a plot.

    Parameters:
        ax: Matplotlib axes object
        start_time: Start of the window
        end_time: End of the window
        color: Fill color
        label: Label for legend (optional)
        alpha: Transparency (uses WINDOW_ALPHA if not specified)
    """
    if alpha is None:
        alpha = WINDOW_ALPHA
    ax.axvspan(
        float(mdates.date2num(start_time)),
        float(mdates.date2num(end_time)),
        alpha=alpha,
        color=color,
        label=label,
    )
