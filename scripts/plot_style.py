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

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
) -> Tuple[Figure, Axes]:
    """
    Create a figure with consistent styling.

    Parameters:
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
