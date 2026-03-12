#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Styling Configuration and Core Utilities
=============================================

Centralised styling constants and helper functions for generating
publication-quality figures across the EPA Legionella project. All
calling scripts apply these shared settings so that every plot maintains
consistent fonts, colors, line widths, and DPI regardless of which
analysis module produces it.

Key Functions:
    - apply_style(): Set global matplotlib rcParams (fonts, DPI, grid, spines)
    - create_figure(): Create figures with standard size, layout, and style applied
    - save_figure(): Write figures to disk at FIGURE_DPI with tight bounding box
    - format_datetime_axis(): Format datetime x-axis with major/minor tick locators
    - format_title(): Build event figure titles with optional event number and datetime
    - format_test_name_for_filename(): Sanitize test name for filesystem-safe filenames
    - format_test_name_for_title(): Convert test name to space-separated title form
    - get_config_color(): Map a W## config key to a WATER_TEMP_COLORS hex color
    - add_vertical_marker(): Draw a vertical line at a specific datetime on an axes
    - add_shower_on_marker(): Add shower ON marker using SHOWER_ON_STYLE (green dotted)
    - add_shower_off_marker(): Add shower OFF marker using SHOWER_OFF_STYLE (red dotted)
    - add_shaded_window(): Shade a time window between two datetimes on an axes

Processing Features:
    - COLORS dict: 17 named colorblind-friendly entries for CO2, environmental sensors,
      shower events, and wind data (single source of truth for shower_on/off colors)
    - SENSOR_COLORS list: 16 distinct colors for multi-sensor time-series plots
    - WATER_TEMP_COLORS dict: 12 temperature keys (11–53 °C) mapping to a cool-blue →
      green → amber → dark-red diverging palette for W## water temperature groups
    - CONFIG_COLORS dict: door-position (Open/Closed/Partial) and fan-status colors
    - BOXPLOT_CONFIG dict: centralized geometry and axis ranges shared by all five
      summary boxplot functions (figsize, temp x-axis, box widths, alpha, flier style)
    - SHOWER_ON_STYLE / SHOWER_OFF_STYLE: dotted ":" line dicts distinguishing shower
      event markers from fitted/predicted "--" lines; reference COLORS dict for single
      source of truth on shower_on/off hex values
    - get_config_color() strips W##b / W##pw suffixes before nearest-neighbor lookup
      in WATER_TEMP_COLORS, falling back to CONFIG_KEY_COLORS for unrecognized keys
    - format_datetime_axis() guards against MinuteLocator crash on spans exceeding 2 days

Methodology:
    1. Calling scripts invoke apply_style() once at startup to set rcParams project-wide
    2. create_figure() produces figures with constrained_layout and auto-sized defaults
    3. Per-series colors are drawn from COLORS, SENSOR_COLORS, or get_config_color()
    4. Shower-timing annotations use add_shower_on_marker() / add_shower_off_marker()
       and add_shaded_window() with styles centrally defined in this module
    5. format_title() and format_test_name_*() standardize titles and output filenames
    6. save_figure() writes output at FIGURE_DPI = 300 dpi with tight bounding box

Output Files:
    None; consumed by plot_co2.py, plot_particle.py, plot_environmental.py,
    plot_utils.py, and the three analysis scripts. All figure files are saved
    by the calling scripts.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union, overload

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 18
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 11
FONT_SIZE_ANNOTATION = 11

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
    "shower_on": "#2ca02c",  # Green
    "shower_off": "#d62728",  # Red
    "pre_shower": "#1f77b4",  # Blue
    "post_shower": "#ff7f0e",  # Orange
    # Wind data colors
    "wind_speed": "#9467bd",  # Purple
    "wind_direction": "#17becf",  # Cyan
}

# Extended color list for multi-sensor plots (16 distinct colorblind-friendly colors)
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
    "#393b79",  # Dark indigo
    "#637939",  # Olive green
    "#e7ba52",  # Gold
    "#ad494a",  # Dark rose
    "#7b4173",  # Plum
    "#ce6dbd",  # Orchid
]

# Configuration-based colors for grouping by test conditions
# Maps configuration keys to colors for consistent visualization
CONFIG_COLORS = {
    # Door position colors
    "Open": "#2ca02c",  # Green for Open
    "Closed": "#ff7f0e",  # Orange for Closed
    "Partial": "#bcbd22",  # Yellow-green for Partial
    # Fan status colors
    "FanOn": "#e377c2",  # Pink for Fan On
    "FanOff": "#7f7f7f",  # Gray for Fan Off
}

# Temperature-based color palette for W## water temperature codes.
# Colors progress from cool blues (cold) through greens/yellows (warm) to reds (hot),
# using a perceptually uniform diverging palette suitable for scientific plots.
# Keys are numeric temperatures in degrees C; fractional lookup uses nearest match.
WATER_TEMP_COLORS = {
    11: "#313695",  # Dark blue       (11 °C)
    14: "#4575b4",  # Blue            (14 °C)
    22: "#74add1",  # Light blue      (22 °C)
    23: "#abd9e9",  # Pale blue       (23 °C)
    25: "#74c476",  # Green           (25 °C)
    30: "#fee090",  # Light amber     (30 °C)
    37: "#fdae61",  # Orange          (37 °C)
    38: "#fca35c",  # Orange-amber    (38 °C — W38pw Pepco wide spray)
    43: "#f46d43",  # Dark orange     (43 °C)
    48: "#d73027",  # Red             (48 °C)
    52: "#af0a26",  # Deep red        (52 °C — W52pw Pepco wide spray)
    53: "#a50026",  # Dark red        (53 °C)
}

# Color palette for different config_key values (fallback for unknown keys)
CONFIG_KEY_COLORS = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#17becf",  # Cyan
    "#bcbd22",  # Yellow-green
]


def get_config_color(config_key: str, index: int = 0) -> str:
    """
    Get color for a configuration key based on water temperature.

    Extracts the numeric temperature from a W## code in the config_key
    (e.g., "W48_DoorOpen_FanOff" -> 48 -> red, "W48b_DoorOpen_FanOff" -> 48 -> red,
    "W52pw_DoorOpen_FanOff" -> 52 -> deep red) and returns the nearest color from
    WATER_TEMP_COLORS.  Any non-digit suffix on the W## code (single-letter repeats
    such as W48b, or multi-character variants such as W52pw) is stripped before
    lookup.  Falls back to CONFIG_KEY_COLORS for unrecognised keys.

    Parameters:
        config_key: Configuration key string (e.g., "W48_DoorOpen_FanOff")
        index: Fallback index if no W## pattern is found

    Returns:
        Hex color string
    """
    # Extract numeric temperature from W## codes; strip any suffix (W##b, W##pw, etc.)
    if config_key and config_key[0] == "W":
        water_part = config_key.split("_")[0]  # e.g., "W48b"
        numeric_str = "".join(c for c in water_part[1:] if c.isdigit())
        if numeric_str:
            try:
                temp_num = int(numeric_str)
                nearest = min(WATER_TEMP_COLORS, key=lambda t: abs(t - temp_num))
                return WATER_TEMP_COLORS[nearest]
            except (ValueError, TypeError):
                pass

    # Fallback to indexed color
    return CONFIG_KEY_COLORS[index % len(CONFIG_KEY_COLORS)]


# Line styles
LINE_WIDTH_DATA = 1.5
LINE_WIDTH_FIT = 2.0
LINE_WIDTH_ANNOTATION = 1.0

# Marker settings
MARKER_SIZE = 4

# Shower/Activation event marker styles (centralized for consistency).
# Dotted (:) distinguishes shower event markers from fitted/predicted lines (--).
SHOWER_ON_STYLE = {
    "color": COLORS["shower_on"],  # Green (from COLORS dict)
    "linestyle": ":",
    "linewidth": 2.0,
    "alpha": 0.8,
}

SHOWER_OFF_STYLE = {
    "color": COLORS["shower_off"],  # Red (from COLORS dict)
    "linestyle": ":",
    "linewidth": 2.0,
    "alpha": 0.8,
}

# Analysis window shaded region styles
WINDOW_ALPHA = 0.15  # Transparency for shaded analysis windows

# =============================================================================
# Boxplot Configuration
# =============================================================================

# Shared settings for all summary boxplot functions (temperature-vs-metric figures).
# Centralised here so that visual style is consistent across plot_emission_boxplot,
# plot_other_process_rate_boxplot, plot_emission_rate_boxplot, and
# plot_penetration_factor_boxplot without duplicating literal values.
BOXPLOT_CONFIG = {
    # Figure size (width, height) in inches
    "figsize": (16, 9),
    # Fixed water-temperature x-axis for W## config groups
    "temp_xmin": 5,
    "temp_xmax": 60,
    "temp_xtick_step": 5,
    # Box width range — linspace(box_width_min, box_width_max, n_bins)
    "box_width_max": 2.5,
    "box_width_min": 0.4,
    # Box and whisker transparency
    "box_alpha": 0.7,
    # Median line appearance
    "median_color": "black",
    "median_linewidth": 1.5,
    # Flier (outlier point) appearance
    "flier_marker": "o",
    "flier_markersize": 3,
    "flier_alpha": 0.5,
}


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


# Overload signatures for create_figure() to provide precise return types
@overload
def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Optional[list] = None,
) -> Tuple[Figure, Axes]:
    """Single subplot case returns single Axes."""
    ...


@overload
def create_figure(
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Optional[list] = None,
) -> Tuple[Figure, np.ndarray]:
    """Multi-subplot case returns ndarray of Axes."""
    ...


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Optional[list] = None,
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
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

    # squeeze=True (matplotlib default): single subplot returns Axes directly;
    # multi-subplot returns a squeezed ndarray (1-D for a single row or column).
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw if gridspec_kw else None,
        constrained_layout=True,
        squeeze=True,
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

    Major ticks are aligned to wall-clock boundaries (e.g., :00 and :30 for
    interval_minutes=30). Minor ticks are placed every 10 minutes.

    Parameters:
        ax: Matplotlib axes object
        interval_minutes: Interval between major ticks in minutes (must divide 60)
    """
    major_byminute = list(range(0, 60, interval_minutes))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=major_byminute))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Only set minute-level minor ticks if the axis spans ≤2 days; otherwise
    # MinuteLocator generates hundreds of thousands of ticks and crashes.
    x_min, x_max = ax.get_xlim()
    if (x_max - x_min) <= 2.0:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=list(range(0, 60, 10))))
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
    Also sanitizes invalid filename characters.
    Example: "0115_W48_Open_Day_R01" -> "0115_w48_open_day"
    Example: "0123_W11_Open_Night_R??" -> "0123_w11_open_night"

    Parameters:
        test_name: Original test name (e.g., "0115_W48_Open_Day_R01")

    Returns:
        Formatted filename string (lowercase, underscores, no replicate)
    """
    import re

    # Remove replicate number (_R01, _R02, _R??, etc.)
    name = re.sub(r"_R[\d?]+$", "", test_name)
    # Remove invalid Windows filename characters: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    # Convert to lowercase
    return name.lower()


def format_test_name_for_title(test_name: str) -> str:
    """
    Format test name for use in figure titles.

    Removes replicate number and converts to proper case with spaces.
    Example: "0115_W48_Open_Day_R01" -> "0115 W48 Open Day"

    Parameters:
        test_name: Original test name (e.g., "0115_W48_Open_Day_R01")

    Returns:
        Formatted title string (proper case, spaces, no replicate)
    """
    # Remove replicate number (_R01, _R02, etc.)
    import re

    name = re.sub(r"_R\d+$", "", test_name)
    # Replace underscores with spaces
    return name.replace("_", " ")


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
