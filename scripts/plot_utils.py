#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Utilities for EPA Legionella Project
==============================================

This module provides consistent styling and helper functions for generating
publication-quality figures across the EPA Legionella project. All plots
produced by this module follow a unified visual style suitable for scientific
publications, presentations, and reports.

This is the main entry point that re-exports all plotting functions from
specialized submodules for backward compatibility:
    - plot_style: Core styling constants and utilities
    - plot_co2: CO2 decay analysis plots
    - plot_environmental: Environmental data plots (RH, Temp, Wind)

Key Functions:
    - create_figure: Create consistently styled matplotlib figures
    - save_figure: Save figures with proper DPI and formatting
    - format_datetime_axis: Apply standard datetime axis formatting
    - add_injection_marker: Add CO2 injection event markers to plots
    - add_shower_markers: Add shower ON/OFF event markers to plots
    - plot_co2_decay_event: Generate CO2 decay plots (numerical method)
    - plot_co2_decay_event_analytical: Generate CO2 decay plots (analytical method)
    - plot_lambda_summary: Generate summary bar charts of air-change rates
    - plot_environmental_time_series: Time series plots for RH/Temp/Wind data
    - plot_pre_post_comparison: Box plots comparing pre vs post shower periods
    - plot_sensor_summary_bars: Bar charts comparing sensor readings

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

# =============================================================================
# Re-export all public symbols from submodules for backward compatibility
# =============================================================================

# Core styling constants and utilities
from scripts.plot_style import (
    COLORS,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_ANNOTATION,
    LINE_WIDTH_DATA,
    LINE_WIDTH_FIT,
    MARKER_SIZE,
    SENSOR_COLORS,
    add_vertical_marker,
    apply_style,
    create_figure,
    format_datetime_axis,
    save_figure,
)

# CO2 decay analysis plots
from scripts.plot_co2 import (
    add_injection_marker,
    plot_co2_decay_event,
    plot_co2_decay_event_analytical,
    plot_lambda_summary,
)

# Environmental data plots
from scripts.plot_environmental import (
    add_analysis_windows,
    add_shower_markers,
    plot_environmental_time_series,
    plot_pre_post_comparison,
    plot_sensor_summary_bars,
)

# Define __all__ for explicit public API
__all__ = [
    # Style constants
    "COLORS",
    "FIGURE_DPI",
    "FIGURE_FORMAT",
    "FONT_FAMILY",
    "FONT_SIZE_ANNOTATION",
    "FONT_SIZE_LABEL",
    "FONT_SIZE_LEGEND",
    "FONT_SIZE_TICK",
    "FONT_SIZE_TITLE",
    "LINE_WIDTH_ANNOTATION",
    "LINE_WIDTH_DATA",
    "LINE_WIDTH_FIT",
    "MARKER_SIZE",
    "SENSOR_COLORS",
    # Core utilities
    "add_vertical_marker",
    "apply_style",
    "create_figure",
    "format_datetime_axis",
    "save_figure",
    # CO2 plots
    "add_injection_marker",
    "plot_co2_decay_event",
    "plot_co2_decay_event_analytical",
    "plot_lambda_summary",
    # Environmental plots
    "add_analysis_windows",
    "add_shower_markers",
    "plot_environmental_time_series",
    "plot_pre_post_comparison",
    "plot_sensor_summary_bars",
]
