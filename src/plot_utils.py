#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Utilities for EPA Legionella Project
==============================================

Backward-compatibility re-export module that aggregates all public plotting
symbols from four specialized submodules (plot_style, plot_co2,
plot_environmental, plot_particle) into a single namespace. Callers that
previously imported from plot_utils continue to work without modification
while each submodule remains independently importable.

Key Functions:
    plot_style — core styling constants and low-level helpers:
        - create_figure: Create consistently styled matplotlib figures
        - save_figure: Save figures with proper DPI and format
        - format_datetime_axis: Apply standard datetime axis formatting
        - add_vertical_marker / add_shaded_window: Annotate time axes
        - add_shower_on_marker / add_shower_off_marker: Shower event markers
        - apply_style / format_test_name_for_filename / format_test_name_for_title

    plot_co2 — CO2 decay analysis plots:
        - plot_co2_decay_event: CO2 decay curves (numerical method)
        - plot_co2_decay_event_analytical: CO2 decay curves (analytical method)
        - plot_lambda_summary: Summary bar charts of air-change rates
        - add_injection_marker: CO2 injection event markers

    plot_environmental — environmental sensor time series and comparisons:
        - plot_environmental_time_series: RH / temperature / wind time series
        - plot_pre_post_comparison: Box plots comparing pre vs. post shower
        - plot_sensor_summary_bars: Bar charts comparing sensor readings
        - add_shower_markers / add_analysis_windows: Overlay event annotations

    plot_particle — particle decay and emission analysis plots:
        - plot_particle_decay_event: Per-event / per-bin decay curves
        - plot_penetration_summary: Penetration factors by particle size
        - plot_deposition_summary: Deposition rates by particle size
        - plot_emission_summary: Emission rates by particle size
        - plot_size_distribution_summary: Multi-panel summary of all metrics

Module Features:
    - Single import point for all project plotting functions
    - __all__ defines the explicit public API surface
    - Style constants (COLORS, FIGURE_DPI, FONT_SIZE_*, etc.) also re-exported
    - Shower marker styles (SHOWER_ON_STYLE, SHOWER_OFF_STYLE) and SENSOR_COLORS

Methodology:
    1. Each submodule is imported at module load time via explicit named imports
    2. __all__ enumerates every re-exported symbol for IDE autocompletion and
       wildcard-import safety
    3. No processing logic resides in this file; all behavior is delegated to
       the respective submodules

Output Files:
    None — this module contains no standalone execution logic. Output files
    (PNG / PDF figures) are produced by the individual plotting functions when
    called from analysis scripts.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

# =============================================================================
# Re-export all public symbols from submodules for backward compatibility
# =============================================================================

# Core styling constants and utilities
from src.plot_style import (
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
    SHOWER_OFF_STYLE,
    SHOWER_ON_STYLE,
    WINDOW_ALPHA,
    add_shaded_window,
    add_shower_off_marker,
    add_shower_on_marker,
    add_vertical_marker,
    apply_style,
    create_figure,
    format_datetime_axis,
    format_test_name_for_filename,
    format_test_name_for_title,
    save_figure,
)

# CO2 decay analysis plots
from src.plot_co2 import (
    add_injection_marker,
    plot_co2_decay_event,
    plot_co2_decay_event_analytical,
    plot_lambda_summary,
)

# Environmental data plots
from src.plot_environmental import (
    add_analysis_windows,
    add_shower_markers,
    plot_environmental_time_series,
    plot_pre_post_comparison,
    plot_sensor_summary_bars,
)

# Particle decay and emission plots
from src.plot_particle import (
    plot_deposition_summary,
    plot_emission_summary,
    plot_particle_decay_event,
    plot_penetration_summary,
    plot_size_distribution_summary,
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
    "SHOWER_OFF_STYLE",
    "SHOWER_ON_STYLE",
    "WINDOW_ALPHA",
    # Core utilities
    "add_shaded_window",
    "add_shower_off_marker",
    "add_shower_on_marker",
    "add_vertical_marker",
    "apply_style",
    "create_figure",
    "format_datetime_axis",
    "format_test_name_for_filename",
    "format_test_name_for_title",
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
    # Particle plots
    "plot_deposition_summary",
    "plot_emission_summary",
    "plot_particle_decay_event",
    "plot_penetration_summary",
    "plot_size_distribution_summary",
]
