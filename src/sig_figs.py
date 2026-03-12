#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Significant Figures Utility
============================

This module provides significant figure rounding and formatting for the EPA
Legionella project analysis scripts. It enforces consistent reporting precision
across all calculated output data (CSV/Excel files and figure annotations). The
default behavior applies rounding when scripts run normally; passing --no-sig-figs
to any analysis script disables rounding and preserves full floating-point precision
for debugging or when downstream calculations require exact values.

Key Functions:
    - set_enabled: Enable or disable sig fig rounding globally (call once at startup)
    - is_enabled: Query the current enabled state
    - round_sig_figs: Round a single numeric value to n significant figures
    - fmt_fig: Format a value as a string for figure annotations (uses SIG_FIGS_FIGURE)
    - apply_sig_figs_to_df: Apply rounding to all float columns in a DataFrame

Processing Features:
    - Global enabled flag: set once at script startup, applies to all subsequent calls
    - Two precision constants: SIG_FIGS_DATA=3 for CSV/Excel, SIG_FIGS_FIGURE=2 for figures
    - Column-type awareness: only float64/float32 columns are rounded; integers, strings,
      object, and datetime columns always pass through unchanged
    - Special-value passthrough: NaN, Inf, and zero values are never modified
    - Bedroom_Conditions sheet in rh_temp_wind_summary.xlsx intentionally exempted from
      rounding because those values feed downstream calculations
    - skip_cols parameter in apply_sig_figs_to_df allows per-call column exclusions

Methodology:
    1. Call sf.set_enabled(not args.no_sig_figs) once at the start of each analysis
       script's run function to configure the global state
    2. Before writing DataFrames to CSV or Excel, call sf.apply_sig_figs_to_df(df) to
       round all float columns to SIG_FIGS_DATA significant figures
    3. In plot annotation strings, replace fixed-precision format strings with
       sf.fmt_fig(value, fallback='.4f') to respect the enabled/disabled state
    4. For individual value rounding outside a DataFrame context, call
       sf.round_sig_figs(value, n_sig_figs) directly

Output Files:
    - None; pure utility module — all output is returned to the calling script for
      use in CSV/Excel writes or figure annotation strings

Configuration Constants:
    - SIG_FIGS_DATA: Significant figures for CSV and Excel output (default: 3)
    - SIG_FIGS_FIGURE: Significant figures for figure annotation text (default: 2)

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import math
from typing import List, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION — adjust these constants to change reporting precision
# =============================================================================

# Significant figures for values written to CSV and Excel output files.
# Applies to all calculated float columns (penetration factor, emission rate,
# air-change rate, etc.).  Does NOT apply to integer columns, string columns,
# or the Bedroom_Conditions sheet (used for downstream calculations).
SIG_FIGS_DATA = 3

# Significant figures for numeric text displayed in figure annotations
# (e.g., λ values, R² values, emission rates on boxplots and event plots).
SIG_FIGS_FIGURE = 2

# =============================================================================
# MODULE STATE
# =============================================================================

# Global enabled flag.  Analysis scripts call set_enabled() once at startup
# based on the --no-sig-figs command-line argument.
_enabled: bool = True


# =============================================================================
# PUBLIC API
# =============================================================================


def set_enabled(enabled: bool) -> None:
    """Enable or disable significant figure rounding globally.

    Call this once at the start of an analysis script's run function based on
    the --no-sig-figs argument:

        sf.set_enabled(not args.no_sig_figs)

    Parameters:
        enabled (bool): True to apply sig figs (default behavior),
            False to use full floating-point precision (--no-sig-figs mode).
    """
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    """Return whether significant figure rounding is currently enabled.

    Returns:
        bool: True if sig figs are applied, False if disabled.
    """
    return _enabled


def round_sig_figs(value: float, n_sig_figs: int = SIG_FIGS_DATA) -> float:
    """Round a numeric value to n significant figures.

    Parameters:
        value (float): The numeric value to round.
        n_sig_figs (int): Number of significant figures. Defaults to SIG_FIGS_DATA.

    Returns:
        float: Value rounded to n_sig_figs significant figures, or the
            original value if it is NaN, Inf, or zero.
    """
    if value is None:
        return value
    try:
        if not math.isfinite(value):
            return value
        if value == 0.0:
            return 0.0
        magnitude = math.floor(math.log10(abs(value)))
        factor = 10 ** (n_sig_figs - 1 - magnitude)
        return round(value * factor) / factor
    except (TypeError, ValueError):
        return value


def fmt_fig(
    value: float,
    n_sig_figs: int = SIG_FIGS_FIGURE,
    fallback: str = ".4g",
) -> str:
    """Format a numeric value as a string for figure annotations.

    When sig figs are enabled, rounds to n_sig_figs significant figures and
    returns using the g format specifier (handles both large and small numbers).
    When disabled (--no-sig-figs), uses the fallback format string to preserve
    the original formatting behavior.

    Parameters:
        value (float): The numeric value to format.
        n_sig_figs (int): Number of significant figures when enabled.
            Defaults to SIG_FIGS_FIGURE.
        fallback (str): Python format specifier used when sig figs are disabled
            (e.g., '.4f', '.3f', '.1e'). Defaults to '.4g'.

    Returns:
        str: Formatted string representation of the value.

    Examples:
        sf.fmt_fig(0.12345)            -> '0.12'  (2 sig figs, enabled)
        sf.fmt_fig(0.12345, fallback='.4f')  -> '0.1234'  (when disabled)
        sf.fmt_fig(1.23e6)             -> '1.2e+06'  (2 sig figs, enabled)
    """
    if value is None:
        return "None"
    try:
        if not math.isfinite(value):
            return str(value)
    except TypeError:
        return str(value)

    if not _enabled:
        return f"{value:{fallback}}"

    rounded = round_sig_figs(value, n_sig_figs)
    return f"{rounded:.{n_sig_figs}g}"


def apply_sig_figs_to_df(
    df: pd.DataFrame,
    n_sig_figs: int = SIG_FIGS_DATA,
    skip_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply significant figure rounding to all float columns in a DataFrame.

    Only float64 and float32 columns are modified.  Integer, string, object,
    and datetime columns are always preserved unchanged.  Explicitly listed
    columns in skip_cols are also skipped.

    When sig figs are disabled (set_enabled(False) / --no-sig-figs), this
    function returns the DataFrame unchanged.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        n_sig_figs (int): Number of significant figures. Defaults to SIG_FIGS_DATA.
        skip_cols (list of str, optional): Column names to exclude from rounding
            in addition to non-float columns. Useful for geographic coordinates,
            model parameters, or any other float column that should retain full
            precision.

    Returns:
        pd.DataFrame: A copy of df with float columns rounded to n_sig_figs
            significant figures (or the original df if disabled).
    """
    if not _enabled:
        return df

    result = df.copy()
    skip = set(skip_cols or [])

    for col in result.columns:
        if col in skip:
            continue
        # Only round true floating-point columns
        if not pd.api.types.is_float_dtype(result[col]):
            continue
        result[col] = result[col].apply(
            lambda v: round_sig_figs(v, n_sig_figs) if pd.notna(v) else v
        )

    return result
