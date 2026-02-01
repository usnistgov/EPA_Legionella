#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2 Decay Analysis Plotting Functions
======================================

This module provides plotting functions for CO2 decay and air-change rate (λ)
analysis in the EPA Legionella project. All plots follow a consistent visual
style suitable for scientific publications.

Key Functions:
    - add_injection_marker: Add CO2 injection event markers to axes
    - plot_co2_decay_event: Plot decay with numerical method exponential fit
    - plot_co2_decay_event_analytical: Plot decay with linear regression fit
    - plot_lambda_summary: Summary bar chart of λ values across events

Plot Features:
    - Two-panel figures showing CO2 concentrations and linearized regression
    - Color-coded sensor traces (Bedroom, Entry, Outside)
    - Exponential fit curves with uncertainty bands
    - Injection and decay window markers
    - Configuration-based grouping for multi-configuration experiments

Methodology:
    1. Extract data window around injection event (±hours_before/after)
    2. Plot raw CO2 concentrations from all sensors
    3. Calculate and overlay exponential fit: C(t) = C_avg + (C_0 - C_avg) * exp(-λ*t)
    4. Add linearized regression panel showing y = -ln[(C(t) - C_avg)/(C_0 - C_avg)] vs t
    5. Display λ value, R², and uncertainty in legend

Output Files:
    - Individual event plots: {test_name}_co2_decay.png
    - Summary chart: co2_lambda_summary.png

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

from scripts.plot_style import (
    COLORS,
    CONFIG_KEY_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TITLE,
    LINE_WIDTH_ANNOTATION,
    LINE_WIDTH_DATA,
    LINE_WIDTH_FIT,
    TITLE_FONTWEIGHT,
    apply_style,
    get_config_color,
    create_figure,
    format_datetime_axis,
    format_test_name_for_title,
    format_title,
    save_figure,
)


def add_injection_marker(
    ax,
    injection_time: datetime,
    label: str = "CO2 Injection",
    color: Optional[str] = None,
) -> None:
    """
    Add a vertical line marking CO2 injection time.

    Parameters:
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
    Calculate fitted exponential decay curve with uncertainty bounds.

    Uses: C(t) = C_source + (C_0 - C_source) * exp(-λ*t)

    Parameters:
        co2_data: DataFrame with CO2 concentrations
        decay_start: Start of decay window
        decay_end: End of decay window
        lambda_value: Calculated air-change rate (h^-1)
        lambda_std: Standard deviation of lambda
        alpha: Fraction from outside
        beta: Fraction from entry

    Returns:
        DataFrame with datetime, C_fit, C_fit_upper, C_fit_lower columns
    """
    mask = (co2_data["datetime"] >= decay_start) & (co2_data["datetime"] <= decay_end)
    decay_data = co2_data[mask].copy()

    if len(decay_data) < 2:
        return None

    c_bedroom_0 = float(decay_data["C_bedroom"].iloc[0])
    c_outside: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_outside"].values, dtype=np.float64
    )
    c_entry: npt.NDArray[np.float64] = np.asarray(
        decay_data["C_entry"].values, dtype=np.float64
    )

    c_source: npt.NDArray[np.float64] = alpha * c_outside + beta * c_entry
    c_source_mean = float(np.mean(c_source))

    t0 = decay_data["datetime"].iloc[0]
    t_hours: npt.NDArray[np.float64] = np.asarray(
        (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0,
        dtype=np.float64,
    )

    delta_c = c_bedroom_0 - c_source_mean
    c_fit: npt.NDArray[np.float64] = c_source_mean + delta_c * np.exp(
        -lambda_value * t_hours
    )

    lambda_upper = max(lambda_value - lambda_std, 0.001)
    lambda_lower = lambda_value + lambda_std

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
    Plot CO2 decay data with fitted exponential decay (numerical method).

    Parameters:
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
    window_start = injection_time - pd.Timedelta(hours=hours_before)
    window_end = injection_time + pd.Timedelta(hours=hours_after)

    mask = (co2_data["datetime"] >= window_start) & (co2_data["datetime"] <= window_end)
    plot_data = co2_data[mask].copy()

    if len(plot_data) == 0:
        return None

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

    # Add fitted exponential decay curve
    if not np.isnan(lambda_value):
        fit_curve = _calculate_exponential_decay(
            co2_data, decay_start, decay_end, lambda_value, lambda_std, alpha, beta
        )
        if fit_curve is not None and len(fit_curve) > 0:
            if not np.isnan(lambda_std) and lambda_std > 0:
                ax.fill_between(
                    fit_curve["datetime"],
                    fit_curve["C_fit_lower"],
                    fit_curve["C_fit_upper"],
                    color=COLORS["fit"],
                    alpha=0.2,
                    label=f"Fit uncertainty (±{lambda_std:.3f} h⁻¹)",
                )
            ax.plot(
                fit_curve["datetime"],
                fit_curve["C_fit"],
                color=COLORS["fit"],
                linewidth=LINE_WIDTH_FIT,
                linestyle="--",
                label=f"Fit (λ={lambda_value:.3f}±{lambda_std:.3f} h⁻¹)",
            )

    # Add markers
    add_injection_marker(ax, injection_time)
    ax.axvline(
        float(mdates.date2num(decay_start)),
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
        label="Decay window",
    )
    ax.axvline(
        float(mdates.date2num(decay_end)),
        color=COLORS["lambda"],
        linestyle=":",
        linewidth=LINE_WIDTH_ANNOTATION,
        alpha=0.7,
    )

    ax.set_ylabel("CO$_2$ Concentration (ppm)")
    ax.set_xlabel("Time")

    # Use consistent title formatting
    title = format_title(
        "CO$_2$ Decay Analysis (Numerical Method)",
        event_number=event_number,
        event_datetime=injection_time,
    )
    ax.set_title(title, fontweight=TITLE_FONTWEIGHT)

    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    ax.set_ylim(bottom=0)

    format_datetime_axis(ax, interval_minutes=30)

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig


def plot_co2_decay_event_analytical(
    co2_data: pd.DataFrame,
    event: dict,
    result: dict,
    output_path: Optional[Path] = None,
    event_number: Optional[int] = None,
    test_name: Optional[str] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Optional[Figure]:
    """
    Plot CO2 decay with analytical linear regression fit (two-panel figure).

    Top panel: CO2 concentrations vs time with exponential fit curve
    Bottom panel: Linearized plot (y vs t) with regression line

    Parameters:
        co2_data: DataFrame with columns: datetime, C_bedroom, C_entry, C_outside
        event: Dict with injection_start, decay_start, decay_end
        result: Dict with lambda_average_mean, lambda_average_std,
                lambda_average_r_squared, c_source_mean, _y_values, _t_values
        output_path: Path to save figure (optional)
        event_number: Event number for title
        test_name: Test name for title (e.g., "0114_HW_Morning_R01")
        alpha: Fraction of infiltration from outside
        beta: Fraction of infiltration from entry zone

    Returns:
        Matplotlib figure object or None if no data
    """
    # Use create_figure with height_ratios for consistent styling
    fig, (ax1, ax2) = create_figure(
        nrows=2, ncols=1, figsize=(10, 8), height_ratios=[2, 1]
    )

    injection_time = event["injection_start"]
    decay_start = event["decay_start"]
    decay_end = event["decay_end"]
    lambda_value = result.get("lambda_average_mean", np.nan)
    lambda_std = result.get("lambda_average_std", np.nan)

    hours_before = 1.0
    hours_after = 0.5
    window_start = injection_time - pd.Timedelta(hours=hours_before)
    window_end = decay_end + pd.Timedelta(hours=hours_after)

    mask = (co2_data["datetime"] >= window_start) & (co2_data["datetime"] <= window_end)
    plot_data = co2_data[mask].copy()

    if len(plot_data) == 0:
        plt.close(fig)
        return None

    # Top panel: CO2 concentrations
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

    # Add fitted exponential decay curve
    if not np.isnan(lambda_value):
        decay_mask = (co2_data["datetime"] >= decay_start) & (
            co2_data["datetime"] <= decay_end
        )
        decay_data = co2_data[decay_mask].copy()

        if len(decay_data) > 0:
            c_0 = float(decay_data["C_bedroom"].iloc[0])
            c_avg = result.get("c_source_mean", result.get("c_outside_mean", 400))

            t0 = decay_data["datetime"].iloc[0]
            t_hours = (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0
            c_fit = c_avg + (c_0 - c_avg) * np.exp(-lambda_value * t_hours)

            fit_label = (
                f"Fit (λ={lambda_value:.3f}±{lambda_std:.3f} h⁻¹)"
                if not np.isnan(lambda_std)
                else f"Fit (λ={lambda_value:.3f} h⁻¹)"
            )

            ax1.plot(
                decay_data["datetime"],
                c_fit,
                color=COLORS["fit"],
                linewidth=LINE_WIDTH_FIT,
                linestyle="--",
                label=fit_label,
            )

    # Add markers
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

    # Use consistent title formatting: "Event 01 - 0114 HW Morning: CO2 Decay"
    if test_name:
        formatted_name = format_test_name_for_title(test_name)
        title = f"Event {event_number:02d} - {formatted_name}: CO$_2$ Decay"
    else:
        title = format_title(
            "CO$_2$ Decay Analysis (Analytical Method)",
            event_number=event_number,
            event_datetime=injection_time,
        )
    ax1.set_title(title, fontweight=TITLE_FONTWEIGHT)

    format_datetime_axis(ax1, interval_minutes=30)

    # Bottom panel: Linearized regression plot
    y_values = result.get("_y_values", [])
    t_values = result.get("_t_values", [])

    if y_values and t_values and not np.isnan(lambda_value):
        y_arr = np.array(y_values)
        t_arr = np.array(t_values)

        ax2.scatter(
            t_arr, y_arr, color=COLORS["bedroom"], alpha=0.5, s=10, label="Data"
        )

        t_line = np.array([0, t_arr.max()])
        y_line = lambda_value * t_line
        r_squared = result.get("lambda_average_r_squared", np.nan)

        reg_label = (
            f"λ = {lambda_value:.4f} h⁻¹ (R² = {r_squared:.4f})"
            if not np.isnan(r_squared)
            else f"λ = {lambda_value:.4f} h⁻¹"
        )

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


def plot_lambda_summary(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Figure:
    """
    Plot summary bar chart of lambda values across all events.

    If config_key column exists, creates subplots (one per configuration)
    and colors bars by configuration. Otherwise, creates a single plot.

    Parameters:
        results_df: DataFrame with lambda_average_mean, lambda_average_std,
                   and optionally config_key for grouping
        output_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    apply_style()

    # Check if we have configuration data for subplots
    has_config = "config_key" in results_df.columns
    if has_config:
        config_keys = results_df["config_key"].dropna().unique()
        n_configs = len(config_keys)
    else:
        config_keys = ["All"]
        n_configs = 1

    # Create figure with subplots (one row per configuration)
    if n_configs > 1:
        fig, axes = plt.subplots(
            n_configs, 1,
            figsize=(12, 4 * n_configs),
            sharex=False,
            squeeze=False
        )
        axes = axes.flatten()
    else:
        fig, ax = create_figure(figsize=(12, 5))
        axes = [ax]

    # Plot each configuration
    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]

        # Filter data for this configuration
        if has_config and config_key != "All":
            config_df = results_df[results_df["config_key"] == config_key]
        else:
            config_df = results_df

        if len(config_df) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Get event numbers
        if "event_number" in config_df.columns:
            event_numbers = config_df["event_number"].values
        else:
            event_numbers = np.arange(1, len(config_df) + 1)

        lambda_avg: npt.NDArray[np.float64] = np.asarray(
            config_df["lambda_average_mean"].values, dtype=np.float64
        )
        lambda_std: npt.NDArray[np.float64] = np.asarray(
            config_df["lambda_average_std"].values, dtype=np.float64
        )

        # Get color for this configuration
        bar_color = get_config_color(config_key, idx)

        valid_mask = ~np.isnan(lambda_avg)
        x_valid = [int(e) for e, v in zip(event_numbers, valid_mask) if v]
        y_valid = lambda_avg[valid_mask]
        yerr_valid = lambda_std[valid_mask]

        # Plot bars with configuration-specific color
        if len(x_valid) > 0:
            ax.bar(
                x_valid,
                y_valid,
                yerr=yerr_valid,
                color=bar_color,
                alpha=0.7,
                capsize=3,
                label=f"λ ({config_key})" if n_configs > 1 else "λ (average method)",
            )

        # Mark skipped events
        x_skipped = [int(e) for e, v in zip(event_numbers, valid_mask) if not v]
        if x_skipped:
            ax.scatter(
                x_skipped,
                [0] * len(x_skipped),
                marker="x",
                color=COLORS["lambda"],
                s=50,
                label="Skipped",
                zorder=5,
            )

        # Add mean line for this configuration
        config_mean = np.nanmean(lambda_avg)
        if not np.isnan(config_mean):
            ax.axhline(
                float(config_mean),
                color=COLORS["fit"],
                linestyle="--",
                linewidth=LINE_WIDTH_FIT,
                label=f"Mean: {config_mean:.3f} h⁻¹",
            )

        ax.set_xlabel("Event Number", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Air-Change Rate λ (h⁻¹)", fontsize=FONT_SIZE_LABEL)

        # Set title with configuration info
        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(
                "Air-Change Rate Summary Across All Events",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )

        ax.set_xticks([int(e) for e in event_numbers])
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis="y")

    # Add overall title if multiple configurations
    if n_configs > 1:
        fig.suptitle(
            "Air-Change Rate Summary by Configuration",
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
            y=1.02,
        )

    plt.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path, close=False)

    return fig
