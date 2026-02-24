#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Plotting Functions
=====================================

This module provides specialized plotting functions for particle decay and
emission analysis in the EPA Legionella project. Plots are designed for
analyzing aerosol behavior during and after shower events.

Key Functions:
    - plot_particle_decay_event: Individual event decay curves per bin (two panels)
    - plot_penetration_summary: Bar chart of penetration factors by size
    - plot_deposition_summary: Bar chart of deposition rates by size
    - plot_emission_summary: Bar chart of emission rates by size
    - plot_emission_boxplot: Box-and-whisker of E_total by water temperature and bin
    - plot_size_distribution_summary: Multi-panel summary of all metrics

Plot Features:
    - Two-panel event plots: concentration time series (top) + emission rates (bottom)
    - Color-coded particle size bins (0.35-3.0 µm)
    - Shaded deposition analysis window
    - Shower ON/OFF markers (dotted lines) distinct from fitted/predicted lines (dashed)
    - Decay R² values listed in top-panel text box alongside lambda and valid-bin count
    - Log-scale concentration axis for wide dynamic range
    - Emission subplot shows per-step E_t lines, E_mean dashed lines, and R² annotation
    - Emission subplot x-axis matches concentration panel; y-axis clipped to 2nd-98th percentile
    - Configuration-based subplot grouping; temperature-based colors from get_config_color()

Methodology:
    1. Extract data window around shower event (2 hr before to 1 hr after deposition end)
    2. Top panel: plot particle concentrations for all 7 size bins (solid = valid beta,
       dashed = invalid beta)
    3. Shade deposition window (2 hr post-shower)
    4. Overlay continuous predicted Ct curves (emission phase + decay phase as one
       unbroken dashed line per valid bin); decay phase starts from the predicted
       concentration at peak_time, not the measured peak value
    5. Bottom panel: per-step E_t as faint lines, E_mean as dashed horizontal lines
       spanning shower_on to peak_time per bin, R² annotation
    6. Display λ (air change rate) and valid-bin count in text box

Output Files:
    - Individual event plots: {test_name}_particle_decay.png
    - Summary charts: penetration_summary.png, deposition_summary.png,
      emission_summary.png, size_distribution_summary.png

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.event_manager import sort_config_keys_by_water_temp
from scripts.plot_style import (
    COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_DATA,
    LINE_WIDTH_FIT,
    SENSOR_COLORS,
    TITLE_FONTWEIGHT,
    WINDOW_ALPHA,
    add_shaded_window,
    add_shower_off_marker,
    add_shower_on_marker,
    apply_style,
    create_figure,
    format_datetime_axis,
    format_test_name_for_title,
    format_title,
    get_config_color,
    save_figure,
)


def plot_particle_decay_event(
    particle_data: pd.DataFrame,
    event: Dict,
    particle_bins: Dict,
    result: Dict,
    output_path: Path,
    event_number: int,
    test_name: Optional[str] = None,
) -> None:
    """
    Plot particle concentration decay for a single event showing all bins.

    Creates a two-panel figure matching the CO2 analytical plot style:

    Top panel (ax1) — concentration time series:
      - Measured particle concentrations for all 7 bins (solid = valid beta,
        dashed = invalid beta)
      - Continuous predicted Ct curve per valid bin (emission phase from
        shower_on to peak_time, then decay phase from peak_time to
        deposition_end, both as dashed lines of the same colour forming one
        unbroken model prediction; decay starts from predicted concentration
        at peak, not from measured peak value)
      - Shower ON/OFF markers and shaded deposition window

    Bottom panel (ax2) — per-step emission rates:
      - Per-step E_t values as faint lines for each valid bin
      - E_mean as a horizontal dashed line spanning shower_on to peak_time
      - Emission R² annotation in the panel

    Parameters:
        particle_data: DataFrame with particle concentrations
        event: Event timing dictionary
        particle_bins: Dictionary of all particle bin information
        result: Analysis results for this event (all bins)
        output_path: Path to save the figure
        event_number: Event number for title
        test_name: Test name for title (e.g., "0114_HW_Morning_R01")
    """
    apply_style()

    # Extract data for plotting window (2 hours before shower to 3 hours after)
    plot_start = event["shower_on"] - timedelta(hours=2)
    plot_end = event["deposition_end"] + timedelta(hours=1)

    mask = (particle_data["datetime"] >= plot_start) & (
        particle_data["datetime"] <= plot_end
    )
    plot_data = particle_data[mask].copy()

    if plot_data.empty:
        print(f"    Warning: No data for event {event_number}")
        return

    # Two-panel figure: concentration (top, 2x height) + emission (bottom, 1x)
    fig, (ax1, ax2) = create_figure(
        nrows=2, ncols=1, figsize=(12, 9), height_ratios=[2, 1]
    )

    lambda_ach = result.get("lambda_ach", np.nan)

    # =========================================================================
    # Top panel: Particle concentrations with decay predictions
    # =========================================================================
    for bin_num, bin_info in particle_bins.items():
        col_inside = f"{bin_info['column']}_inside"
        color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

        if col_inside in plot_data.columns:
            # Check if this bin has valid decay results
            beta_val = result.get(f"bin{bin_num}_beta", np.nan)
            is_valid = not np.isnan(beta_val)
            linestyle = "-" if is_valid else "--"
            alpha = 0.9 if is_valid else 0.4

            # Plot raw data
            ax1.plot(
                plot_data["datetime"],
                plot_data[col_inside],
                label=f"Bin {bin_num} ({bin_info['name']} µm)",
                color=color,
                linewidth=LINE_WIDTH_DATA,
                linestyle=linestyle,
                alpha=alpha,
            )

            # Plot continuous predicted Ct: emission phase then decay phase
            # as two segments of the same simulation (they connect at peak_time)
            emission_dts = result.get(f"bin{bin_num}_emission_datetimes", [])
            emission_pred = result.get(f"bin{bin_num}_emission_predicted", [])
            decay_dts = result.get(f"bin{bin_num}_decay_datetimes", [])
            decay_pred = result.get(f"bin{bin_num}_decay_predicted", [])

            if len(emission_dts) > 0 and len(emission_pred) > 0:
                ax1.plot(
                    pd.to_datetime(emission_dts),
                    np.array(emission_pred),
                    color=color,
                    linewidth=LINE_WIDTH_FIT,
                    linestyle="--",
                    alpha=0.8,
                )
            if len(decay_dts) > 0 and len(decay_pred) > 0:
                ax1.plot(
                    pd.to_datetime(decay_dts),
                    np.array(decay_pred),
                    color=color,
                    linewidth=LINE_WIDTH_FIT,
                    linestyle="--",
                    alpha=0.8,
                )

    # Add single legend entry for predicted Ct lines
    has_predictions = any(
        len(result.get(f"bin{bn}_emission_predicted", [])) > 0
        or len(result.get(f"bin{bn}_decay_predicted", [])) > 0
        for bn in particle_bins.keys()
    )
    if has_predictions:
        ax1.plot(
            [],
            [],
            color="gray",
            linestyle="--",
            linewidth=LINE_WIDTH_FIT,
            label="Predicted Ct",
        )

    # Add shaded window for deposition analysis period
    add_shaded_window(
        ax1,
        event["shower_off"],
        event["deposition_end"],
        color=COLORS["post_shower"],
        label="Deposition window (2 hr)",
        alpha=WINDOW_ALPHA,
    )

    # Add shower ON/OFF markers
    add_shower_on_marker(ax1, event["shower_on"], label="Shower ON")
    add_shower_off_marker(ax1, event["shower_off"], label="Shower OFF")

    # Axis formatting
    ax1.set_ylabel("Particle Concentration (#/cm³)", fontsize=FONT_SIZE_LABEL)

    # Use consistent title formatting
    if test_name:
        formatted_name = format_test_name_for_title(test_name)
        title = f"Event {event_number:02d} - {formatted_name}: PM Decay"
    else:
        title = format_title(
            "Particle Decay - All Size Bins",
            event_number=event_number,
            event_datetime=event["shower_on"],
        )
    ax1.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT)

    # Count valid bins and build decay R² summary for the text box
    valid_bins = 0
    decay_r2_lines = []
    for bin_num in particle_bins.keys():
        beta_val = result.get(f"bin{bin_num}_beta", np.nan)
        if not np.isnan(beta_val):
            valid_bins += 1
            r2_val = result.get(f"bin{bin_num}_beta_r_squared", np.nan)
            r2_str = f"{r2_val:.3f}" if not np.isnan(r2_val) else "N/A"
            decay_r2_lines.append(f" B{bin_num}: R²={r2_str}")

    # Build text box content: lambda, valid-bin count, and per-bin decay R²
    textstr = f"λ = {lambda_ach:.4f} h⁻¹\n"
    textstr += f"Valid bins: {valid_bins}/{len(particle_bins)}\n"
    textstr += "(Solid=valid, Dashed=invalid)"
    if decay_r2_lines:
        textstr += "\n\nDecay R²:\n" + "\n".join(decay_r2_lines)

    props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
    ax1.text(
        0.02,
        0.98,
        textstr,
        transform=ax1.transAxes,
        fontsize=FONT_SIZE_LEGEND,
        verticalalignment="top",
        bbox=props,
    )

    ax1.legend(
        loc="upper right",
        fontsize=FONT_SIZE_LEGEND - 1,
        framealpha=0.9,
        ncol=2,
    )
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.001)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    format_datetime_axis(ax1)

    # =========================================================================
    # Bottom panel: Per-step emission rates
    # =========================================================================
    has_emission_data = False
    E_r2_lines = []  # Collect R² annotations for valid bins

    for bin_num, bin_info in particle_bins.items():
        color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

        E_times = result.get(f"bin{bin_num}_E_times", [])
        E_per_step = result.get(f"bin{bin_num}_E_per_step", [])
        E_mean_val = result.get(f"bin{bin_num}_E_mean", np.nan)
        E_r2_val = result.get(f"bin{bin_num}_E_r_squared", np.nan)
        peak_time = result.get(f"bin{bin_num}_peak_time", None)

        if len(E_times) > 0 and len(E_per_step) > 0:
            has_emission_data = True
            # Per-step E_t as a faint line (all values, including negative)
            ax2.plot(
                pd.to_datetime(E_times),
                np.array(E_per_step),
                color=color,
                linewidth=0.8,
                alpha=0.35,
            )

        # E_mean as horizontal dashed line spanning shower_on → peak_time
        if not np.isnan(E_mean_val) and peak_time is not None:
            has_emission_data = True
            ax2.hlines(
                E_mean_val,
                event["shower_on"],
                pd.Timestamp(peak_time),
                color=color,
                linewidth=LINE_WIDTH_FIT,
                linestyle="--",
                alpha=0.9,
            )
            r2_str = f"{E_r2_val:.3f}" if not np.isnan(E_r2_val) else "N/A"
            E_r2_lines.append(f"B{bin_num}: R²={r2_str}")

    # Add shower ON/OFF markers to emission panel
    add_shower_on_marker(ax2, event["shower_on"])
    add_shower_off_marker(ax2, event["shower_off"])

    ax2.set_ylabel("Emission Rate E\n(#/cm³·min)", fontsize=FONT_SIZE_LABEL - 1)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)

    # Match x-axis range to the full plot window (same as the concentration panel)
    ax2.set_xlim(ax1.get_xlim())
    format_datetime_axis(ax2, interval_minutes=30)

    # Percentile-based y-axis limits to avoid extreme noise spikes dominating
    all_e_steps = []
    for bn in particle_bins.keys():
        all_e_steps.extend(result.get(f"bin{bn}_E_per_step", []))
    if all_e_steps:
        e_arr = np.array([v for v in all_e_steps if not np.isnan(v)], dtype=float)
        if len(e_arr) >= 4:
            p2 = float(np.percentile(e_arr, 2))
            p98 = float(np.percentile(e_arr, 98))
            margin = max((p98 - p2) * 0.15, abs(p98) * 0.05, 1e-6)
            ax2.set_ylim(p2 - margin, p98 + margin)

    # Add E R² annotation box to emission panel
    if E_r2_lines:
        r2_text = "Emission R²:\n" + "\n".join(E_r2_lines)
        props_r2 = dict(
            boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"
        )
        ax2.text(
            0.02,
            0.98,
            r2_text,
            transform=ax2.transAxes,
            fontsize=FONT_SIZE_LEGEND - 1,
            verticalalignment="top",
            bbox=props_r2,
        )

    if not has_emission_data:
        ax2.text(
            0.5,
            0.5,
            "No emission data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=FONT_SIZE_LABEL,
            color="gray",
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_penetration_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing penetration factors across all bins.

    If config_key column exists, creates subplots (one per configuration).

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Check if we have configuration data for subplots
    has_config = "config_key" in results_df.columns
    if has_config:
        config_keys = sort_config_keys_by_water_temp(
            list(results_df["config_key"].dropna().unique())
        )
        n_configs = len(config_keys)
    else:
        config_keys = ["All"]
        n_configs = 1

    # Create figure with subplots
    if n_configs > 1:
        fig, axes = plt.subplots(
            n_configs, 1, figsize=(12, 5 * n_configs), squeeze=False
        )
        axes = axes.flatten()
    else:
        fig, ax = create_figure(figsize=(10, 6))
        axes = [ax]

    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]

        # Filter data for this configuration
        if has_config and config_key != "All":
            config_df = results_df[results_df["config_key"] == config_key]
        else:
            config_df = results_df

        # Get color for this configuration
        bar_color = get_config_color(config_key, idx)

        # Calculate mean and std for each bin
        p_means = []
        p_stds = []
        for bin_num in bin_nums:
            col = f"bin{bin_num}_p_mean"
            if col in config_df.columns:
                valid_values = config_df[col].dropna()
                p_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
                p_stds.append(valid_values.std() if len(valid_values) > 0 else 0)
            else:
                p_means.append(0)
                p_stds.append(0)

        x = np.arange(len(bin_nums))
        bars = ax.bar(
            x,
            p_means,
            yerr=p_stds,
            capsize=5,
            color=bar_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, p_means, p_stds)):
            height = bar.get_height()
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std + 0.02,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE_TICK - 1,
                )

        ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Penetration Factor (p)", fontsize=FONT_SIZE_LABEL)

        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(
                "Penetration Factor by Particle Size\n(Mean ± Std Dev)",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)

        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    if n_configs > 1:
        fig.suptitle(
            "Penetration Factor by Particle Size\n(Mean ± Std Dev)",
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
            y=1.02,
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_deposition_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing deposition rates across all bins.

    If config_key column exists, creates subplots (one per configuration).

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Check if we have configuration data for subplots
    has_config = "config_key" in results_df.columns
    if has_config:
        config_keys = sort_config_keys_by_water_temp(
            list(results_df["config_key"].dropna().unique())
        )
        n_configs = len(config_keys)
    else:
        config_keys = ["All"]
        n_configs = 1

    # Create figure with subplots
    if n_configs > 1:
        fig, axes = plt.subplots(
            n_configs, 1, figsize=(12, 5 * n_configs), squeeze=False
        )
        axes = axes.flatten()
    else:
        fig, ax = create_figure(figsize=(10, 6))
        axes = [ax]

    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]

        # Filter data for this configuration
        if has_config and config_key != "All":
            config_df = results_df[results_df["config_key"] == config_key]
        else:
            config_df = results_df

        # Get color for this configuration
        bar_color = get_config_color(config_key, idx)

        # Calculate mean and std for each bin
        beta_means = []
        beta_stds = []
        for bin_num in bin_nums:
            col = f"bin{bin_num}_beta"
            if col in config_df.columns:
                valid_values = config_df[col].dropna()
                beta_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
                beta_stds.append(valid_values.std() if len(valid_values) > 0 else 0)
            else:
                beta_means.append(0)
                beta_stds.append(0)

        x = np.arange(len(bin_nums))
        bars = ax.bar(
            x,
            beta_means,
            yerr=beta_stds,
            capsize=5,
            color=bar_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, beta_means, beta_stds)):
            height = bar.get_height()
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std + 0.1,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE_TICK - 1,
                )

        ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Deposition Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)

        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(
                "Deposition Rate by Particle Size\n(Mean ± Std Dev)",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    if n_configs > 1:
        fig.suptitle(
            "Deposition Rate by Particle Size\n(Mean ± Std Dev)",
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
            y=1.02,
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_emission_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create bar chart summarizing emission rates across all bins.

    If config_key column exists, creates subplots (one per configuration).

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Check if we have configuration data for subplots
    has_config = "config_key" in results_df.columns
    if has_config:
        config_keys = sort_config_keys_by_water_temp(
            list(results_df["config_key"].dropna().unique())
        )
        n_configs = len(config_keys)
    else:
        config_keys = ["All"]
        n_configs = 1

    # Create figure with subplots
    if n_configs > 1:
        fig, axes = plt.subplots(
            n_configs, 1, figsize=(12, 5 * n_configs), squeeze=False
        )
        axes = axes.flatten()
    else:
        fig, ax = create_figure(figsize=(10, 6))
        axes = [ax]

    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]

        # Filter data for this configuration
        if has_config and config_key != "All":
            config_df = results_df[results_df["config_key"] == config_key]
        else:
            config_df = results_df

        # Get color for this configuration
        bar_color = get_config_color(config_key, idx)

        # Calculate mean and std for each bin
        E_means = []
        E_stds = []
        for bin_num in bin_nums:
            col = f"bin{bin_num}_E_mean"
            if col in config_df.columns:
                valid_values = config_df[col].dropna()
                E_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
                E_stds.append(valid_values.std() if len(valid_values) > 0 else 0)
            else:
                E_means.append(0)
                E_stds.append(0)

        x = np.arange(len(bin_nums))
        bars = ax.bar(
            x,
            E_means,
            yerr=E_stds,
            capsize=5,
            color=bar_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on bars (scientific notation)
        for i, (bar, mean, std) in enumerate(zip(bars, E_means, E_stds)):
            height = bar.get_height()
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std,
                    f"{mean:.1e}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE_TICK - 1,
                    rotation=0,
                )

        ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Emission Rate E (#/min)", fontsize=FONT_SIZE_LABEL)

        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(
                "Shower Emission Rate by Particle Size\n(Mean ± Std Dev)",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")

        # Use log scale if range is large
        if (
            max(E_means) > 0
            and max(E_means) / min([m for m in E_means if m > 0] + [1]) > 100
        ):
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    if n_configs > 1:
        fig.suptitle(
            "Shower Emission Rate by Particle Size\n(Mean ± Std Dev)",
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
            y=1.02,
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_size_distribution_summary(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create multi-panel figure showing all three metrics vs particle size.

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_centers = [
        (particle_bins[i]["min"] + particle_bins[i]["max"]) / 2 for i in bin_nums
    ]

    fig, axes = create_figure(nrows=1, ncols=3, figsize=(15, 5))

    # Panel 1: Penetration factor
    p_means = []
    p_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_p_mean"
        valid_values = results_df[col].dropna()
        p_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        p_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[0].errorbar(
        bin_centers,
        p_means,
        yerr=p_stds,
        marker="o",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[0],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[0].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("Penetration Factor (p)", fontsize=FONT_SIZE_LABEL)
    axes[0].set_title(
        "(a) Penetration Factor", fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Panel 2: Deposition rate
    beta_means = []
    beta_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_beta"
        valid_values = results_df[col].dropna()
        beta_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        beta_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[1].errorbar(
        bin_centers,
        beta_means,
        yerr=beta_stds,
        marker="s",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[1],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[1].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("Deposition Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_title(
        "(b) Deposition Rate", fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT
    )
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Emission rate
    E_means = []
    E_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_E_mean"
        valid_values = results_df[col].dropna()
        E_means.append(valid_values.mean() if len(valid_values) > 0 else np.nan)
        E_stds.append(valid_values.std() if len(valid_values) > 0 else np.nan)

    axes[2].errorbar(
        bin_centers,
        E_means,
        yerr=E_stds,
        marker="^",
        markersize=8,
        capsize=5,
        color=SENSOR_COLORS[2],
        linewidth=LINE_WIDTH_DATA,
    )
    axes[2].set_xlabel("Particle Size (µm)", fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel("Emission Rate E (#/min)", fontsize=FONT_SIZE_LABEL)
    axes[2].set_title(
        "(c) Emission Rate", fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT
    )
    axes[2].grid(True, alpha=0.3)

    # Apply log scale if needed
    if max([e for e in E_means if not np.isnan(e)] + [0]) > 0:
        valid_E = [e for e in E_means if not np.isnan(e) and e > 0]
        if valid_E and max(valid_E) / min(valid_E) > 100:
            axes[2].set_yscale("log")

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_emission_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
) -> None:
    """
    Create a grouped box-and-whisker plot of total particle emission by water temperature.

    For each water temperature configuration (x-axis), draws one box per particle size
    bin (grouped side-by-side) so that bin-specific emission distributions can be
    compared across temperatures.  Each bin is assigned a consistent color from
    SENSOR_COLORS.  Configurations are sorted from coldest to hottest water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and
                    bin{n}_E_total columns)
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    from matplotlib.patches import Patch

    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return

    config_keys = sort_config_keys_by_water_temp(
        [k for k in results_df["config_key"].dropna().unique()]
    )

    if not config_keys:
        return

    n_configs = len(config_keys)
    bin_nums = list(particle_bins.keys())
    n_bins = len(bin_nums)

    # Layout: boxes 0.12 wide, 0.14 apart within a group, 1.4 between group centers
    box_width = 0.12
    bin_gap = 0.14
    group_spacing = max(1.4, n_bins * bin_gap * 1.3)
    group_centers = np.arange(n_configs) * group_spacing

    fig, ax = create_figure(figsize=(max(10, n_configs * 2.2), 7))

    for bin_idx, bin_num in enumerate(bin_nums):
        color = SENSOR_COLORS[bin_idx % len(SENSOR_COLORS)]
        col = f"bin{bin_num}_E_total"
        if col not in results_df.columns:
            continue

        # Offset from group center; bin_idx 0 is at the far left of the group
        offset = (bin_idx - (n_bins - 1) / 2.0) * bin_gap
        positions = []
        data = []

        for cfg_idx, config_key in enumerate(config_keys):
            values = results_df.loc[
                results_df["config_key"] == config_key, col
            ].dropna().values
            if len(values) > 0:
                positions.append(group_centers[cfg_idx] + offset)
                data.append(values)

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.5, color=color),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(color)
                line.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(config_keys, rotation=45, ha="right", fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("Water Temperature Configuration", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Total Emission E_total (#)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Particle Emission by Water Temperature and Size Bin\n(Box = median/IQR, whiskers = 1.5×IQR)",
        fontsize=FONT_SIZE_TITLE,
        fontweight=TITLE_FONTWEIGHT,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Legend: one entry per bin
    legend_elements = [
        Patch(
            facecolor=SENSOR_COLORS[i % len(SENSOR_COLORS)],
            alpha=0.7,
            label=f"Bin {b} ({particle_bins[b]['name']} µm)",
        )
        for i, b in enumerate(bin_nums)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=FONT_SIZE_LEGEND - 1,
        ncol=2,
    )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
