#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Plotting Functions
=====================================

This module provides specialized plotting functions for particle decay and
emission analysis in the EPA Legionella project.

Key Functions:
    - plot_particle_decay_event: Individual event decay curves per bin
    - plot_penetration_summary: Bar chart of penetration factors
    - plot_deposition_summary: Bar chart of deposition rates
    - plot_emission_summary: Bar chart of emission rates

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

from scripts.plot_style import (
    COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_DATA,
    LINE_WIDTH_FIT,
    SENSOR_COLORS,
    SHOWER_OFF_STYLE,
    SHOWER_ON_STYLE,
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
    save_figure,
)


def _calculate_exponential_fit_curve(
    particle_data: pd.DataFrame,
    event: Dict,
    bin_num: int,
    bin_info: Dict,
    result: Dict,
    lambda_ach: float,
) -> Optional[pd.DataFrame]:
    """
    Calculate fitted exponential decay curve for a particle bin.

    Uses: C(t) = C_ss + (C_0 - C_ss) * exp(-(λ + β)*t)
    where C_ss is the steady-state concentration.

    Parameters:
        particle_data: DataFrame with particle concentrations
        event: Event timing dictionary
        bin_num: Particle bin number
        bin_info: Bin information dictionary
        result: Analysis results for this event
        lambda_ach: Air change rate (h⁻¹)

    Returns:
        DataFrame with datetime and C_fit columns, or None if insufficient data
    """
    beta_mean = result.get(f"bin{bin_num}_beta_mean", np.nan)
    c_steady_state = result.get(f"bin{bin_num}_c_steady_state", np.nan)

    if np.isnan(beta_mean) or np.isnan(c_steady_state):
        return None

    col_inside = f"{bin_info['column']}_inside"

    # Get deposition window data
    decay_start = event["shower_off"]
    decay_end = event["deposition_end"]

    mask = (particle_data["datetime"] >= decay_start) & (
        particle_data["datetime"] <= decay_end
    )
    decay_data = particle_data[mask].copy()

    if len(decay_data) < 2 or col_inside not in decay_data.columns:
        return None

    c_0 = float(decay_data[col_inside].iloc[0])
    if np.isnan(c_0) or c_0 <= c_steady_state:
        return None

    # Calculate time in hours from decay start
    t0 = decay_data["datetime"].iloc[0]
    t_hours = (decay_data["datetime"] - t0).dt.total_seconds() / 3600.0

    # Calculate fitted concentration: C(t) = C_ss + (C_0 - C_ss) * exp(-(λ + β)*t)
    total_loss_rate = lambda_ach + beta_mean
    c_fit = c_steady_state + (c_0 - c_steady_state) * np.exp(-total_loss_rate * t_hours)

    return pd.DataFrame({
        "datetime": decay_data["datetime"].values,
        "C_fit": c_fit.values,
    })


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

    Creates a two-panel figure similar to CO2 decay plots:
    - Top panel: Particle concentrations with exponential fit curves
    - Bottom panel: Linearized regression plot for all bins

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

    # Create two-panel figure (like CO2 plots)
    fig, axes = create_figure(
        nrows=2, ncols=1, figsize=(12, 10), height_ratios=[2, 1]
    )
    ax1, ax2 = axes  # type: ignore[misc]

    lambda_ach = result.get("lambda_ach", np.nan)

    # =========================================================================
    # Top panel: Particle concentrations with exponential fit curves
    # =========================================================================
    for bin_num, bin_info in particle_bins.items():
        col_inside = f"{bin_info['column']}_inside"
        color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

        if col_inside in plot_data.columns:
            # Check if this bin has valid results
            E_mean = result.get(f"bin{bin_num}_E_mean", np.nan)
            beta_mean = result.get(f"bin{bin_num}_beta_mean", np.nan)
            is_valid = not np.isnan(E_mean)
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

            # Add exponential fit curve if valid
            if is_valid and not np.isnan(beta_mean):
                fit_curve = _calculate_exponential_fit_curve(
                    particle_data, event, bin_num, bin_info, result, lambda_ach
                )
                if fit_curve is not None and len(fit_curve) > 0:
                    ax1.plot(
                        fit_curve["datetime"],
                        fit_curve["C_fit"],
                        color=color,
                        linewidth=LINE_WIDTH_FIT,
                        linestyle=":",
                        alpha=0.8,
                    )

    # Add shaded windows for analysis periods
    add_shaded_window(
        ax1,
        event["penetration_start"],
        event["shower_on"],
        color=COLORS["pre_shower"],
        label="Penetration window (1 hr)",
        alpha=WINDOW_ALPHA,
    )
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

    # Top panel formatting
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

    # Add results text box with summary
    textstr = f"λ = {lambda_ach:.4f} h⁻¹\n\n"

    # Count valid bins and build beta summary
    valid_bins = 0
    beta_values = []
    for bin_num in particle_bins.keys():
        if not np.isnan(result.get(f"bin{bin_num}_E_mean", np.nan)):
            valid_bins += 1
            beta_val = result.get(f"bin{bin_num}_beta_mean", np.nan)
            r2_val = result.get(f"bin{bin_num}_beta_r_squared", np.nan)
            if not np.isnan(beta_val):
                beta_values.append((bin_num, beta_val, r2_val))

    textstr += f"Valid bins: {valid_bins}/{len(particle_bins)}\n"
    textstr += "(Solid=data, Dotted=fit)"

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
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    format_datetime_axis(ax1)

    # =========================================================================
    # Bottom panel: Linearized regression plot
    # =========================================================================
    has_regression_data = False

    for bin_num, bin_info in particle_bins.items():
        color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

        # Get fit data from results
        t_values = result.get(f"bin{bin_num}_fit_t_values", [])
        y_values = result.get(f"bin{bin_num}_fit_y_values", [])
        beta_mean = result.get(f"bin{bin_num}_beta_mean", np.nan)
        r_squared = result.get(f"bin{bin_num}_beta_r_squared", np.nan)

        if t_values and y_values and not np.isnan(beta_mean):
            has_regression_data = True
            t_arr = np.array(t_values)
            y_arr = np.array(y_values)

            # Plot scatter points
            ax2.scatter(
                t_arr,
                y_arr,
                color=color,
                alpha=0.4,
                s=8,
            )

            # Plot regression line
            if len(t_arr) > 0:
                t_line = np.array([0, t_arr.max()])
                # slope = λ + β
                slope = lambda_ach + beta_mean
                y_line = slope * t_line

                label = f"Bin {bin_num}: β={beta_mean:.2f} h⁻¹"
                if not np.isnan(r_squared):
                    label += f" (R²={r_squared:.3f})"

                ax2.plot(
                    t_line,
                    y_line,
                    color=color,
                    linewidth=LINE_WIDTH_FIT,
                    label=label,
                )

    if has_regression_data:
        ax2.set_xlabel("Time since decay start (hours)", fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel("$-\\ln[(C(t) - C_{ss}) / (C_0 - C_{ss})]$", fontsize=FONT_SIZE_LABEL)
        ax2.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND - 1, ncol=2)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient data for linearized regression",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=FONT_SIZE_LABEL,
        )
        ax2.set_xlabel("Time since decay start (hours)", fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel("$-\\ln[(C(t) - C_{ss}) / (C_0 - C_{ss})]$", fontsize=FONT_SIZE_LABEL)

    ax2.tick_params(labelsize=FONT_SIZE_TICK)

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

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    p_means = []
    p_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_p_mean"
        valid_values = results_df[col].dropna()
        p_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        p_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        p_means,
        yerr=p_stds,
        capsize=5,
        color=SENSOR_COLORS[0],
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
                fontsize=FONT_SIZE_TICK,
            )

    ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Penetration Factor (p)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Penetration Factor by Particle Size\n(Mean ± Std Dev)",
        fontsize=FONT_SIZE_TITLE,
        fontweight=TITLE_FONTWEIGHT,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(
        y=0.7,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Expected range",
    )
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND)

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

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    beta_means = []
    beta_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_beta_mean"
        valid_values = results_df[col].dropna()
        beta_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        beta_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        beta_means,
        yerr=beta_stds,
        capsize=5,
        color=SENSOR_COLORS[1],
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
                fontsize=FONT_SIZE_TICK,
            )

    ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Deposition Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "Deposition Rate by Particle Size\n(Mean ± Std Dev)",
        fontsize=FONT_SIZE_TITLE,
        fontweight=TITLE_FONTWEIGHT,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=FONT_SIZE_TICK)

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

    Parameters:
        results_df: DataFrame with analysis results
        particle_bins: Dictionary of particle bin information
        output_path: Path to save the figure
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    # Calculate mean and std for each bin
    E_means = []
    E_stds = []
    for bin_num in bin_nums:
        col = f"bin{bin_num}_E_mean"
        valid_values = results_df[col].dropna()
        E_means.append(valid_values.mean() if len(valid_values) > 0 else 0)
        E_stds.append(valid_values.std() if len(valid_values) > 0 else 0)

    fig, ax = create_figure(figsize=(10, 6))

    x = np.arange(len(bin_nums))
    bars = ax.bar(
        x,
        E_means,
        yerr=E_stds,
        capsize=5,
        color=SENSOR_COLORS[2],
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
        col = f"bin{bin_num}_beta_mean"
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
