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

import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from scripts.event_manager import sort_config_keys_by_water_temp
from scripts.plot_style import (
    BOXPLOT_CONFIG,
    COLORS,
    FONT_SIZE_ANNOTATION,
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


# Configuration for the three per-bin summary bar charts.
# Keys: col_template, ylabel, title, label_fmt, label_offset, ylim, log_scale
_BAR_CHART_CONFIG = {
    "penetration": dict(
        col_template="bin{n}_p_mean",
        ylabel="Penetration Factor (p)",
        title="Penetration Factor by Particle Size\n(Mean ± Std Dev)",
        label_fmt=".3f",
        label_offset=0.02,
        ylim=(0, 1.1),
        log_scale=False,
    ),
    "deposition": dict(
        col_template="bin{n}_beta",
        ylabel="Deposition Rate β (h⁻¹)",
        title="Deposition Rate by Particle Size\n(Mean ± Std Dev)",
        label_fmt=".2f",
        label_offset=0.1,
        ylim=None,
        log_scale=False,
    ),
    "emission": dict(
        col_template="bin{n}_E_mean",
        ylabel="Emission Rate E (#/min)",
        title="Shower Emission Rate by Particle Size\n(Mean ± Std Dev)",
        label_fmt=".1e",
        label_offset=0,
        ylim=None,
        log_scale="auto",
    ),
}


def _plot_summary_bar_chart(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    cfg: dict,
) -> None:
    """
    Shared implementation for per-bin summary bar charts.

    Handles all three metric types (penetration, deposition, emission) via a
    configuration dict.  If config_key is present, creates one subplot per
    configuration; otherwise draws a single panel.

    Parameters:
        results_df: DataFrame with analysis results.
        particle_bins: Dictionary of particle bin information.
        output_path: Path to save the figure.
        cfg: Configuration dict from _BAR_CHART_CONFIG.
    """
    apply_style()

    bin_nums = list(particle_bins.keys())
    bin_labels = [particle_bins[i]["name"] for i in bin_nums]

    has_config = "config_key" in results_df.columns
    if has_config:
        config_keys = sort_config_keys_by_water_temp(
            list(results_df["config_key"].dropna().unique())
        )
        n_configs = len(config_keys)
    else:
        config_keys = ["All"]
        n_configs = 1

    if n_configs > 1:
        fig, axes = plt.subplots(
            n_configs, 1, figsize=(12, 5 * n_configs), squeeze=False
        )
        axes = axes.flatten()
    else:
        fig, _ax = create_figure(figsize=(10, 6))
        if isinstance(_ax, list):
            _ax = _ax[0]
        axes = [_ax]

    for idx, config_key in enumerate(config_keys):
        ax = axes[idx]
        config_df = (
            results_df[results_df["config_key"] == config_key]
            if has_config and config_key != "All"
            else results_df
        )
        bar_color = get_config_color(config_key, idx)

        means, stds = [], []
        for bin_num in bin_nums:
            col = cfg["col_template"].format(n=bin_num)
            if col in config_df.columns:
                valid = config_df[col].dropna()
                means.append(float(valid.mean()) if len(valid) > 0 else 0)
                stds.append(float(valid.std()) if len(valid) > 0 else 0)
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(bin_nums))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=bar_color, alpha=0.7, edgecolor="black", linewidth=1,
        )

        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + cfg["label_offset"],
                    f"{mean:{cfg['label_fmt']}}",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICK - 1,
                )

        ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(cfg["ylabel"], fontsize=FONT_SIZE_LABEL)
        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(cfg["title"], fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")

        if cfg["ylim"] is not None:
            ax.set_ylim(*cfg["ylim"])
        elif cfg["log_scale"] == "auto":
            valid_means = [m for m in means if m > 0]
            if valid_means and max(valid_means) / min(valid_means) > 100:
                ax.set_yscale("log")

        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=FONT_SIZE_TICK)

    if n_configs > 1:
        fig.suptitle(
            cfg["title"], fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT, y=1.02,
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_penetration_summary(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path
) -> None:
    """Bar chart of penetration factors across all bins (mean ± std per bin).

    Parameters:
        results_df: DataFrame with analysis results.
        particle_bins: Dictionary of particle bin information.
        output_path: Path to save the figure.
    """
    _plot_summary_bar_chart(results_df, particle_bins, output_path, _BAR_CHART_CONFIG["penetration"])


def plot_deposition_summary(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path
) -> None:
    """Bar chart of deposition rates across all bins (mean ± std per bin).

    Parameters:
        results_df: DataFrame with analysis results.
        particle_bins: Dictionary of particle bin information.
        output_path: Path to save the figure.
    """
    _plot_summary_bar_chart(results_df, particle_bins, output_path, _BAR_CHART_CONFIG["deposition"])


def plot_emission_summary(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path
) -> None:
    """Bar chart of emission rates across all bins (mean ± std per bin).

    Parameters:
        results_df: DataFrame with analysis results.
        particle_bins: Dictionary of particle bin information.
        output_path: Path to save the figure.
    """
    _plot_summary_bar_chart(results_df, particle_bins, output_path, _BAR_CHART_CONFIG["emission"])


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


# =============================================================================
# PRIVATE HELPERS — shared by all temperature-axis boxplot functions
# =============================================================================


def _is_base_config(key: str) -> bool:
    """Return True if config_key is a base W## with no letter suffix (e.g. W48, not W48b)."""
    return bool(re.match(r"^W\d+(_|$)", str(key)))


def _extract_config_temp(key: str) -> "Optional[float]":
    """Extract numeric temperature from a W## config_key string."""
    m = re.match(r"^W(\d+)", str(key))
    return float(m.group(1)) if m else None


def _get_rh_at_shower_on(
    shower_on_series: "pd.Series",
    rh_data: "pd.DataFrame",
    tolerance_min: float = 5.0,
) -> float:
    """
    Return the average Aranet4 Bedroom RH (%) at the shower-on times in *shower_on_series*.

    Parameters:
        shower_on_series: Series of shower_on timestamps for one temperature group.
        rh_data: DataFrame with columns 'datetime' and 'RH_bedroom'.
        tolerance_min: Maximum allowed time difference (minutes) for a nearest-
            neighbour match.

    Returns:
        Mean RH across matched events, or np.nan if no matches found.
    """
    rh_vals = []
    for t in shower_on_series.dropna():
        t_pd = pd.Timestamp(t)
        diffs = (rh_data["datetime"] - t_pd).abs().dt.total_seconds() / 60.0
        nearest_idx = int(diffs.idxmin())
        if float(diffs.iloc[nearest_idx]) <= tolerance_min:
            rh_vals.append(float(rh_data["RH_bedroom"].iloc[nearest_idx]))
    return float(np.mean(rh_vals)) if rh_vals else np.nan


def _annotate_temp_groups(
    ax: "plt.Axes",
    temp_stats: dict,
    rh_data: "Optional[pd.DataFrame]",
    font_size: int,
) -> None:
    """
    Add 'n=#\\nRH=##%' annotations above the highest data point in each
    temperature group.

    Parameters:
        ax: The axes on which boxes have already been drawn.
        temp_stats: Dict mapping numeric temperature → dict with keys
            'n' (event count), 'shower_on' (pd.Series of shower-on times),
            'max_val' (highest plotted data value across all bins).
        rh_data: Aranet4 Bedroom RH DataFrame (datetime + RH_bedroom) or None.
        font_size: Font size for annotation text.
    """
    if not temp_stats:
        return

    # Extend y-axis headroom so annotations aren't clipped (3 lines: W##, n=, RH=)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.25 * y_range)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = 0.02 * y_range

    for temp, stats in sorted(temp_stats.items()):
        n = stats["n"]
        max_val = stats.get("max_val", np.nan)
        if np.isnan(max_val):
            continue

        # Prepend W## label if a config_key is stored in the stats entry
        text_lines = []
        config_key = stats.get("config_key", "")
        if config_key and str(config_key).upper().startswith("W"):
            w_part = str(config_key).split("_")[0]  # e.g. "W48b"
            digits = "".join(c for c in w_part[1:] if c.isdigit())
            if digits:
                text_lines.append(f"W{digits}")
        text_lines.append(f"n={n}")
        if rh_data is not None and "shower_on" in stats:
            avg_rh = _get_rh_at_shower_on(stats["shower_on"], rh_data)
            if not np.isnan(avg_rh):
                text_lines.append(f"RH={avg_rh:.0f}%")

        ax.text(
            temp,
            max_val + offset,
            "\n".join(text_lines),
            ha="center",
            va="bottom",
            fontsize=font_size,
            color="black",
        )


def _build_temp_stats(
    base_df: pd.DataFrame,
    config_keys: list,
    temp_map: dict,
    value_cols: list,
) -> dict:
    """
    Build per-temperature statistics used for the annotation pass.

    Parameters:
        base_df: Filtered results DataFrame (base W## configs only).
        config_keys: Ordered list of config keys to include.
        temp_map: Mapping config_key → numeric temperature.
        value_cols: List of column names whose values contribute to max_val
            (e.g. all E_total or beta_raw_mean columns for the figure).

    Returns:
        Dict mapping numeric temperature → {n, shower_on (Series), max_val}.
    """
    temp_stats: dict = {}
    for config_key in config_keys:
        temp = temp_map.get(config_key)
        if temp is None:
            continue
        group_df = base_df[base_df["config_key"] == config_key]

        # Aggregate values for max-val tracking
        all_vals: list = []
        for col in value_cols:
            if col in group_df.columns:
                all_vals.extend(group_df[col].dropna().values.tolist())

        n = len(group_df)
        max_val = float(np.max(all_vals)) if all_vals else np.nan

        if temp in temp_stats:
            temp_stats[temp]["n"] += n
            if not np.isnan(max_val):
                cur = temp_stats[temp].get("max_val", np.nan)
                temp_stats[temp]["max_val"] = (
                    max(cur, max_val) if not np.isnan(cur) else max_val
                )
            if "shower_on" in group_df.columns:
                temp_stats[temp]["shower_on"] = pd.concat(
                    [temp_stats[temp]["shower_on"], group_df["shower_on"]]
                )
        else:
            temp_stats[temp] = {
                "n": n,
                "max_val": max_val,
                "shower_on": group_df["shower_on"].copy()
                if "shower_on" in group_df.columns
                else pd.Series([], dtype="object"),
                "config_key": config_key,
            }
    return temp_stats


# Configuration for the four fixed-temperature-axis boxplot functions.
# Keys: col_template, ylabel, title_metric, title_note, hline (None or float)
_TEMP_BOXPLOT_CONFIG = {
    "emission_etotal": dict(
        col_template="bin{n}_E_total",
        ylabel="Total Emission E_total (#)",
        title_metric="Particle Emission",
        title_note="(Box = median/IQR, whiskers = 1.5×IQR)",
        hline=None,
    ),
    "deposition_rate": dict(
        col_template="bin{n}_beta_raw_mean",
        ylabel="Deposition Rate β (h⁻¹)",
        title_metric="Particle Deposition Rate",
        title_note="(Box = median/IQR, whiskers = 1.5×IQR; β = unclamped trimmed mean)",
        hline=0.0,
    ),
    "emission_rate": dict(
        col_template="bin{n}_E_mean",
        ylabel="Emission Rate E (#/min)",
        title_metric="Particle Emission Rate",
        title_note="(Box = median/IQR, whiskers = 1.5×IQR; E = mean over shower-on to peak)",
        hline=None,
    ),
    "penetration_factor": dict(
        col_template="bin{n}_p_mean",
        ylabel="Penetration Factor p",
        title_metric="Particle Penetration Factor",
        title_note="(Box = median/IQR, whiskers = 1.5×IQR; p capped at 1)",
        hline=1.0,
    ),
}


def _draw_temp_axis_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    cfg: dict,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """
    Shared implementation for all four fixed-temperature-axis boxplot functions.

    Produces two output files (bin0-2 and bin3-6) by appending the bin-range
    suffix to *output_path*.  Only base W## events are included.  X-axis is the
    fixed 5–60 °C range from BOXPLOT_CONFIG.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; suffix ``_bin0-2`` / ``_bin3-6`` is appended.
        cfg: Configuration dict from _TEMP_BOXPLOT_CONFIG.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' columns.
    """
    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return

    base_df = results_df[results_df["config_key"].apply(_is_base_config)].copy()
    if base_df.empty:
        return

    config_keys = sort_config_keys_by_water_temp(
        [k for k in base_df["config_key"].dropna().unique()]
    )
    if not config_keys:
        return

    temp_map = {k: _extract_config_temp(k) for k in config_keys}
    all_bin_nums = list(particle_bins.keys())
    all_bin_widths = np.linspace(
        BOXPLOT_CONFIG["box_width_min"],
        BOXPLOT_CONFIG["box_width_max"],
        len(all_bin_nums),
    )
    bin_groups = [
        ([b for b in all_bin_nums if b <= 2], "bin0-2"),
        ([b for b in all_bin_nums if b >= 3], "bin3-6"),
    ]

    for group_bins, group_label in bin_groups:
        if not group_bins:
            continue

        value_cols = [cfg["col_template"].format(n=b) for b in group_bins]
        temp_stats = _build_temp_stats(base_df, config_keys, temp_map, value_cols)

        fig, ax = create_figure(figsize=BOXPLOT_CONFIG["figsize"])
        if isinstance(ax, list):
            ax = ax[0]

        for bin_num in group_bins:
            global_idx = all_bin_nums.index(bin_num)
            color = SENSOR_COLORS[global_idx % len(SENSOR_COLORS)]
            col = cfg["col_template"].format(n=bin_num)
            if col not in base_df.columns:
                continue

            positions, data = [], []
            for config_key in config_keys:
                temp = temp_map.get(config_key)
                if temp is None:
                    continue
                values = base_df[base_df["config_key"] == config_key][col].dropna().values
                if len(values) > 0:
                    positions.append(temp)
                    data.append(values)

            if not data:
                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=float(all_bin_widths[global_idx]),
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker=BOXPLOT_CONFIG["flier_marker"],
                    markersize=BOXPLOT_CONFIG["flier_markersize"],
                    alpha=BOXPLOT_CONFIG["flier_alpha"],
                    color=color,
                ),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(BOXPLOT_CONFIG["box_alpha"])
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(color)
                    line.set_alpha(BOXPLOT_CONFIG["box_alpha"])
            for med in bp["medians"]:
                med.set_color(BOXPLOT_CONFIG["median_color"])
                med.set_linewidth(BOXPLOT_CONFIG["median_linewidth"])

        _annotate_temp_groups(ax, temp_stats, rh_data, font_size=FONT_SIZE_ANNOTATION)

        bxmin = BOXPLOT_CONFIG["temp_xmin"]
        bxmax = BOXPLOT_CONFIG["temp_xmax"]
        bxstep = BOXPLOT_CONFIG["temp_xtick_step"]
        bxticks = range(bxmin, bxmax + bxstep, bxstep)
        ax.set_xlim(bxmin, bxmax)
        ax.set_xticks(bxticks)
        ax.set_xticklabels([f"{t}°C" for t in bxticks], fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Water Temperature (°C)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(cfg["ylabel"], fontsize=FONT_SIZE_LABEL)
        bin_label = group_label.replace("-", "–").replace("bin", "Bin ")
        ax.set_title(
            f"{cfg['title_metric']} by Water Temperature — {bin_label}"
            f"\n{cfg['title_note']}",
            fontsize=FONT_SIZE_TITLE,
            fontweight=TITLE_FONTWEIGHT,
        )
        if cfg["hline"] is not None:
            ax.axhline(cfg["hline"], color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=FONT_SIZE_TICK)

        legend_elements = [
            Patch(
                facecolor=SENSOR_COLORS[all_bin_nums.index(b) % len(SENSOR_COLORS)],
                alpha=0.7,
                label=f"Bin {b} ({particle_bins[b]['name']} µm)",
            )
            for b in group_bins
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=FONT_SIZE_LEGEND - 1, ncol=1)

        group_output = output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        plt.tight_layout()
        save_figure(fig, group_output)
        plt.close(fig)


def plot_emission_boxplot(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Two box-and-whisker figures of total particle emission (E_total) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_total).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
    """
    _draw_temp_axis_boxplot(results_df, particle_bins, output_path, _TEMP_BOXPLOT_CONFIG["emission_etotal"], rh_data)


def plot_deposition_rate_boxplot(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Two box-and-whisker figures of unclamped deposition rate (beta_raw_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_beta_raw_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
    """
    _draw_temp_axis_boxplot(results_df, particle_bins, output_path, _TEMP_BOXPLOT_CONFIG["deposition_rate"], rh_data)


def plot_emission_rate_boxplot(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Two box-and-whisker figures of mean emission rate (E_mean, #/min) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
    """
    _draw_temp_axis_boxplot(results_df, particle_bins, output_path, _TEMP_BOXPLOT_CONFIG["emission_rate"], rh_data)


def plot_penetration_factor_boxplot(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Two box-and-whisker figures of penetration factor (p_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_p_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
    """
    _draw_temp_axis_boxplot(results_df, particle_bins, output_path, _TEMP_BOXPLOT_CONFIG["penetration_factor"], rh_data)


def plot_emission_etotal_by_metric_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    metric_col: str,
    metric_label: str,
    rh_data: "Optional[pd.DataFrame]" = None,
    x_range: "Optional[tuple]" = None,
) -> None:
    """
    Create two box-and-whisker figures of E_total positioned along a continuous metric axis.

    Produces one figure for small bins (Bin 0–2) and one for large bins (Bin 3–6).
    Unlike :func:`plot_emission_boxplot`, the x-axis is **not** fixed to the
    5–60 °C water-temperature range; instead each water-temperature group (W##) is
    centred at the *group mean* of *metric_col*, and the x-axis auto-scales from data
    unless *x_range* is provided.
    Box widths also scale proportionally to the data range.

    Intended for exploring how E_total relates to continuous predictors such as
    bedroom RH, bedroom temperature, air-change rate, deposition rate, or penetration
    factor — see the 10-figure Task 7 series.

    Only base W## events are included (letter-suffix repeats excluded).

    Parameters:
        results_df: DataFrame with analysis results; must contain 'config_key',
                    bin{n}_E_total columns, and *metric_col*.
        particle_bins: Dictionary of particle bin information.
        output_path: Base path used to derive the two output filenames (suffix
                     ``_bin0-2`` and ``_bin3-6`` are appended to the stem).
        metric_col: Column name in *results_df* used to position each
                    temperature-group box along the x-axis (group mean).
        metric_label: Human-readable x-axis label (e.g. 'Bedroom RH (%)').
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for the
                 n=/RH= annotation (same as other boxplot functions).
        x_range: Optional (xmin, xmax, xtick_step) tuple to fix the x-axis
                 range and tick positions. When None the axis auto-scales.
    """
    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return
    if metric_col not in results_df.columns:
        return

    base_df = results_df[results_df["config_key"].apply(_is_base_config)].copy()

    if base_df.empty:
        return

    config_keys = sort_config_keys_by_water_temp(
        [k for k in base_df["config_key"].dropna().unique()]
    )

    if not config_keys:
        return

    all_bin_nums = list(particle_bins.keys())

    bin_groups = [
        ([b for b in all_bin_nums if b <= 2], "bin0-2"),
        ([b for b in all_bin_nums if b >= 3], "bin3-6"),
    ]

    for group_bins, group_label in bin_groups:
        if not group_bins:
            continue

        # Compute per-group x positions (mean of metric_col for each W## group)
        group_x_pos: dict = {}
        for config_key in config_keys:
            group_df = base_df[base_df["config_key"] == config_key]
            x_vals = group_df[metric_col].dropna().values
            if len(x_vals) > 0:
                group_x_pos[config_key] = float(np.mean(x_vals))

        if not group_x_pos:
            continue

        # Scale box widths proportionally to the x-axis data range.
        # Targets roughly the same visual proportion as the fixed-axis boxplots
        # (widest bin ≈ 4.5% of range, narrowest ≈ 0.7%).
        all_x_vals = sorted(group_x_pos.values())
        x_data_span = max(all_x_vals[-1] - all_x_vals[0], 0.01)
        width_max = 0.045 * x_data_span
        width_min = 0.007 * x_data_span
        all_bin_widths = np.linspace(width_min, width_max, len(all_bin_nums))

        # Build annotation stats keyed by x position (not temperature)
        value_cols = [f"bin{b}_E_total" for b in group_bins]
        annot_stats: dict = {}
        for config_key, x_pos in group_x_pos.items():
            group_df = base_df[base_df["config_key"] == config_key]
            all_vals: list = []
            for col in value_cols:
                if col in group_df.columns:
                    all_vals.extend(group_df[col].dropna().values.tolist())
            n = len(group_df)
            max_val = float(np.max(all_vals)) if all_vals else np.nan
            annot_stats[x_pos] = {
                "n": n,
                "max_val": max_val,
                "shower_on": group_df["shower_on"].copy()
                if "shower_on" in group_df.columns
                else pd.Series([], dtype="object"),
                "config_key": config_key,
            }

        fig, ax = create_figure(figsize=BOXPLOT_CONFIG["figsize"])
        if isinstance(ax, list):
            ax = ax[0]

        for bin_num in group_bins:
            global_idx = all_bin_nums.index(bin_num)
            color = SENSOR_COLORS[global_idx % len(SENSOR_COLORS)]
            col = f"bin{bin_num}_E_total"
            if col not in base_df.columns:
                continue

            box_width = float(all_bin_widths[global_idx])
            positions = []
            data = []

            for config_key in config_keys:
                x_pos = group_x_pos.get(config_key)
                if x_pos is None:
                    continue
                values = (
                    base_df[base_df["config_key"] == config_key][col].dropna().values
                )
                if len(values) > 0:
                    positions.append(x_pos)
                    data.append(values)

            if not data:
                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=BOXPLOT_CONFIG["flier_marker"], markersize=BOXPLOT_CONFIG["flier_markersize"], alpha=BOXPLOT_CONFIG["flier_alpha"], color=color),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(BOXPLOT_CONFIG["box_alpha"])
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(color)
                    line.set_alpha(BOXPLOT_CONFIG["box_alpha"])
            for med in bp["medians"]:
                med.set_color(BOXPLOT_CONFIG["median_color"])
                med.set_linewidth(BOXPLOT_CONFIG["median_linewidth"])

        _annotate_temp_groups(ax, annot_stats, rh_data, font_size=FONT_SIZE_ANNOTATION)

        # Apply fixed x-axis range when provided; otherwise auto-scale
        if x_range is not None:
            xmin, xmax, xstep = x_range
            ticks = np.arange(xmin, xmax + xstep * 0.5, xstep)
            ticks = np.round(ticks, 10)  # avoid floating-point drift
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f"{t:g}" for t in ticks],
                fontsize=FONT_SIZE_TICK,
                rotation=45,
                ha="right",
            )

        ax.set_xlabel(metric_label, fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Total Emission E_total (#)", fontsize=FONT_SIZE_LABEL)
        ax.set_title(
            f"Particle Emission by {metric_label} — "
            f"{group_label.replace('-', '–').replace('bin', 'Bin ')}"
            "\n(Box = median/IQR, whiskers = 1.5×IQR; x = group mean of metric)",
            fontsize=FONT_SIZE_TITLE,
            fontweight=TITLE_FONTWEIGHT,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
        if x_range is None:
            ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)

        legend_elements_metric = [
            Patch(
                facecolor=SENSOR_COLORS[all_bin_nums.index(b) % len(SENSOR_COLORS)],
                alpha=0.7,
                label=f"Bin {b} ({particle_bins[b]['name']} µm)",
            )
            for b in group_bins
        ]
        ax.legend(
            handles=legend_elements_metric,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND - 1,
            ncol=1,
        )

        metric_output = (
            output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        )
        plt.tight_layout()
        save_figure(fig, metric_output)
        plt.close(fig)
