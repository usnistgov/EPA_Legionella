#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Plotting Functions
=====================================

This module provides specialized plotting functions for particle decay and
emission analysis in the EPA Legionella project. Functions are called by
particle_decay_analysis.py and produce individual event figures, cross-event
summary bar charts, and water-temperature/metric-axis boxplots for all twelve
particle size bins (0.35–10.0 µm).

Key Functions:
    - plot_particle_decay_event: Four-panel individual event figure (concentration + three emission groups)
    - plot_penetration_summary: Bar chart of penetration factors by size bin
    - plot_deposition_summary: Bar chart of other process (deposition) rates by size bin
    - plot_emission_summary: Bar chart of emission rates by size bin
    - plot_size_distribution_summary: Multi-panel summary of all metrics by size bin
    - plot_emission_boxplot: E_total box-and-whisker by water temperature (fixed 5–60 °C axis)
    - plot_deposition_rate_boxplot: beta_raw_mean box-and-whisker by water temperature
    - plot_emission_rate_boxplot: E_mean box-and-whisker by water temperature
    - plot_penetration_factor_boxplot: p_mean box-and-whisker by water temperature
    - plot_emission_etotal_by_metric_boxplot: E_total vs. continuous metric axis (RH, temp, ACR, beta, p)
    - plot_emission_etotal_by_showerhead_boxplot: E_total grouped by shower head type

Plot Features:
    - Four-panel event plots: concentration time series (top) + three per-step emission panels
      (bins 0–2 small, 3–6 medium, 7–11 large)
    - Color-coded particle size bins; all bins plotted as solid lines regardless of beta validity
    - Optional outdoor PM overlay: dotted lines at 55% alpha showing outdoor concentrations
      for events listed in OUTDOOR_PM_EVENTS (e.g., event 77)
    - Shaded deposition analysis window (2 hr post-shower)
    - Shower ON/OFF dotted markers distinct from fit/predicted dashed lines
    - Decay R² values listed in top-panel text box alongside lambda and valid-bin count
    - Log-scale concentration axis for wide dynamic range
    - Emission panel: per-step E_t as faint lines, E_mean as dashed horizontal lines
      spanning shower_on to peak_time; y-axis clipped to 2nd–98th percentile of E_per_step data
    - Emission panel x-axis matches concentration panel (shared time axis)
    - Fixed water-temperature axis boxplots (5–60 °C, configurable); box widths scale
      with bin particle size (Bin 0 narrowest → Bin 11 widest)
    - Metric-axis boxplots: each W## group centred at the group mean of the predictor column;
      box widths proportional to data range
    - Shower-head boxplot: categorical x-axis with three temperature clusters and visible gaps
    - W##\nn=#\nRH=##% annotation above each group on all boxplot figures
    - Temperature-based colors from get_config_color(); RH from rh_temp_wind_summary.xlsx

Methodology:
    1. Extract data window around shower event (1 hr before to 0.5 hr after deposition end)
    2. Top panel: plot measured particle concentrations for all 12 size bins
    3. Shade deposition window and overlay continuous predicted Ct curves per valid bin
       (emission phase shower_on→peak_time, then decay phase; decay starts from predicted
       concentration at peak_time, not the measured peak value)
    4. Bottom panels (3): small bins 0–2, medium bins 3–6, large bins 7–11; per-step E_t
       as faint lines; E_mean as dashed horizontal lines per valid bin; y-axis uses 1e#
       notation; R² annotation displayed in the small-bins emission panel
    5. Summary bar charts: compute mean ± std across all events for each metric and bin
    6. Fixed-axis boxplots: group events by W## config key; draw one box per group per bin;
       annotate with W##, n count, and mean bedroom RH
    7. Metric-axis boxplots: position each W## group at group mean of continuous predictor
       with proportional box widths; otherwise same grouping and annotation logic
    8. Shower-head boxplot: categorical grouping with temperature-ordered clusters

Output Files:
    - {test_name}_particle_decay.png: Individual event concentration + emission figure
    - penetration_summary.png: Bar chart of penetration factors across all events
    - deposition_summary.png: Bar chart of other process (beta) rates across all events
    - emission_summary.png: Bar chart of emission rates across all events
    - size_distribution_summary.png: Multi-panel summary of all metrics
    - emission_etotal_boxplot_{bin0-2,bin3-6,bin7-11}.png: E_total by water temperature
    - deposition_rate_boxplot_{bin0-2,bin3-6,bin7-11}.png: beta_raw_mean by water temperature
    - emission_rate_boxplot_{bin0-2,bin3-6,bin7-11}.png: E_mean by water temperature
    - penetration_factor_boxplot_{bin0-2,bin3-6,bin7-11}.png: p_mean by water temperature
    - emission_etotal_by_{metric}_{bin0-2,bin3-6,bin7-11}.png: E_total vs. continuous metric
      (15 figures: bedroom_rh, bedroom_temp, acr, beta, p × three bin groups)
    - emission_etotal_by_showerhead_{bin0-2,bin3-6,bin7-11}.png: E_total by shower head type

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import re
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

import src.sig_figs as sf
from src.event_manager import sort_config_keys_by_water_temp
from src.plot_style import (
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
    show_outdoor: bool = False,
) -> None:
    """
    Plot particle concentration decay for a single event showing all bins.

    Creates a two-panel figure matching the CO2 analytical plot style:

    Top panel (ax1) — concentration time series:
      - Measured indoor particle concentrations for all bins
      - Continuous predicted Ct curve per valid bin (emission + decay phases)
      - Shower ON/OFF markers and shaded deposition window
      - Optional outdoor PM overlay (when show_outdoor=True): plots
        opc_binN_outside columns using the same per-bin colours with
        reduced alpha and dotted lines

    Bottom panels (ax2–ax4) — per-step emission rates by bin group:
      - Per-step E_t values as faint lines for each valid bin
      - E_mean as a horizontal dashed line spanning shower_on to peak_time
      - Emission R² annotation in each panel

    Parameters:
        particle_data: DataFrame with particle concentrations
        event: Event timing dictionary
        particle_bins: Dictionary of all particle bin information
        result: Analysis results for this event (all bins)
        output_path: Path to save the figure
        event_number: Event number for title
        test_name: Test name for title (e.g., "0114_HW_Morning_R01")
        show_outdoor: If True, overlay outdoor PM concentration on the top
            panel using opc_binN_outside columns (same colours, dotted lines).
    """
    apply_style()

    # Extract data for plotting window (1 hour before shower to 1 hour after deposition end)
    plot_start = event["shower_on"] - timedelta(hours=1)
    plot_end = event["deposition_end"] + timedelta(hours=1)

    mask = (particle_data["datetime"] >= plot_start) & (particle_data["datetime"] <= plot_end)
    plot_data = particle_data[mask].copy()

    if plot_data.empty:
        print(f"    Warning: No data for event {event_number}")
        return

    # Four-panel figure: concentration (top) + three emission panels by bin group
    fig, axes_array = create_figure(nrows=4, ncols=1, figsize=(12, 14), height_ratios=[2, 1, 1, 1])
    ax1, ax2, ax3, ax4 = cast(np.ndarray, axes_array)

    lambda_ach = result.get("lambda_ach", np.nan)

    # =========================================================================
    # Top panel: Particle concentrations with decay predictions
    # =========================================================================
    for bin_num, bin_info in particle_bins.items():
        col_inside = f"{bin_info['column']}_inside"
        color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

        if col_inside in plot_data.columns:
            # Plot raw instrument data — always solid regardless of beta validity
            ax1.plot(
                plot_data["datetime"],
                plot_data[col_inside],
                label=f"Bin {bin_num} ({bin_info['name']} µm)",
                color=color,
                linewidth=LINE_WIDTH_DATA,
                linestyle="-",
                alpha=0.9,
            )

            # Plot continuous predicted Ct: emission phase then decay phase
            # as two segments of the same simulation (they connect at peak_time)
            emission_dts = result.get(f"bin{bin_num}_emission_datetimes", [])
            emission_pred = result.get(f"bin{bin_num}_emission_predicted", [])
            decay_dts = result.get(f"bin{bin_num}_decay_datetimes", [])
            decay_pred = result.get(f"bin{bin_num}_decay_predicted", [])

            if len(emission_dts) > 0 and len(emission_pred) > 0:
                ep_arr = np.array(emission_pred, dtype=float)
                ep_arr = np.where(ep_arr > 0, ep_arr, np.nan)
                if np.any(np.isfinite(ep_arr)):
                    ax1.plot(
                        pd.to_datetime(emission_dts),
                        ep_arr,
                        color=color,
                        linewidth=LINE_WIDTH_FIT,
                        linestyle="--",
                        alpha=0.8,
                    )
            if len(decay_dts) > 0 and len(decay_pred) > 0:
                dp_arr = np.array(decay_pred, dtype=float)
                dp_arr = np.where(dp_arr > 0, dp_arr, np.nan)
                if np.any(np.isfinite(dp_arr)):
                    ax1.plot(
                        pd.to_datetime(decay_dts),
                        dp_arr,
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

    # Optional outdoor PM concentration overlay
    if show_outdoor:
        outdoor_added = False
        for bin_num, bin_info in particle_bins.items():
            col_outside = f"{bin_info['column']}_outside"
            color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]
            if col_outside in plot_data.columns:
                ax1.plot(
                    plot_data["datetime"],
                    plot_data[col_outside],
                    color=color,
                    linewidth=LINE_WIDTH_DATA * 0.8,
                    linestyle=":",
                    alpha=0.55,
                )
                outdoor_added = True
        if outdoor_added:
            ax1.plot(
                [],
                [],
                color="gray",
                linestyle=":",
                linewidth=LINE_WIDTH_DATA * 0.8,
                alpha=0.55,
                label="Outdoor Concentration",
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
        beta_val = result.get(f"bin{bin_num}_beta_other", np.nan)
        if not np.isnan(beta_val):
            valid_bins += 1
            r2_val = result.get(f"bin{bin_num}_beta_other_r_squared", np.nan)
            r2_str = sf.fmt_fig(r2_val, fallback=".3f") if not np.isnan(r2_val) else "N/A"
            decay_r2_lines.append(f" B{bin_num}: R²={r2_str}")

    # Build text box content: lambda, valid-bin count, and per-bin decay R²
    textstr = f"λ = {sf.fmt_fig(lambda_ach, fallback='.4f')} h⁻¹\n"
    if decay_r2_lines:
        textstr += "\nDecay R²:\n" + "\n".join(decay_r2_lines)

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
        ncol=1,
    )
    # Compute y limits from all positive plotted values (measured + predicted)
    _all_pos_y = []
    for _bn, _bi in particle_bins.items():
        _col = f"{_bi['column']}_inside"
        if _col in plot_data.columns:
            _vals = plot_data[_col].dropna().values
            _all_pos_y.extend(_vals[_vals > 0].tolist())
        for _key in (f"bin{_bn}_emission_predicted", f"bin{_bn}_decay_predicted"):
            _all_pos_y.extend(
                [
                    v
                    for v in result.get(_key, [])
                    if isinstance(v, (int, float)) and v > 0 and not np.isnan(v)
                ]
            )
    ax1.set_yscale("log")
    if _all_pos_y:
        _ylo = min(_all_pos_y) * 0.3
        _yhi = max(_all_pos_y) * 3.0
        ax1.set_ylim(bottom=max(_ylo, 1e-6), top=_yhi)
    else:
        ax1.set_ylim(bottom=1e-3)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    format_datetime_axis(ax1)
    ax1.set_xlim(plot_start, plot_end)

    # =========================================================================
    # Emission panels: per-step emission rates split by bin group
    # ax2 = Bins 0–2  (small particles)
    # ax3 = Bins 3–6  (medium particles)
    # ax4 = Bins 7–11 (large particles)
    # Each panel has its own y-axis scale so all groups are readable.
    # =========================================================================
    bins_small = [bn for bn in particle_bins.keys() if bn <= 2]
    bins_medium = [bn for bn in particle_bins.keys() if 3 <= bn <= 6]
    bins_large = [bn for bn in particle_bins.keys() if bn >= 7]

    # Shared x-axis limits: 10 min before shower_on to 10 min after latest peak_time
    peak_times = []
    for bn in particle_bins.keys():
        pt = result.get(f"bin{bn}_peak_time", None)
        if pt is not None:
            peak_times.append(pd.Timestamp(pt))
    ax_e_left = event["shower_on"] - timedelta(minutes=10)
    ax_e_right = (
        max(peak_times) + timedelta(minutes=10)
        if peak_times
        else event["shower_off"] + timedelta(minutes=10)
    )

    def _populate_emission_panel(ax, bin_group, panel_label):
        """Plot E_per_step scatter and E_mean dashed line for a subset of bins."""
        has_data = False
        r2_lines = []
        e_steps_panel = []
        e_means_panel = []

        for bin_num in bin_group:
            bin_info = particle_bins[bin_num]
            color = SENSOR_COLORS[bin_num % len(SENSOR_COLORS)]

            E_times = result.get(f"bin{bin_num}_E_times", [])
            E_per_step = result.get(f"bin{bin_num}_E_per_step", [])
            E_mean_val = result.get(f"bin{bin_num}_E_mean", np.nan)
            E_r2_val = result.get(f"bin{bin_num}_E_r_squared", np.nan)
            peak_time = result.get(f"bin{bin_num}_peak_time", None)

            if len(E_times) > 0 and len(E_per_step) > 0:
                has_data = True
                e_steps_panel.extend(E_per_step)
                # Per-step E_t as a faint line (all values, including negative)
                ax.plot(
                    pd.to_datetime(E_times),
                    np.array(E_per_step),
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )

            # E_mean as horizontal dashed line spanning shower_on → peak_time
            if not np.isnan(E_mean_val) and peak_time is not None:
                has_data = True
                e_means_panel.append(E_mean_val)
                ax.hlines(
                    E_mean_val,
                    event["shower_on"],
                    pd.Timestamp(peak_time),
                    color=color,
                    linewidth=LINE_WIDTH_FIT,
                    linestyle="--",
                    alpha=0.9,
                )
                r2_str = sf.fmt_fig(E_r2_val, fallback=".3f") if not np.isnan(E_r2_val) else "N/A"
                r2_lines.append(f"B{bin_num}: R²={r2_str}")

        add_shower_on_marker(ax, event["shower_on"])
        add_shower_off_marker(ax, event["shower_off"])

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_ylabel(f"E (#/cm³·min)\n{panel_label}", fontsize=FONT_SIZE_LABEL - 1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.set_xlim(ax_e_left, ax_e_right)
        format_datetime_axis(ax, interval_minutes=5)

        # Per-panel percentile y-limits so E_mean dashed lines are never clipped
        if e_steps_panel:
            e_arr = np.array([v for v in e_steps_panel if not np.isnan(v)], dtype=float)
            if len(e_arr) >= 4:
                p2 = float(np.percentile(e_arr, 2))
                p98 = float(np.percentile(e_arr, 98))
                if e_means_panel:
                    p2 = min(p2, min(e_means_panel))
                    p98 = max(p98, max(e_means_panel))
                margin = max((p98 - p2) * 0.15, abs(p98) * 0.05, 1e-6)
                ax.set_ylim(p2 - margin, p98 + margin)

        if r2_lines:
            r2_text = "Emission R²:\n" + "\n".join(r2_lines)
            props_r2 = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
            ax.text(
                0.02,
                0.98,
                r2_text,
                transform=ax.transAxes,
                fontsize=FONT_SIZE_LEGEND - 1,
                verticalalignment="top",
                bbox=props_r2,
            )

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No emission data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=FONT_SIZE_LABEL,
                color="gray",
            )

    _populate_emission_panel(ax2, bins_small, "Bins 0–2")
    _populate_emission_panel(ax3, bins_medium, "Bins 3–6")
    _populate_emission_panel(ax4, bins_large, "Bins 7–11")

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
        col_template="bin{n}_beta_other",
        ylabel="Other Process Rate β (h⁻¹)",
        title="Other Process Rate by Particle Size\n(Mean ± Std Dev)",
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

    Handles all three metric types (penetration, other process, emission) via a
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

    # Scale figure width with bin count (at least 1.4 in per bin, min 14 in wide)
    fig_width = max(14, len(bin_nums) * 1.4)
    if n_configs > 1:
        fig, axes = plt.subplots(n_configs, 1, figsize=(fig_width, 6 * n_configs), squeeze=False)
        axes = axes.flatten()
    else:
        fig, _ax = create_figure(figsize=(fig_width, 6))
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
            x,
            means,
            yerr=stds,
            capsize=4,
            color=bar_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Only add value labels if the number of bins is small enough to avoid overlap
        if len(bin_nums) <= 8:
            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + cfg["label_offset"],
                        sf.fmt_fig(mean, fallback=cfg["label_fmt"]),
                        ha="center",
                        va="bottom",
                        fontsize=FONT_SIZE_TICK - 2,
                    )

        ax.set_xlabel("Particle Size Bin (µm)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(cfg["ylabel"], fontsize=FONT_SIZE_LABEL)
        if n_configs > 1:
            ax.set_title(
                f"Configuration: {config_key} (n={len(config_df)})",
                fontsize=FONT_SIZE_TITLE,
                fontweight=TITLE_FONTWEIGHT,
            )
        else:
            ax.set_title(cfg["title"], fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1)

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
            cfg["title"],
            fontsize=FONT_SIZE_TITLE + 2,
            fontweight=TITLE_FONTWEIGHT,
        )

    plt.tight_layout(pad=2.0)
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
    _plot_summary_bar_chart(
        results_df, particle_bins, output_path, _BAR_CHART_CONFIG["penetration"]
    )


def plot_deposition_summary(
    results_df: pd.DataFrame, particle_bins: Dict, output_path: Path
) -> None:
    """Bar chart of other process rates across all bins (mean ± std per bin).

    Parameters:
        results_df: DataFrame with analysis results.
        particle_bins: Dictionary of particle bin information.
        output_path: Path to save the figure.
    """
    _plot_summary_bar_chart(results_df, particle_bins, output_path, _BAR_CHART_CONFIG["deposition"])


def plot_emission_summary(results_df: pd.DataFrame, particle_bins: Dict, output_path: Path) -> None:
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
    bin_centers = [(particle_bins[i]["min"] + particle_bins[i]["max"]) / 2 for i in bin_nums]

    fig, axes_temp = create_figure(nrows=1, ncols=3, figsize=(15, 5))
    axes = cast(np.ndarray, axes_temp)

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

    # Panel 2: Other Process Rate
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
    axes[1].set_ylabel("Other Process Rate β (h⁻¹)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_title(
        "(b) Other Process Rate", fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT
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
    axes[2].set_title("(c) Emission Rate", fontsize=FONT_SIZE_TITLE, fontweight=TITLE_FONTWEIGHT)
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
    """Return True if config_key is the baseline temperature-sweep config.

    Requires: Standard head (no Pepco/FilterWand/Used), no Mannequin,
    Door Open, Fan Off.  Only these events are plotted on the fixed
    water-temperature axis so that temperature is the sole variable.
    """
    key_str = str(key)
    return (
        bool(re.match(r"^W\d+(_|$)", key_str))
        and "_Pepco" not in key_str
        and "_FilterWand" not in key_str
        and "_Used" not in key_str
        and "_Mannequin" not in key_str
        and "_DoorClosed" not in key_str
        and "_DoorPartial" not in key_str
        and "_FanOn" not in key_str
    )


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
    ax: Axes,
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
                text_lines.append(f"RH={sf.fmt_fig(avg_rh, fallback='.0f')}%")

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
                temp_stats[temp]["max_val"] = max(cur, max_val) if not np.isnan(cur) else max_val
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
        col_template="bin{n}_beta_other_raw_mean",
        ylabel="Other Process Rate β (h⁻¹)",
        title_metric="Particle Other Process Rate",
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
    x_range: "Optional[tuple]" = None,
) -> None:
    """
    Shared implementation for all four fixed-temperature-axis boxplot functions.

    Produces two output files (bin0-2 and bin3-6) by appending the bin-range
    suffix to *output_path*.  Only base W## events are included.  X-axis is the
    fixed temperature range from BOXPLOT_CONFIG (5–60 °C) unless *x_range* is
    provided.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; suffix ``_bin0-2`` / ``_bin3-6`` is appended.
        cfg: Configuration dict from _TEMP_BOXPLOT_CONFIG.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' columns.
        x_range: Optional (xmin, xmax, xtick_step) tuple in °C to override the
                 BOXPLOT_CONFIG temp axis range.
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
        ([b for b in all_bin_nums if 3 <= b <= 6], "bin3-6"),
        ([b for b in all_bin_nums if b >= 7], "bin7-11"),
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

        if x_range is not None:
            bxmin, bxmax, bxstep = x_range
        else:
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
            f"{cfg['title_metric']} by Water Temperature — {bin_label}\n{cfg['title_note']}",
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
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND - 1,
            ncol=1,
        )

        group_output = output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        plt.tight_layout()
        save_figure(fig, group_output)
        plt.close(fig)


def plot_emission_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
    x_range: "Optional[tuple]" = None,
) -> None:
    """Two box-and-whisker figures of total particle emission (E_total) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_total).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
        x_range: Optional (xmin, xmax, xtick_step) in °C to override the default 5–60 °C axis.
    """
    _draw_temp_axis_boxplot(
        results_df,
        particle_bins,
        output_path,
        _TEMP_BOXPLOT_CONFIG["emission_etotal"],
        rh_data,
        x_range,
    )


def plot_deposition_rate_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
    x_range: "Optional[tuple]" = None,
) -> None:
    """Two box-and-whisker figures of unclamped other process rate (beta_raw_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_beta_raw_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
        x_range: Optional (xmin, xmax, xtick_step) in °C to override the default 5–60 °C axis.
    """
    _draw_temp_axis_boxplot(
        results_df,
        particle_bins,
        output_path,
        _TEMP_BOXPLOT_CONFIG["deposition_rate"],
        rh_data,
        x_range,
    )


def plot_emission_rate_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
    x_range: "Optional[tuple]" = None,
) -> None:
    """Two box-and-whisker figures of mean emission rate (E_mean, #/min) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
        x_range: Optional (xmin, xmax, xtick_step) in °C to override the default 5–60 °C axis.
    """
    _draw_temp_axis_boxplot(
        results_df,
        particle_bins,
        output_path,
        _TEMP_BOXPLOT_CONFIG["emission_rate"],
        rh_data,
        x_range,
    )


def plot_penetration_factor_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
    x_range: "Optional[tuple]" = None,
) -> None:
    """Two box-and-whisker figures of penetration factor (p_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_p_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` suffixes are appended.
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for RH annotation.
        x_range: Optional (xmin, xmax, xtick_step) in °C to override the default 5–60 °C axis.
    """
    _draw_temp_axis_boxplot(
        results_df,
        particle_bins,
        output_path,
        _TEMP_BOXPLOT_CONFIG["penetration_factor"],
        rh_data,
        x_range,
    )


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
    bedroom RH, bedroom temperature, air-change rate, other process rate, or penetration
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
        ([b for b in all_bin_nums if 3 <= b <= 6], "bin3-6"),
        ([b for b in all_bin_nums if b >= 7], "bin7-11"),
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
                values = base_df[base_df["config_key"] == config_key][col].dropna().values
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

        metric_output = output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        plt.tight_layout()
        save_figure(fig, metric_output)
        plt.close(fig)


def plot_emission_etotal_by_showerhead_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """
    Create two box-and-whisker figures of E_total grouped by shower head type.

    Produces one figure for small bins (Bin 0–2) and one for large bins (Bin 3–6).
    Configs are ordered by temperature and grouped into three clusters separated
    by visible gaps on the categorical x-axis:
      - Cluster 1: W37, W40_Pepco_Narrow, W40_Pepco_Wide, W40_Pepco_Mid, W43
      - Cluster 2: W48, W49_Pepco_Wide
      - Cluster 3: W53, W52_Pepco_Wide
    Keys match compound config_key prefixes (e.g. "W40_Pepco_Narrow" matches
    "W40_Pepco_Narrow_DoorOpen_FanOff").  Annotations follow the standard
    'W##\\nn=#\\nRH=##%' style.

    Parameters:
        results_df: DataFrame with analysis results; must contain 'config_key'
                    and bin{n}_E_total columns.
        particle_bins: Dictionary of particle bin information.
        output_path: Base path used to derive the two output filenames (suffix
                     ``_bin0-2`` and ``_bin3-6`` are appended to the stem).
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for the
                 n=/RH= annotation (same as other boxplot functions).
    """
    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return

    # Ordered mapping: compound config_key prefix → (categorical x position, tick label)
    # Keys are the leading portion of the full config_key (everything before _Door…).
    # Standard-head events match on the W## token alone; Pepco events match on
    # W##_Pepco_<Pattern>.  Positions leave a gap of 1 unit between clusters.
    SHOWERHEAD_CONFIGS = {
        # Cluster 1: base W37 + Pepco narrow/wide/mid near 40°C + base W43
        "W37": (0, "W37"),
        "W40_Pepco_Narrow": (1, "W40\nP.Narrow"),
        "W40_Pepco_Wide": (2, "W40\nP.Wide"),
        "W40_Pepco_Mid": (3, "W40\nP.Mid"),
        "W43": (4, "W43"),
        # gap at 5
        # Cluster 2: base W48 + Pepco wide near 49°C
        "W48": (6, "W48"),
        "W49_Pepco_Wide": (7, "W49\nP.Wide"),
        # gap at 8
        # Cluster 3: base W53 + Pepco wide near 52°C
        "W53": (9, "W53"),
        "W52_Pepco_Wide": (10, "W52\nP.Wide"),
    }

    # Match each row's config_key to a SHOWERHEAD_CONFIGS group.
    # A config_key like "W40_Pepco_Narrow_DoorOpen_FanOff" matches the group
    # key "W40_Pepco_Narrow" because it starts with that prefix followed by "_".
    # A standard-head key like "W37_DoorOpen_FanOff" matches "W37" the same way.
    def _sh_group(ck: str) -> "Optional[str]":
        for sh_key in SHOWERHEAD_CONFIGS:
            if ck == sh_key or ck.startswith(sh_key + "_"):
                return sh_key
        return None

    sh_df = results_df.copy()
    sh_df["_sh_group"] = sh_df["config_key"].apply(_sh_group)
    sh_df = sh_df[sh_df["_sh_group"].notna()].copy()
    if sh_df.empty:
        return

    all_bin_nums = list(particle_bins.keys())

    # Proportional box widths scaled to the x-axis data span (0–9)
    x_positions_all = [x for (x, _) in SHOWERHEAD_CONFIGS.values()]
    x_data_span = float(max(x_positions_all) - min(x_positions_all))
    width_max = 0.045 * x_data_span
    width_min = 0.007 * x_data_span
    all_bin_widths = np.linspace(width_min, width_max, len(all_bin_nums))

    bin_groups = [
        ([b for b in all_bin_nums if b <= 2], "bin0-2"),
        ([b for b in all_bin_nums if 3 <= b <= 6], "bin3-6"),
        ([b for b in all_bin_nums if b >= 7], "bin7-11"),
    ]

    for group_bins, group_label in bin_groups:
        if not group_bins:
            continue

        value_cols = [f"bin{b}_E_total" for b in group_bins]

        # Build annotation stats keyed by categorical x position
        annot_stats: dict = {}
        for config_key, (x_pos, tick_label) in SHOWERHEAD_CONFIGS.items():
            group_df = sh_df[sh_df["_sh_group"] == config_key]
            if group_df.empty:
                continue
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
                "w_label": tick_label,  # e.g. "W40\nP.Narrow", "W52\nP.Wide"
            }

        fig, ax = create_figure(figsize=BOXPLOT_CONFIG["figsize"])
        if isinstance(ax, list):
            ax = ax[0]

        for bin_num in group_bins:
            global_idx = all_bin_nums.index(bin_num)
            color = SENSOR_COLORS[global_idx % len(SENSOR_COLORS)]
            col = f"bin{bin_num}_E_total"
            if col not in sh_df.columns:
                continue

            box_width = float(all_bin_widths[global_idx])
            positions = []
            data = []

            for config_key, (x_pos, _) in SHOWERHEAD_CONFIGS.items():
                values = sh_df[sh_df["_sh_group"] == config_key][col].dropna().values
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

        # Inline annotation: use tick label from SHOWERHEAD_CONFIGS (e.g. "W40\nP.Narrow")
        if annot_stats:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + 0.25 * y_range)
            y_min, y_max = ax.get_ylim()
            offset = 0.02 * (y_max - y_min)

            for x_pos, stats in sorted(annot_stats.items()):
                n = stats["n"]
                max_val = stats.get("max_val", np.nan)
                if np.isnan(max_val):
                    continue
                text_lines = [stats.get("w_label", "")]
                text_lines.append(f"n={n}")
                if rh_data is not None and "shower_on" in stats:
                    avg_rh = _get_rh_at_shower_on(stats["shower_on"], rh_data)
                    if not np.isnan(avg_rh):
                        text_lines.append(f"RH={sf.fmt_fig(avg_rh, fallback='.0f')}%")
                ax.text(
                    x_pos,
                    max_val + offset,
                    "\n".join(text_lines),
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                    color="black",
                )

        # Categorical x-axis: show only positions that have data or are defined
        x_tick_positions = [cfg[0] for cfg in SHOWERHEAD_CONFIGS.values()]
        x_tick_labels = [cfg[1] for cfg in SHOWERHEAD_CONFIGS.values()]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=FONT_SIZE_TICK - 1)
        ax.set_xlim(-0.5, max(x_tick_positions) + 0.5)

        ax.set_xlabel("Configuration", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Total Emission E_total (#)", fontsize=FONT_SIZE_LABEL)
        bin_label = group_label.replace("-", "–").replace("bin", "Bin ")
        ax.set_title(
            f"Particle Emission by Shower Head Type — {bin_label}"
            "\n(Box = median/IQR, whiskers = 1.5×IQR)",
            fontsize=FONT_SIZE_TITLE,
            fontweight=TITLE_FONTWEIGHT,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

        legend_elements = [
            Patch(
                facecolor=SENSOR_COLORS[all_bin_nums.index(b) % len(SENSOR_COLORS)],
                alpha=0.7,
                label=f"Bin {b} ({particle_bins[b]['name']} µm)",
            )
            for b in group_bins
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND - 1,
            ncol=1,
        )

        sh_output = output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        plt.tight_layout()
        save_figure(fig, sh_output)
        plt.close(fig)


# =============================================================================
# CATEGORICAL COMPARISON BOXPLOTS — spray pattern, head type, mannequin,
# door position, fan status
# =============================================================================

# GroupDef: (group_key, tick_label, filter_fn)
#   group_key  — unique identifier used as dict key
#   tick_label — text shown on x-axis (may contain \n for wrapping)
#   filter_fn  — callable(config_key: str) -> bool; True means the row belongs here
_GroupDef = Tuple[str, str, Callable[[str], bool]]


def _draw_categorical_comparison_boxplot(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    output_path: Path,
    cfg: dict,
    group_defs: "List[_GroupDef]",
    title_base: str,
    x_label: str,
    rh_data: "Optional[pd.DataFrame]" = None,
    temp_filter: "Optional[Tuple[float, float]]" = None,
) -> None:
    """
    Shared implementation for all five condition-comparison boxplot families.

    Draws three output files (bin0-2, bin3-6, bin7-11) with a categorical
    x-axis whose groups are defined by *group_defs*.  Each group may pool
    multiple config_key values selected by an arbitrary filter function.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11``
            are appended to the stem.
        cfg: One of the ``_TEMP_BOXPLOT_CONFIG`` entries (supplies col_template,
            ylabel, title_note, hline).
        group_defs: Ordered list of (group_key, tick_label, filter_fn) tuples.
        title_base: Figure title prefix (e.g. "Spray Pattern Effect on Emission").
        x_label: x-axis label string.
        rh_data: Optional Bedroom_Conditions RH DataFrame for n=/RH= annotations.
        temp_filter: Optional (min_temp, max_temp) in °C; rows whose config_key
            encodes a temperature outside this range are excluded before grouping.
    """
    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return

    # Optional temperature pre-filter (±2 °C rule)
    if temp_filter is not None:
        t_lo, t_hi = temp_filter

        def _in_range(ck: str) -> bool:
            t = _extract_config_temp(ck)
            return t is not None and t_lo <= t <= t_hi

        work_df = results_df[results_df["config_key"].apply(_in_range)].copy()
    else:
        work_df = results_df.copy()

    if work_df.empty:
        return

    # Assign each row to the first matching group
    def _assign(ck: str) -> "Optional[str]":
        for gk, _, fn in group_defs:
            if fn(ck):
                return gk
        return None

    work_df["_cmp_group"] = work_df["config_key"].apply(_assign)
    work_df = work_df[work_df["_cmp_group"].notna()].copy()
    if work_df.empty:
        return

    x_pos = {gk: i for i, (gk, _, _) in enumerate(group_defs)}
    tick_label_map = {gk: lbl for gk, lbl, _ in group_defs}
    n_pos = len(group_defs)

    all_bin_nums = list(particle_bins.keys())
    # Fixed-width boxes scaled to unit categorical spacing
    all_bin_widths = np.linspace(0.10, 0.38, len(all_bin_nums))

    bin_groups = [
        ([b for b in all_bin_nums if b <= 2], "bin0-2"),
        ([b for b in all_bin_nums if 3 <= b <= 6], "bin3-6"),
        ([b for b in all_bin_nums if b >= 7], "bin7-11"),
    ]

    for group_bins, group_label in bin_groups:
        if not group_bins:
            continue

        value_cols = [cfg["col_template"].format(n=b) for b in group_bins]

        # Per-group annotation stats (keyed by categorical x position)
        annot_stats: dict = {}
        for gk, tick_lbl, _ in group_defs:
            gdf = work_df[work_df["_cmp_group"] == gk]
            if gdf.empty:
                continue
            all_vals: list = []
            for col in value_cols:
                if col in gdf.columns:
                    all_vals.extend(gdf[col].dropna().tolist())
            annot_stats[x_pos[gk]] = {
                "n": len(gdf),
                "max_val": float(np.max(all_vals)) if all_vals else np.nan,
                "shower_on": gdf["shower_on"].copy()
                if "shower_on" in gdf.columns
                else pd.Series([], dtype="object"),
                "tick_label": tick_lbl,
            }

        fig, ax = create_figure(figsize=BOXPLOT_CONFIG["figsize"])
        if isinstance(ax, list):
            ax = ax[0]

        for bin_num in group_bins:
            global_idx = all_bin_nums.index(bin_num)
            color = SENSOR_COLORS[global_idx % len(SENSOR_COLORS)]
            col = cfg["col_template"].format(n=bin_num)
            if col not in work_df.columns:
                continue

            box_width = float(all_bin_widths[global_idx])
            positions: list = []
            data: list = []
            for gk, _, _ in group_defs:
                gdf = work_df[work_df["_cmp_group"] == gk]
                vals = gdf[col].dropna().values if not gdf.empty else np.array([])
                if len(vals) > 0:
                    positions.append(x_pos[gk])
                    data.append(vals)

            if not data:
                continue

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_width,
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

        # Annotations above each group
        if annot_stats:
            y_min, y_max = ax.get_ylim()
            y_rng = y_max - y_min
            ax.set_ylim(y_min, y_max + 0.25 * y_rng)
            y_min, y_max = ax.get_ylim()
            offset = 0.02 * (y_max - y_min)
            for xp, stats in sorted(annot_stats.items()):
                max_val = stats.get("max_val", np.nan)
                if np.isnan(max_val):
                    continue
                lines = [stats["tick_label"], f"n={stats['n']}"]
                if rh_data is not None:
                    avg_rh = _get_rh_at_shower_on(stats["shower_on"], rh_data)
                    if not np.isnan(avg_rh):
                        lines.append(f"RH={sf.fmt_fig(avg_rh, fallback='.0f')}%")
                ax.text(
                    xp,
                    max_val + offset,
                    "\n".join(lines),
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                    color="black",
                )

        all_xp = [x_pos[gk] for gk, _, _ in group_defs]
        all_tl = [tick_label_map[gk] for gk, _, _ in group_defs]
        ax.set_xticks(all_xp)
        ax.set_xticklabels(all_tl, fontsize=FONT_SIZE_TICK - 1, ha="center")
        ax.set_xlim(-0.6, n_pos - 0.4)

        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(cfg["ylabel"], fontsize=FONT_SIZE_LABEL)
        bin_lbl = group_label.replace("-", "–").replace("bin", "Bin ")
        ax.set_title(
            f"{title_base} — {bin_lbl}\n{cfg['title_note']}",
            fontsize=FONT_SIZE_TITLE,
            fontweight=TITLE_FONTWEIGHT,
        )
        if cfg.get("hline") is not None:
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
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=FONT_SIZE_LEGEND - 1,
            ncol=1,
        )

        out = output_path.parent / f"{output_path.stem}_{group_label}{output_path.suffix}"
        plt.tight_layout()
        save_figure(fig, out)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Metric keys used by all five comparison families:
#   emission_etotal  → bin{n}_E_total
#   penetration_factor → bin{n}_p_mean
#   deposition_rate  → bin{n}_beta_other_raw_mean
# ---------------------------------------------------------------------------
_COMPARISON_METRICS: "List[Tuple[str, str]]" = [
    ("emission_etotal", "emission_etotal"),
    ("penetration_factor", "penetration_factor"),
    ("deposition_rate", "other_process_rate"),
]


def _run_comparison_family(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    stem_prefix: str,
    group_defs: "List[_GroupDef]",
    title_base: str,
    x_label: str,
    rh_data: "Optional[pd.DataFrame]" = None,
    temp_filter: "Optional[Tuple[float, float]]" = None,
) -> None:
    """Loop over the three comparison metrics and call the shared drawing function."""
    for cfg_key, file_metric in _COMPARISON_METRICS:
        cfg = _TEMP_BOXPLOT_CONFIG[cfg_key]
        base_path = plot_dir / f"{stem_prefix}_{file_metric}_boxplot.png"
        _draw_categorical_comparison_boxplot(
            results_df,
            particle_bins,
            base_path,
            cfg,
            group_defs,
            title_base=f"{title_base} — {cfg['title_metric']}",
            x_label=x_label,
            rh_data=rh_data,
            temp_filter=temp_filter,
        )


# ---------------------------------------------------------------------------
# 1. Spray Pattern comparison  (W36–42 °C, Pepco head, no mannequin, open, fan off)
# ---------------------------------------------------------------------------

def plot_spray_pattern_comparison_boxplots(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Boxplots comparing Wide / Narrow / Mid Pepco spray patterns.

    Fixed conditions: Pepco head, W36–42 °C, no mannequin, door open, fan off.
    Produces nine figures: 3 metrics × 3 bin groups.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        plot_dir: Directory for output files.
        rh_data: Optional RH DataFrame for annotations.
    """
    group_defs: "List[_GroupDef]" = [
        (
            "Wide",
            "Pepco\nWide",
            lambda ck: "_Pepco_Wide_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
        (
            "Narrow",
            "Pepco\nNarrow",
            lambda ck: "_Pepco_Narrow_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
        (
            "Mid",
            "Pepco\nMid",
            lambda ck: "_Pepco_Mid_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
    ]
    _run_comparison_family(
        results_df,
        particle_bins,
        plot_dir,
        stem_prefix="spray_pattern",
        group_defs=group_defs,
        title_base="Spray Pattern Effect",
        x_label="Spray Pattern",
        rh_data=rh_data,
        temp_filter=(36.0, 42.0),
    )


# ---------------------------------------------------------------------------
# 2. Shower Head comparison  (W36–42 °C, door open, fan off)
# ---------------------------------------------------------------------------

def plot_shower_head_comparison_boxplots(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Boxplots comparing Standard, Pepco (Wide/Narrow/Mid), FilterWand, and Used heads.

    Fixed conditions: W36–42 °C, door open, fan off.  Standard-head events
    at W37 satisfy the ±2 °C rule relative to W38/W40.
    Produces nine figures: 3 metrics × 3 bin groups.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        plot_dir: Directory for output files.
        rh_data: Optional RH DataFrame for annotations.
    """
    group_defs: "List[_GroupDef]" = [
        (
            "Standard",
            "Standard",
            lambda ck: (
                "_Pepco" not in ck
                and "_FilterWand" not in ck
                and "_Used" not in ck
                and "_Mannequin" not in ck
                and "_DoorOpen_FanOff" in ck
            ),
        ),
        (
            "Pepco_Wide",
            "Pepco\nWide",
            lambda ck: "_Pepco_Wide_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
        (
            "Pepco_Narrow",
            "Pepco\nNarrow",
            lambda ck: "_Pepco_Narrow_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
        (
            "Pepco_Mid",
            "Pepco\nMid",
            lambda ck: "_Pepco_Mid_DoorOpen_FanOff" in ck and "_Mannequin" not in ck,
        ),
        (
            "FilterWand",
            "Filter\nWand",
            lambda ck: "_FilterWand_" in ck and "_DoorOpen_FanOff" in ck,
        ),
        (
            "Used_rainfall",
            "Used\nRainfall",
            lambda ck: "_Used_rainfall_DoorOpen_FanOff" in ck,
        ),
        (
            "Used_12Nozzle",
            "Used\n12Nozzle",
            lambda ck: "_Used_12Nozzle_DoorOpen_FanOff" in ck,
        ),
        (
            "Used_SingleWide",
            "Used\nSingleWide",
            lambda ck: "_Used_SingleWide_DoorOpen_FanOff" in ck,
        ),
    ]
    _run_comparison_family(
        results_df,
        particle_bins,
        plot_dir,
        stem_prefix="head_type",
        group_defs=group_defs,
        title_base="Shower Head Type Effect",
        x_label="Shower Head Configuration",
        rh_data=rh_data,
        temp_filter=(36.0, 42.0),
    )


# ---------------------------------------------------------------------------
# 3. Mannequin comparison  (W36–42 °C, Pepco head, door open, fan off)
# ---------------------------------------------------------------------------

def plot_mannequin_comparison_boxplots(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Boxplots comparing with-mannequin vs. without-mannequin conditions.

    Fixed conditions: Pepco head, W36–42 °C, door open, fan off.
    Produces nine figures: 3 metrics × 3 bin groups.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        plot_dir: Directory for output files.
        rh_data: Optional RH DataFrame for annotations.
    """
    group_defs: "List[_GroupDef]" = [
        (
            "No_Mannequin",
            "No\nMannequin",
            lambda ck: (
                "_Pepco_" in ck
                and "_Mannequin" not in ck
                and "_DoorOpen_FanOff" in ck
            ),
        ),
        (
            "Mannequin",
            "With\nMannequin",
            lambda ck: "_Pepco_" in ck and "_Mannequin_DoorOpen_FanOff" in ck,
        ),
    ]
    _run_comparison_family(
        results_df,
        particle_bins,
        plot_dir,
        stem_prefix="mannequin",
        group_defs=group_defs,
        title_base="Mannequin Presence Effect",
        x_label="Mannequin",
        rh_data=rh_data,
        temp_filter=(36.0, 42.0),
    )


# ---------------------------------------------------------------------------
# 4. Door Position comparison  (W36–42 °C, Pepco Wide, fan on)
# ---------------------------------------------------------------------------

def plot_door_comparison_boxplots(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Boxplots comparing door-open vs. door-closed conditions.

    Fixed conditions: Pepco Wide, W36–42 °C, fan on (the door-closed period
    coincided with fan-on operation, so fan status is held constant here).
    Produces nine figures: 3 metrics × 3 bin groups.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        plot_dir: Directory for output files.
        rh_data: Optional RH DataFrame for annotations.
    """
    group_defs: "List[_GroupDef]" = [
        (
            "DoorOpen_FanOn",
            "Door Open\n(Fan On)",
            lambda ck: "_Pepco_Wide_DoorOpen_FanOn" in ck,
        ),
        (
            "DoorClosed_FanOn",
            "Door Closed\n(Fan On)",
            lambda ck: "_Pepco_Wide_DoorClosed_FanOn" in ck,
        ),
    ]
    _run_comparison_family(
        results_df,
        particle_bins,
        plot_dir,
        stem_prefix="door_position",
        group_defs=group_defs,
        title_base="Door Position Effect",
        x_label="Door Position",
        rh_data=rh_data,
        temp_filter=(36.0, 42.0),
    )


# ---------------------------------------------------------------------------
# 5. Fan Status comparison  (W36–42 °C, Pepco Wide, door open, no mannequin)
# ---------------------------------------------------------------------------

def plot_fan_comparison_boxplots(
    results_df: pd.DataFrame,
    particle_bins: Dict,
    plot_dir: Path,
    rh_data: "Optional[pd.DataFrame]" = None,
) -> None:
    """Boxplots comparing fan-off vs. fan-on conditions.

    Fixed conditions: Pepco Wide, W36–42 °C, door open, no mannequin.
    Produces nine figures: 3 metrics × 3 bin groups.

    Parameters:
        results_df: Full particle analysis results DataFrame.
        particle_bins: Dict of bin metadata.
        plot_dir: Directory for output files.
        rh_data: Optional RH DataFrame for annotations.
    """
    group_defs: "List[_GroupDef]" = [
        (
            "FanOff",
            "Fan Off",
            lambda ck: (
                "_Pepco_Wide_DoorOpen_FanOff" in ck and "_Mannequin" not in ck
            ),
        ),
        (
            "FanOn",
            "Fan On",
            lambda ck: "_Pepco_Wide_DoorOpen_FanOn" in ck,
        ),
    ]
    _run_comparison_family(
        results_df,
        particle_bins,
        plot_dir,
        stem_prefix="fan_status",
        group_defs=group_defs,
        title_base="Bath Fan Effect",
        x_label="Bath Fan Status",
        rh_data=rh_data,
        temp_filter=(36.0, 42.0),
    )
