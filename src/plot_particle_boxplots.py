#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis — Temperature-Axis Boxplot Functions
=======================================================

Private helpers and four fixed-temperature-axis boxplot functions shared across
all water-temperature comparison figures, plus the shower-head categorical boxplot.
Called by plot_particle.py (re-exported) and plot_comparison.py.

Functions:
    - plot_emission_boxplot: E_total box-and-whisker by water temperature
    - plot_deposition_rate_boxplot: beta_raw_mean box-and-whisker by water temperature
    - plot_emission_rate_boxplot: E_mean box-and-whisker by water temperature
    - plot_penetration_factor_boxplot: p_mean box-and-whisker by water temperature
    - plot_emission_etotal_by_metric_boxplot: E_total vs. continuous metric axis
    - plot_emission_etotal_by_showerhead_boxplot: E_total grouped by shower head type

Private helpers (used by plot_comparison.py):
    - _is_base_config: identifies baseline temperature-sweep events
    - _extract_config_temp: extracts numeric temperature from W## config key
    - _get_rh_at_shower_on: averages RH near shower-on timestamps
    - _annotate_temp_groups: draws W##/n=/RH= annotations above each group
    - _build_temp_stats: builds per-temperature annotation statistics
    - _TEMP_BOXPLOT_CONFIG: metric configuration dict for boxplot functions

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

import src.sig_figs as sf
from src.event_manager import sort_config_keys_by_water_temp
from src.plot_style import (
    BOXPLOT_CONFIG,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    SENSOR_COLORS,
    TITLE_FONTWEIGHT,
    apply_style,
    create_figure,
    save_figure,
)


# =============================================================================
# PRIVATE HELPERS — shared by all temperature-axis boxplot functions
# =============================================================================


def _is_base_config(key: str) -> bool:
    """Return True if config_key is the baseline temperature-sweep config.

    Requires: Standard head (no Pepco/FilterWand/Used), no Mannequin,
    BathDoor Open, BdrmDoor Closed, Fan Off.  Only these events are plotted
    on the fixed water-temperature axis so that temperature is the sole variable.
    """
    key_str = str(key)
    return (
        bool(re.match(r"^W\d+(_|$)", key_str))
        and "_Pepco" not in key_str
        and "_FilterWand" not in key_str
        and "_Used" not in key_str
        and "_Mannequin" not in key_str
        and "_BathDoorClosed" not in key_str
        and "_BathDoorPartial" not in key_str
        and "_BdrmDoorOpen" not in key_str
        and "_BdrmDoorAjar" not in key_str
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

    # Extend y-axis headroom so annotations aren't clipped
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
        bin_counts = stats.get("bin_counts", {})
        if bin_counts:
            for col in sorted(bin_counts, key=lambda c: int(c.split("_")[0][3:])):
                bin_label = col.split("_")[0].capitalize()
                text_lines.append(f"{bin_label} n={bin_counts[col]}")
        else:
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
        bin_counts = {
            col: int(group_df[col].notna().sum()) if col in group_df.columns else 0
            for col in value_cols
        }

        if temp in temp_stats:
            temp_stats[temp]["n"] += n
            for col, cnt in bin_counts.items():
                temp_stats[temp]["bin_counts"][col] = (
                    temp_stats[temp]["bin_counts"].get(col, 0) + cnt
                )
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
                "bin_counts": bin_counts,
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


def _write_boxplot_companion_md(
    output_path: Path,
    title: str,
    groups: list,
) -> None:
    """Write a Markdown file listing the events that make up each box.

    Parameters:
        output_path: Base figure path; the .md file is written alongside it
            with the same stem and no bin-group suffix.
        title: Figure title used as the document heading.
        groups: Ordered list of dicts, each with:
            - 'header': str — the label shown above the box (e.g. "W11 | 11°C | n=3")
            - 'events': list of (event_number, test_name, config_key) tuples
    """
    from datetime import date as _date

    md_path = output_path.parent / f"{output_path.stem}.md"
    lines = [
        f"# {title} — Event Membership",
        "",
        f"Generated: {_date.today().isoformat()}",
        "",
        "Each entry below corresponds to one box in the figure.",
        "Events are sorted by event number within each group.",
        "",
        "---",
        "",
    ]
    for group in groups:
        lines.append(f"## {group['header']}")
        lines.append("")
        events = group["events"]
        if events:
            lines.append("| Event | Test Name | Config Key |")
            lines.append("|------:|-----------|------------|")
            for ev_num, test_name, config_key in events:
                try:
                    ev_str = str(int(ev_num)) if not pd.isna(ev_num) else "—"
                except (TypeError, ValueError):
                    ev_str = "—"
                lines.append(f"| {ev_str} | {test_name} | {config_key} |")
        else:
            lines.append("*(no events)*")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


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

    Produces three output files (bin0-2, bin3-6, bin7-11) by appending the
    bin-range suffix to *output_path*.  Only base W## events are included.
    X-axis is the fixed temperature range from BOXPLOT_CONFIG (5–60 °C) unless
    *x_range* is provided.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; suffix ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11``
            is appended.
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

    # Build companion .md once (event membership is identical across bin groups)
    md_groups = []
    for ck in config_keys:
        temp = temp_map.get(ck)
        group_df = base_df[base_df["config_key"] == ck]
        n = len(group_df)
        rh_str = ""
        if rh_data is not None and "shower_on" in group_df.columns:
            avg_rh = _get_rh_at_shower_on(group_df["shower_on"], rh_data)
            if not np.isnan(avg_rh):
                rh_str = f" | RH={avg_rh:.0f}%"
        w_label = ck.split("_")[0]
        temp_label = f"{temp:.0f}°C" if temp is not None else ""
        header = f"{w_label} | {temp_label} | n={n}{rh_str}"
        events_list = []
        sort_col = "event_number" if "event_number" in group_df.columns else None
        iter_df = group_df.sort_values(sort_col) if sort_col else group_df
        for _, row in iter_df.iterrows():
            events_list.append((
                row.get("event_number", ""),
                row.get("test_name", ""),
                ck,
            ))
        md_groups.append({"header": header, "events": events_list})
    _write_boxplot_companion_md(output_path, cfg["title_metric"], md_groups)

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
    """Three box-and-whisker figures of total particle emission (E_total) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_total).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11`` suffixes are appended.
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
    """Three box-and-whisker figures of unclamped other process rate (beta_raw_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_beta_raw_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11`` suffixes are appended.
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
    """Three box-and-whisker figures of mean emission rate (E_mean, #/min) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_E_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11`` suffixes are appended.
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
    """Three box-and-whisker figures of penetration factor (p_mean) by water temperature.

    Parameters:
        results_df: DataFrame with analysis results (must contain config_key and bin{n}_p_mean).
        particle_bins: Dictionary of particle bin information.
        output_path: Base path; ``_bin0-2`` / ``_bin3-6`` / ``_bin7-11`` suffixes are appended.
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
    Create three box-and-whisker figures of E_total positioned along a continuous metric axis.

    Produces one figure for each bin group (Bin 0–2, 3–6, 7–11).
    Unlike :func:`plot_emission_boxplot`, the x-axis is not fixed to the
    5–60 °C water-temperature range; instead each water-temperature group (W##) is
    centred at the *group mean* of *metric_col*, and the x-axis auto-scales from data
    unless *x_range* is provided.
    Box widths also scale proportionally to the data range.

    Only base W## events are included (letter-suffix repeats excluded).

    Parameters:
        results_df: DataFrame with analysis results; must contain 'config_key',
                    bin{n}_E_total columns, and *metric_col*.
        particle_bins: Dictionary of particle bin information.
        output_path: Base path used to derive the three output filenames (suffix
                     ``_bin0-2``, ``_bin3-6``, ``_bin7-11`` are appended to the stem).
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

    # Build companion .md once (event membership is identical across bin groups)
    md_groups = []
    for ck in config_keys:
        group_df = base_df[base_df["config_key"] == ck]
        n = len(group_df)
        rh_str = ""
        if rh_data is not None and "shower_on" in group_df.columns:
            avg_rh = _get_rh_at_shower_on(group_df["shower_on"], rh_data)
            if not np.isnan(avg_rh):
                rh_str = f" | RH={avg_rh:.0f}%"
        metric_vals = group_df[metric_col].dropna().values
        metric_str = f" | {metric_label}={np.mean(metric_vals):.1f}" if len(metric_vals) > 0 else ""
        w_label = ck.split("_")[0]
        header = f"{w_label} | n={n}{metric_str}{rh_str}"
        events_list = []
        sort_col = "event_number" if "event_number" in group_df.columns else None
        iter_df = group_df.sort_values(sort_col) if sort_col else group_df
        for _, row in iter_df.iterrows():
            events_list.append((
                row.get("event_number", ""),
                row.get("test_name", ""),
                ck,
            ))
        md_groups.append({"header": header, "events": events_list})
    fig_title = f"E_total by {metric_label}"
    _write_boxplot_companion_md(output_path, fig_title, md_groups)

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
    Create three box-and-whisker figures of E_total grouped by shower head type.

    Produces one figure for each bin group (Bin 0–2, 3–6, 7–11).
    Configs are ordered by temperature and grouped into four clusters separated
    by visible gaps on the categorical x-axis:
      - Cluster 1: W37, W38_Pepco_Narrow, W38_Pepco_Wide, W40_Pepco_Narrow,
                   W40_Pepco_Wide, W43
      - Cluster 2: W38_FilterWand, W38_Used_rainfall, W38_Used_12Nozzle,
                   W38_Used_SingleWide
      - Cluster 3: W48, W49_Pepco_Wide
      - Cluster 4: W52_Pepco_Wide, W53
    Keys match compound config_key prefixes (e.g. "W40_Pepco_Narrow" matches
    "W40_Pepco_Narrow_BathDoorOpen_BdrmDoorClosed_FanOff").  Annotations follow
    the standard 'W##\\nn=#\\nRH=##%' style.

    Parameters:
        results_df: DataFrame with analysis results; must contain 'config_key'
                    and bin{n}_E_total columns.
        particle_bins: Dictionary of particle bin information.
        output_path: Base path used to derive the three output filenames (suffix
                     ``_bin0-2``, ``_bin3-6``, ``_bin7-11`` are appended to the stem).
        rh_data: Optional DataFrame with 'datetime' and 'RH_bedroom' for the
                 n=/RH= annotation (same as other boxplot functions).
    """
    apply_style()

    if results_df.empty or "config_key" not in results_df.columns:
        return

    # Ordered mapping: compound config_key prefix → (categorical x position, tick label)
    # Keys are the leading portion of the full config_key (everything before _BathDoor…).
    # Standard-head events match on the W## token alone; Pepco/FilterWand/Used events
    # match on the full head+pattern prefix.  Positions leave a gap of 1 unit between
    # clusters.
    SHOWERHEAD_CONFIGS = {
        # Cluster 1: standard W37, Pepco near 38–40°C, standard W43
        "W37": (0, "W37"),
        "W38_Pepco_Narrow": (1, "W38\nP.Narrow"),
        "W38_Pepco_Wide": (2, "W38\nP.Wide"),
        "W40_Pepco_Narrow": (3, "W40\nP.Narrow"),
        "W40_Pepco_Wide": (4, "W40\nP.Wide"),
        "W43": (5, "W43"),
        # gap at 6
        # Cluster 2: FilterWand and Used heads near 38°C
        "W38_FilterWand": (7, "W38\nFilter\nWand"),
        "W38_Used_rainfall": (8, "W38\nUsed\nRainfall"),
        "W38_Used_12Nozzle": (9, "W38\nUsed\n12Nozzle"),
        "W38_Used_SingleWide": (10, "W38\nUsed\nSingleWide"),
        # gap at 11
        # Cluster 3: standard W48 + Pepco wide near 49°C
        "W48": (12, "W48"),
        "W49_Pepco_Wide": (13, "W49\nP.Wide"),
        # gap at 14
        # Cluster 4: Pepco wide near 52°C + standard W53 (low→high temp)
        "W52_Pepco_Wide": (15, "W52\nP.Wide"),
        "W53": (16, "W53"),
    }

    # Match each row's config_key to a SHOWERHEAD_CONFIGS group.
    # NaN or non-string config_key values are skipped.
    def _sh_group(ck: object) -> "Optional[str]":
        if not isinstance(ck, str) or not ck:
            return None
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

    # Build companion .md once (event membership is identical across bin groups)
    md_groups = []
    for sh_key, (_, tick_label) in SHOWERHEAD_CONFIGS.items():
        group_df = sh_df[sh_df["_sh_group"] == sh_key]
        if group_df.empty:
            continue
        n = len(group_df)
        rh_str = ""
        if rh_data is not None and "shower_on" in group_df.columns:
            avg_rh = _get_rh_at_shower_on(group_df["shower_on"], rh_data)
            if not np.isnan(avg_rh):
                rh_str = f" | RH={avg_rh:.0f}%"
        label = tick_label.replace("\n", " ")
        header = f"{label} | n={n}{rh_str}"
        events_list = []
        sort_col = "event_number" if "event_number" in group_df.columns else None
        iter_df = group_df.sort_values(sort_col) if sort_col else group_df
        for _, row in iter_df.iterrows():
            events_list.append((
                row.get("event_number", ""),
                row.get("test_name", ""),
                row.get("config_key", sh_key),
            ))
        md_groups.append({"header": header, "events": events_list})
    _write_boxplot_companion_md(output_path, "E_total by Shower Head Type", md_groups)

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
            bin_counts = {
                col: int(group_df[col].notna().sum()) if col in group_df.columns else 0
                for col in value_cols
            }
            annot_stats[x_pos] = {
                "n": n,
                "bin_counts": bin_counts,
                "max_val": max_val,
                "shower_on": group_df["shower_on"].copy()
                if "shower_on" in group_df.columns
                else pd.Series([], dtype="object"),
                "config_key": config_key,
                "w_label": tick_label,  # e.g. "W40\nP.Narrow", "W52\nP.Wide"
            }

        n_sh_positions = len(SHOWERHEAD_CONFIGS)
        sh_fig_width = max(BOXPLOT_CONFIG["figsize"][0], 1.4 * n_sh_positions)
        fig, ax = create_figure(figsize=(sh_fig_width, BOXPLOT_CONFIG["figsize"][1]))
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
                bin_counts = stats.get("bin_counts", {})
                if bin_counts:
                    for col in sorted(bin_counts, key=lambda c: int(c.split("_")[0][3:])):
                        bin_label = col.split("_")[0].capitalize()
                        text_lines.append(f"{bin_label} n={bin_counts[col]}")
                else:
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

        legend_elements: List[Patch] = [
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
