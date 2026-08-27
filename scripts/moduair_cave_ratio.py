#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Inside Sensor vs Position-Weighted Fleet Average
============================================================

Compares the inside bedroom sensor (MOD-PM-00195) against a position-weighted
average of the IAQ&V MODULAIR-PM fleet (C_room), per particle-size bin, over the
fleet co-location window (2026-06-03 to 2026-07-16).

The fleet is grouped by vertical position (labels from src.plot_style
MODUAIR_SENSOR_LABELS):

    C_high = mean(402, 816)
    C_mid  = mean(814, 554, 942, 401, 815)     # excludes 555 "Middle Bathroom"
    C_bed  = mean(195, 813)
    C_low  = mean(515, 465, 943, 516, 467)
    C_room = 0.32*C_high + 0.28*C_mid + 0.20*C_bed + 0.20*C_low
    C_bed1 = sensor 00195

All concentrations are the raw opc_bin{N} number concentration on a shared
1-minute grid (no smoothing), one value per size bin.

">8 Quants" filter
------------------
"Quants" are reporting MODULAIR-PM sensors. Data are used only where more than
eight fleet sensors report a valid (finite, positive) reading:
    - Deliverable 1 applies the filter per 1-minute timestamp.
    - Deliverable 2 applies the filter per 1-minute timestamp before averaging
      each event's aligned curve.

Deliverables
------------
1. C_bed1 vs C_room scatter, one interactive Bokeh HTML per bin, over all
   >8-sensor 1-minute samples in the window. Overlays a Deming (orthogonal)
   fit (reused from moduair_correction_factor.fit_deming) and a 1:1 line. A
   per-bin fit table (slope, intercept, r^2, n) is written to CSV.

2. C_bed1,average / C_room,average versus time, one interactive Bokeh HTML per
   bin, with three group curves. Events are grouped by registry water_temp:
   W38 (the W38-W41 runs), W49 (two runs), and W24 (one set). Each event is
   aligned on minute index (0 = shower on) from 15 min before shower-on through
   the 2-hour deposition window. C_bed1 and C_room are each averaged across the
   group's events at every minute, and the ratio of those two averaged curves
   is plotted.

Input
-----
    - MODULAIR-PM fleet chunks (src.moduair_loader.load_fleet_bins)
    - Event registry (scripts.event_registry.load_event_registry) for the
      shower-on times and water_temp codes of numbered events in the window.

Output Files
------------
    <output>/plots/moduair_room/c_bed1_vs_c_room_bin{N}.html          (12 figures)
    <output>/plots/moduair_room/c_bed1_c_room_ratio_time_bin{N}.html  (12 figures)
    <output>/moduair_room_ratio_fit.csv                           (per-bin fit)

Usage
-----
    python scripts/moduair_cave_ratio.py
    python scripts/moduair_cave_ratio.py --start "2026-06-03 00:00:00"

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-08-17
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

# Ensure stdout/stderr use UTF-8 on Windows (log files default to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Slope, Span
from bokeh.plotting import figure, output_file, save

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_registry import load_event_registry  # noqa: E402
from scripts.moduair_correction_factor import fit_deming  # noqa: E402
from src.data_paths import get_data_root  # noqa: E402
from src.moduair_loader import (  # noqa: E402
    BIN_COLUMNS,
    N_BINS,
    list_available_sensors,
    load_fleet_bins,
)
from src.particle_calculations import PARTICLE_BINS  # noqa: E402
from src.plot_style import COLORS  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

# Co-location window (inclusive), matching the fleet export scripts.
DEFAULT_START = "2026-06-03 00:00:00"
DEFAULT_END = "2026-07-16 23:59:59"

# Position groups keyed by 5-digit sensor ID. The vertical-position membership
# follows MODUAIR_SENSOR_LABELS in src.plot_style; 00555 ("Middle Bathroom") is
# deliberately left out of C_mid because it is not a bedroom mid-height sensor.
POSITION_GROUPS = {
    "high": ["00402", "00816"],
    "mid":  ["00814", "00554", "00942", "00401", "00815"],
    "bed":  ["00195", "00813"],
    "low":  ["00515", "00465", "00943", "00516", "00467"],
}

# Position weights for the fleet average C_room.
POSITION_WEIGHTS = {"high": 0.32, "mid": 0.28, "bed": 0.20, "low": 0.20}

# Inside sensor of interest.
TARGET_SN = "00195"

# All fleet sensors that count toward the ">8 sensors reporting" filter: every
# sensor in any position group.
FLEET_SNS = sorted({sn for group in POSITION_GROUPS.values() for sn in group})

# Minimum reporting sensors (strictly greater than 8).
MIN_QUANTS = 8

# Event-window alignment: 15 min before shower-on through the 2-hour deposition.
PRE_SHOWER_LEAD = pd.Timedelta(minutes=15)
DEPOSITION_HOURS = 2.0

# Water-temperature groups for the ratio-vs-time figure. Keys are display
# labels; values are the registry water_temp codes each group collects.
WATER_TEMP_GROUPS = {
    "W38-W41": ["W38", "W39", "W40", "W41"],
    "W24": ["W24"],
    "W49": ["W49"],
}

# One color per group (single source of truth: plot_style COLORS).
GROUP_COLORS = {
    "W38-W41": COLORS["lambda"],
    "W24": COLORS["bedroom"],
    "W49": COLORS["outside"],
}


# =============================================================================
# Concentration Assembly
# =============================================================================


def build_position_totals(fleet: dict) -> dict:
    """
    Build per-sensor bin concentration frames aligned on a common time index.

    Parameters
    ----------
    fleet : dict
        Mapping of 5-digit sensor ID -> DataFrame(datetime + opc_bin0..11) at
        1-minute cadence, as returned by load_fleet_bins.

    Returns
    -------
    dict
        Mapping of sensor ID -> DataFrame indexed by datetime with the 12
        opc_bin columns. Sensors absent from ``fleet`` are omitted.
    """
    totals = {}
    for sn in FLEET_SNS + [TARGET_SN]:
        df = fleet.get(sn)
        if df is None or df.empty:
            continue
        totals[sn] = df.set_index("datetime")[BIN_COLUMNS].sort_index()
    return totals


def compute_room_frame(totals: dict, bin_col: str) -> pd.DataFrame:
    """
    Assemble C_bed1, the four position means, C_room, and the sensor count for one bin.

    Position means treat a sensor as present only where its reading is finite
    and strictly positive; a zero (a common dead-sensor sentinel) does not count
    toward the sensor tally and is excluded from the position mean.

    Parameters
    ----------
    totals : dict
        Mapping of sensor ID -> per-bin DataFrame from build_position_totals.
    bin_col : str
        Bin column name (e.g. "opc_bin2").

    Returns
    -------
    pd.DataFrame
        Indexed by datetime with columns: C_bed1, C_high, C_mid, C_bed, C_low,
        C_room, n_sensors. Position means and C_room are NaN where no member
        sensor reported; n_sensors counts all fleet sensors reporting there.
    """
    # Per-sensor valid-reading series (finite and > 0) for this bin.
    valid = {}
    for sn, df in totals.items():
        s = df[bin_col]
        valid[sn] = s.where(np.isfinite(s) & (s > 0))

    if TARGET_SN not in valid:
        return pd.DataFrame()

    # Common 1-minute index spanning all sensors.
    index = None
    for s in valid.values():
        index = s.index if index is None else index.union(s.index)

    aligned = {sn: s.reindex(index) for sn, s in valid.items()}

    # Position means over available member sensors.
    out = pd.DataFrame(index=index)
    for group, members in POSITION_GROUPS.items():
        cols = [aligned[sn] for sn in members if sn in aligned]
        if cols:
            out[f"C_{group}"] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
        else:
            out[f"C_{group}"] = np.nan

    out["C_room"] = sum(
        POSITION_WEIGHTS[g] * out[f"C_{g}"] for g in POSITION_GROUPS
    )
    out["C_bed1"] = aligned[TARGET_SN]

    # ">8 Quants": number of fleet sensors reporting a valid reading here.
    fleet_cols = [aligned[sn] for sn in FLEET_SNS if sn in aligned]
    out["n_sensors"] = pd.concat(fleet_cols, axis=1).notna().sum(axis=1)

    return out


# =============================================================================
# Deliverable 1: C_bed1 vs C_room scatter
# =============================================================================


def plot_scatter(cave: pd.DataFrame, bin_index: int, plot_dir: Path) -> dict:
    """
    Plot C_bed1 (y) vs C_room (x) for one bin over all >8-sensor 1-minute samples.

    Parameters
    ----------
    cave : pd.DataFrame
        Output of compute_room_frame for this bin.
    bin_index : int
        Particle-size bin index (0-11).
    plot_dir : Path
        Directory to write the HTML figure into.

    Returns
    -------
    dict
        Deming fit result (fit_deming output) plus a "bin" key, or a NaN-filled
        record when there are too few paired points.
    """
    bin_name = PARTICLE_BINS[bin_index]["name"]

    paired = (
        cave.loc[cave["n_sensors"] > MIN_QUANTS, ["C_room", "C_bed1"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    fit = {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "n": 0}
    if paired.empty:
        print(f"    [WARN] No >8-sensor paired data for bin {bin_index}; skipping scatter")
        fit["bin"] = bin_index
        return fit

    fit = fit_deming(paired["C_room"], paired["C_bed1"])
    fit["bin"] = bin_index

    out_path = plot_dir / f"c_bed1_vs_c_room_bin{bin_index}.html"
    output_file(str(out_path), title=f"C_bed1 vs C_room bin {bin_index}")

    title = (
        f"C_bed1 vs C_room, bin {bin_index} ({bin_name} µm), >8 sensors | "
        f"C_bed1 = {fit['slope']:.3f}·C_room + {fit['intercept']:.3f}, "
        f"r² = {fit['r_squared']:.3f}, n = {int(fit['n'])}"
    )
    fig = figure(
        width=650,
        height=600,
        title=title,
        x_axis_label="C_room, position-weighted fleet average (#/cm³)",
        y_axis_label="C_bed1, inside sensor (#/cm³)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    source = ColumnDataSource(
        data={"x": paired["C_room"], "y": paired["C_bed1"], "t": paired.index}
    )
    pts = fig.scatter(
        "x", "y", source=source, size=3, alpha=0.35,
        color=COLORS["bedroom"], legend_label="1-min samples (>8 sensors)",
    )
    fig.add_tools(
        HoverTool(
            renderers=[pts],
            tooltips=[
                ("C_room (x)", "@x{0.000}"),
                ("C_bed1 (y)", "@y{0.000}"),
                ("Time", "@t{%F %H:%M}"),
            ],
            formatters={"@t": "datetime"},
        )
    )

    lo = float(min(paired["C_room"].min(), paired["C_bed1"].min()))
    hi = float(max(paired["C_room"].max(), paired["C_bed1"].max()))
    fig.line([lo, hi], [lo, hi], line_color=COLORS["grid"],
             line_dash="dashed", line_width=1.0, legend_label="1:1")

    if pd.notna(fit["slope"]):
        fig.add_layout(
            Slope(gradient=float(fit["slope"]), y_intercept=float(fit["intercept"]),
                  line_color=COLORS["lambda"], line_width=2.0)
        )
        # Legend proxy for the fit line (Slope has no legend entry).
        fig.line([lo, hi],
                 [fit["slope"] * lo + fit["intercept"],
                  fit["slope"] * hi + fit["intercept"]],
                 line_color=COLORS["lambda"], line_width=2.0,
                 legend_label="Deming fit")

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    save(fig)
    print(f"    Saved {out_path.name}")
    return fit


# =============================================================================
# Deliverable 2: C_bed1,avg / C_room,avg versus time by water-temperature group
# =============================================================================


def load_group_events(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """
    Group numbered registry events in the window by water-temperature label.

    Parameters
    ----------
    start, end : pd.Timestamp
        Inclusive shower-on window.

    Returns
    -------
    dict
        Mapping of group label (WATER_TEMP_GROUPS key) -> list of event dicts
        {shower_on, deposition_end}. Empty groups are omitted.
    """
    registry = load_event_registry()
    mask = (
        registry["event_number"].notna()
        & registry["shower_on"].notna()
        & (registry["shower_on"] >= start)
        & (registry["shower_on"] <= end)
    )
    valid = registry[mask].copy()

    # Fill missing deposition_end from shower_off (same fallback as fleet export).
    if "deposition_end" not in valid.columns:
        valid["deposition_end"] = pd.NaT
    missing = valid["deposition_end"].isna()
    valid.loc[missing, "deposition_end"] = (
        valid.loc[missing, "shower_off"] + timedelta(hours=DEPOSITION_HOURS)
    )

    code_to_group = {
        code: label for label, codes in WATER_TEMP_GROUPS.items() for code in codes
    }

    groups = {label: [] for label in WATER_TEMP_GROUPS}
    for _, row in valid.iterrows():
        label = code_to_group.get(row.get("water_temp"))
        if label is None:
            continue
        groups[label].append(
            {"shower_on": row["shower_on"], "deposition_end": row["deposition_end"]}
        )

    return {label: evs for label, evs in groups.items() if evs}


def align_event_series(cave: pd.DataFrame, event: dict, column: str) -> pd.Series:
    """
    Slice one column of the bin frame to an event and index it by minute offset.

    The >8-sensor filter is applied before slicing so only well-sampled minutes
    contribute. Minute 0 is shower-on; the pre-shower lead-in is negative.

    Parameters
    ----------
    cave : pd.DataFrame
        Output of compute_room_frame for one bin.
    event : dict
        Event dict with shower_on and deposition_end.
    column : str
        Column to extract ("C_bed1" or "C_room").

    Returns
    -------
    pd.Series
        Indexed by integer minute offset from shower-on.
    """
    shower_on = event["shower_on"]
    window_start = shower_on - PRE_SHOWER_LEAD
    window_end = event["deposition_end"]

    filtered = cave[cave["n_sensors"] > MIN_QUANTS]
    mask = (filtered.index >= window_start) & (filtered.index <= window_end)
    s = filtered.loc[mask, column].dropna()
    if s.empty:
        return pd.Series(dtype=float)

    minute = ((s.index - shower_on).total_seconds() / 60).round().astype(int)
    s = pd.Series(s.values, index=minute)
    return s[~s.index.duplicated(keep="first")]


def group_average_curve(cave: pd.DataFrame, events: list, column: str) -> pd.Series:
    """
    Average one concentration column across a group's events at each minute.

    Parameters
    ----------
    cave : pd.DataFrame
        Output of compute_room_frame for one bin.
    events : list
        Event dicts for the group.
    column : str
        "C_bed1" or "C_room".

    Returns
    -------
    pd.Series
        Minute-indexed mean across events (NaN where no event contributes).
    """
    per_event = [align_event_series(cave, ev, column) for ev in events]
    per_event = [s for s in per_event if not s.empty]
    if not per_event:
        return pd.Series(dtype=float)
    return pd.concat(per_event, axis=1).mean(axis=1, skipna=True).sort_index()


def plot_ratio_time(cave: pd.DataFrame, groups: dict, bin_index: int,
                    plot_dir: Path) -> None:
    """
    Plot C_bed1,average / C_room,average versus minute, one line per group, one bin.

    For each group, C_bed1 and C_room are averaged across events at each minute
    (>8-sensor minutes only) and the ratio of those two averaged curves is
    drawn. Minute 0 is shower-on.

    Parameters
    ----------
    cave : pd.DataFrame
        Output of compute_room_frame for this bin.
    groups : dict
        Group label -> list of event dicts.
    bin_index : int
        Particle-size bin index (0-11).
    plot_dir : Path
        Directory to write the HTML figure into.
    """
    bin_name = PARTICLE_BINS[bin_index]["name"]

    out_path = plot_dir / f"c_bed1_c_room_ratio_time_bin{bin_index}.html"
    output_file(str(out_path), title=f"C_bed1/C_room vs time bin {bin_index}")

    fig = figure(
        width=1000,
        height=600,
        title=f"C_bed1,avg / C_room,avg vs time, bin {bin_index} ({bin_name} µm), >8 sensors",
        x_axis_label="Minutes from shower on",
        y_axis_label="C_bed1,average / C_room,average",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    any_line = False
    for label, events in groups.items():
        c195 = group_average_curve(cave, events, "C_bed1")
        cavg = group_average_curve(cave, events, "C_room")
        if c195.empty or cavg.empty:
            print(f"    [WARN] Bin {bin_index}, group {label}: no data; skipping curve")
            continue

        ratio = (c195 / cavg.where(cavg != 0)).dropna().sort_index()
        if ratio.empty:
            continue

        source = ColumnDataSource(
            data={"minute": ratio.index, "ratio": ratio.values}
        )
        color = GROUP_COLORS.get(label, COLORS["grid"])
        r = fig.line(
            "minute", "ratio", source=source, line_width=2.0, color=color,
            legend_label=f"{label} (n={len(events)})",
        )
        fig.scatter("minute", "ratio", source=source, size=4, color=color)
        fig.add_tools(
            HoverTool(
                renderers=[r],
                tooltips=[("Minute", "@minute"), ("Ratio", "@ratio{0.000}")],
                mode="vline",
            )
        )
        any_line = True

    if not any_line:
        print(f"    [WARN] Bin {bin_index}: no group curves; skipping figure")
        return

    # Reference lines: unity ratio and shower-on.
    fig.add_layout(Span(location=1.0, dimension="width",
                        line_color=COLORS["grid"], line_dash="dashed", line_width=1.0))
    fig.add_layout(Span(location=0, dimension="height",
                        line_color=COLORS["grid"], line_dash="dotted", line_width=1.0))

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"

    save(fig)
    print(f"    Saved {out_path.name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM inside sensor (C_bed1) vs position-weighted fleet "
        "average (C_room): per-bin scatter with Deming fit and per-group "
        "C_bed1/C_room ratio-vs-time curves."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end datetime.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"
    plot_dir = output_dir / "plots" / "moduair_room"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("MODULAIR-PM Inside (C_bed1) vs Position-Weighted Fleet Average")
    print("=" * 70)
    print(f"Window: {start} to {end}")
    print(f"Fleet sensors: {', '.join(FLEET_SNS)}")
    print(f">8-sensor filter: keep minutes with more than {MIN_QUANTS} reporting sensors")

    # Load fleet data (restrict to the sensors we actually use).
    available = set(list_available_sensors("raw"))
    wanted = [sn for sn in (FLEET_SNS + [TARGET_SN]) if sn in available]
    missing = [sn for sn in (FLEET_SNS + [TARGET_SN]) if sn not in available]
    if missing:
        print(f"  Sensors with no chunks (skipped): {', '.join(sorted(set(missing)))}")

    print("\nLoading fleet data...")
    fleet = load_fleet_bins(wanted, start=start, end=end)
    if TARGET_SN not in fleet:
        print(f"ERROR: target sensor {TARGET_SN} has no data in the window.")
        sys.exit(1)

    totals = build_position_totals(fleet)

    # Event groups for the ratio-vs-time figure.
    print("\nLoading event groups from registry...")
    groups = load_group_events(start, end)
    for label, evs in groups.items():
        print(f"  {label}: {len(evs)} events")
    if not groups:
        print("  No matching events found; ratio-vs-time figures will be skipped.")

    # Per-bin: build C_room frame, then both deliverables.
    print("\nBuilding per-bin figures...")
    fit_rows = []
    for i in range(N_BINS):
        bin_col = f"opc_bin{i}"
        print(f"  Bin {i} ({PARTICLE_BINS[i]['name']} µm):")
        cave = compute_room_frame(totals, bin_col)
        if cave.empty:
            print("    [WARN] No data; skipping bin")
            fit_rows.append({"bin": i, "slope": np.nan, "intercept": np.nan,
                             "r_squared": np.nan, "n": 0})
            continue

        fit = plot_scatter(cave, i, plot_dir)
        fit_rows.append({
            "bin": i,
            "slope": fit.get("slope", np.nan),
            "intercept": fit.get("intercept", np.nan),
            "r_squared": fit.get("r_squared", np.nan),
            "n": fit.get("n", 0),
        })

        if groups:
            plot_ratio_time(cave, groups, i, plot_dir)

    # Per-bin Deming fit table.
    fit_table = pd.DataFrame(fit_rows)
    fit_path = output_dir / "moduair_room_ratio_fit.csv"
    fit_table.to_csv(fit_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved fit table: {fit_path}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
