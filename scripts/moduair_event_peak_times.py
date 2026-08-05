#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Event Delta Peak Times
==================================

For every shower event since June 4, 2026 (00:00), this script computes the
"delta peak time" for each MODULAIR-PM sensor in the IAQ&V fleet: the elapsed
time from shower ON to the moment the summed bin0-11 particle concentration
reaches its maximum within the post-shower window.

Delta peak time per sensor per event:
    delta_peak = peak_time - shower_on
where peak_time is the timestamp of the maximum of the summed raw analysis
bins (opc_bin0 + ... + opc_bin11) inside the analysis window
[shower_on, shower_off + PEAK_WINDOW_HOURS].

Shower events are read from the project shower log (state-change log) using the
same ON-transition logic as the particle decay workflow.

Sensor set
----------
The figure shows the co-located bedroom sensors only. The outside sensor
(MOD-PM-00785) and MOD-PM-00555 are dropped. The four bedroom sensors added
on 2026-06-25 (515, 465, 516, 467) are included; each is gated to its install
time so that events before it went live are left blank for that sensor only.

MOD-PM-00401 is retained, but its raw bins read exactly 0 for roughly 90 % of
this period (a near-dead sensor), so its delta-peak values collapse to the
window start. The script prints a warning to flag that 401 is unreliable here.

Usage
-----
    python scripts/moduair_event_peak_times.py
    python scripts/moduair_event_peak_times.py --start "2026-06-04 00:00:00"

Arguments
---------
    --start STR   Inclusive start datetime for events (default 2026-06-04 00:00:00).
    --end STR     Inclusive end datetime for events (default 2026-07-16 23:59:59).
    --window-hours FLOAT  Hours after shower OFF to search for the peak (default 2.0).
    --output-dir PATH     Override output directory.

Output Files
------------
    <output>/moduair_event_peak_times.csv         (event x sensor, minutes)
    <output>/plots/moduair_correction/event_delta_peak_times.html

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version.
    2026-07-06 (Nathan Lima): Switch the delta-peak figure to an interactive
        Bokeh plot with click-to-hide legend so individual sensor traces can be
        toggled on and off; output is now an HTML file.
    2026-08-04 (Nathan Lima): Extend the default event window to 2026-06-04
        through 2026-07-16, restrict the plotted fleet to the co-located
        bedroom sensors (drop 555 and 785), add the four bedroom sensors
        installed 2026-06-25 (515, 465, 516, 467) with per-sensor install
        gating, and warn when 401 is a near-dead sensor for the window.
    2026-08-05 (Nathan Lima): Double the figure size (2200x1000) and color
        sensors by group (four colorblind-safe hue families, distinct shade
        per member); trace and legend order now follow the group sequence.
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

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_common_file, get_data_root  # noqa: E402
from src.moduair_loader import (  # noqa: E402
    BIN_COLUMNS,
    list_available_sensors,
    load_fleet_bins,
)

DEFAULT_START = "2026-06-04 00:00:00"
DEFAULT_END = "2026-07-16 23:59:59"
DEFAULT_PEAK_WINDOW_HOURS = 2.0

# Co-located bedroom sensors shown on the figure. The outside sensor (00785)
# and 00555 are excluded. 00401 is kept but is a near-dead sensor for this
# window (see module docstring); a warning is printed at run time.
PLOT_SENSORS = [
    "00195",
    "00401",
    "00402",
    "00465",
    "00467",
    "00515",
    "00516",
    "00554",
    "00813",
    "00814",
    "00815",
    "00816",
    "00942",
    "00943",
]

# Per-sensor install cutoffs: an event is blanked for that sensor if its
# shower-on time falls before the cutoff (sensor not yet in the test area).
# Matches SENSOR_INSTALL_CUTOFFS in scripts/export_config_timeseries_fleet.py;
# 00467 shares the 2026-07-08 window with the other three late installs.
SENSOR_INSTALL_CUTOFFS = {
    "00465": pd.Timestamp("2026-07-08 13:00:00"),
    "00467": pd.Timestamp("2026-07-08 13:00:00"),
    "00515": pd.Timestamp("2026-07-08 13:00:00"),
    "00516": pd.Timestamp("2026-07-08 13:00:00"),
}

# Sensor groups for the delta-peak figure. Each group shares a colorblind-safe
# base hue, and members are given distinct shades of that hue so co-located
# sensors read as one family on the plot. The plotting order below (group by
# group, in this sequence) also sets the legend order. Keys are 3-digit sensor
# labels (the last three digits of the sensor ID).
SENSOR_GROUP_COLORS = {
    # Group A: blues
    "465": "#08306b",
    "467": "#08519c",
    "515": "#2171b5",
    "516": "#4292c6",
    "943": "#6baed6",
    # Group B: oranges
    "402": "#e6550d",
    "816": "#fd8d3c",
    # Group C: greens
    "195": "#238b45",
    "813": "#74c476",
    # Group D: purples
    "401": "#3f007d",
    "554": "#6a51a3",
    "814": "#807dba",
    "815": "#9e9ac8",
    "942": "#bcbddc",
}

# Legend/draw order for the figure, matching the group sequence above.
SENSOR_PLOT_ORDER = list(SENSOR_GROUP_COLORS.keys())


def get_shower_events(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """
    Read shower ON events from the shower log within [start, end].

    A shower event is a 0 -> >0 transition of the 'shower' column, matching
    identify_shower_events() in src.particle_data_loader. The following OFF
    transition (back to 0) gives shower_off; if none is found within a short
    window, a 10-minute default duration is assumed.

    Parameters:
        start: Inclusive lower bound on shower_on.
        end: Inclusive upper bound on shower_on.

    Returns:
        List of dicts: {shower_on, shower_off}.
    """
    log_path = get_common_file("shower_log_file")
    if not log_path.exists():
        raise FileNotFoundError(
            f"Shower log not found: {log_path}\nRun scripts/process_shower_log.py first."
        )

    df = pd.read_csv(log_path)
    df["datetime_EDT"] = pd.to_datetime(df["datetime_EDT"])
    df = df.sort_values("datetime_EDT").reset_index(drop=True)

    events = []
    for i in range(len(df) - 1):
        if df.iloc[i]["shower"] == 0 and df.iloc[i + 1]["shower"] > 0:
            shower_on = df.iloc[i + 1]["datetime_EDT"]
            if shower_on < start or shower_on > end:
                continue

            shower_off = None
            for j in range(i + 2, min(i + 30, len(df))):
                if df.iloc[j]["shower"] == 0:
                    shower_off = df.iloc[j]["datetime_EDT"]
                    break
            if shower_off is None:
                shower_off = shower_on + timedelta(minutes=10)

            events.append({"shower_on": shower_on, "shower_off": shower_off})

    return events


def compute_delta_peaks(
    events: list,
    fleet: dict,
    window_hours: float,
) -> pd.DataFrame:
    """
    Compute delta peak time (minutes) per event per sensor.

    For each event and sensor, the analysis window runs from shower_on to
    shower_off + window_hours. The summed bin0-11 concentration is evaluated on
    the sensor's 1-minute series within that window; the delta peak time is the
    minutes between shower_on and the window's maximum-concentration timestamp.

    Parameters:
        events: List of {shower_on, shower_off} dicts.
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)}.
        window_hours: Hours after shower_off to include in the search window.

    Returns:
        DataFrame, one row per event, columns:
        shower_on, shower_off, then delta_peak_min_<sn> per sensor.
    """
    sensor_ids = sorted(fleet.keys())

    # Pre-compute the summed bin0-11 total per sensor on its datetime index.
    totals = {}
    for sn in sensor_ids:
        df = fleet[sn].set_index("datetime")
        totals[sn] = df[BIN_COLUMNS].sum(axis=1, skipna=True)

    rows = []
    for ev in events:
        shower_on = ev["shower_on"]
        shower_off = ev["shower_off"]
        win_end = shower_off + timedelta(hours=window_hours)

        row = {"shower_on": shower_on, "shower_off": shower_off}
        for sn in sensor_ids:
            # Blank events that precede this sensor's install time.
            cutoff = SENSOR_INSTALL_CUTOFFS.get(sn)
            if cutoff is not None and shower_on < cutoff:
                row[f"delta_peak_min_{sn}"] = float("nan")
                continue
            series = totals[sn]
            mask = (series.index >= shower_on) & (series.index <= win_end)
            window = series[mask].dropna()
            if window.empty:
                row[f"delta_peak_min_{sn}"] = float("nan")
                continue
            peak_time = window.idxmax()
            row[f"delta_peak_min_{sn}"] = (peak_time - shower_on).total_seconds() / 60.0
        rows.append(row)

    return pd.DataFrame(rows)


def plot_delta_peaks(peaks: pd.DataFrame, sensor_ids: list, output_dir: Path) -> None:
    """
    Plot delta peak time per event for each sensor as an interactive Bokeh figure.

    Each sensor is a separate line+scatter renderer. The legend uses a
    click-to-hide policy so individual sensor traces can be toggled on and off
    in the resulting HTML file.

    Parameters:
        peaks: DataFrame from compute_delta_peaks().
        sensor_ids: Sorted sensor IDs to plot.
        output_dir: Analysis output directory.
    """
    plot_dir = output_dir / "plots" / "moduair_correction"
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_path = plot_dir / "event_delta_peak_times.html"
    output_file(str(out_path), title="Delta peak time per shower event")

    fig = figure(
        width=1600,
        height=800,
        x_axis_type="datetime",
        title="Delta peak time per shower event (summed bin0-11)",
        x_axis_label="Shower ON",
        y_axis_label="Delta peak time (min since shower ON)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    # Draw sensors in the grouped legend order; any plotted sensor not covered
    # by SENSOR_GROUP_COLORS is appended afterward so it is never silently dropped.
    label_by_sn = {sn: sn[-3:] for sn in sensor_ids}
    ordered = [sn for lbl in SENSOR_PLOT_ORDER for sn in sensor_ids if label_by_sn[sn] == lbl]
    ordered += [sn for sn in sensor_ids if sn not in ordered]

    for sn in ordered:
        col = f"delta_peak_min_{sn}"
        source = ColumnDataSource(data={"x": peaks["shower_on"], "y": peaks[col]})
        label = label_by_sn[sn]
        color = SENSOR_GROUP_COLORS.get(label, "#7f7f7f")

        line = fig.line("x", "y", source=source, line_width=1.0, color=color, legend_label=label)
        fig.scatter("x", "y", source=source, size=5, color=color, legend_label=label)
        # Hover reports the sensor, event time, and delta peak for the line only.
        fig.add_tools(
            HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sensor", label),
                    ("Shower ON", "@x{%F %H:%M}"),
                    ("Delta peak (min)", "@y{0.0}"),
                ],
                formatters={"@x": "datetime"},
                mode="vline",
            )
        )

    fig.legend.title = "Sensor"
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"

    save(fig)
    print(f"  Saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MODULAIR-PM delta peak times per shower event.")
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end datetime.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_PEAK_WINDOW_HOURS,
        help="Hours after shower OFF to search for the peak.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Event Delta Peak Times")
    print("=" * 70)
    print(f"Events: {start} to {end}  Peak window: shower OFF + {args.window_hours} h")

    events = get_shower_events(start, end)
    print(f"\nShower events in window: {len(events)}")
    if not events:
        print("No events found; nothing to do.")
        return

    available = list_available_sensors("raw")
    # Restrict to the co-located bedroom sensors we plot, keeping only those
    # that actually have chunk files in the share.
    sensor_ids = [sn for sn in PLOT_SENSORS if sn in available]
    missing = [sn for sn in PLOT_SENSORS if sn not in available]
    if missing:
        print(f"Requested sensors with no chunks (skipped): {', '.join(missing)}")
    print(f"Plotting sensors: {', '.join(sensor_ids)}")

    # Load fleet data spanning the full event range plus the peak window.
    load_end = max(e["shower_off"] for e in events) + timedelta(hours=args.window_hours + 1)
    print("\nLoading fleet data...")
    fleet = load_fleet_bins(sensor_ids, start=start, end=load_end)

    # Keep only sensors we intend to plot and that loaded successfully.
    sensor_ids = [sn for sn in sensor_ids if sn in fleet]

    print("\nComputing delta peak times...")
    peaks = compute_delta_peaks(events, fleet, args.window_hours)

    # Flag 401 if it is effectively dead for this window: its summed bins are
    # zero for a large share of minutes, so delta-peaks collapse to the start.
    if "00401" in fleet:
        total_401 = fleet["00401"].set_index("datetime")[BIN_COLUMNS].sum(axis=1)
        zero_frac = float((total_401 == 0).mean())
        if zero_frac > 0.3:
            print(
                f"\n[WARN] MOD-PM-00401 reads exactly 0 for {zero_frac * 100:.0f}% of "
                "minutes in this window (near-dead sensor). Its delta-peak values "
                "collapse to the window start and should not be interpreted."
            )

    csv_path = output_dir / "moduair_event_peak_times.csv"
    peaks.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    print("\nPlotting...")
    plot_delta_peaks(peaks, sensor_ids, output_dir)

    # Brief summary: mean delta peak per sensor
    print("\nMean delta peak time (min) per sensor:")
    for sn in sensor_ids:
        col = f"delta_peak_min_{sn}"
        print(f"  {sn[-3:]}: {peaks[col].mean():.1f}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
