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

Usage
-----
    python scripts/moduair_event_peak_times.py
    python scripts/moduair_event_peak_times.py --start "2026-06-04 00:00:00"

Arguments
---------
    --start STR   Inclusive start datetime for events (default 2026-06-04 00:00:00).
    --window-hours FLOAT  Hours after shower OFF to search for the peak (default 2.0).
    --output-dir PATH     Override output directory.

Output Files
------------
    <output>/moduair_event_peak_times.csv         (event x sensor, minutes)
    <output>/plots/moduair_correction/event_delta_peak_times.png

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version.
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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_common_file, get_data_root  # noqa: E402
from src.moduair_loader import (  # noqa: E402
    BIN_COLUMNS,
    list_available_sensors,
    load_fleet_bins,
)
from src.plot_style import SENSOR_COLORS, create_figure, save_figure  # noqa: E402

DEFAULT_START = "2026-06-04 00:00:00"
DEFAULT_PEAK_WINDOW_HOURS = 2.0


def get_shower_events(start: pd.Timestamp) -> list:
    """
    Read shower ON events from the shower log on or after ``start``.

    A shower event is a 0 -> >0 transition of the 'shower' column, matching
    identify_shower_events() in src.particle_data_loader. The following OFF
    transition (back to 0) gives shower_off; if none is found within a short
    window, a 10-minute default duration is assumed.

    Parameters:
        start: Inclusive lower bound on shower_on.

    Returns:
        List of dicts: {shower_on, shower_off}.
    """
    log_path = get_common_file("shower_log_file")
    if not log_path.exists():
        raise FileNotFoundError(
            f"Shower log not found: {log_path}\n"
            "Run scripts/process_shower_log.py first."
        )

    df = pd.read_csv(log_path)
    df["datetime_EDT"] = pd.to_datetime(df["datetime_EDT"])
    df = df.sort_values("datetime_EDT").reset_index(drop=True)

    events = []
    for i in range(len(df) - 1):
        if df.iloc[i]["shower"] == 0 and df.iloc[i + 1]["shower"] > 0:
            shower_on = df.iloc[i + 1]["datetime_EDT"]
            if shower_on < start:
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
    Plot delta peak time per event for each sensor.

    Parameters:
        peaks: DataFrame from compute_delta_peaks().
        sensor_ids: Sorted sensor IDs to plot.
        output_dir: Analysis output directory.
    """
    plot_dir = output_dir / "plots" / "moduair_correction"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = create_figure(figsize=(12, 5))
    x = peaks["shower_on"]

    for i, sn in enumerate(sensor_ids):
        col = f"delta_peak_min_{sn}"
        ax.plot(
            x,
            peaks[col],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color=SENSOR_COLORS[i % len(SENSOR_COLORS)],
            label=sn[-3:],
        )

    ax.set_title("Delta peak time per shower event (summed bin0-11)")
    ax.set_xlabel("Shower ON")
    ax.set_ylabel("Delta peak time (min since shower ON)")
    ax.legend(loc="upper right", ncol=2, title="Sensor")
    fig.autofmt_xdate()

    out_path = plot_dir / "event_delta_peak_times.png"
    save_figure(fig, out_path)
    print(f"  Saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM delta peak times per shower event."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_PEAK_WINDOW_HOURS,
        help="Hours after shower OFF to search for the peak.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Event Delta Peak Times")
    print("=" * 70)
    print(f"Start: {start}  Peak window: shower OFF + {args.window_hours} h")

    events = get_shower_events(start)
    print(f"\nShower events since {start.date()}: {len(events)}")
    if not events:
        print("No events found; nothing to do.")
        return

    available = list_available_sensors("raw")
    print(f"Available raw sensors: {', '.join(available)}")

    # Load fleet data spanning the full event range plus the peak window.
    load_end = max(e["shower_off"] for e in events) + timedelta(hours=args.window_hours + 1)
    print("\nLoading fleet data...")
    fleet = load_fleet_bins(available, start=start, end=load_end)

    print("\nComputing delta peak times...")
    peaks = compute_delta_peaks(events, fleet, args.window_hours)
    sensor_ids = sorted(fleet.keys())

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
