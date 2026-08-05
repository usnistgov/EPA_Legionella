#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Bin-2 Time Series (Bedroom Fleet)
=============================================

Plots the opc_bin2 (0.66-1.0 µm) particle concentration for the co-located
bedroom MODULAIR-PM sensors over 2026-06-04 through 2026-07-16, as a single
interactive Bokeh figure styled after the weekly QuantAQ PM figures produced by
the NIST_moduair-pm repository (quantaq_pm25.html): one line per sensor on a
shared datetime axis with a click-to-hide legend.

Sensor set
----------
The co-located bedroom sensors only. The outside sensor (MOD-PM-00785) is
excluded, and MOD-PM-00401 is dropped because it reads exactly 0 for a large
share of this window (near-dead sensor). The four sensors installed
2026-06-25/2026-07-08 (515, 465, 516, 467) are included; their traces simply
begin when their data does.

Usage
-----
    python scripts/moduair_bin2_timeseries.py
    python scripts/moduair_bin2_timeseries.py --start "2026-06-04" --end "2026-07-16"

Arguments
---------
    --start STR   Inclusive start datetime (default: 2026-06-04 00:00:00).
    --end STR     Inclusive end datetime (default: 2026-07-16 23:59:59).
    --output-dir  Override the figure output directory.

Output Files
------------
    <output>/plots/moduair_correction/bin2_timeseries.html

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-08-04
"""

import argparse
import sys
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

from src.data_paths import get_data_root  # noqa: E402
from src.moduair_loader import list_available_sensors, load_fleet_bins  # noqa: E402
from src.particle_calculations import PARTICLE_BINS  # noqa: E402
from src.plot_style import SENSOR_COLORS  # noqa: E402

DEFAULT_START = "2026-06-04 00:00:00"
DEFAULT_END = "2026-07-16 23:59:59"

# Particle-size bin to plot (opc_bin2, 0.66-1.0 µm).
BIN_INDEX = 2
BIN_COLUMN = f"opc_bin{BIN_INDEX}"

# Co-located bedroom sensors. The outside sensor (00785) and the near-dead
# 00401 are excluded so the figure shows only trustworthy bedroom traces.
PLOT_SENSORS = [
    "00195",
    "00402",
    "00465",
    "00467",
    "00515",
    "00516",
    "00554",
    "00555",
    "00813",
    "00814",
    "00815",
    "00816",
    "00942",
    "00943",
]


def plot_bin2(fleet: dict, sensor_ids: list, output_dir: Path,
              start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Plot opc_bin2 per sensor as a single interactive Bokeh figure.

    Each sensor is one line renderer with its own color and a click-to-hide
    legend entry, matching the weekly QuantAQ PM figure style.

    Parameters:
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)}.
        sensor_ids: Sensor IDs to plot, in legend order.
        output_dir: Analysis output directory.
        start: Window start (for the title).
        end: Window end (for the title).
    """
    plot_dir = output_dir / "plots" / "moduair_correction"
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_path = plot_dir / "bin2_timeseries.html"
    output_file(str(out_path), title="Bin 2 concentration, bedroom fleet")

    bin_name = PARTICLE_BINS[BIN_INDEX]["name"]
    fig = figure(
        width=1400,
        height=650,
        x_axis_type="datetime",
        title=(
            f"MODULAIR-PM bin {BIN_INDEX} ({bin_name} µm) concentration, "
            f"bedroom fleet ({start.date()} to {end.date()})"
        ),
        x_axis_label="Time",
        y_axis_label=f"Bin {BIN_INDEX} ({bin_name} µm) (# / cm³)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    for idx, sn in enumerate(sensor_ids):
        df = fleet[sn]
        source = ColumnDataSource(data={"x": df["datetime"], "y": df[BIN_COLUMN]})
        color = SENSOR_COLORS[idx % len(SENSOR_COLORS)]
        label = f"MOD-PM-{sn}"

        line = fig.line(
            "x", "y", source=source, line_width=1.0,
            color=color, legend_label=label,
        )
        fig.add_tools(
            HoverTool(
                renderers=[line],
                tooltips=[
                    ("Sensor", label),
                    ("Time", "@x{%F %H:%M}"),
                    (f"Bin {BIN_INDEX}", "@y{0.000}"),
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
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM bin-2 time series for the bedroom fleet."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end datetime.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Bin-2 Time Series (bedroom fleet)")
    print("=" * 70)
    print(f"Window: {start} to {end}  Bin: {BIN_COLUMN} ({PARTICLE_BINS[BIN_INDEX]['name']} µm)")

    available = list_available_sensors("raw")
    sensor_ids = [sn for sn in PLOT_SENSORS if sn in available]
    missing = [sn for sn in PLOT_SENSORS if sn not in available]
    if missing:
        print(f"Requested sensors with no chunks (skipped): {', '.join(missing)}")
    print(f"Plotting sensors: {', '.join(sensor_ids)}")

    print("\nLoading fleet data...")
    fleet = load_fleet_bins(sensor_ids, start=start, end=end)
    sensor_ids = [sn for sn in sensor_ids if sn in fleet]

    print("\nPlotting...")
    plot_bin2(fleet, sensor_ids, output_dir, start, end)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
