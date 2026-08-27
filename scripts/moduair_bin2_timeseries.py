#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Per-Bin Time Series (Bedroom Fleet)
===============================================

Plots the opc particle-size-bin concentration for the co-located bedroom
MODULAIR-PM sensors over 2026-06-04 through 2026-07-16, as one interactive
Bokeh figure per bin (bins 0-10), styled after the weekly QuantAQ PM figures
produced by the NIST_moduair-pm repository (quantaq_pm25.html): one line per
sensor on a shared datetime axis with a click-to-hide legend. Figure geometry,
font size, trace colors, and the descriptive sensor legend come from the shared
MODULAIR-PM helpers in src.plot_style.

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
    <output>/plots/moduair_correction/bin0_timeseries.html
    ...
    <output>/plots/moduair_correction/bin10_timeseries.html

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-08-04
Update log:
    2026-08-07 (Nathan Lima): Produce one figure per bin (0-10) instead of only
        bin 2; move figure size, font, colors, and the descriptive sensor
        legend into the shared MODULAIR-PM Bokeh helpers in src.plot_style
        (1600x800, no title, 12pt); remove x-axis range padding; and cap the
        x-axis at 2026-07-16 10:00.
    2026-08-17 (Nathan Lima): Overlay the position-weighted fleet average
        C_room = 0.32*C_high + 0.28*C_mid + 0.20*C_bed + 0.20*C_low as a black
        trace on each bin figure (position groups defined in
        scripts.moduair_cave_ratio).
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
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.plotting import figure, output_file, save

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_data_root  # noqa: E402
from src.moduair_loader import list_available_sensors, load_fleet_bins  # noqa: E402
from src.particle_calculations import PARTICLE_BINS  # noqa: E402
from src.plot_style import (  # noqa: E402
    moduair_color,
    moduair_label,
    order_moduair_sensors,
    style_moduair_figure,
)

# Position-weighted fleet average (single source of truth for the groups,
# weights, and assembly logic lives in scripts.moduair_cave_ratio).
from scripts.moduair_cave_ratio import (  # noqa: E402
    FLEET_SNS,
    build_position_totals,
    compute_room_frame,
)

# Color and label for the C_room overlay trace.
CAVE_COLOR = "#000000"
CAVE_LABEL = "C_room (0.32 high + 0.28 mid + 0.20 bed + 0.20 low)"

DEFAULT_START = "2026-06-04 00:00:00"
DEFAULT_END = "2026-07-16 23:59:59"

# Particle-size bins to plot (opc_bin0 through opc_bin10), one figure per bin.
BIN_INDICES = list(range(0, 11))

# Right edge of the x-axis, shared by every bin figure.
X_AXIS_END = pd.Timestamp("2026-07-16 10:00:00")

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


def plot_bin(fleet: dict, sensor_ids: list, bin_index: int, output_dir: Path,
             start: pd.Timestamp, room_series: pd.Series | None = None) -> None:
    """
    Plot one opc bin per sensor as a single interactive Bokeh figure.

    Each sensor is one line renderer with its shared MODULAIR-PM color and a
    click-to-hide legend entry, matching the weekly QuantAQ PM figure style.
    The x-axis is set explicitly with no range padding, from ``start`` to the
    shared X_AXIS_END (2026-07-16 10:00). When ``room_series`` is supplied, the
    position-weighted fleet average C_room is drawn as a black overlay trace.

    Parameters:
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)}.
        sensor_ids: Sensor IDs to plot, in legend order.
        bin_index: opc bin index to plot (0-10).
        output_dir: Analysis output directory.
        start: Window start (left edge of the x-axis).
        room_series: Optional datetime-indexed C_room series for this bin.
    """
    plot_dir = output_dir / "plots" / "moduair_correction"
    plot_dir.mkdir(parents=True, exist_ok=True)

    bin_column = f"opc_bin{bin_index}"
    bin_name = PARTICLE_BINS[bin_index]["name"]

    out_path = plot_dir / f"bin{bin_index}_timeseries.html"
    output_file(str(out_path), title=f"Bin {bin_index} concentration, bedroom fleet")

    fig = figure(
        x_axis_type="datetime",
        x_axis_label="Time",
        y_axis_label=f"Bin {bin_index} ({bin_name} µm) (# / cm³)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )

    # Explicit x-range with no padding, capped at X_AXIS_END.
    fig.x_range = Range1d(start=start, end=X_AXIS_END)

    for sn in order_moduair_sensors(sensor_ids):
        df = fleet[sn]
        # Clip to the visible x-window before plotting so the auto-scaled y-axis
        # is driven only by data shown in the figure, not by points past
        # X_AXIS_END (2026-07-16 10:00).
        df = df[(df["datetime"] >= start) & (df["datetime"] <= X_AXIS_END)]
        source = ColumnDataSource(data={"x": df["datetime"], "y": df[bin_column]})
        color = moduair_color(sn)
        label = moduair_label(sn)

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
                    (f"Bin {bin_index}", "@y{0.000}"),
                ],
                formatters={"@x": "datetime"},
                mode="vline",
            )
        )

    style_moduair_figure(fig, legend_location="top_left")

    # Overlay the position-weighted fleet average C_room as a black trace, drawn
    # last so it sits on top of the individual sensor lines.
    if room_series is not None and not room_series.empty:
        cs = room_series[(room_series.index >= start) & (room_series.index <= X_AXIS_END)]
        cs = cs.dropna()
        if not cs.empty:
            source = ColumnDataSource(data={"x": cs.index, "y": cs.values})
            line = fig.line(
                "x", "y", source=source, line_width=2.0,
                color=CAVE_COLOR, legend_label=CAVE_LABEL,
            )
            fig.add_tools(
                HoverTool(
                    renderers=[line],
                    tooltips=[
                        ("Trace", "C_room"),
                        ("Time", "@x{%F %H:%M}"),
                        (f"Bin {bin_index}", "@y{0.000}"),
                    ],
                    formatters={"@x": "datetime"},
                    mode="vline",
                )
            )

    save(fig)
    print(f"  Saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM per-bin (0-10) time series for the bedroom fleet."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end datetime.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Per-Bin Time Series (bedroom fleet)")
    print("=" * 70)
    print(f"Window: {start} to {end}  Bins: 0-10")

    available = list_available_sensors("raw")
    sensor_ids = [sn for sn in PLOT_SENSORS if sn in available]
    missing = [sn for sn in PLOT_SENSORS if sn not in available]
    if missing:
        print(f"Requested sensors with no chunks (skipped): {', '.join(missing)}")
    print(f"Plotting sensors: {', '.join(sensor_ids)}")

    print("\nLoading fleet data...")
    fleet = load_fleet_bins(sensor_ids, start=start, end=end)
    sensor_ids = [sn for sn in sensor_ids if sn in fleet]

    # Load the C_room sensor set (may include sensors not plotted, e.g. 00401)
    # and build one C_room series per bin. Sensors already loaded above are
    # reused; any extra ones are loaded here.
    room_wanted = [sn for sn in FLEET_SNS if sn in available]
    room_missing = [sn for sn in room_wanted if sn not in fleet]
    room_fleet = dict(fleet)
    if room_missing:
        print(f"\nLoading extra sensors for C_room: {', '.join(room_missing)}")
        room_fleet.update(load_fleet_bins(room_missing, start=start, end=end))

    room_totals = build_position_totals(room_fleet)
    room_by_bin = {}
    for bin_index in BIN_INDICES:
        frame = compute_room_frame(room_totals, f"opc_bin{bin_index}")
        room_by_bin[bin_index] = frame["C_room"] if not frame.empty else None

    print("\nPlotting...")
    for bin_index in BIN_INDICES:
        plot_bin(fleet, sensor_ids, bin_index, output_dir, start,
                 room_series=room_by_bin.get(bin_index))

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
