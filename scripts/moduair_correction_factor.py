#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Inter-Sensor Correction Factor (Concentration Ratios)
=================================================================

Computes and plots candidate correction factors for the inside MODULAIR-PM
sensor (MOD-PM-00195) by taking bin-wise concentration ratios against other
sensors in the IAQ&V fleet, using the raw analysis bins (opc_bin0-opc_bin11).

Three ratios are produced, per particle-size bin, on a shared time base:

    Ratio 813:   195 / 813
        Direct ratio against reference sensor MOD-PM-00813.

    Ratio quad:  195 / mean(515, 465, 943, 516)
        Ratio against the average of the four co-located reference sensors
        515, 465, 943, and 516.

    Ratio others: 195 / mean(remaining bedroom sensors)
        Ratio against the average of the co-located bedroom sensors that are
        not already used above: every sensor except the target (195), the
        outside sensor (785), the direct reference (813), and the four
        quad-average sensors (515, 465, 943, 516).

A value near 1.0 means the inside sensor agrees with the reference; a sustained
offset is a candidate multiplicative correction factor.

One figure is produced per bin (0-11) over the full analysis window
(2026-06-04 through 2026-07-16 by default), with all three ratios drawn on the
same axes so they can be compared directly. Figures are interactive Bokeh HTML
files with a click-to-hide legend.

Usage
-----
    python scripts/moduair_correction_factor.py
    python scripts/moduair_correction_factor.py --start "2026-06-04 00:00:00"

Arguments
---------
    --start STR   Inclusive start datetime (default: 2026-06-04 00:00:00).
    --end STR     Inclusive end datetime (default: 2026-07-16 23:59:59).
    --output-dir  Override the figure output directory.

Output Files
------------
    <output>/plots/moduair_correction/jun04_jul16/correction_factor_bin{N}.html
    <output>/moduair_correction_factor_ratios_jun04_jul16.csv
    <output>/moduair_correction_factor_summary_jun04_jul16.csv
    <output>/moduair_correction_factor_195_813_jun04_jul16.csv
        Focused 195/813 table: per-bin mean, standard deviation, and count.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version.
    2026-07-06 (Nathan Lima): Switch to three ratios (195/813, 195/943,
        195/mean(all others), excluding 785); add last-week and last-24h
        windowed figure sets and per-window ratio and summary tables.
    2026-08-04 (Nathan Lima): Fix the analysis to a single 2026-06-04 through
        2026-07-16 window (drop last-week and last-24h). Redefine the ratios:
        195/813, 195/mean(515,465,943,516), and 195/mean(remaining bedroom
        sensors). Switch the per-bin figures to interactive Bokeh plots.
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
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure, output_file, save

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_data_root  # noqa: E402
from src.moduair_loader import (  # noqa: E402
    BIN_COLUMNS,
    N_BINS,
    list_available_sensors,
    load_fleet_bins,
)
from src.particle_calculations import PARTICLE_BINS  # noqa: E402
from src.plot_style import COLORS  # noqa: E402

# Sensor of interest (inside) and the direct-ratio reference sensor.
TARGET_SN = "00195"
REFERENCE_SN = "00813"

# The four co-located sensors averaged for the "quad" ratio: 195 / mean(these).
QUAD_AVERAGE_SNS = ["00515", "00465", "00943", "00516"]

# Sensors excluded from the "others" average: the target (195), the outside
# sensor (785), the direct reference (813), and the four quad-average sensors.
# Every remaining sensor with data forms the "others" set.
EXCLUDE_FROM_OTHERS = {"00195", "00785", "00813", "00515", "00465", "00943", "00516"}

DEFAULT_START = "2026-06-04 00:00:00"
DEFAULT_END = "2026-07-16 23:59:59"
WINDOW_KEY = "jun04_jul16"
WINDOW_LABEL = "2026-06-04 to 2026-07-16"

# Line colors for the three ratios (single source of truth: plot_style COLORS)
COLOR_RATIO_813 = COLORS["lambda"]    # red:    195 / 813
COLOR_RATIO_QUAD = COLORS["outside"]  # green:  195 / mean(515,465,943,516)
COLOR_RATIO_OTHERS = COLORS["bedroom"]  # blue:  195 / mean(remaining bedroom)

# Ratio metadata driving both plotting and table columns. The "key" is the
# suffix used in the ratio_<key>_bin{N} column names; "label" is the figure
# legend and table label (kept in sync with compute_ratios()).
RATIO_SPECS = [
    {"key": "813", "color": COLOR_RATIO_813, "label": "195 / 813"},
    {"key": "quad", "color": COLOR_RATIO_QUAD, "label": "195 / mean(515,465,943,516)"},
    {"key": "others", "color": COLOR_RATIO_OTHERS, "label": "195 / mean(other bedroom)"},
]


def compute_ratios(fleet: dict) -> pd.DataFrame:
    """
    Build a tidy DataFrame of all three bin-wise ratios on a common time index.

    Ratios per bin N:
        ratio_813_bin{N}    = 195 / 813
        ratio_quad_bin{N}   = 195 / mean(515, 465, 943, 516)
        ratio_others_bin{N} = 195 / mean(remaining bedroom sensors)

    Parameters:
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)} from
            src.moduair_loader.load_fleet_bins().

    Returns:
        DataFrame indexed by datetime with columns ratio_813_bin{N},
        ratio_quad_bin{N}, and ratio_others_bin{N} for N in 0..N_BINS-1.
    """
    if TARGET_SN not in fleet:
        raise ValueError(f"Target sensor {TARGET_SN} has no data in the fleet.")
    if REFERENCE_SN not in fleet:
        raise ValueError(f"Reference sensor {REFERENCE_SN} has no data in the fleet.")

    target = fleet[TARGET_SN].set_index("datetime")[BIN_COLUMNS]
    reference = fleet[REFERENCE_SN].set_index("datetime")[BIN_COLUMNS]

    # Quad-average sensors that actually have data.
    quad_ids = [sid for sid in QUAD_AVERAGE_SNS if sid in fleet]
    missing_quad = [sid for sid in QUAD_AVERAGE_SNS if sid not in fleet]
    if missing_quad:
        print(f"  [WARN] Quad-average sensors with no data (dropped): {', '.join(missing_quad)}")

    # "Others" set: every remaining sensor with data.
    other_ids = [sid for sid in fleet if sid not in EXCLUDE_FROM_OTHERS]

    print(f"  Direct ratio reference: {REFERENCE_SN}")
    print(f"  Quad ratio averages over {len(quad_ids)} sensors: {', '.join(quad_ids)}")
    print(f"  Others ratio averages over {len(other_ids)} sensors: {', '.join(other_ids)}")

    def _mean_stack(sensor_ids: list) -> pd.DataFrame:
        """Per-bin mean across a set of sensors, aligned to the target index."""
        stack = pd.concat(
            [fleet[sid].set_index("datetime")[BIN_COLUMNS] for sid in sensor_ids],
            axis=1,
            keys=sensor_ids,
        )
        return stack

    quad_stack = _mean_stack(quad_ids) if quad_ids else None
    others_stack = _mean_stack(other_ids) if other_ids else None

    out = pd.DataFrame(index=target.index)
    for i in range(N_BINS):
        col = f"opc_bin{i}"

        # Direct ratio: 195 / 813
        ref_aligned = reference[col].reindex(target.index)
        out[f"ratio_813_bin{i}"] = target[col] / ref_aligned.where(ref_aligned != 0)

        # Quad ratio: 195 / mean(515, 465, 943, 516)
        if quad_stack is not None:
            quad_bin = quad_stack.xs(col, axis=1, level=1)
            quad_mean = quad_bin.mean(axis=1, skipna=True).reindex(target.index)
            out[f"ratio_quad_bin{i}"] = target[col] / quad_mean.where(quad_mean != 0)
        else:
            out[f"ratio_quad_bin{i}"] = float("nan")

        # Others ratio: 195 / mean(remaining bedroom sensors)
        if others_stack is not None:
            others_bin = others_stack.xs(col, axis=1, level=1)
            others_mean = others_bin.mean(axis=1, skipna=True).reindex(target.index)
            out[f"ratio_others_bin{i}"] = target[col] / others_mean.where(others_mean != 0)
        else:
            out[f"ratio_others_bin{i}"] = float("nan")

    return out


def plot_bin_ratios(ratios: pd.DataFrame, plot_dir: Path, window_label: str) -> None:
    """
    Plot all three ratios per bin (one interactive Bokeh figure per bin).

    Each figure shows the three ratio traces on a shared datetime axis with a
    dashed reference line at 1.0 (perfect agreement) and a click-to-hide
    legend. Output is one HTML file per bin.

    Parameters:
        ratios: DataFrame from compute_ratios(), datetime-indexed.
        plot_dir: Directory to write the per-bin HTML figures into.
        window_label: Human-readable window name for figure titles.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    times = ratios.index

    for i in range(N_BINS):
        bin_name = PARTICLE_BINS[i]["name"]
        out_path = plot_dir / f"correction_factor_bin{i}.html"
        output_file(str(out_path), title=f"Correction factor bin {i}")

        fig = figure(
            width=1100,
            height=450,
            x_axis_type="datetime",
            title=(
                f"MODULAIR-PM correction factor, bin {i} ({bin_name} µm) "
                f"— {window_label}"
            ),
            x_axis_label="Date",
            y_axis_label="Concentration ratio",
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )

        for spec in RATIO_SPECS:
            col = f"ratio_{spec['key']}_bin{i}"
            source = ColumnDataSource(data={"x": times, "y": ratios[col]})
            line = fig.line(
                "x", "y", source=source, line_width=1.2,
                color=spec["color"], legend_label=spec["label"],
            )
            fig.add_tools(
                HoverTool(
                    renderers=[line],
                    tooltips=[
                        ("Ratio", spec["label"]),
                        ("Time", "@x{%F %H:%M}"),
                        ("Value", "@y{0.000}"),
                    ],
                    formatters={"@x": "datetime"},
                    mode="vline",
                )
            )

        # Reference line at 1.0 (perfect agreement).
        fig.add_layout(
            Span(location=1.0, dimension="width", line_color=COLORS["grid"],
                 line_dash="dashed", line_width=1.0)
        )

        fig.legend.title = "Ratio"
        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"

        save(fig)
        print(f"    Saved {out_path.name}")


def summarize_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-bin, per-ratio summary table (mean, std, median, count).

    Parameters:
        ratios: DataFrame from compute_ratios().

    Returns:
        Long-format DataFrame with columns bin, bin_name_um, ratio, mean, std,
        median, count.
    """
    rows = []
    for i in range(N_BINS):
        for spec in RATIO_SPECS:
            series = ratios[f"ratio_{spec['key']}_bin{i}"].dropna()
            rows.append(
                {
                    "bin": i,
                    "bin_name_um": PARTICLE_BINS[i]["name"],
                    "ratio": spec["label"],
                    "mean": series.mean(),
                    "std": series.std(),
                    "median": series.median(),
                    "count": int(series.count()),
                }
            )
    return pd.DataFrame(rows)


def summarize_813(ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Build the focused 195/813 correction-factor table (Task 3).

    One row per particle-size bin with the mean 195/813 ratio, its standard
    deviation, median, and the number of contributing 1-minute samples over the
    analysis window.

    Parameters:
        ratios: DataFrame from compute_ratios().

    Returns:
        DataFrame with columns bin, bin_name_um, correction_factor_mean,
        correction_factor_std, correction_factor_median, count.
    """
    rows = []
    for i in range(N_BINS):
        series = ratios[f"ratio_813_bin{i}"].dropna()
        rows.append(
            {
                "bin": i,
                "bin_name_um": PARTICLE_BINS[i]["name"],
                "correction_factor_mean": series.mean(),
                "correction_factor_std": series.std(),
                "correction_factor_median": series.median(),
                "count": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM inter-sensor correction factor (bin-wise ratios)."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive end datetime.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"
    plot_root = output_dir / "plots" / "moduair_correction"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Correction Factor Analysis")
    print("=" * 70)
    print(f"Window: {start} to {end}")

    available = list_available_sensors("raw")
    print(f"\nAvailable raw sensors: {', '.join(available)}")

    print("\nLoading fleet data...")
    fleet = load_fleet_bins(available, start=start, end=end)

    print("\nComputing ratios...")
    ratios = compute_ratios(fleet)
    if ratios.empty:
        print("No overlapping ratio data; nothing to do.")
        return

    print(f"  Time range: {ratios.index.min()} to {ratios.index.max()}")

    print(f"\n[{WINDOW_LABEL}] plots and tables...")
    plot_bin_ratios(ratios, plot_root / WINDOW_KEY, WINDOW_LABEL)

    ratios_path = output_dir / f"moduair_correction_factor_ratios_{WINDOW_KEY}.csv"
    ratios.reset_index().to_csv(ratios_path, index=False)
    print(f"  Saved {ratios_path.name}")

    summary_path = output_dir / f"moduair_correction_factor_summary_{WINDOW_KEY}.csv"
    summarize_ratios(ratios).to_csv(summary_path, index=False)
    print(f"  Saved {summary_path.name}")

    # Task 3: focused 195/813 per-bin correction factor with standard deviation.
    table_813_path = output_dir / f"moduair_correction_factor_195_813_{WINDOW_KEY}.csv"
    table_813 = summarize_813(ratios)
    table_813.to_csv(table_813_path, index=False)
    print(f"  Saved {table_813_path.name}")
    print("\n  195/813 correction factor per bin (mean ± std):")
    for _, r in table_813.iterrows():
        print(
            f"    bin {int(r['bin']):>2} ({r['bin_name_um']:>9} µm): "
            f"{r['correction_factor_mean']:.3f} ± {r['correction_factor_std']:.3f}"
        )

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
