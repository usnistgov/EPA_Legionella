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

    Ratio 943:   195 / 943
        Direct ratio against reference sensor MOD-PM-00943.

    Ratio mean:  195 / mean(all others)
        Ratio against the average of all other fleet sensors, excluding the
        target (195), the two reference sensors (813, 943), and 785. The
        averaging set is every remaining sensor with data.

A value near 1.0 means the inside sensor agrees with the reference; a sustained
offset is a candidate multiplicative correction factor.

Figures are produced for two trailing windows measured back from the latest
timestamp present in the loaded fleet data: the last week and the last 24
hours. One figure is produced per bin (0-11) per window, with all three ratios
drawn on the same axes so they can be compared directly.

Usage
-----
    python scripts/moduair_correction_factor.py
    python scripts/moduair_correction_factor.py --start "2026-06-04 00:00:00"

Arguments
---------
    --start STR   Inclusive start datetime (default: 2026-06-04 00:00:00).
    --end STR     Inclusive end datetime (default: latest available).
    --output-dir  Override the figure output directory.

Output Files
------------
    <output>/plots/moduair_correction/last_week/correction_factor_bin{N}.png
    <output>/plots/moduair_correction/last_24h/correction_factor_bin{N}.png
    <output>/moduair_correction_factor_ratios_last_week.csv
    <output>/moduair_correction_factor_ratios_last_24h.csv
    <output>/moduair_correction_factor_summary_last_week.csv
    <output>/moduair_correction_factor_summary_last_24h.csv

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version.
    2026-07-06 (Nathan Lima): Switch to three ratios (195/813, 195/943,
        195/mean(all others), excluding 785); add last-week and last-24h
        windowed figure sets and per-window ratio and summary tables.
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

import matplotlib.dates as mdates
import pandas as pd

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
from src.plot_style import COLORS, create_figure, save_figure  # noqa: E402

# Sensor of interest (inside) and the two reference sensors for direct ratios.
TARGET_SN = "00195"
REFERENCE_SNS = ["00813", "00943"]

# Sensors excluded from the "mean (all others)" average: the two reference
# sensors (813, 943) and 785. The target (195) is always excluded implicitly.
EXCLUDE_FROM_AVERAGE = {"00785", "00813", "00943"}

DEFAULT_START = "2026-06-04 00:00:00"

# Trailing figure windows, measured back from the latest available timestamp.
WINDOWS = {
    "last_week": {
        "label": "last 7 days",
        "delta": timedelta(days=7),
        "date_fmt": "%m-%d",
    },
    "last_24h": {
        "label": "last 24 hours",
        "delta": timedelta(hours=24),
        "date_fmt": "%m-%d %H:%M",
    },
}

# Line colors for the three ratios (single source of truth: plot_style COLORS)
COLOR_RATIO_813 = COLORS["lambda"]    # red:    195 / 813
COLOR_RATIO_943 = COLORS["outside"]   # green:  195 / 943
COLOR_RATIO_MEAN = COLORS["bedroom"]  # blue:   195 / mean(all others)

# Ratio metadata driving both plotting and table columns.
RATIO_SPECS = [
    {"key": "813", "color": COLOR_RATIO_813, "label": "195 / 813"},
    {"key": "943", "color": COLOR_RATIO_943, "label": "195 / 943"},
    {"key": "mean", "color": COLOR_RATIO_MEAN, "label": "195 / mean(others)"},
]


def compute_ratios(fleet: dict) -> pd.DataFrame:
    """
    Build a tidy DataFrame of all three bin-wise ratios on a common time index.

    Parameters:
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)} from
            src.moduair_loader.load_fleet_bins().

    Returns:
        DataFrame indexed by datetime with columns ratio_813_bin{N},
        ratio_943_bin{N}, and ratio_mean_bin{N} for N in 0..N_BINS-1.
    """
    if TARGET_SN not in fleet:
        raise ValueError(f"Target sensor {TARGET_SN} has no data in the fleet.")
    for ref in REFERENCE_SNS:
        if ref not in fleet:
            raise ValueError(f"Reference sensor {ref} has no data in the fleet.")

    target = fleet[TARGET_SN].set_index("datetime")[BIN_COLUMNS]
    references = {
        ref: fleet[ref].set_index("datetime")[BIN_COLUMNS] for ref in REFERENCE_SNS
    }

    # Sensors that form the "others" average for the mean ratio.
    other_ids = [
        sid for sid in fleet
        if sid != TARGET_SN and sid not in EXCLUDE_FROM_AVERAGE
    ]
    print(f"  Direct ratio references: {', '.join(REFERENCE_SNS)}")
    print(f"  Mean ratio average over {len(other_ids)} sensors: {', '.join(other_ids)}")

    # Average the "others" per bin across sensors on the shared 1-min index.
    others_stack = pd.concat(
        [fleet[sid].set_index("datetime")[BIN_COLUMNS] for sid in other_ids],
        axis=1,
        keys=other_ids,
    )

    out = pd.DataFrame(index=target.index)
    for i in range(N_BINS):
        col = f"opc_bin{i}"

        # Direct ratios: 195 / 813 and 195 / 943
        for ref in REFERENCE_SNS:
            ref_aligned = references[ref][col].reindex(target.index)
            out[f"ratio_{ref[-3:]}_bin{i}"] = (
                target[col] / ref_aligned.where(ref_aligned != 0)
            )

        # Mean ratio: 195 / mean(others), averaging across the sensor level
        others_bin = others_stack.xs(col, axis=1, level=1)
        others_mean = others_bin.mean(axis=1, skipna=True).reindex(target.index)
        out[f"ratio_mean_bin{i}"] = target[col] / others_mean.where(others_mean != 0)

    return out


def slice_window(ratios: pd.DataFrame, delta: timedelta) -> pd.DataFrame:
    """
    Return the trailing slice of ``ratios`` covering the last ``delta``.

    The window is anchored to the latest timestamp in the ratio index, so it
    tracks the most recent data regardless of any lag behind wall-clock time.

    Parameters:
        ratios: DataFrame from compute_ratios(), datetime-indexed.
        delta: Trailing window length.

    Returns:
        Row-subset of ``ratios`` with index in [latest - delta, latest].
    """
    if ratios.empty:
        return ratios
    latest = ratios.index.max()
    return ratios.loc[ratios.index >= (latest - delta)]


def plot_bin_ratios(ratios: pd.DataFrame, plot_dir: Path, window_label: str, date_fmt: str) -> None:
    """
    Plot all three ratios per bin (one figure per bin) for a single window.

    Parameters:
        ratios: Windowed DataFrame from slice_window().
        plot_dir: Directory to write the per-bin figures into.
        window_label: Human-readable window name for figure titles.
        date_fmt: strftime format for x-axis major tick labels.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    times = ratios.index

    for i in range(N_BINS):
        bin_name = PARTICLE_BINS[i]["name"]
        fig, ax = create_figure(figsize=(11, 4))

        for spec in RATIO_SPECS:
            ax.plot(
                times,
                ratios[f"ratio_{spec['key']}_bin{i}"],
                color=spec["color"],
                linewidth=1.2,
                label=spec["label"],
            )

        # Reference line at 1.0 (perfect agreement)
        ax.axhline(1.0, color=COLORS["grid"], linewidth=1.0, linestyle="--", zorder=0)

        ax.set_title(
            f"MODULAIR-PM correction factor, bin {i} ({bin_name} µm) — {window_label}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Concentration ratio")
        ax.legend(loc="upper right")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        fig.autofmt_xdate()

        out_path = plot_dir / f"correction_factor_bin{i}.png"
        save_figure(fig, out_path)
        print(f"    Saved {out_path.name}")


def summarize_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-bin, per-ratio summary table (mean, median, count).

    Parameters:
        ratios: Windowed DataFrame from slice_window().

    Returns:
        Long-format DataFrame with columns bin, ratio, mean, median, count.
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
                    "median": series.median(),
                    "count": int(series.count()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MODULAIR-PM inter-sensor correction factor (bin-wise ratios)."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive start datetime.")
    parser.add_argument("--end", default=None, help="Inclusive end datetime.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else None

    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"
    plot_root = output_dir / "plots" / "moduair_correction"

    print("\n" + "=" * 70)
    print("MODULAIR-PM Correction Factor Analysis")
    print("=" * 70)
    print(f"Start: {start}  End: {end if end else 'latest available'}")

    available = list_available_sensors("raw")
    print(f"\nAvailable raw sensors: {', '.join(available)}")

    print("\nLoading fleet data...")
    fleet = load_fleet_bins(available, start=start, end=end)

    print("\nComputing ratios...")
    ratios = compute_ratios(fleet)
    if ratios.empty:
        print("No overlapping ratio data; nothing to do.")
        return

    latest = ratios.index.max()
    print(f"  Latest timestamp: {latest}")

    for win_key, win in WINDOWS.items():
        print(f"\n[{win['label']}] plots and tables...")
        windowed = slice_window(ratios, win["delta"])
        if windowed.empty:
            print("  No data in this window; skipping.")
            continue

        plot_bin_ratios(
            windowed,
            plot_root / win_key,
            win["label"],
            win["date_fmt"],
        )

        ratios_path = output_dir / f"moduair_correction_factor_ratios_{win_key}.csv"
        windowed.reset_index().to_csv(ratios_path, index=False)
        print(f"  Saved {ratios_path.name}")

        summary_path = output_dir / f"moduair_correction_factor_summary_{win_key}.csv"
        summarize_ratios(windowed).to_csv(summary_path, index=False)
        print(f"  Saved {summary_path.name}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
