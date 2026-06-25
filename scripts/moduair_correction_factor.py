#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Inter-Sensor Correction Factor (Concentration Ratios)
=================================================================

Computes and plots candidate correction factors for the inside MODULAIR-PM
sensor (MOD-PM-00195) by taking bin-wise concentration ratios against other
sensors in the IAQ&V fleet, using the raw analysis bins (opc_bin0-opc_bin11).

Two ratios are produced, per particle-size bin, on a shared time base:

    Ratio A:  195 / 402
        Direct ratio against a single reference sensor (MOD-PM-00402).

    Ratio B:  195 / mean(other sensors)
        Ratio against the average of all other fleet sensors, excluding the
        reference (402) and the original Gaithersburg inside/outside pair
        (785, 555). The averaging set is every remaining sensor with data.

A value near 1.0 means the inside sensor agrees with the reference; a sustained
offset is a candidate multiplicative correction factor.

One figure is produced per bin (0-11). Both ratios are drawn on the same axes
as two time series so they can be compared directly.

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
    <output>/plots/moduair_correction/correction_factor_bin{N}.png  (N = 0..11)
    <output>/moduair_correction_factor_ratios.csv  (long-format ratios)

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version.
"""

import argparse
import sys
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

# Sensor of interest (inside) and the single reference sensor for Ratio A.
TARGET_SN = "00195"
REFERENCE_SN = "00402"

# Sensors excluded from the Ratio B "other sensors" average:
# the reference (402) and the original Gaithersburg pair (785, 555).
EXCLUDE_FROM_AVERAGE = {"00402", "00785", "00555"}

DEFAULT_START = "2026-06-04 00:00:00"

# Line colors for the two ratios (single source of truth: plot_style COLORS)
COLOR_RATIO_A = COLORS["lambda"]   # red:   195 / 402
COLOR_RATIO_B = COLORS["bedroom"]  # blue:  195 / mean(others)


def compute_ratios(
    fleet: dict,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of both bin-wise ratios on a common time index.

    Parameters:
        fleet: Dict of {sensor_id: DataFrame(datetime + opc_bin0..11)} from
            src.moduair_loader.load_fleet_bins().

    Returns:
        DataFrame indexed by datetime with columns
        ratioA_bin{N} and ratioB_bin{N} for N in 0..11.
    """
    if TARGET_SN not in fleet:
        raise ValueError(f"Target sensor {TARGET_SN} has no data in the fleet.")
    if REFERENCE_SN not in fleet:
        raise ValueError(f"Reference sensor {REFERENCE_SN} has no data in the fleet.")

    target = fleet[TARGET_SN].set_index("datetime")[BIN_COLUMNS]
    reference = fleet[REFERENCE_SN].set_index("datetime")[BIN_COLUMNS]

    # Sensors that form the "others" average for Ratio B
    other_ids = [
        sid for sid in fleet
        if sid != TARGET_SN and sid not in EXCLUDE_FROM_AVERAGE
    ]
    print(f"  Ratio A reference: {REFERENCE_SN}")
    print(f"  Ratio B average over {len(other_ids)} sensors: {', '.join(other_ids)}")

    # Average the "others" per bin across sensors on the shared 1-min index.
    # Concatenate along a new axis and take the mean ignoring NaNs.
    others_stack = pd.concat(
        [fleet[sid].set_index("datetime")[BIN_COLUMNS] for sid in other_ids],
        axis=1,
        keys=other_ids,
    )

    out = pd.DataFrame(index=target.index)
    for i in range(N_BINS):
        col = f"opc_bin{i}"
        # Ratio A: 195 / 402
        ref_aligned = reference[col].reindex(target.index)
        ratio_a = target[col] / ref_aligned.where(ref_aligned != 0)

        # Ratio B: 195 / mean(others), averaging across the sensor level
        others_bin = others_stack.xs(col, axis=1, level=1)
        others_mean = others_bin.mean(axis=1, skipna=True).reindex(target.index)
        ratio_b = target[col] / others_mean.where(others_mean != 0)

        out[f"ratioA_bin{i}"] = ratio_a
        out[f"ratioB_bin{i}"] = ratio_b

    return out


def plot_bin_ratios(ratios: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot both ratios per bin (one figure per bin) and save to disk.

    Parameters:
        ratios: DataFrame from compute_ratios().
        output_dir: Analysis output directory (figures go to
            output_dir/plots/moduair_correction/).
    """
    plot_dir = output_dir / "plots" / "moduair_correction"
    plot_dir.mkdir(parents=True, exist_ok=True)

    times = ratios.index

    for i in range(N_BINS):
        bin_name = PARTICLE_BINS[i]["name"]
        fig, ax = create_figure(figsize=(11, 4))

        ax.plot(
            times,
            ratios[f"ratioA_bin{i}"],
            color=COLOR_RATIO_A,
            linewidth=1.2,
            label=f"{TARGET_SN[-3:]} / {REFERENCE_SN[-3:]}",
        )
        ax.plot(
            times,
            ratios[f"ratioB_bin{i}"],
            color=COLOR_RATIO_B,
            linewidth=1.2,
            label=f"{TARGET_SN[-3:]} / mean(others)",
        )

        # Reference line at 1.0 (perfect agreement)
        ax.axhline(1.0, color=COLORS["grid"], linewidth=1.0, linestyle="--", zorder=0)

        ax.set_title(f"MODULAIR-PM correction factor, bin {i} ({bin_name} µm)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Concentration ratio")
        ax.legend(loc="upper right")

        # Multi-day span: let matplotlib auto-locate day ticks.
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.autofmt_xdate()

        out_path = plot_dir / f"correction_factor_bin{i}.png"
        save_figure(fig, out_path)
        print(f"  Saved {out_path.name}")


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

    print("\nPlotting per-bin figures...")
    plot_bin_ratios(ratios, output_dir)

    # Save the underlying ratio data (long format) for reference
    csv_path = output_dir / "moduair_correction_factor_ratios.csv"
    ratios.reset_index().to_csv(csv_path, index=False)
    print(f"\nSaved ratio data: {csv_path}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
