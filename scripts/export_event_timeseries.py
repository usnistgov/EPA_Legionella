#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Event Time-Series to Excel
==================================

One-off utility script that runs the particle decay analysis for a single
event and writes the predicted per-step emission rate (E_t) and predicted
indoor concentration (Ct) for all analysed size bins to a time-indexed
Excel workbook.

Intended use: provide time-series data to collaborators who want the
model predictions without regenerating all figures.

Output Excel workbook contains two sheets:
    - predicted_ct:  Timestamp | bin0_Ct (#/cm³) | bin1_Ct | … | bin11_Ct
    - predicted_et:  Timestamp | bin0_Et (#/cm³·min) | bin1_Et | … | bin11_Et

Usage
-----
    # Export event 101 using defaults:
    python scripts/export_event_timeseries.py --event 101

    # Specify output path explicitly:
    python scripts/export_event_timeseries.py --event 101 --output my_output.xlsx

    # Skip sig-fig rounding:
    python scripts/export_event_timeseries.py --event 101 --no-sig-figs

Arguments
---------
    --event INT       Event number to export (required).
    --output PATH     Path for the output Excel file.  Defaults to
                      <output_dir>/event_{NN}_timeseries.xlsx.
    --output-dir PATH Analysis output directory (default: data_root/output).
    --no-sig-figs     Disable significant-figure rounding on exported values.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import argparse
import sys
import warnings
from pathlib import Path

# Ensure stdout/stderr use UTF-8 on Windows (log files default to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.sig_figs as sf  # noqa: E402
from src.data_paths import get_data_root  # noqa: E402
from src.particle_calculations import PARTICLE_BINS  # noqa: E402
from src.particle_data_loader import (  # noqa: E402
    get_events_from_registry,
    load_and_merge_quantaq_data,
    load_co2_lambda_results,
)


def _find_event(events: list, target_event_num: int) -> dict:
    """Return the event dict matching *target_event_num*, or raise ValueError."""
    for ev in events:
        if ev.get("event_number") == target_event_num:
            return ev
    available = sorted(
        [e.get("event_number") for e in events if e.get("event_number") is not None]
    )
    raise ValueError(
        f"Event {target_event_num} not found. Available events: {available}"
    )


def build_ct_dataframe(result: dict) -> pd.DataFrame:
    """
    Build a time-indexed DataFrame of predicted Ct (#/cm³) for all bins.

    Uses the continuous ct_datetimes / ct_predicted arrays produced by
    calculate_ct_prediction (emission phase + decay phase concatenated).

    Parameters
    ----------
    result : dict
        Output of analyze_event_all_bins for a single event.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp; one column per bin (bin0_Ct … binN_Ct).
    """
    frames = {}
    for bin_num in PARTICLE_BINS.keys():
        # Prefer the full ct arrays; fall back to emission+decay separately
        ct_dts = result.get(f"bin{bin_num}_ct_datetimes", [])
        ct_pred = result.get(f"bin{bin_num}_ct_predicted", [])

        if not ct_dts or not ct_pred:
            # Try reconstructing from emission + decay segments
            em_dts = result.get(f"bin{bin_num}_emission_datetimes", [])
            em_pred = result.get(f"bin{bin_num}_emission_predicted", [])
            dc_dts = result.get(f"bin{bin_num}_decay_datetimes", [])
            dc_pred = result.get(f"bin{bin_num}_decay_predicted", [])
            ct_dts = list(em_dts) + list(dc_dts)
            ct_pred = list(em_pred) + list(dc_pred)

        if ct_dts and ct_pred:
            ts = pd.to_datetime(ct_dts)
            vals = np.array(ct_pred, dtype=float)
            frames[f"bin{bin_num}_Ct"] = pd.Series(vals, index=ts)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df


def build_et_dataframe(result: dict) -> pd.DataFrame:
    """
    Build a time-indexed DataFrame of per-step emission rate E_t (#/cm³·min)
    for all bins.

    Parameters
    ----------
    result : dict
        Output of analyze_event_all_bins for a single event.

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp; one column per bin (bin0_Et … binN_Et).
    """
    frames = {}
    for bin_num in PARTICLE_BINS.keys():
        e_times = result.get(f"bin{bin_num}_E_times", [])
        e_per_step = result.get(f"bin{bin_num}_E_per_step", [])

        if e_times and e_per_step:
            ts = pd.to_datetime(e_times)
            vals = np.array(e_per_step, dtype=float)
            frames[f"bin{bin_num}_Et"] = pd.Series(vals, index=ts)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df


def export_event_timeseries(
    event_number: int,
    output_path: Path,
    output_dir: Path,
    apply_sig_figs: bool = True,
) -> None:
    """
    Run particle decay analysis for a single event and export time-series data.

    Parameters
    ----------
    event_number : int
        Event number to export.
    output_path : Path
        Destination Excel file path.
    output_dir : Path
        Analysis output directory (for loading lambda results).
    apply_sig_figs : bool
        Whether to apply significant-figure rounding to exported values.
    """
    sf.set_enabled(apply_sig_figs)

    print("=" * 60)
    print(f"Event Time-Series Export — Event {event_number:02d}")
    print("=" * 60)

    # ── Load events from registry ──────────────────────────────────────────
    print("\nLoading event registry...")
    events, used_registry = get_events_from_registry(output_dir)
    if not used_registry or not events:
        print("ERROR: Could not load event registry.")
        print("  Run 'python scripts/event_registry.py' first.")
        sys.exit(1)
    print(f"  Loaded {len(events)} events from registry.")

    # ── Find target event ─────────────────────────────────────────────────
    try:
        event = _find_event(events, event_number)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    test_name = event.get("test_name", f"event_{event_number:02d}")
    print(f"  Event: {test_name}")

    # ── Check excluded flag ────────────────────────────────────────────────
    if event.get("is_excluded", False):
        reason = event.get("exclusion_reason", "unknown")
        print(f"WARNING: Event {event_number} is excluded ({reason}).")
        print("  Proceeding with export anyway.")

    # ── Load lambda ────────────────────────────────────────────────────────
    lambda_ach = event.get("lambda_ach", np.nan)
    if np.isnan(lambda_ach):
        print("\nNote: lambda_ach not in registry; loading from CO2 results...")
        try:
            lambda_df = load_co2_lambda_results(output_dir)
            row = lambda_df[lambda_df["event_number"] == event_number]
            if not row.empty:
                lambda_ach = float(row["lambda_average_mean"].iloc[0])
                print(f"  Loaded λ = {lambda_ach:.4f} h⁻¹")
            else:
                print(f"  WARNING: No lambda found for event {event_number}.")
        except Exception as exc:
            print(f"  WARNING: Could not load lambda: {exc}")

    if np.isnan(lambda_ach):
        print("ERROR: Cannot export without a valid air-change rate (λ).")
        print("  Run 'python scripts/co2_decay_analysis.py' first.")
        sys.exit(1)

    # ── Load particle data ─────────────────────────────────────────────────
    print("\nLoading QuantAQ particle data...")
    particle_data = load_and_merge_quantaq_data(output_dir)
    if particle_data is None or particle_data.empty:
        print("ERROR: Could not load particle data.")
        sys.exit(1)
    print(f"  Loaded {len(particle_data):,} records.")

    # ── Run analysis for this event ────────────────────────────────────────
    from scripts.particle_decay_analysis import analyze_event_all_bins  # noqa: E402

    print(f"\nRunning particle decay analysis for event {event_number:02d}...")
    result = analyze_event_all_bins(particle_data, event, lambda_ach)

    # ── Build DataFrames ───────────────────────────────────────────────────
    ct_df = build_ct_dataframe(result)
    et_df = build_et_dataframe(result)

    if ct_df.empty and et_df.empty:
        print("WARNING: No predicted data available for this event.")
        print("  Check that p, beta, and E were computed successfully.")
        sys.exit(1)

    # ── Apply sig figs ─────────────────────────────────────────────────────
    if apply_sig_figs:
        ct_df = sf.apply_sig_figs_to_df(ct_df)
        et_df = sf.apply_sig_figs_to_df(et_df)

    # ── Write Excel ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting Excel workbook: {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not ct_df.empty:
            ct_out = ct_df.reset_index()
            ct_out.to_excel(writer, sheet_name="predicted_ct", index=False)
            print(f"  predicted_ct: {len(ct_out)} rows × {len(ct_out.columns)} columns")
        else:
            print("  predicted_ct: no data (skipping sheet)")

        if not et_df.empty:
            et_out = et_df.reset_index()
            et_out.to_excel(writer, sheet_name="predicted_et", index=False)
            print(f"  predicted_et: {len(et_out)} rows × {len(et_out.columns)} columns")
        else:
            print("  predicted_et: no data (skipping sheet)")

    print(f"\nDone. Output: {output_path}")

    # ── Summary of valid bins ──────────────────────────────────────────────
    valid_bins = [
        bn for bn in PARTICLE_BINS.keys()
        if not np.isnan(result.get(f"bin{bn}_E_mean", np.nan))
    ]
    print(
        f"\nValid bins: {len(valid_bins)}/{len(PARTICLE_BINS)} "
        f"({', '.join('Bin ' + str(b) for b in valid_bins)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export predicted emission (E_t) and concentration (Ct) "
        "time-series for a single particle decay event to Excel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--event",
        type=int,
        required=True,
        help="Event number to export (e.g. 101).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output Excel file.  Defaults to "
        "<output_dir>/event_NN_timeseries.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Analysis output directory (default: data_root/output).",
    )
    parser.add_argument(
        "--no-sig-figs",
        action="store_true",
        help="Disable significant-figure rounding on exported values.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"event_{args.event:02d}_timeseries.xlsx"

    export_event_timeseries(
        event_number=args.event,
        output_path=output_path,
        output_dir=output_dir,
        apply_sig_figs=not args.no_sig_figs,
    )


if __name__ == "__main__":
    main()
