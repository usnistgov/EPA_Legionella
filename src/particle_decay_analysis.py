#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Decay & Emission Analysis
===================================

This script analyzes particle concentration decay data from QuantAQ MODULAIR-PM
sensors to calculate particle penetration factors, deposition rates, and shower
emission rates for the EPA Legionella study. The analysis uses a numerical
approach to solve the mass balance equation for seven particle size bins.

Particle size bins analyzed (um):
    - Bin 0: 0.35-0.46
    - Bin 1: 0.46-0.66
    - Bin 2: 0.66-1.0
    - Bin 3: 1.0-1.3
    - Bin 4: 1.3-1.7
    - Bin 5: 1.7-2.3
    - Bin 6: 2.3-3.0

Key Metrics Calculated:
    - p: Particle penetration factor (dimensionless, 0-1 range)
    - beta_deposition: Effective deposition loss rate (h-1)
    - E: Shower emission rate (particles/minute)
    - lambda: Air change rate from CO2 analysis (h-1)

Analysis Features:
    - Numerical solution of time-dependent mass balance equation
    - Integration with CO2-derived air change rates
    - Per-bin analysis for size-dependent behavior
    - Statistical summaries across all shower events
    - Comprehensive visualization of decay curves and emissions

Methodology:
    The mass balance equation for indoor particle concentration:
        V dC/dt = pQC_out - QC - beta_deposition CV + E
        dC/dt = p*lambda*C_out - lambda*C - beta_deposition*C + E/V

    1. Calculate penetration factor (p):
       - Use two averaging windows around each shower event (before and after)
       - For Night events:
           Before: 9pm (day before) to 2am (day of)
           After:  9am (day of) to 2pm (day of)
       - For Day events:
           Before: 9am (day of) to 2pm (day of)
           After:  9pm (day of) to 2am (next day)
       - p = C_inside / C_outside (averaged over each window, zeros excluded)
       - Final p = average of before and after window p values
       - Allowable range: 0-1 (values > 1 are capped at 1)

    2. Obtain air change rate (lambda):
       - Load from CO2 decay analysis results
       - Units: h-1

    3. Calculate deposition rate (beta_deposition) when E=0:
       - Use 2-hour window after shower ends
       - Start time from peak concentration within the window to end of window
       - Solve numerically for each time step:
         beta_deposition = 1/dt - lambda - C_t(i+1)/(C_t dt) + (p*lambda*C_out,t)/C_t
       - Average beta over the window

    4. Calculate emission rate (E) from shower start to peak concentration:
       - Use shower ON to peak concentration time within analysis window
       - Solve numerically for each time step:
         E = p*lambda*V*C_out,t + V(C_t - C_t(i+1))/dt - lambda*V*C_t - beta*V*C_t
       - Average E over the shower-to-peak period

    5. Predict concentration Ct using forward Euler simulation:
       - Window: shower ON to 2 hours after shower OFF
       - C_t(i+1) = C_t + dt[p*lambda*C_out,t - C_t(lambda + beta) + E/V]
       - E = E_mean from shower ON to peak, then E = 0
       - C_0 = measured bin concentration at shower ON
       - Plot predicted Ct on particle decay figures

Output Files:
    - particle_analysis_summary.xlsx: Multi-sheet workbook with:
        * p_penetration: Penetration factors per event and bin
        * beta_deposition: Deposition rates per event and bin
        * E_emission: Emission rates per event and bin
        * overall_summary: Aggregated statistics
    - plots/event_XX_bin_Y_decay.png: Individual decay curves
    - plots/penetration_summary.png: Summary of p values
    - plots/deposition_summary.png: Summary of beta values
    - plots/emission_summary.png: Summary of E values

Module Structure:
    - particle_calculations.py: Pure computation functions (p, beta, E, Ct)
    - particle_data_loader.py: Data loading and event identification
    - particle_decay_analysis.py: Orchestration and main pipeline (this file)

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_manager import (  # noqa: E402
    is_event_excluded,
    process_events_with_management,
)
from src.data_paths import get_data_root  # noqa: E402
from src.particle_calculations import (  # noqa: E402
    BEDROOM_VOLUME_M3,
    DEPOSITION_WINDOW_HOURS,
    MAX_DEPOSITION_RATE,
    MIN_CONCENTRATION_RATIO,
    MIN_POINTS_DEPOSITION,
    MIN_POINTS_EMISSION,
    MIN_POINTS_PENETRATION,
    PARTICLE_BINS,
    TIME_STEP_MINUTES,
    calculate_ct_prediction,
    calculate_deposition_rate,
    calculate_emission_rate,
    calculate_penetration_factor,
)
from src.particle_data_loader import (  # noqa: E402
    get_events_from_registry,
    identify_shower_events,
    load_and_merge_quantaq_data,
    load_co2_lambda_results,
    load_shower_log,
)


# =============================================================================
# Event Analysis Orchestration
# =============================================================================


def analyze_event_all_bins(
    particle_data: pd.DataFrame,
    event: Dict,
    lambda_ach: float,
) -> Dict:
    """
    Analyze all particle bins for a single shower event.

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        event (Dict): Event timing information
        lambda_ach (float): Air change rate (h-1)

    Returns:
        Dict: Results for all bins
    """
    results = {
        "event_number": event.get("event_number", 0),
        "test_name": event.get("test_name", ""),
        "config_key": event.get("config_key", ""),
        "water_temp": event.get("water_temp", ""),
        "door_position": event.get("door_position", ""),
        "planned_fan": event.get("planned_fan", ""),
        "time_of_day": event.get("time_of_day", ""),
        "fan_during_test": event.get("fan_during_test", False),
        "replicate_num": event.get("replicate_num", 0),
        "shower_on": event["shower_on"],
        "shower_off": event["shower_off"],
        "shower_duration_min": event.get(
            "shower_duration_min", event.get("duration_min", 0)
        ),
        "lambda_ach": lambda_ach,
        "co2_event_idx": event.get("co2_event_idx", None),
    }

    time_of_day = event.get("time_of_day", "")

    for bin_num in PARTICLE_BINS.keys():
        # Calculate penetration factor using before/after windows
        p_result = calculate_penetration_factor(
            particle_data,
            event["shower_on"],
            time_of_day,
            bin_num,
        )

        results[f"bin{bin_num}_p_mean"] = p_result.get("p_mean", np.nan)
        results[f"bin{bin_num}_p_std"] = p_result.get("p_std", np.nan)

        # Skip further calculations if p is invalid
        if np.isnan(p_result.get("p_mean", np.nan)):
            results[f"bin{bin_num}_beta_mean"] = np.nan
            results[f"bin{bin_num}_beta_std"] = np.nan
            results[f"bin{bin_num}_beta_r_squared"] = np.nan
            results[f"bin{bin_num}_beta_fit"] = np.nan
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = p_result.get(
                "skip_reason", "Unknown"
            )
            # Store empty fit data for plotting
            results[f"bin{bin_num}_fit_t_values"] = []
            results[f"bin{bin_num}_fit_y_values"] = []
            results[f"bin{bin_num}_fit_slope"] = np.nan
            results[f"bin{bin_num}_fit_intercept"] = 0.0
            results[f"bin{bin_num}_c_steady_state"] = np.nan
            results[f"bin{bin_num}_peak_time"] = None
            continue

        p_mean = p_result["p_mean"]

        # Calculate deposition rate
        beta_result = calculate_deposition_rate(
            particle_data,
            event["deposition_start"],
            event["deposition_end"],
            bin_num,
            p_mean,
            lambda_ach,
        )

        results[f"bin{bin_num}_beta_mean"] = beta_result.get("beta_mean", np.nan)
        results[f"bin{bin_num}_beta_std"] = beta_result.get("beta_std", np.nan)
        results[f"bin{bin_num}_beta_r_squared"] = beta_result.get(
            "beta_r_squared", np.nan
        )
        results[f"bin{bin_num}_beta_fit"] = beta_result.get(
            "beta_fit", np.nan
        )  # From linearized regression

        # Store fit data for plotting (even if beta is valid, we want the data)
        results[f"bin{bin_num}_fit_t_values"] = beta_result.get("_t_values", [])
        results[f"bin{bin_num}_fit_y_values"] = beta_result.get("_y_values", [])
        results[f"bin{bin_num}_fit_slope"] = beta_result.get(
            "_fit_slope", np.nan
        )  # Actual regression slope
        results[f"bin{bin_num}_fit_intercept"] = beta_result.get(
            "_fit_intercept", 0.0
        )  # Regression intercept
        results[f"bin{bin_num}_c_steady_state"] = beta_result.get(
            "c_steady_state", np.nan
        )
        results[f"bin{bin_num}_peak_time"] = beta_result.get("peak_time", None)

        # Skip emission calculation if beta is invalid
        if np.isnan(beta_result.get("beta_mean", np.nan)):
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = beta_result.get(
                "skip_reason", "Unknown"
            )
            continue

        beta_mean = beta_result["beta_mean"]

        # Use peak_time from deposition calculation as E window endpoint
        peak_time = beta_result.get("peak_time")
        if peak_time is None:
            peak_time = event["shower_off"]

        # Calculate emission rate (shower_on to peak_time)
        E_result = calculate_emission_rate(
            particle_data,
            event["shower_on"],
            peak_time,
            bin_num,
            p_mean,
            lambda_ach,
            beta_mean,
        )

        results[f"bin{bin_num}_E_mean"] = E_result.get("E_mean", np.nan)
        results[f"bin{bin_num}_E_std"] = E_result.get("E_std", np.nan)
        results[f"bin{bin_num}_E_total"] = E_result.get("E_total", np.nan)
        results[f"bin{bin_num}_skip_reason"] = E_result.get("skip_reason", None)

        # Calculate Ct prediction (forward Euler simulation)
        E_mean_val = E_result.get("E_mean", np.nan)
        if not np.isnan(E_mean_val) and not np.isnan(beta_mean):
            ct_result = calculate_ct_prediction(
                particle_data,
                event["shower_on"],
                event["shower_off"],
                event["deposition_end"],
                bin_num,
                p_mean,
                lambda_ach,
                beta_mean,
                E_mean_val,
                peak_time,
            )
            results[f"bin{bin_num}_ct_datetimes"] = ct_result.get("datetimes", [])
            results[f"bin{bin_num}_ct_predicted"] = ct_result.get("predicted_ct", [])
            results[f"bin{bin_num}_decay_datetimes"] = ct_result.get("decay_datetimes", [])
            results[f"bin{bin_num}_decay_predicted"] = ct_result.get("decay_predicted", [])
        else:
            results[f"bin{bin_num}_ct_datetimes"] = []
            results[f"bin{bin_num}_ct_predicted"] = []
            results[f"bin{bin_num}_decay_datetimes"] = []
            results[f"bin{bin_num}_decay_predicted"] = []

    return results


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_particle_analysis(
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
) -> pd.DataFrame:
    """
    Run the complete particle decay and emission analysis.

    Parameters:
        output_dir (Path): Optional output directory (defaults to data_root/output)
        generate_plots (bool): If True, generate plots for each event and summary

    Returns:
        pd.DataFrame: DataFrame with analysis results for all events and bins
    """
    print("=" * 80)
    print("Particle Decay & Emission Analysis")
    print("Numerical Approach - Seven Particle Size Bins")
    print("=" * 80)
    print(f"Bedroom volume: {BEDROOM_VOLUME_M3} m^3")
    print(f"Time step: {TIME_STEP_MINUTES} minute(s)")
    print("Penetration factor: averaged before/after windows (p capped at 1)")
    print(f"Deposition window: {DEPOSITION_WINDOW_HOURS} hour(s) after shower")
    print("\nValidation thresholds:")
    print(f"  Max deposition rate (beta): {MAX_DEPOSITION_RATE} h^-1")
    print(f"  Min concentration ratio: {MIN_CONCENTRATION_RATIO}")
    print(
        f"  Min data points: p={MIN_POINTS_PENETRATION}, beta={MIN_POINTS_DEPOSITION}, E={MIN_POINTS_EMISSION}"
    )

    # Set output directory
    if output_dir is None:
        output_dir = get_data_root() / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load particle data
    particle_data = load_and_merge_quantaq_data()

    # Try to load events from unified registry first (for consistent numbering)
    events, co2_results, used_registry = get_events_from_registry(output_dir)

    if used_registry:
        print("  Using unified event registry for consistent event numbering")
    else:
        # Fall back to existing event management system
        print("\nNote: Registry not found. Using process_events_with_management().")
        print("  Run 'python scripts/event_registry.py' for unified numbering.\n")

        # Load shower log and identify events
        print("Loading shower log...")
        shower_log = load_shower_log()
        raw_events = identify_shower_events(shower_log)
        print(f"Found {len(raw_events)} raw shower events")

        # Load CO2 lambda results
        co2_results = load_co2_lambda_results()

        # Process events using the enhanced event management system
        print("\nProcessing events with event management system...")
        events, co2_events_processed, event_log = process_events_with_management(
            raw_events,
            [],  # CO2 events (will be loaded from co2_results)
            shower_log,
            co2_results,
            output_dir,
            create_synthetic=False,
        )

    # Print event matching summary
    print("\nEvent Matching Summary:")
    matched_count = 0
    excluded_count = 0
    missing_lambda_count = 0

    for event in events:
        shower_time = event["shower_on"]

        # Check if excluded
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if is_excluded_flag:
            excluded_count += 1
            print(
                f"  Event {event.get('event_number', '?')} "
                f"({shower_time.strftime('%Y-%m-%d %H:%M')}): "
                f"EXCLUDED - {exclusion_reason}"
            )
            continue

        # Check if has lambda value
        lambda_val = event.get("lambda_ach", np.nan)
        if not np.isnan(lambda_val):
            matched_count += 1
            co2_idx = event.get("co2_event_idx")
            if co2_idx is not None and co2_idx < len(co2_results):
                co2_time = co2_results.iloc[co2_idx]["injection_start"]
                print(
                    f"  {event.get('test_name', 'Event ' + str(event.get('event_number', '?')))} "
                    f"({shower_time.strftime('%m/%d %H:%M')}) "
                    f"-> CO2 {co2_idx + 1} ({co2_time.strftime('%H:%M')}), "
                    f"lambda={lambda_val:.4f} h^-1"
                )
        else:
            missing_lambda_count += 1
            print(
                f"  {event.get('test_name', 'Event ' + str(event.get('event_number', '?')))} "
                f"({shower_time.strftime('%m/%d %H:%M')}): "
                f"No lambda value available"
            )

    print(
        f"\nTotal: {len(events)} events | Matched: {matched_count} | "
        f"Excluded: {excluded_count} | Missing lambda: {missing_lambda_count}"
    )

    # Analyze each event
    print("\nAnalyzing shower events...")
    results = []

    # Setup plot directory
    plot_dir = output_dir / "plots"
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)

    for event in events:
        event_num = event.get("event_number", 0)
        test_name = event.get("test_name", f"Event_{event_num}")
        shower_time = event["shower_on"]
        lambda_ach = event.get("lambda_ach", np.nan)

        # Skip excluded events
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if is_excluded_flag:
            print(f"  {test_name}: Skipped (excluded: {exclusion_reason})")
            continue

        # Skip events without lambda
        if np.isnan(lambda_ach):
            print(f"  {test_name}: Skipped (no lambda from CO2 analysis)")
            continue

        print(
            f"  {test_name} ({shower_time.strftime('%m/%d %H:%M')}): "
            f"lambda={lambda_ach:.4f} h^-1"
        )

        result = analyze_event_all_bins(particle_data, event, lambda_ach)
        results.append(result)

        # Print summary for this event with detailed skip reasons
        valid_bins = 0
        skipped_bins = []
        for bin_num in PARTICLE_BINS.keys():
            if not np.isnan(result.get(f"bin{bin_num}_E_mean", np.nan)):
                valid_bins += 1
            else:
                skip_reason = result.get(f"bin{bin_num}_skip_reason", "Unknown")
                skipped_bins.append((bin_num, skip_reason))

        print(f"    Successfully analyzed {valid_bins}/{len(PARTICLE_BINS)} bins")

        # Print skip reasons for failed bins (up to 3 for brevity)
        if skipped_bins and valid_bins < len(PARTICLE_BINS):
            for bin_num, reason in skipped_bins[:3]:
                bin_name = PARTICLE_BINS[bin_num]["name"]
                # Truncate long reasons
                if len(reason) > 80:
                    reason = reason[:77] + "..."
                print(f"      Bin {bin_num} ({bin_name} um): {reason}")
            if len(skipped_bins) > 3:
                print(f"      ... and {len(skipped_bins) - 3} more bins skipped")

        # Generate individual event plot if enabled (all bins on one plot)
        if generate_plots and valid_bins > 0:
            try:
                from scripts.plot_particle import plot_particle_decay_event
                from scripts.plot_style import format_test_name_for_filename

                # Format filename: event_01-0114_hw_morning_pm_decay.png
                formatted_name = format_test_name_for_filename(test_name)
                plot_path = (
                    plot_dir / f"event_{event_num:02d}-{formatted_name}_pm_decay.png"
                )
                plot_particle_decay_event(
                    particle_data=particle_data,
                    event=event,
                    particle_bins=PARTICLE_BINS,
                    result=result,
                    output_path=plot_path,
                    event_number=event_num,
                    test_name=test_name,
                )
            except ImportError:
                pass  # Already warned about missing plot module
            except Exception as e:
                print(f"    Warning: Failed to generate plot for {test_name}: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print overall statistics
    _print_overall_summary(results_df, results)

    # Save results
    _save_results(results_df, output_dir)

    # Generate summary plots
    if generate_plots and not results_df.empty:
        _generate_summary_plots(results_df, output_dir)
    elif generate_plots and results_df.empty:
        print("\nSkipping plot generation - no results to plot.")

    return results_df


def _print_overall_summary(results_df: pd.DataFrame, results: list) -> None:
    """Print overall statistics summary to console."""
    print("\n" + "=" * 80)
    print("Overall Results Summary")
    print("=" * 80)

    if results_df.empty:
        print(
            "\nNo events were analyzed (all skipped due to missing lambda or exclusions)."
        )
        return

    for bin_num, bin_info in PARTICLE_BINS.items():
        bin_name = bin_info["name"]
        p_col = f"bin{bin_num}_p_mean"
        beta_col = f"bin{bin_num}_beta_mean"
        E_col = f"bin{bin_num}_E_mean"

        valid_p = results_df[p_col].dropna()
        valid_beta = results_df[beta_col].dropna()
        valid_E = results_df[E_col].dropna()

        print(f"\nBin {bin_num} ({bin_name} um):")
        if len(valid_p) > 0:
            print(
                f"  p (penetration):     {valid_p.mean():.3f} +/- {valid_p.std():.3f}"
            )
        if len(valid_beta) > 0:
            print(
                f"  beta (deposition):   {valid_beta.mean():.3f} +/- {valid_beta.std():.3f} h^-1"
            )
        if len(valid_E) > 0:
            print(
                f"  E (emission):        {valid_E.mean():.2e} +/- {valid_E.std():.2e} #/min"
            )
        print(f"  Valid events:        {len(valid_E)}/{len(results)}")


def _save_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save analysis results to Excel workbook."""
    output_file = output_dir / "particle_analysis_summary.xlsx"

    if results_df.empty:
        print(f"\nNo results to save - skipping {output_file}")
        return

    # Create column rename mapping for units
    column_rename = {
        "shower_duration_min": "shower_duration (min)",
        "lambda_ach": "lambda_ach (h-1)",
    }
    for bin_num in PARTICLE_BINS.keys():
        column_rename[f"bin{bin_num}_p_mean"] = f"bin{bin_num}_p_mean (-)"
        column_rename[f"bin{bin_num}_p_std"] = f"bin{bin_num}_p_std (-)"
        column_rename[f"bin{bin_num}_beta_mean"] = f"bin{bin_num}_beta_mean (h-1)"
        column_rename[f"bin{bin_num}_beta_std"] = f"bin{bin_num}_beta_std (h-1)"
        column_rename[f"bin{bin_num}_beta_fit"] = f"bin{bin_num}_beta_fit (h-1)"
        column_rename[f"bin{bin_num}_E_mean"] = f"bin{bin_num}_E_mean (#/min)"
        column_rename[f"bin{bin_num}_E_std"] = f"bin{bin_num}_E_std (#/min)"
        column_rename[f"bin{bin_num}_E_total"] = f"bin{bin_num}_E_total (#)"

    results_df_export = results_df.rename(columns=column_rename)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Main results
        results_df_export.to_excel(writer, sheet_name="all_results", index=False)

        # Separate sheets for each metric
        p_cols = ["event_number", "shower_on"] + [
            f"bin{i}_p_mean (-)" for i in PARTICLE_BINS.keys()
        ]
        beta_cols = ["event_number", "shower_on"] + [
            f"bin{i}_beta_mean (h-1)" for i in PARTICLE_BINS.keys()
        ]
        beta_r2_cols = ["event_number", "shower_on"] + [
            f"bin{i}_beta_r_squared" for i in PARTICLE_BINS.keys()
        ]
        E_cols = ["event_number", "shower_on"] + [
            f"bin{i}_E_mean (#/min)" for i in PARTICLE_BINS.keys()
        ]

        results_df_export[p_cols].to_excel(
            writer, sheet_name="p_penetration", index=False
        )
        results_df_export[beta_cols].to_excel(
            writer, sheet_name="beta_deposition", index=False
        )
        results_df_export[beta_r2_cols].to_excel(
            writer, sheet_name="beta_r_squared", index=False
        )
        results_df_export[E_cols].to_excel(
            writer, sheet_name="E_emission", index=False
        )

    print(f"\nResults saved to: {output_file}")


def _generate_summary_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary plots for penetration, deposition, and emission."""
    print("\nGenerating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    try:
        from scripts.plot_particle import (
            plot_deposition_summary,
            plot_emission_summary,
            plot_penetration_summary,
        )
    except ImportError:
        print("  Warning: plot_particle module not found. Skipping plots.")
        return

    for plot_func, filename in [
        (plot_penetration_summary, "penetration_summary.png"),
        (plot_deposition_summary, "deposition_summary.png"),
        (plot_emission_summary, "emission_summary.png"),
    ]:
        try:
            plot_func(results_df, PARTICLE_BINS, plot_dir / filename)
            print(f"  Generated: {filename}")
        except Exception as e:
            print(f"  Error generating {filename}: {e}")

    print(f"  Plots saved to: {plot_dir}")


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Particle Decay & Emission Analysis")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data_root/output)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_particle_analysis(
        output_dir=output_dir,
        generate_plots=not args.no_plot,
    )


if __name__ == "__main__":
    main()
