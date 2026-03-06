#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Decay & Emission Analysis
===================================

This script analyzes particle concentration decay data from QuantAQ MODULAIR-PM
sensors to calculate particle penetration factors, other process rates, and shower
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
    - beta_other: Effective other process loss rate (h-1)
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

    3. Calculate other process rate (beta_other) when E=0:
       - Use 2-hour window after shower ends (DEPOSITION_WINDOW_HOURS)
       - Start time from peak concentration within the window to end of window
       - Solve numerically for each time step:
           beta = 1/dt - lambda - C_{t+1}/(C_t*dt) + p*lambda*(C_out,t/C_t)
       - Collect all estimates <= MAX_OTHER_PROCESS_RATE (no lower bound to
         avoid upward bias from excluding negative/noisy steps)
       - Apply 5th-95th percentile trim to remove extreme outliers symmetrically
       - If trimmed-mean beta < 0, return NaN (skip_reason set); do NOT clamp
         to 0, as a zero beta is physically different from insufficient data
       - Report mean beta over the trimmed set

    4. Calculate emission rate (E) from shower start to peak concentration:
       - Use shower ON to peak concentration time within analysis window
       - Solve numerically for E_t at each time step by rearranging the mass balance:
           (C_{t+1} - C_t)/dt = p*lambda*C_out,t - lambda*C_t - beta*C_t + E_t/V
           E_t/V = (C_{t+1} - C_t)/dt - p*lambda*C_out,t + lambda*C_t + beta*C_t
           E_t = V*(C_{t+1} - C_t)/dt - p*lambda*V*C_out,t + lambda*V*C_t + beta*V*C_t
       - Report E_mean and E_std from positive E_t values over the window
       - E_times and E_per_step (all steps including negative) stored for plotting

    5. Predict concentration Ct using forward Euler simulation:
       - Window: shower ON to 2 hours after shower OFF
       - Single continuous simulation using time-varying outdoor concentration:
           C_t(i+1) = C_t + dt*[p*lambda*C_out,t - C_t*(lambda + beta) + E_t/V]
       - E_t = E_mean from shower ON to peak_time, then E_t = 0
       - When E_mean is unavailable (emission calc failed), E_t = 0 throughout
         so a decay-only prediction is still generated for valid-beta bins
       - C_0 = measured bin concentration at shower ON
       - Returned as two continuous segments: emission phase (shower ON to peak)
         and decay phase (peak to deposition end); decay starts from predicted
         concentration at peak, not from the measured peak value
       - Plot both phases as a single continuous predicted Ct curve on figures
       - E_r_squared: R² of emission-phase forward Euler vs. measured concentration

    6. Calculate total emission (E_total) for each bin:
       - Area under the E_t vs. time curve using the trapezoidal rule:
           E_total = dt * sum[(E_t + E_t(i+1)) / 2]
       - Summed over all time steps from shower ON to peak concentration
       - Negative per-step contributions clipped to 0 before integration

Output Files:
    - particle_analysis_summary.xlsx: Multi-sheet workbook with:
        * all_results: Full results table (all metrics per event and bin)
        * p_penetration: Penetration factors per event and bin (includes test_name)
        * beta_other: Other process rates per event and bin (includes test_name)
        * beta_r_squared: R² of forward Euler decay simulation (includes test_name)
        * E_emission: Emission rates per event and bin (includes test_name)
        * E_total_particles: Total emitted particle counts (E_total) per bin (includes test_name)
        * E_r_squared: R² of forward Euler emission-phase simulation (includes test_name)
        * peak_comparison: Measured vs. predicted concentration at peak_time and
          deposition_end for each bin, with percent difference (wide format,
          one row per event)
    - plots/event_XX-YYYYYY_pm_decay.png: Individual event decay curves
      (two-panel): top panel shows measured concentrations and continuous
      predicted Ct (emission + decay phases) with decay R² in text box;
      bottom panel shows per-step E_t, E_mean dashed lines, and emission R²
      annotation; shower markers use dotted lines; E plot x-axis matches
      concentration panel; E plot y-axis clipped to 2nd-98th percentile
    - plots/penetration_summary.png: Summary of p values
    - plots/deposition_summary.png: Summary of beta values
    - plots/emission_summary.png: Summary of E values
    - plots/emission_etotal_boxplot.png: Box-and-whisker of E_total by water
      temperature configuration and particle size bin

Module Structure:
    - src/particle_calculations.py: Pure computation functions (p, beta, E, Ct)
    - src/particle_data_loader.py: Data loading and event identification
    - scripts/particle_decay_analysis.py: Orchestration and main pipeline (this file)

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

import src.sig_figs as sf  # noqa: E402
from src.event_manager import (  # noqa: E402
    is_event_excluded,
    process_events_with_management,
)
from src.data_paths import get_data_root, get_event_figures_dir  # noqa: E402
from src.particle_calculations import (  # noqa: E402
    BEDROOM_VOLUME_M3,
    DEPOSITION_WINDOW_HOURS,
    MAX_OTHER_PROCESS_RATE,
    MIN_POINTS_EMISSION,
    MIN_POINTS_OTHER_PROCESS,
    MIN_POINTS_PENETRATION,
    PARTICLE_BINS,
    TIME_STEP_MINUTES,
    calculate_ct_prediction,
    calculate_emission_rate,
    calculate_other_process_rate,
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

    For each bin the following are computed in order:
        1. Penetration factor (p) from before/after windows
        2. Other process rate (beta_other) using the numerical step-by-step approach
        3. Emission rate (E_mean, E_std, E_total) from shower_on to peak
        4. Forward Euler Ct prediction (emission + decay phases); decay-only
           prediction (E=0) is generated even when emission calc fails, so
           that a predicted curve is available for every bin with a valid beta

    Per-bin result keys include:
        bin{n}_p_mean, bin{n}_p_std
        bin{n}_beta_other, bin{n}_beta_other_raw_mean, bin{n}_beta_other_std, bin{n}_beta_other_r_squared
        bin{n}_E_mean, bin{n}_E_std, bin{n}_E_total
        bin{n}_emission_datetimes, bin{n}_emission_predicted  (shower→peak)
        bin{n}_decay_datetimes, bin{n}_decay_predicted        (peak→deposition_end)
        bin{n}_ct_datetimes, bin{n}_ct_predicted              (full window)
        bin{n}_skip_reason, bin{n}_c_steady_state, bin{n}_peak_time

    Parameters:
        particle_data (pd.DataFrame): DataFrame with particle concentrations
        event (Dict): Event timing information
        lambda_ach (float): Air change rate (h⁻¹)

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
            results[f"bin{bin_num}_beta_other"] = np.nan
            results[f"bin{bin_num}_beta_other_raw_mean"] = np.nan
            results[f"bin{bin_num}_beta_other_std"] = np.nan
            results[f"bin{bin_num}_beta_other_r_squared"] = np.nan
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_E_r_squared"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = p_result.get(
                "skip_reason", "Unknown"
            )
            results[f"bin{bin_num}_c_steady_state"] = np.nan
            results[f"bin{bin_num}_peak_time"] = None
            results[f"bin{bin_num}_ct_datetimes"] = []
            results[f"bin{bin_num}_ct_predicted"] = []
            results[f"bin{bin_num}_emission_datetimes"] = []
            results[f"bin{bin_num}_emission_predicted"] = []
            results[f"bin{bin_num}_decay_datetimes"] = []
            results[f"bin{bin_num}_decay_predicted"] = []
            results[f"bin{bin_num}_E_times"] = []
            results[f"bin{bin_num}_E_per_step"] = []
            results[f"bin{bin_num}_peak_measured"] = np.nan
            results[f"bin{bin_num}_peak_predicted"] = np.nan
            results[f"bin{bin_num}_deposition_end_measured"] = np.nan
            results[f"bin{bin_num}_deposition_end_predicted"] = np.nan
            continue

        p_mean = p_result["p_mean"]

        # Calculate other process rate
        beta_result = calculate_other_process_rate(
            particle_data,
            event["deposition_start"],
            event["deposition_end"],
            bin_num,
            p_mean,
            lambda_ach,
        )

        results[f"bin{bin_num}_beta_other"] = beta_result.get("beta", np.nan)
        results[f"bin{bin_num}_beta_other_raw_mean"] = beta_result.get(
            "beta_raw_mean", np.nan
        )
        results[f"bin{bin_num}_beta_other_std"] = beta_result.get("beta_std", np.nan)
        results[f"bin{bin_num}_beta_other_r_squared"] = beta_result.get(
            "beta_r_squared", np.nan
        )
        results[f"bin{bin_num}_c_steady_state"] = beta_result.get(
            "c_steady_state", np.nan
        )
        results[f"bin{bin_num}_peak_time"] = beta_result.get("peak_time", None)

        # Skip emission calculation if beta is invalid
        # (beta_raw_mean already stored above from beta_result)
        if np.isnan(beta_result.get("beta", np.nan)):
            results[f"bin{bin_num}_E_mean"] = np.nan
            results[f"bin{bin_num}_E_std"] = np.nan
            results[f"bin{bin_num}_E_total"] = np.nan
            results[f"bin{bin_num}_E_r_squared"] = np.nan
            results[f"bin{bin_num}_skip_reason"] = beta_result.get(
                "skip_reason", "Unknown"
            )
            results[f"bin{bin_num}_ct_datetimes"] = []
            results[f"bin{bin_num}_ct_predicted"] = []
            results[f"bin{bin_num}_emission_datetimes"] = []
            results[f"bin{bin_num}_emission_predicted"] = []
            results[f"bin{bin_num}_decay_datetimes"] = []
            results[f"bin{bin_num}_decay_predicted"] = []
            results[f"bin{bin_num}_E_times"] = []
            results[f"bin{bin_num}_E_per_step"] = []
            results[f"bin{bin_num}_peak_measured"] = np.nan
            results[f"bin{bin_num}_peak_predicted"] = np.nan
            results[f"bin{bin_num}_deposition_end_measured"] = np.nan
            results[f"bin{bin_num}_deposition_end_predicted"] = np.nan
            continue

        beta_val = beta_result["beta"]

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
            beta_val,
        )

        results[f"bin{bin_num}_E_mean"] = E_result.get("E_mean", np.nan)
        results[f"bin{bin_num}_E_std"] = E_result.get("E_std", np.nan)
        results[f"bin{bin_num}_E_total"] = E_result.get("E_total", np.nan)
        results[f"bin{bin_num}_skip_reason"] = E_result.get("skip_reason", None)
        results[f"bin{bin_num}_E_times"] = E_result.get("E_times", [])
        results[f"bin{bin_num}_E_per_step"] = E_result.get("E_per_step", [])

        # Calculate Ct prediction (forward Euler simulation).
        # Uses E_mean when available; falls back to E=0 (decay-only) so that
        # a predicted curve is always generated for any bin with a valid beta.
        E_mean_val = E_result.get("E_mean", np.nan)
        effective_E = E_mean_val if not np.isnan(E_mean_val) else 0.0
        ct_result = calculate_ct_prediction(
            particle_data,
            event["shower_on"],
            event["shower_off"],
            event["deposition_end"],
            bin_num,
            p_mean,
            lambda_ach,
            beta_val,
            effective_E,
            peak_time,
        )
        results[f"bin{bin_num}_ct_datetimes"] = ct_result.get("datetimes", [])
        results[f"bin{bin_num}_ct_predicted"] = ct_result.get("predicted_ct", [])
        results[f"bin{bin_num}_emission_datetimes"] = ct_result.get(
            "emission_datetimes", []
        )
        results[f"bin{bin_num}_emission_predicted"] = ct_result.get(
            "emission_predicted", []
        )
        results[f"bin{bin_num}_decay_datetimes"] = ct_result.get("decay_datetimes", [])
        results[f"bin{bin_num}_decay_predicted"] = ct_result.get("decay_predicted", [])
        results[f"bin{bin_num}_E_r_squared"] = ct_result.get("E_r_squared", np.nan)

        # Peak and deposition-end concentration comparisons (measured vs. predicted)
        col_inside = f"{PARTICLE_BINS[bin_num]['column']}_inside"
        emission_pred = ct_result.get("emission_predicted", [])
        decay_pred = ct_result.get("decay_predicted", [])

        # Measured concentration nearest to peak_time.
        # Use argmin() + to_numpy() to avoid pandas Scalar typing ambiguity.
        col_values = particle_data[col_inside].to_numpy(dtype=float, na_value=np.nan)
        if peak_time is not None and col_inside in particle_data.columns:
            time_diffs = (particle_data["datetime"] - pd.Timestamp(peak_time)).abs()
            peak_row = int(time_diffs.argmin())
            results[f"bin{bin_num}_peak_measured"] = float(col_values[peak_row])
        else:
            results[f"bin{bin_num}_peak_measured"] = np.nan

        # Predicted concentration at peak_time (last point of emission phase)
        results[f"bin{bin_num}_peak_predicted"] = (
            float(emission_pred[-1]) if len(emission_pred) > 0 else np.nan
        )

        # Measured concentration nearest to deposition_end
        depo_end = event["deposition_end"]
        if col_inside in particle_data.columns:
            time_diffs_end = (particle_data["datetime"] - pd.Timestamp(depo_end)).abs()
            end_row = int(time_diffs_end.argmin())
            results[f"bin{bin_num}_deposition_end_measured"] = float(
                col_values[end_row]
            )
        else:
            results[f"bin{bin_num}_deposition_end_measured"] = np.nan

        # Predicted concentration at deposition_end (last point of decay phase)
        results[f"bin{bin_num}_deposition_end_predicted"] = (
            float(decay_pred[-1]) if len(decay_pred) > 0 else np.nan
        )

    return results


# =============================================================================
# Main Analysis Pipeline
# =============================================================================


def run_particle_analysis(
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
    apply_sig_figs: bool = True,
) -> pd.DataFrame:
    """
    Run the complete particle decay and emission analysis.

    Parameters:
        output_dir (Path): Optional output directory (defaults to data_root/output)
        generate_plots (bool): If True, generate plots for each event and summary
        apply_sig_figs (bool): If True (default), round calculated float columns to
            SIG_FIGS_DATA significant figures before writing Excel output files and
            apply SIG_FIGS_FIGURE significant figures to figure annotations.
            Pass False (via --no-sig-figs) to preserve full floating-point precision.

    Returns:
        pd.DataFrame: DataFrame with analysis results for all events and bins
    """
    sf.set_enabled(apply_sig_figs)
    print("=" * 80)
    print("Particle Decay & Emission Analysis")
    print("Numerical Approach - Seven Particle Size Bins")
    print("=" * 80)
    print(f"Bedroom volume: {BEDROOM_VOLUME_M3} m^3")
    print(f"Time step: {TIME_STEP_MINUTES} minute(s)")
    print("Penetration factor: averaged before/after windows (p capped at 1)")
    print(f"Deposition window: {DEPOSITION_WINDOW_HOURS} hour(s) after shower")
    print(
        "Beta selection: R²-based 3-step (unclamped → clamped ≥ 0 → 0; threshold 0.80)"
    )
    print("\nValidation thresholds:")
    print(f"  Max other process rate (beta_other): {MAX_OTHER_PROCESS_RATE} h^-1")
    print(
        f"  Min data points: p={MIN_POINTS_PENETRATION}, beta_other={MIN_POINTS_OTHER_PROCESS}, E={MIN_POINTS_EMISSION}"
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

        # Check if excluded (time-based or duration-based)
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if not is_excluded_flag:
            is_excluded_flag = event.get("is_excluded", False)
            exclusion_reason = event.get("exclusion_reason", "")
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
    event_figures_dir = get_event_figures_dir(output_dir)
    if generate_plots:
        plot_dir.mkdir(exist_ok=True)
        event_figures_dir.mkdir(parents=True, exist_ok=True)

    for event in events:
        event_num = event.get("event_number", 0)
        test_name = event.get("test_name", f"Event_{event_num}")
        shower_time = event["shower_on"]
        lambda_ach = event.get("lambda_ach", np.nan)

        # Skip excluded events (time-based or duration-based)
        is_excluded_flag, exclusion_reason = is_event_excluded(shower_time)
        if not is_excluded_flag:
            is_excluded_flag = event.get("is_excluded", False)
            exclusion_reason = event.get("exclusion_reason", "")
        if is_excluded_flag:
            print(f"  {test_name}: Skipped (excluded: {exclusion_reason})")
            # Generate raw PM plot for excluded events that have a real event_number
            if generate_plots and event_num:
                try:
                    from src.plot_particle import plot_particle_decay_event
                    from src.plot_style import format_test_name_for_filename

                    excluded_dir = event_figures_dir / "excluded_events"
                    excluded_dir.mkdir(parents=True, exist_ok=True)

                    # Ensure deposition_end is set (fall back to shower_off + 2h)
                    if event.get("deposition_end") is None:
                        event = dict(event)
                        event["deposition_end"] = event["shower_off"] + timedelta(hours=2)

                    empty_result = {}
                    formatted_name = format_test_name_for_filename(test_name)
                    plot_path = (
                        excluded_dir
                        / f"event_{event_num:02d}-{formatted_name}_pm_decay.png"
                    )
                    plot_particle_decay_event(
                        particle_data=particle_data,
                        event=event,
                        particle_bins=PARTICLE_BINS,
                        result=empty_result,
                        output_path=plot_path,
                        event_number=event_num,
                        test_name=test_name,
                    )
                except Exception as e:
                    print(f"    Warning: Failed to generate excluded plot for {test_name}: {e}")
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
                from src.plot_particle import plot_particle_decay_event
                from src.plot_style import format_test_name_for_filename

                # Format filename: event_01-0114_hw_morning_pm_decay.png
                formatted_name = format_test_name_for_filename(test_name)
                plot_path = (
                    event_figures_dir
                    / f"event_{event_num:02d}-{formatted_name}_pm_decay.png"
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
        beta_col = f"bin{bin_num}_beta_other"
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
    """Save analysis results to Excel workbook with one sheet per metric.

    Sheets written:
        all_results       - Full results table (all metrics per event and bin)
        p_penetration     - Penetration factors
        beta_other        - Other process rates (h⁻¹): beta_other (clamped ≥ 0) and
                            beta_other_raw_mean (unclamped trimmed mean) per bin
        beta_other_r_squared    - R² of forward Euler decay simulation
        E_emission        - Mean emission rates (#/min)
        E_total_particles - Total emitted particle counts per bin (E_total, #)
        E_r_squared       - R² of forward Euler emission-phase simulation
    """
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
        column_rename[f"bin{bin_num}_beta_other"] = f"bin{bin_num}_beta_other (h-1)"
        column_rename[f"bin{bin_num}_beta_other_raw_mean"] = (
            f"bin{bin_num}_beta_other_raw_mean (h-1)"
        )
        column_rename[f"bin{bin_num}_beta_other_std"] = (
            f"bin{bin_num}_beta_other_std (h-1)"
        )
        column_rename[f"bin{bin_num}_E_mean"] = f"bin{bin_num}_E_mean (#/min)"
        column_rename[f"bin{bin_num}_E_std"] = f"bin{bin_num}_E_std (#/min)"
        column_rename[f"bin{bin_num}_E_total"] = f"bin{bin_num}_E_total (#)"

    results_df_export = results_df.rename(columns=column_rename)
    results_df_export = sf.apply_sig_figs_to_df(results_df_export)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Main results
        results_df_export.to_excel(writer, sheet_name="all_results", index=False)

        # Shared ID columns present on every sheet
        id_cols = ["event_number", "test_name", "shower_on"]

        # Separate sheets for each metric (all include test_name for cross-referencing)
        p_cols = id_cols + [f"bin{i}_p_mean (-)" for i in PARTICLE_BINS.keys()]
        beta_cols = id_cols + [
            col
            for i in PARTICLE_BINS.keys()
            for col in (f"bin{i}_beta_other (h-1)", f"bin{i}_beta_other_raw_mean (h-1)")
        ]
        beta_r2_cols = id_cols + [
            f"bin{i}_beta_other_r_squared" for i in PARTICLE_BINS.keys()
        ]
        E_cols = id_cols + [f"bin{i}_E_mean (#/min)" for i in PARTICLE_BINS.keys()]
        E_total_cols = id_cols + [f"bin{i}_E_total (#)" for i in PARTICLE_BINS.keys()]
        E_total_cols = [c for c in E_total_cols if c in results_df_export.columns]

        E_r2_cols = id_cols + [f"bin{i}_E_r_squared" for i in PARTICLE_BINS.keys()]
        E_r2_cols = [c for c in E_r2_cols if c in results_df_export.columns]

        results_df_export[p_cols].to_excel(
            writer, sheet_name="p_penetration", index=False
        )
        results_df_export[beta_cols].to_excel(
            writer, sheet_name="beta_deposition", index=False
        )
        results_df_export[beta_r2_cols].to_excel(
            writer, sheet_name="beta_r_squared", index=False
        )
        results_df_export[E_cols].to_excel(writer, sheet_name="E_emission", index=False)
        results_df_export[E_total_cols].to_excel(
            writer, sheet_name="E_total_particles", index=False
        )
        if E_r2_cols:
            results_df_export[E_r2_cols].to_excel(
                writer, sheet_name="E_r_squared", index=False
            )

        # Peak comparison sheet: measured vs. predicted at peak_time and deposition_end.
        # Wide format: one row per event, bins as column groups.
        peak_df = results_df[["event_number", "test_name"]].copy()
        for bin_num in PARTICLE_BINS.keys():
            meas_pk = f"bin{bin_num}_peak_measured"
            pred_pk = f"bin{bin_num}_peak_predicted"
            meas_de = f"bin{bin_num}_deposition_end_measured"
            pred_de = f"bin{bin_num}_deposition_end_predicted"

            if meas_pk not in results_df.columns:
                continue

            peak_df[f"bin{bin_num}_peak_measured (#/cm3)"] = results_df[meas_pk].values
            peak_df[f"bin{bin_num}_peak_predicted (#/cm3)"] = results_df[pred_pk].values
            with np.errstate(invalid="ignore", divide="ignore"):
                pct_pk = (
                    (results_df[pred_pk] - results_df[meas_pk])
                    / results_df[meas_pk]
                    * 100.0
                )
            peak_df[f"bin{bin_num}_peak_pct_diff (%)"] = pct_pk.values

            peak_df[f"bin{bin_num}_deposition_end_measured (#/cm3)"] = results_df[
                meas_de
            ].values
            peak_df[f"bin{bin_num}_deposition_end_predicted (#/cm3)"] = results_df[
                pred_de
            ].values
            with np.errstate(invalid="ignore", divide="ignore"):
                pct_de = (
                    (results_df[pred_de] - results_df[meas_de])
                    / results_df[meas_de]
                    * 100.0
                )
            peak_df[f"bin{bin_num}_deposition_end_pct_diff (%)"] = pct_de.values

        peak_df = sf.apply_sig_figs_to_df(peak_df)
        peak_df.to_excel(writer, sheet_name="peak_comparison", index=False)

    print(f"\nResults saved to: {output_file}")


def _generate_summary_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary plots for penetration, deposition, emission, and E_total boxplot."""
    print("\nGenerating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    try:
        from src.plot_particle import (
            plot_deposition_rate_boxplot,
            plot_deposition_summary,
            plot_emission_boxplot,
            plot_emission_etotal_by_metric_boxplot,
            plot_emission_etotal_by_showerhead_boxplot,
            plot_emission_rate_boxplot,
            plot_emission_summary,
            plot_penetration_factor_boxplot,
            plot_penetration_summary,
        )
    except ImportError:
        print("  Warning: plot_particle module not found. Skipping plots.")
        return

    # Load Bedroom_Conditions RH data once for n= / RH= boxplot annotations.
    # This is best-effort: if the summary file is unavailable the annotation
    # will show n= only (no RH line).  The Bedroom_Conditions sheet has columns
    # "shower_on" (datetime) and "rh_mean (%)" which are renamed to the
    # "datetime" / "RH_bedroom" convention expected by the boxplot helpers.
    rh_data = None
    try:
        import pandas as _pd

        from src.data_paths import get_common_file

        _summary_path = get_common_file("rh_temp_wind_summary")
        _bc = _pd.read_excel(_summary_path, sheet_name="Bedroom_Conditions")
        _bc = _bc.rename(columns={"shower_on": "datetime", "rh_mean (%)": "RH_bedroom"})
        rh_data = _bc[["datetime", "RH_bedroom"]].copy()
        rh_data["datetime"] = _pd.to_datetime(rh_data["datetime"])
        print("  Loaded Bedroom_Conditions RH for boxplot annotations.")
    except Exception as _rh_err:
        print(
            f"  Note: Could not load Bedroom_Conditions RH data (n= only annotations): {_rh_err}"
        )

    # Bar-chart summary plots (no RH annotation needed)
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

    # ── Task 7: emission E_total vs. continuous metric (10 figures) ───────────
    # Add per-event computed columns to a local copy so results_df is not mutated.
    _df7 = results_df.copy()
    _bin_nums = list(PARTICLE_BINS.keys())

    # Average beta and p across all 7 bins per event
    _df7["avg_beta"] = _df7[
        [
            f"bin{b}_beta_other"
            for b in _bin_nums
            if f"bin{b}_beta_other" in _df7.columns
        ]
    ].mean(axis=1)
    _df7["avg_p"] = _df7[
        [f"bin{b}_p_mean" for b in _bin_nums if f"bin{b}_p_mean" in _df7.columns]
    ].mean(axis=1)

    # Merge bedroom RH and temperature from Bedroom_Conditions sheet (best-effort)
    try:
        import pandas as _pd7

        from src.data_paths import get_common_file as _gcf7

        _bc7 = _pd7.read_excel(
            _gcf7("rh_temp_wind_summary"), sheet_name="Bedroom_Conditions"
        )
        _bc7 = _bc7[["event_number", "rh_mean (%)", "temp_mean (degC)"]].rename(
            columns={"rh_mean (%)": "bedroom_rh", "temp_mean (degC)": "bedroom_temp"}
        )
        _df7 = _df7.merge(_bc7, on="event_number", how="left")
        print("  Merged Bedroom_Conditions for metric-axis figures.")
    except Exception as _e7:
        _df7["bedroom_rh"] = np.nan
        _df7["bedroom_temp"] = np.nan
        print(f"  Note: Bedroom_Conditions not merged for metric-axis figures: {_e7}")

    # ── Boxplot x-range configuration ────────────────────────────────────────
    # Fixed water-temperature axis: x_range=(xmin, xmax, xtick_step) in °C
    _fixed_axis_boxplots = [
        (plot_emission_boxplot, "emission_etotal_boxplot.png", (5, 55, 5)),
        (plot_deposition_rate_boxplot, "other_process_rate_boxplot.png", (5, 55, 5)),
        (plot_emission_rate_boxplot, "emission_rate_boxplot.png", (5, 55, 5)),
        (plot_penetration_factor_boxplot, "penetration_factor_boxplot.png", (5, 55, 5)),
    ]
    for plot_func, filename, x_range in _fixed_axis_boxplots:
        try:
            plot_func(
                results_df,
                PARTICLE_BINS,
                plot_dir / filename,
                rh_data=rh_data,
                x_range=x_range,
            )
            print(f"  Generated: {filename}")
        except Exception as e:
            print(f"  Error generating {filename}: {e}")

    # Continuous metric axis: x_range=(xmin, xmax, step) in metric units
    _metric_axes = [
        # (metric_col, metric_label, filename, x_range=(xmin, xmax, step))
        (
            "bedroom_rh",
            "Bedroom RH (%)",
            "emission_etotal_by_bedroom_rh_boxplot.png",
            (22, 44, 2),
        ),
        (
            "bedroom_temp",
            "Bedroom Temperature (°C)",
            "emission_etotal_by_bedroom_temp_boxplot.png",
            (15, 19, 0.2),
        ),
        (
            "lambda_ach",
            "Air Change Rate λ (h⁻¹)",
            "emission_etotal_by_acr_boxplot.png",
            (0.7, 1.70, 0.1),
        ),
        (
            "avg_beta",
            "Avg. Other Process Rate β (h⁻¹)",
            "emission_etotal_by_beta_boxplot.png",
            (-0.35, 0.35, 0.05),
        ),
        (
            "avg_p",
            "Avg. Penetration Factor p",
            "emission_etotal_by_p_boxplot.png",
            (0.4, 0.8, 0.05),
        ),
    ]
    for metric_col, metric_label, filename, x_range in _metric_axes:
        try:
            plot_emission_etotal_by_metric_boxplot(
                _df7,
                PARTICLE_BINS,
                plot_dir / filename,
                metric_col=metric_col,
                metric_label=metric_label,
                rh_data=rh_data,
                x_range=x_range,
            )
            print(f"  Generated: {filename} (bin0-2 and bin3-6)")
        except Exception as e:
            print(f"  Error generating {filename}: {e}")

    # Shower head type comparison: W53 (base) vs. W52pw (Pepco)
    try:
        _sh_filename = "emission_etotal_by_showerhead_boxplot.png"
        plot_emission_etotal_by_showerhead_boxplot(
            results_df,
            PARTICLE_BINS,
            plot_dir / _sh_filename,
            rh_data=rh_data,
        )
        print(f"  Generated: {_sh_filename} (bin0-2 and bin3-6)")
    except Exception as e:
        print(f"  Error generating emission_etotal_by_showerhead_boxplot: {e}")

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
    parser.add_argument(
        "--no-sig-figs",
        action="store_true",
        help="Disable significant figure rounding on output data and figure annotations "
        "(default: 3 sig figs for files, 2 sig figs for figures)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_particle_analysis(
        output_dir=output_dir,
        generate_plots=not args.no_plot,
        apply_sig_figs=not args.no_sig_figs,
    )


if __name__ == "__main__":
    main()
