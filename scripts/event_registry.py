#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Event Registry
=======================

This module provides a single source of truth for event identification, naming,
and matching across all analysis scripts (CO2, particle, RH/temp).

Key Features:
    - Bidirectional synthetic event creation (shower<->CO2)
    - Consistent event numbering and naming
    - Duration inference from neighboring events
    - User prompts for ambiguous cases
    - Comprehensive logging of all events and decisions

The registry ensures that event_07 has the same test_name across all analysis
scripts, eliminating naming inconsistencies.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: January 2026
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_manager import (
    EXPECTED_CO2_BEFORE_SHOWER,
    EXPERIMENT_START_DATE,
    check_fan_during_test,
    filter_events_by_date,
    generate_test_name,
    get_time_of_day,
    get_water_temperature_code,
    is_event_excluded,
)

# =============================================================================
# Configuration
# =============================================================================

# CO2 injection duration change date (4 min before, 6 min after)
CO2_DURATION_CHANGE_DATE = datetime(2026, 1, 22, 14, 0, 0)
CO2_DURATION_BEFORE = 4.0  # minutes
CO2_DURATION_AFTER = 6.0  # minutes

# Default shower duration if no neighbors available
DEFAULT_SHOWER_DURATION = 10.0  # minutes

# Time tolerance for matching (minutes)
MATCH_TOLERANCE_MINUTES = 10.0

# Registry output file
REGISTRY_FILENAME = "event_registry.csv"


# =============================================================================
# Duration Inference Functions
# =============================================================================


def infer_duration_from_neighbors(
    event_time: datetime,
    events: List[Dict],
    duration_key: str,
    event_type: str,
    prompt_user: bool = True,
) -> float:
    """
    Infer duration for a synthetic event based on neighboring events.

    Parameters:
        event_time: Time of the synthetic event
        events: List of real events with duration information
        duration_key: Key in event dict containing duration (e.g., 'duration_min')
        event_type: 'shower' or 'co2' (for user prompt context)
        prompt_user: Whether to prompt user if durations differ

    Returns:
        Inferred duration in minutes
    """
    if not events:
        if event_type == "co2":
            # Use historical default based on date
            if event_time < CO2_DURATION_CHANGE_DATE:
                return CO2_DURATION_BEFORE
            else:
                return CO2_DURATION_AFTER
        return DEFAULT_SHOWER_DURATION

    # Find events before and after
    before_events = [
        e
        for e in events
        if e.get("shower_on", e.get("injection_start", datetime.max)) < event_time
    ]
    after_events = [
        e
        for e in events
        if e.get("shower_on", e.get("injection_start", datetime.min)) > event_time
    ]

    before_duration = None
    after_duration = None

    if before_events:
        closest_before = max(
            before_events, key=lambda e: e.get("shower_on", e.get("injection_start"))
        )
        before_duration = closest_before.get(duration_key)

    if after_events:
        closest_after = min(
            after_events, key=lambda e: e.get("shower_on", e.get("injection_start"))
        )
        after_duration = closest_after.get(duration_key)

    # If both are available and differ, we may need to prompt
    if before_duration is not None and after_duration is not None:
        if abs(before_duration - after_duration) < 0.5:
            # Close enough, use average
            return (before_duration + after_duration) / 2
        elif prompt_user:
            # Durations differ - prompt user
            return _prompt_for_duration(
                event_time, event_type, before_duration, after_duration
            )
        else:
            # No prompt, use average
            return (before_duration + after_duration) / 2
    elif before_duration is not None:
        return before_duration
    elif after_duration is not None:
        return after_duration
    else:
        # No neighbors, use default
        if event_type == "co2":
            if event_time < CO2_DURATION_CHANGE_DATE:
                return CO2_DURATION_BEFORE
            else:
                return CO2_DURATION_AFTER
        return DEFAULT_SHOWER_DURATION


def _prompt_for_duration(
    event_time: datetime, event_type: str, before_duration: float, after_duration: float
) -> float:
    """
    Prompt user to choose duration when neighbors differ.

    Parameters:
        event_time: Time of the synthetic event
        event_type: 'shower' or 'co2'
        before_duration: Duration from event before
        after_duration: Duration from event after

    Returns:
        Selected duration in minutes
    """
    print(f"\n{'=' * 60}")
    print(f"Duration Selection Required for Synthetic {event_type.upper()} Event")
    print(f"{'=' * 60}")
    print(f"Event time: {event_time.strftime('%Y-%m-%d %H:%M')}")
    print("\nNeighboring events have different durations:")
    print(f"  [1] Before: {before_duration:.1f} minutes")
    print(f"  [2] After:  {after_duration:.1f} minutes")
    print(f"  [3] Average: {(before_duration + after_duration) / 2:.1f} minutes")
    print("  [4] Enter custom duration")

    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            if choice == "1":
                return before_duration
            elif choice == "2":
                return after_duration
            elif choice == "3":
                return (before_duration + after_duration) / 2
            elif choice == "4":
                custom = float(input("Enter duration in minutes: ").strip())
                if custom > 0:
                    return custom
                print("Duration must be positive.")
            else:
                print("Please enter 1, 2, 3, or 4.")
        except (ValueError, EOFError):
            print("Invalid input. Using average.")
            return (before_duration + after_duration) / 2


# =============================================================================
# Synthetic Event Creation
# =============================================================================


def create_synthetic_shower_event(
    co2_injection_time: datetime,
    event_number: int,
    shower_events: List[Dict],
    prompt_user: bool = True,
) -> Dict:
    """
    Create a synthetic shower event for a CO2 injection that has no matching shower.

    Parameters:
        co2_injection_time: Datetime of CO2 injection start
        event_number: Event number for this synthetic shower event
        shower_events: List of real shower events (for duration inference)
        prompt_user: Whether to prompt user for duration if ambiguous

    Returns:
        Dictionary with synthetic shower event structure
    """
    # Expected shower: 20 minutes after CO2 injection
    shower_on = co2_injection_time + timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)

    # Infer duration from neighboring events
    duration_min = infer_duration_from_neighbors(
        shower_on, shower_events, "duration_min", "shower", prompt_user
    )

    shower_off = shower_on + timedelta(minutes=duration_min)

    # Calculate analysis windows
    pre_start = shower_on - timedelta(minutes=30)
    post_end = shower_off + timedelta(hours=2)

    # Particle analysis windows
    penetration_start = shower_on - timedelta(hours=1)
    penetration_end = shower_on
    deposition_start = shower_off
    deposition_end = shower_off + timedelta(hours=2)

    return {
        "event_number": event_number,
        "shower_on": shower_on,
        "shower_off": shower_off,
        "duration_min": duration_min,
        "shower_duration_min": duration_min,
        "pre_start": pre_start,
        "post_end": post_end,
        "penetration_start": penetration_start,
        "penetration_end": penetration_end,
        "deposition_start": deposition_start,
        "deposition_end": deposition_end,
        "is_synthetic": True,
    }


def create_synthetic_co2_event(
    shower_time: datetime,
    event_number: int,
    co2_events: List[Dict],
    prompt_user: bool = True,
) -> Dict:
    """
    Create a synthetic CO2 event for a shower that has no matching CO2 data.

    Parameters:
        shower_time: Datetime when shower started
        event_number: Event number for this synthetic CO2 event
        co2_events: List of real CO2 events (for duration inference)
        prompt_user: Whether to prompt user for duration if ambiguous

    Returns:
        Dictionary with synthetic CO2 event structure
    """
    # Expected CO2 injection: 20 minutes before shower
    injection_start = shower_time - timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)

    # Infer injection duration from neighboring events
    # Note: We look for 'injection_duration_min' or calculate from injection_end - injection_start
    duration_min = infer_duration_from_neighbors(
        injection_start, co2_events, "injection_duration_min", "co2", prompt_user
    )

    injection_end = injection_start + timedelta(minutes=duration_min)
    fan_off = injection_start + timedelta(
        minutes=duration_min + 1
    )  # Fan off 1 min after injection

    # Decay analysis would start at :50 (10 minutes before next hour)
    hour_after_injection = injection_start.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    decay_start = hour_after_injection + timedelta(minutes=-10)  # At :50
    decay_end = decay_start + timedelta(hours=2)  # 2-hour analysis window

    return {
        "event_number": event_number,
        "injection_start": injection_start,
        "injection_end": injection_end,
        "injection_duration_min": duration_min,
        "fan_off": fan_off,
        "decay_start": decay_start,
        "decay_end": decay_end,
        "decay_duration_hours": 2.0,
        "is_synthetic": True,
    }


# =============================================================================
# Event Matching and Registration
# =============================================================================


def match_events_bidirectional(
    shower_events: List[Dict],
    co2_events: List[Dict],
    tolerance_minutes: float = MATCH_TOLERANCE_MINUTES,
) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]]]:
    """
    Match shower and CO2 events bidirectionally.

    Returns:
        Tuple of:
        - shower_to_co2: Dict mapping shower index to CO2 index (or None)
        - co2_to_shower: Dict mapping CO2 index to shower index (or None)
    """
    shower_to_co2: Dict[int, Optional[int]] = {}
    co2_to_shower: Dict[int, Optional[int]] = {}

    # Initialize all as None
    for i in range(len(shower_events)):
        shower_to_co2[i] = None
    for i in range(len(co2_events)):
        co2_to_shower[i] = None

    # Match showers to CO2
    for i, shower in enumerate(shower_events):
        shower_time = shower["shower_on"]
        expected_co2_time = shower_time - timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)

        best_match = None
        best_diff = float("inf")

        for j, co2 in enumerate(co2_events):
            co2_time = co2["injection_start"]
            diff_minutes = abs((co2_time - expected_co2_time).total_seconds() / 60.0)

            if diff_minutes <= tolerance_minutes and diff_minutes < best_diff:
                best_match = j
                best_diff = diff_minutes

        if best_match is not None:
            shower_to_co2[i] = best_match
            co2_to_shower[best_match] = i

    return shower_to_co2, co2_to_shower


def build_unified_event_registry(
    shower_events: List[Dict],
    co2_events: List[Dict],
    shower_log: pd.DataFrame,
    create_synthetic: bool = True,
    prompt_user: bool = True,
) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    Build a unified event registry with consistent naming and numbering.

    This is the main entry point for creating a consistent event list that
    all analysis scripts should use.

    Parameters:
        shower_events: List of raw shower event dictionaries
        co2_events: List of raw CO2 event dictionaries
        shower_log: DataFrame with shower state changes
        create_synthetic: Whether to create synthetic events for missing data
        prompt_user: Whether to prompt user for duration decisions

    Returns:
        Tuple of:
        - unified_shower_events: List with consistent numbering and naming
        - unified_co2_events: List with consistent numbering
        - event_log: DataFrame documenting all events and decisions
    """
    print("\n" + "=" * 70)
    print("Building Unified Event Registry")
    print("=" * 70)

    # Step 1: Filter by date
    print(f"\nFiltering events (keeping >= {EXPERIMENT_START_DATE.date()})...")
    shower_events = filter_events_by_date(shower_events)
    co2_events = filter_events_by_date(co2_events)
    print(f"  Shower events: {len(shower_events)}")
    print(f"  CO2 events: {len(co2_events)}")

    # Step 2: Match events bidirectionally
    print("\nMatching events bidirectionally...")
    shower_to_co2, co2_to_shower = match_events_bidirectional(shower_events, co2_events)

    showers_without_co2 = [i for i, j in shower_to_co2.items() if j is None]
    co2_without_shower = [i for i, j in co2_to_shower.items() if j is None]

    print(f"  Matched pairs: {sum(1 for v in shower_to_co2.values() if v is not None)}")
    print(f"  Showers without CO2: {len(showers_without_co2)}")
    print(f"  CO2 without shower: {len(co2_without_shower)}")

    # Step 3: Create synthetic events if requested
    synthetic_co2_events = []
    synthetic_shower_events = []

    if create_synthetic:
        if showers_without_co2:
            print(f"\nCreating {len(showers_without_co2)} synthetic CO2 events...")
            for shower_idx in showers_without_co2:
                shower = shower_events[shower_idx]
                synthetic = create_synthetic_co2_event(
                    shower["shower_on"],
                    len(co2_events) + len(synthetic_co2_events) + 1,
                    co2_events,
                    prompt_user,
                )
                synthetic_co2_events.append(synthetic)
                print(
                    f"    Shower {shower_idx + 1} -> Synthetic CO2 at "
                    f"{synthetic['injection_start'].strftime('%m/%d %H:%M')}"
                )

        if co2_without_shower:
            print(f"\nCreating {len(co2_without_shower)} synthetic shower events...")
            for co2_idx in co2_without_shower:
                co2 = co2_events[co2_idx]
                synthetic = create_synthetic_shower_event(
                    co2["injection_start"],
                    len(shower_events) + len(synthetic_shower_events) + 1,
                    shower_events,
                    prompt_user,
                )
                synthetic_shower_events.append(synthetic)
                print(
                    f"    CO2 {co2_idx + 1} -> Synthetic shower at "
                    f"{synthetic['shower_on'].strftime('%m/%d %H:%M')}"
                )

    # Step 4: Combine real and synthetic events
    all_shower_events = shower_events + synthetic_shower_events
    all_co2_events = co2_events + synthetic_co2_events

    # Step 5: Sort by time and assign UNIFIED event numbers
    # Event numbers are based on SHOWER events only (primary source of truth)
    all_shower_events.sort(key=lambda e: e["shower_on"])
    all_co2_events.sort(key=lambda e: e["injection_start"])

    for i, event in enumerate(all_shower_events):
        event["event_number"] = i + 1

    # NOTE: CO2 events get their event_number from matched showers in Step 7 below
    # This ensures unified numbering across all analysis scripts

    # Step 6: Assign test names based on shower events
    print("\nAssigning test names...")
    replicate_counters: Dict[str, int] = {}

    for event in all_shower_events:
        shower_time = event["shower_on"]
        shower_off = event["shower_off"]

        # Skip excluded events for naming (but still number them)
        is_excluded, _ = is_event_excluded(shower_time)

        # Determine test parameters
        water_temp = get_water_temperature_code(shower_time)
        time_of_day = get_time_of_day(shower_time)

        # Check fan status (can't check for synthetic events without real log data)
        if event.get("is_synthetic", False):
            fan_status = False
        else:
            fan_status = check_fan_during_test(shower_time, shower_off, shower_log)

        # Create condition key
        date_str = shower_time.strftime("%m%d")
        condition_key = f"{date_str}_{water_temp}_{time_of_day}"
        if fan_status:
            condition_key += "_Fan"

        # Get replicate number
        replicate_num = replicate_counters.get(condition_key, 0) + 1
        replicate_counters[condition_key] = replicate_num

        # Generate test name
        test_name = generate_test_name(
            shower_time, water_temp, time_of_day, replicate_num, fan_status
        )

        # Add metadata to event
        event["test_name"] = test_name
        event["water_temp"] = water_temp
        event["time_of_day"] = time_of_day
        event["fan_during_test"] = fan_status
        event["replicate_num"] = replicate_num

    # Step 7: Re-match and propagate UNIFIED event numbers and test names to CO2 events
    # This ensures CO2 events use the SAME event_number as their matched shower
    shower_to_co2, co2_to_shower = match_events_bidirectional(
        all_shower_events, all_co2_events
    )

    for co2_idx, shower_idx in co2_to_shower.items():
        if shower_idx is not None:
            co2_event = all_co2_events[co2_idx]
            shower_event = all_shower_events[shower_idx]
            # CRITICAL: Use shower's event_number for unified numbering
            co2_event["event_number"] = shower_event["event_number"]
            co2_event["test_name"] = shower_event["test_name"]
            co2_event["water_temp"] = shower_event.get("water_temp")
            co2_event["time_of_day"] = shower_event.get("time_of_day")
            co2_event["matched_shower_idx"] = shower_idx
        else:
            # CO2 without matched shower - this shouldn't happen if create_synthetic=True
            # Assign None for event_number (will be excluded from analysis)
            co2_event = all_co2_events[co2_idx]
            co2_event["event_number"] = None  # No matching shower
            expected_shower = co2_event["injection_start"] + timedelta(
                minutes=EXPECTED_CO2_BEFORE_SHOWER
            )
            water_temp = get_water_temperature_code(expected_shower)
            time_of_day = get_time_of_day(expected_shower)
            date_str = expected_shower.strftime("%m%d")
            co2_event["test_name"] = f"{date_str}_{water_temp}_{time_of_day}_R??"
            co2_event["water_temp"] = water_temp
            co2_event["time_of_day"] = time_of_day
            co2_event["matched_shower_idx"] = None

    # Step 8: Create event log
    print("\nGenerating event log...")
    log_entries = []

    for event in all_shower_events:
        shower_time = event["shower_on"]
        is_excluded, exclusion_reason = is_event_excluded(shower_time)

        # Find matched CO2
        matched_co2 = None
        for co2_idx, shower_idx in co2_to_shower.items():
            if shower_idx is not None:
                # Get the index of this shower in all_shower_events
                try:
                    this_shower_idx = all_shower_events.index(event)
                    if shower_idx == this_shower_idx:
                        matched_co2 = all_co2_events[co2_idx]
                        break
                except ValueError:
                    pass

        log_entries.append(
            {
                "event_number": event["event_number"],
                "test_name": event.get("test_name", ""),
                "event_type": "shower",
                "datetime": shower_time,
                "shower_on": shower_time,
                "shower_off": event["shower_off"],
                "duration_min": event.get("duration_min", 0),
                "matched_co2_time": matched_co2["injection_start"]
                if matched_co2
                else None,
                "is_synthetic": event.get("is_synthetic", False),
                "is_excluded": is_excluded,
                "exclusion_reason": exclusion_reason or "",
                "water_temp": event.get("water_temp", ""),
                "time_of_day": event.get("time_of_day", ""),
                "fan_during_test": event.get("fan_during_test", False),
            }
        )

    event_log = pd.DataFrame(log_entries)

    # Summary
    n_total = len(all_shower_events)
    n_synthetic = sum(1 for e in all_shower_events if e.get("is_synthetic", False))
    n_excluded = sum(
        1 for e in all_shower_events if is_event_excluded(e["shower_on"])[0]
    )

    print("\nRegistry Summary:")
    print(f"  Total shower events: {n_total}")
    print(f"  Real events: {n_total - n_synthetic}")
    print(f"  Synthetic events: {n_synthetic}")
    print(f"  Excluded events: {n_excluded}")
    print(f"  Total CO2 events: {len(all_co2_events)}")

    print("\n" + "=" * 70)
    print("Event Registry Complete")
    print("=" * 70 + "\n")

    return all_shower_events, all_co2_events, event_log


# =============================================================================
# Registry File I/O Functions
# =============================================================================


def save_event_registry(
    shower_events: List[Dict],
    co2_events: List[Dict],
    co2_results_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Save the unified event registry to a CSV file.

    The registry contains one row per shower event (the primary source of truth
    for event numbering), with matched CO2 data and lambda values.

    Parameters:
        shower_events: List of unified shower event dictionaries
        co2_events: List of CO2 event dictionaries (matched to showers)
        co2_results_df: DataFrame with CO2 analysis results (lambda values)
        output_path: Path to save event_registry.csv

    Returns:
        DataFrame: The registry as a DataFrame
    """
    # Build shower-to-CO2 mapping
    shower_to_co2, _ = match_events_bidirectional(shower_events, co2_events)

    registry_rows = []

    for i, shower in enumerate(shower_events):
        event_num = shower["event_number"]
        shower_time = shower["shower_on"]

        # Check exclusion status
        is_excluded, exclusion_reason = is_event_excluded(shower_time)

        # Find matched CO2 event
        co2_idx = shower_to_co2.get(i)
        co2_event = co2_events[co2_idx] if co2_idx is not None else None

        # Get lambda values from CO2 results if available
        lambda_mean = np.nan
        lambda_std = np.nan
        lambda_r_squared = np.nan

        if co2_event is not None and not co2_results_df.empty:
            # Match by injection_start time
            co2_time = co2_event.get("injection_start")
            if co2_time is not None:
                # Find matching row in co2_results_df
                time_diffs = abs(
                    (co2_results_df["injection_start"] - co2_time).dt.total_seconds()
                )
                if len(time_diffs) > 0 and time_diffs.min() < 60:  # Within 1 minute
                    match_idx = time_diffs.idxmin()
                    result_row = co2_results_df.loc[match_idx]
                    # Handle column name variations
                    if "lambda_average_mean" in result_row:
                        lambda_mean = result_row["lambda_average_mean"]
                    elif "lambda_average_mean (h-1)" in result_row:
                        lambda_mean = result_row["lambda_average_mean (h-1)"]
                    if "lambda_average_std" in result_row:
                        lambda_std = result_row["lambda_average_std"]
                    elif "lambda_average_std (h-1)" in result_row:
                        lambda_std = result_row["lambda_average_std (h-1)"]
                    if "lambda_average_r_squared" in result_row:
                        lambda_r_squared = result_row["lambda_average_r_squared"]

        row = {
            "event_number": event_num,
            "test_name": shower.get("test_name", ""),
            "shower_on": shower["shower_on"],
            "shower_off": shower["shower_off"],
            "shower_duration_min": shower.get(
                "shower_duration_min", shower.get("duration_min", 0)
            ),
            "is_shower_synthetic": shower.get("is_synthetic", False),
            "co2_injection_start": co2_event["injection_start"]
            if co2_event
            else pd.NaT,
            "co2_injection_end": co2_event.get("injection_end")
            if co2_event
            else pd.NaT,
            "is_co2_missing": co2_event is None,
            "is_co2_synthetic": co2_event.get("is_synthetic", False)
            if co2_event
            else False,
            "lambda_average_mean": lambda_mean,
            "lambda_average_std": lambda_std,
            "lambda_r_squared": lambda_r_squared,
            "water_temp": shower.get("water_temp", ""),
            "door_position": shower.get("door_position", "Open"),
            "time_of_day": shower.get("time_of_day", ""),
            "fan_during_test": shower.get("fan_during_test", False),
            "replicate_num": shower.get("replicate_num", 0),
            "is_excluded": is_excluded,
            "exclusion_reason": exclusion_reason or "",
            "penetration_start": shower.get("penetration_start"),
            "penetration_end": shower.get("penetration_end"),
            "deposition_start": shower.get("deposition_start"),
            "deposition_end": shower.get("deposition_end"),
            "decay_start": co2_event.get("decay_start") if co2_event else pd.NaT,
            "decay_end": co2_event.get("decay_end") if co2_event else pd.NaT,
        }
        registry_rows.append(row)

    registry_df = pd.DataFrame(registry_rows)

    # Save to CSV
    registry_df.to_csv(output_path, index=False)
    print(f"\nEvent registry saved to: {output_path}")
    print(f"  Total events: {len(registry_df)}")
    print(f"  With CO2 data: {(~registry_df['is_co2_missing']).sum()}")
    print(f"  With lambda values: {registry_df['lambda_average_mean'].notna().sum()}")
    print(f"  Excluded: {registry_df['is_excluded'].sum()}")

    return registry_df


def load_event_registry(registry_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the unified event registry from CSV.

    This function is used by analysis scripts to get consistent event numbering.

    Parameters:
        registry_path: Path to registry file (defaults to output/event_registry.csv)

    Returns:
        DataFrame: The event registry with parsed datetime columns

    Raises:
        FileNotFoundError: If registry doesn't exist (suggests running this script)
    """
    if registry_path is None:
        from src.data_paths import get_data_root

        registry_path = get_data_root() / "output" / REGISTRY_FILENAME

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Event registry not found: {registry_path}\n"
            "Run 'python scripts/event_registry.py' to generate it."
        )

    # Define datetime columns to parse
    datetime_cols = [
        "shower_on",
        "shower_off",
        "co2_injection_start",
        "co2_injection_end",
        "penetration_start",
        "penetration_end",
        "deposition_start",
        "deposition_end",
        "decay_start",
        "decay_end",
    ]

    df = pd.read_csv(registry_path)

    # Parse datetime columns
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def run_co2_analysis_if_needed(output_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    Run CO2 analysis if results don't exist, return lambda results.

    Parameters:
        output_dir: Output directory
        force: If True, re-run even if results exist

    Returns:
        DataFrame: CO2 analysis results with lambda values
    """
    co2_results_path = output_dir / "co2_lambda_summary.csv"

    if not co2_results_path.exists() or force:
        print("\n" + "=" * 70)
        print("Running CO2 decay analysis to get lambda values...")
        print("=" * 70)

        # Import and run CO2 analysis
        from src.co2_decay_analysis import run_co2_decay_analysis

        run_co2_decay_analysis(output_dir=output_dir, generate_plots=False)

    # Load and return results
    if co2_results_path.exists():
        df = pd.read_csv(co2_results_path)
        df["injection_start"] = pd.to_datetime(df["injection_start"])
        print(f"\nLoaded CO2 results: {len(df)} events with lambda values")
        return df
    else:
        print("\nWarning: CO2 analysis did not produce results")
        return pd.DataFrame()


# =============================================================================
# Main Entry Point (CLI)
# =============================================================================


def main():
    """
    Build and save the unified event registry.

    This is the recommended entry point for generating the event registry
    that all analysis scripts should use for consistent event numbering.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build unified event registry for EPA Legionella project.\n"
        "This creates event_registry.csv with consistent event numbering "
        "that all analysis scripts should use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data_root/output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate registry even if it exists",
    )
    parser.add_argument(
        "--no-co2",
        action="store_true",
        help="Skip CO2 analysis (registry will have no lambda values)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Don't create synthetic events for missing data",
    )

    args = parser.parse_args()

    # Set output directory
    from src.data_paths import get_data_root

    output_dir = Path(args.output_dir) if args.output_dir else get_data_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = output_dir / REGISTRY_FILENAME

    # Check if registry exists
    if registry_path.exists() and not args.force:
        print(f"Registry already exists: {registry_path}")
        print("Use --force to regenerate.")

        # Show summary of existing registry
        existing = load_event_registry(registry_path)
        print(f"\nExisting registry summary:")
        print(f"  Total events: {len(existing)}")
        print(f"  Date range: {existing['shower_on'].min()} to {existing['shower_on'].max()}")
        return

    print("=" * 70)
    print("Building Unified Event Registry")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")

    # STEP 1: Load shower log and identify events
    print("\nLoading shower events...")
    from src.env_data_loader import identify_shower_events, load_shower_log

    shower_log = load_shower_log()
    shower_events = identify_shower_events(shower_log)
    print(f"  Found {len(shower_events)} raw shower events")

    # STEP 2: Load CO2 injection events
    print("\nLoading CO2 injection events...")
    from src.co2_decay_analysis import identify_injection_events, load_co2_injection_log

    co2_log = load_co2_injection_log()
    co2_events = identify_injection_events(co2_log)
    print(f"  Found {len(co2_events)} raw CO2 injection events")

    # STEP 3: Build unified registry (without lambda values first)
    unified_showers, unified_co2, event_log = build_unified_event_registry(
        shower_events,
        co2_events,
        shower_log,
        create_synthetic=not args.no_synthetic,
        prompt_user=False,  # Non-interactive mode for CLI
    )

    # STEP 4: Save registry FIRST (so CO2 analysis can use it for event numbering)
    # This initial save has no lambda values yet
    print("\nSaving initial registry (without lambda values)...")
    registry_df = save_event_registry(
        unified_showers, unified_co2, pd.DataFrame(), registry_path
    )

    # STEP 5: Now run CO2 analysis (which will use the registry for event numbering)
    co2_results_df = pd.DataFrame()
    if not args.no_co2:
        print("\n" + "=" * 70)
        print("Running CO2 analysis with unified event numbering...")
        print("=" * 70)
        co2_results_df = run_co2_analysis_if_needed(output_dir, force=True)

        # STEP 6: Update registry with lambda values from CO2 analysis
        if not co2_results_df.empty:
            print("\nUpdating registry with lambda values...")
            registry_df = save_event_registry(
                unified_showers, unified_co2, co2_results_df, registry_path
            )

    # Save the event log for reference
    event_log_path = output_dir / "event_log.csv"
    event_log.to_csv(event_log_path, index=False)
    print(f"Event log saved to: {event_log_path}")

    print("\n" + "=" * 70)
    print("Event Registry Complete")
    print("=" * 70)
    print(f"\nTo use in analysis scripts:")
    print(f"  from scripts.event_registry import load_event_registry")
    print(f"  registry = load_event_registry()")


if __name__ == "__main__":
    main()
