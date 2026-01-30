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
    EXPERIMENT_START_DATE,
    EXPECTED_CO2_BEFORE_SHOWER,
    filter_events_by_date,
    is_event_excluded,
    get_time_of_day,
    get_water_temperature_code,
    check_fan_during_test,
    generate_test_name,
)


# =============================================================================
# Configuration
# =============================================================================

# CO2 injection duration change date (4 min before, 6 min after)
CO2_DURATION_CHANGE_DATE = datetime(2026, 1, 22, 14, 0, 0)
CO2_DURATION_BEFORE = 4.0  # minutes
CO2_DURATION_AFTER = 6.0   # minutes

# Default shower duration if no neighbors available
DEFAULT_SHOWER_DURATION = 10.0  # minutes

# Time tolerance for matching (minutes)
MATCH_TOLERANCE_MINUTES = 10.0


# =============================================================================
# Duration Inference Functions
# =============================================================================

def infer_duration_from_neighbors(
    event_time: datetime,
    events: List[Dict],
    duration_key: str,
    event_type: str,
    prompt_user: bool = True
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
        if event_type == 'co2':
            # Use historical default based on date
            if event_time < CO2_DURATION_CHANGE_DATE:
                return CO2_DURATION_BEFORE
            else:
                return CO2_DURATION_AFTER
        return DEFAULT_SHOWER_DURATION

    # Find events before and after
    before_events = [e for e in events if e.get('shower_on', e.get('injection_start', datetime.max)) < event_time]
    after_events = [e for e in events if e.get('shower_on', e.get('injection_start', datetime.min)) > event_time]

    before_duration = None
    after_duration = None

    if before_events:
        closest_before = max(before_events, key=lambda e: e.get('shower_on', e.get('injection_start')))
        before_duration = closest_before.get(duration_key)

    if after_events:
        closest_after = min(after_events, key=lambda e: e.get('shower_on', e.get('injection_start')))
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
        if event_type == 'co2':
            if event_time < CO2_DURATION_CHANGE_DATE:
                return CO2_DURATION_BEFORE
            else:
                return CO2_DURATION_AFTER
        return DEFAULT_SHOWER_DURATION


def _prompt_for_duration(
    event_time: datetime,
    event_type: str,
    before_duration: float,
    after_duration: float
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
    print(f"\n{'='*60}")
    print(f"Duration Selection Required for Synthetic {event_type.upper()} Event")
    print(f"{'='*60}")
    print(f"Event time: {event_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"\nNeighboring events have different durations:")
    print(f"  [1] Before: {before_duration:.1f} minutes")
    print(f"  [2] After:  {after_duration:.1f} minutes")
    print(f"  [3] Average: {(before_duration + after_duration) / 2:.1f} minutes")
    print(f"  [4] Enter custom duration")

    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            if choice == '1':
                return before_duration
            elif choice == '2':
                return after_duration
            elif choice == '3':
                return (before_duration + after_duration) / 2
            elif choice == '4':
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
    prompt_user: bool = True
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
        shower_on,
        shower_events,
        'duration_min',
        'shower',
        prompt_user
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
    prompt_user: bool = True
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
        injection_start,
        co2_events,
        'injection_duration_min',
        'co2',
        prompt_user
    )

    injection_end = injection_start + timedelta(minutes=duration_min)
    fan_off = injection_start + timedelta(minutes=duration_min + 1)  # Fan off 1 min after injection

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
    tolerance_minutes: float = MATCH_TOLERANCE_MINUTES
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
        best_diff = float('inf')

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
    prompt_user: bool = True
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
                    prompt_user
                )
                synthetic_co2_events.append(synthetic)
                print(f"    Shower {shower_idx + 1} -> Synthetic CO2 at "
                      f"{synthetic['injection_start'].strftime('%m/%d %H:%M')}")

        if co2_without_shower:
            print(f"\nCreating {len(co2_without_shower)} synthetic shower events...")
            for co2_idx in co2_without_shower:
                co2 = co2_events[co2_idx]
                synthetic = create_synthetic_shower_event(
                    co2["injection_start"],
                    len(shower_events) + len(synthetic_shower_events) + 1,
                    shower_events,
                    prompt_user
                )
                synthetic_shower_events.append(synthetic)
                print(f"    CO2 {co2_idx + 1} -> Synthetic shower at "
                      f"{synthetic['shower_on'].strftime('%m/%d %H:%M')}")

    # Step 4: Combine real and synthetic events
    all_shower_events = shower_events + synthetic_shower_events
    all_co2_events = co2_events + synthetic_co2_events

    # Step 5: Sort by time and re-number
    all_shower_events.sort(key=lambda e: e["shower_on"])
    all_co2_events.sort(key=lambda e: e["injection_start"])

    for i, event in enumerate(all_shower_events):
        event["event_number"] = i + 1

    for i, event in enumerate(all_co2_events):
        event["event_number"] = i + 1

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

    # Step 7: Re-match and propagate test names to CO2 events
    shower_to_co2, co2_to_shower = match_events_bidirectional(
        all_shower_events, all_co2_events
    )

    for co2_idx, shower_idx in co2_to_shower.items():
        if shower_idx is not None:
            co2_event = all_co2_events[co2_idx]
            shower_event = all_shower_events[shower_idx]
            co2_event["test_name"] = shower_event["test_name"]
            co2_event["water_temp"] = shower_event.get("water_temp")
            co2_event["time_of_day"] = shower_event.get("time_of_day")
            co2_event["matched_shower_idx"] = shower_idx
        else:
            # CO2 without matched shower - generate name from expected shower time
            co2_event = all_co2_events[co2_idx]
            expected_shower = co2_event["injection_start"] + timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)
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

        log_entries.append({
            "event_number": event["event_number"],
            "test_name": event.get("test_name", ""),
            "event_type": "shower",
            "datetime": shower_time,
            "shower_on": shower_time,
            "shower_off": event["shower_off"],
            "duration_min": event.get("duration_min", 0),
            "matched_co2_time": matched_co2["injection_start"] if matched_co2 else None,
            "is_synthetic": event.get("is_synthetic", False),
            "is_excluded": is_excluded,
            "exclusion_reason": exclusion_reason or "",
            "water_temp": event.get("water_temp", ""),
            "time_of_day": event.get("time_of_day", ""),
            "fan_during_test": event.get("fan_during_test", False),
        })

    event_log = pd.DataFrame(log_entries)

    # Summary
    n_total = len(all_shower_events)
    n_synthetic = sum(1 for e in all_shower_events if e.get("is_synthetic", False))
    n_excluded = sum(1 for e in all_shower_events
                     if is_event_excluded(e["shower_on"])[0])

    print(f"\nRegistry Summary:")
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
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Unified Event Registry Module")
    print("=" * 40)
    print("\nThis module should be imported and used by analysis scripts.")
    print("\nKey function:")
    print("  build_unified_event_registry(shower_events, co2_events, shower_log)")
    print("\nReturns:")
    print("  (unified_shower_events, unified_co2_events, event_log)")
