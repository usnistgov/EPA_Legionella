#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Matching Utilities
========================

This module provides functions for matching CO2 injection events with shower events.

The experimental protocol follows this sequence within each testing cycle:
1. CO2 injection at :40 (40 minutes into the hour, e.g., 08:40, 14:40)
2. CO2 injection ends at :44 (4-minute injection period)
3. Mixing fan turns off at :45
4. Shower ON at :00 (top of next hour, e.g., 09:00, 15:00)
5. Shower OFF after 5-15 minutes
6. CO2 decay measurement starts at :50 (10 minutes before top of hour after injection)

This creates a timing offset where the CO2 injection occurs approximately 20 minutes
BEFORE the shower starts. The matching logic must account for this timing relationship.

Key Functions:
    - match_shower_to_co2_event: Find the CO2 event corresponding to a shower event
    - get_testing_cycle_hour: Extract the testing cycle hour from an event time
    - build_event_mapping: Create a complete mapping between shower and CO2 events

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


def get_testing_cycle_hour(event_time: datetime) -> datetime:
    """
    Get the testing cycle hour for an event.

    Events within the same testing cycle share the same hour, even if they
    occur at different minutes within that hour. This function normalizes
    the event time to the start of its hour.

    For events that occur at :50 or later, they may belong to the next hour's
    testing cycle if the CO2 injection happens at :40.

    Parameters:
        event_time: Datetime of the event

    Returns:
        Datetime normalized to the hour (minutes=0, seconds=0)
    """
    # Special handling: if event is at :50 or later, it might be part of
    # the current hour's CO2 decay (injection at :40, decay starts :50)
    return event_time.replace(minute=0, second=0, microsecond=0)


def match_shower_to_co2_event(
    shower_time: datetime,
    co2_events_df: pd.DataFrame,
    time_tolerance_before: float = 10.0,  # minutes before expected CO2 time
    time_tolerance_after: float = 10.0,   # minutes after expected CO2 time
    injection_column: str = "injection_start",
) -> Optional[int]:
    """
    Find the CO2 event that corresponds to a shower event.

    The matching logic accounts for the experimental protocol where:
    - CO2 injection occurs at :40 (e.g., 08:40, 14:40)
    - Shower starts at :00 of the next hour (e.g., 09:00, 15:00)
    - Expected timing: CO2 injection is ~20 minutes BEFORE shower start

    Matching criteria:
    1. CO2 injection occurs within tolerance window around expected time (shower_time - 20 minutes)
    2. Prefer the CO2 event closest in time to the expected injection time

    Parameters:
        shower_time: Datetime when shower started
        co2_events_df: DataFrame with CO2 event data (must have injection_column)
        time_tolerance_before: Minutes before expected CO2 time to search (default 10)
        time_tolerance_after: Minutes after expected CO2 time to search (default 10)
        injection_column: Column name for CO2 injection time

    Returns:
        Index (0-based) of matching CO2 event, or None if no match found
    """
    if co2_events_df.empty:
        return None

    # Ensure injection times are datetime
    injection_times = pd.to_datetime(co2_events_df[injection_column])

    # Expected CO2 injection time: 20 minutes BEFORE shower start
    expected_co2_time = shower_time - timedelta(minutes=20)

    # Find CO2 events that occur within the tolerance window around expected time
    min_time = expected_co2_time - timedelta(minutes=time_tolerance_before)
    max_time = expected_co2_time + timedelta(minutes=time_tolerance_after)

    # Find candidates within the time window
    candidates = []
    for idx, inj_time in enumerate(injection_times):
        if min_time <= inj_time <= max_time:
            # Calculate time difference from expected CO2 injection time
            time_diff = (inj_time - expected_co2_time).total_seconds() / 60.0
            candidates.append((idx, abs(time_diff)))

    if not candidates:
        return None

    # Return the closest match (smallest absolute time difference from expected time)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def match_co2_to_shower_event(
    co2_injection_time: datetime,
    shower_events: List[Dict],
    time_tolerance_minutes: float = 10.0,
) -> Optional[int]:
    """
    Find the shower event that corresponds to a CO2 injection event.

    This is the reverse of match_shower_to_co2_event. The matching logic accounts
    for the experimental protocol where CO2 injection occurs ~20 minutes BEFORE
    the shower starts.

    Parameters:
        co2_injection_time: Datetime when CO2 injection started
        shower_events: List of shower event dictionaries with 'shower_on' key
        time_tolerance_minutes: Tolerance window around expected shower time

    Returns:
        Index (0-based) of matching shower event, or None if no match found
    """
    if not shower_events:
        return None

    # Expected shower time: 20 minutes AFTER CO2 injection
    expected_shower_time = co2_injection_time + timedelta(minutes=20)

    # Find shower events within the tolerance window
    candidates = []
    for idx, event in enumerate(shower_events):
        shower_time = event.get("shower_on")
        if shower_time is None:
            continue

        time_diff = abs((shower_time - expected_shower_time).total_seconds() / 60.0)
        if time_diff <= time_tolerance_minutes:
            candidates.append((idx, time_diff))

    if not candidates:
        return None

    # Return the closest match
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def build_event_mapping(
    shower_events: List[Dict],
    co2_events_df: pd.DataFrame,
    time_tolerance_before: float = 10.0,
    time_tolerance_after: float = 10.0,
) -> Dict[int, Optional[int]]:
    """
    Build a complete mapping from shower event indices to CO2 event indices.

    Parameters:
        shower_events: List of shower event dictionaries with 'shower_on' key
        co2_events_df: DataFrame with CO2 analysis results
        time_tolerance_before: Minutes before expected CO2 time to search (default 10)
        time_tolerance_after: Minutes after expected CO2 time to search (default 10)

    Returns:
        Dictionary mapping shower event index (1-based) to CO2 event index (0-based)
        None value indicates no matching CO2 event was found
    """
    mapping = {}

    for i, shower_event in enumerate(shower_events):
        shower_num = i + 1  # 1-based shower event number
        shower_time = shower_event.get("shower_on") or shower_event.get("shower_start")

        if shower_time is None:
            mapping[shower_num] = None
            continue

        co2_idx = match_shower_to_co2_event(
            shower_time,
            co2_events_df,
            time_tolerance_before=time_tolerance_before,
            time_tolerance_after=time_tolerance_after,
        )

        mapping[shower_num] = co2_idx

    return mapping


def get_lambda_for_shower(
    shower_time: datetime,
    co2_events_df: pd.DataFrame,
    lambda_column: str = "lambda_average_mean",
    time_tolerance_before: float = 10.0,
    time_tolerance_after: float = 10.0,
) -> Tuple[Optional[float], Optional[int]]:
    """
    Get the air change rate (λ) for a shower event from CO2 analysis results.

    Parameters:
        shower_time: Datetime when shower started
        co2_events_df: DataFrame with CO2 analysis results
        lambda_column: Column name for λ value
        time_tolerance_before: Minutes before expected CO2 time to search (default 10)
        time_tolerance_after: Minutes after expected CO2 time to search (default 10)

    Returns:
        Tuple of (lambda_value, co2_event_index) or (None, None) if no match
    """
    co2_idx = match_shower_to_co2_event(
        shower_time,
        co2_events_df,
        time_tolerance_before=time_tolerance_before,
        time_tolerance_after=time_tolerance_after,
    )

    if co2_idx is None:
        return None, None

    lambda_value = co2_events_df.iloc[co2_idx][lambda_column]
    return lambda_value, co2_idx


def print_event_matching_summary(
    shower_events: List[Dict],
    co2_events_df: pd.DataFrame,
    mapping: Optional[Dict[int, Optional[int]]] = None,
) -> None:
    """
    Print a summary of event matching for debugging.

    Parameters:
        shower_events: List of shower event dictionaries
        co2_events_df: DataFrame with CO2 analysis results
        mapping: Pre-computed mapping (optional, will compute if not provided)
    """
    if mapping is None:
        mapping = build_event_mapping(shower_events, co2_events_df)

    print("\n" + "=" * 70)
    print("Event Matching Summary")
    print("=" * 70)
    print(
        f"{'Shower #':<10} {'Shower Time':<20} {'CO2 #':<10} {'CO2 Time':<20} {'λ (h⁻¹)':<10}"
    )
    print("-" * 70)

    for shower_num, co2_idx in mapping.items():
        shower_event = shower_events[shower_num - 1]
        shower_time = shower_event.get("shower_on") or shower_event.get("shower_start")
        shower_time_str = (
            shower_time.strftime("%Y-%m-%d %H:%M") if shower_time else "N/A"
        )

        if co2_idx is not None:
            co2_row = co2_events_df.iloc[co2_idx]
            co2_time = pd.to_datetime(co2_row["injection_start"])
            co2_time_str = co2_time.strftime("%Y-%m-%d %H:%M")
            lambda_val = co2_row.get("lambda_average_mean", float("nan"))
            co2_num = co2_idx + 1
            print(
                f"{shower_num:<10} {shower_time_str:<20} {co2_num:<10} {co2_time_str:<20} {lambda_val:.4f}"
            )
        else:
            print(
                f"{shower_num:<10} {shower_time_str:<20} {'N/A':<10} {'No match':<20} {'N/A':<10}"
            )

    # Summary statistics
    matched = sum(1 for v in mapping.values() if v is not None)
    total = len(mapping)
    print("-" * 70)
    print(f"Matched: {matched}/{total} shower events")
    print("=" * 70 + "\n")
