#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Matching Utilities
========================

This module provides functions for matching CO2 injection events with shower events.

The experimental protocol follows this sequence within each testing cycle:
1. Shower ON at :00 (top of hour) or other scheduled time
2. Shower OFF after 5-15 minutes
3. CO2 injection at :40 (40 minutes into the hour)
4. CO2 decay measurement starts at :50

This creates a timing offset where the CO2 injection occurs AFTER the shower ends,
but both belong to the same testing cycle. The matching logic must account for this.

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
    time_tolerance_hours: float = 1.0,
    injection_column: str = "injection_start",
) -> Optional[int]:
    """
    Find the CO2 event that corresponds to a shower event.
    
    The matching logic accounts for the experimental protocol where:
    - Shower events occur at various times (often :00 or :50)
    - CO2 injection occurs at :40 of the same or following hour
    - The CO2 event follows the shower within the same testing cycle
    
    Matching criteria:
    1. CO2 injection must occur AFTER the shower starts
    2. CO2 injection should be within time_tolerance_hours of the shower
    3. Prefer the CO2 event closest in time to the shower
    
    Parameters:
        shower_time: Datetime when shower started
        co2_events_df: DataFrame with CO2 event data (must have injection_column)
        time_tolerance_hours: Maximum hours between shower and CO2 injection
        injection_column: Column name for CO2 injection time
        
    Returns:
        Index (0-based) of matching CO2 event, or None if no match found
    """
    if co2_events_df.empty:
        return None
    
    # Ensure injection times are datetime
    injection_times = pd.to_datetime(co2_events_df[injection_column])
    
    # Find CO2 events that occur within the tolerance window AFTER the shower
    # The CO2 injection should happen after the shower starts
    min_time = shower_time  # CO2 injection must be at or after shower start
    max_time = shower_time + timedelta(hours=time_tolerance_hours)
    
    # Find candidates within the time window
    candidates = []
    for idx, inj_time in enumerate(injection_times):
        if min_time <= inj_time <= max_time:
            time_diff = (inj_time - shower_time).total_seconds() / 3600.0
            candidates.append((idx, time_diff))
    
    if not candidates:
        # Try a more lenient match: same calendar hour
        shower_hour = get_testing_cycle_hour(shower_time)
        for idx, inj_time in enumerate(injection_times):
            inj_hour = get_testing_cycle_hour(inj_time)
            # Check if same hour or next hour (for late showers like :50)
            if inj_hour == shower_hour or inj_hour == shower_hour + timedelta(hours=1):
                time_diff = (inj_time - shower_time).total_seconds() / 3600.0
                if time_diff > 0:  # CO2 must be after shower
                    candidates.append((idx, time_diff))
    
    if not candidates:
        return None
    
    # Return the closest match (smallest positive time difference)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def build_event_mapping(
    shower_events: List[Dict],
    co2_events_df: pd.DataFrame,
    time_tolerance_hours: float = 1.5,
) -> Dict[int, Optional[int]]:
    """
    Build a complete mapping from shower event indices to CO2 event indices.
    
    Parameters:
        shower_events: List of shower event dictionaries with 'shower_on' key
        co2_events_df: DataFrame with CO2 analysis results
        time_tolerance_hours: Maximum hours between shower and CO2 injection
        
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
            time_tolerance_hours=time_tolerance_hours,
        )
        
        mapping[shower_num] = co2_idx
    
    return mapping


def get_lambda_for_shower(
    shower_time: datetime,
    co2_events_df: pd.DataFrame,
    lambda_column: str = "lambda_average_mean",
    time_tolerance_hours: float = 1.5,
) -> Tuple[Optional[float], Optional[int]]:
    """
    Get the air change rate (λ) for a shower event from CO2 analysis results.
    
    Parameters:
        shower_time: Datetime when shower started
        co2_events_df: DataFrame with CO2 analysis results
        lambda_column: Column name for λ value
        time_tolerance_hours: Maximum hours between shower and CO2 injection
        
    Returns:
        Tuple of (lambda_value, co2_event_index) or (None, None) if no match
    """
    co2_idx = match_shower_to_co2_event(
        shower_time, 
        co2_events_df, 
        time_tolerance_hours=time_tolerance_hours,
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
    print(f"{'Shower #':<10} {'Shower Time':<20} {'CO2 #':<10} {'CO2 Time':<20} {'λ (h⁻¹)':<10}")
    print("-" * 70)
    
    for shower_num, co2_idx in mapping.items():
        shower_event = shower_events[shower_num - 1]
        shower_time = shower_event.get("shower_on") or shower_event.get("shower_start")
        shower_time_str = shower_time.strftime("%Y-%m-%d %H:%M") if shower_time else "N/A"
        
        if co2_idx is not None:
            co2_row = co2_events_df.iloc[co2_idx]
            co2_time = pd.to_datetime(co2_row["injection_start"])
            co2_time_str = co2_time.strftime("%Y-%m-%d %H:%M")
            lambda_val = co2_row.get("lambda_average_mean", float("nan"))
            co2_num = co2_idx + 1
            print(f"{shower_num:<10} {shower_time_str:<20} {co2_num:<10} {co2_time_str:<20} {lambda_val:.4f}")
        else:
            print(f"{shower_num:<10} {shower_time_str:<20} {'N/A':<10} {'No match':<20} {'N/A':<10}")
    
    # Summary statistics
    matched = sum(1 for v in mapping.values() if v is not None)
    total = len(mapping)
    print("-" * 70)
    print(f"Matched: {matched}/{total} shower events")
    print("=" * 70 + "\n")
