#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Management System
=======================

This module provides enhanced event matching, filtering, naming, and logging
capabilities for the EPA Legionella project.

Key Features:
    - Date filtering to exclude pre-experiment data
    - Missing event detection and synthetic event creation
    - Test parameter-based naming convention (e.g., 0115_W48_Open_Day_R01)
    - Bath fan status detection for test conditions
    - Event exclusion system with reason logging (duration-based exclusion for water temp testing)
    - Comprehensive event_log.csv for tracking all events and issues

Naming Convention Format:
    MMDD_TempCode_DoorPos_TimeOfDay_RNN

    Components:
    - MMDD: Month and day (e.g., 0114 for January 14)
    - TempCode: Water temperature code (e.g., W48, W11, W25, W48b)
    - DoorPos: Door position (Open, Closed, or Partial)
    - TimeOfDay: Day or Night
    - RNN: Replicate number (R01, R02, etc.)

    Repeat Temperature Convention:
    When the same water temperature is retested (e.g., after a shower head change),
    a suffix is appended to the temperature code: W48 (first run), W48b (second
    run), W48c (third run), etc. Multi-character suffixes are also supported for
    named variants, e.g. W52pw (Pepco wide-spray shower head at 52 °C). The
    numeric temperature value determines sort order. Each variant forms a separate
    config_key group for independent replicate counting and plotting.

    Examples:
    - 0115_W48_Open_Day_R01
    - 0122_W11_Open_Night_R03
    - 0203_W25_Open_Day_R01
    - 0223_W48b_Open_Day_R01  (second W48 run, e.g. with new shower head)
    - 0224_W52pw_Open_Day_R01 (Pepco wide-spray shower head at 52 °C)

Time of Day Categories:
    - Day: 5am - 5pm
    - Night: 5pm - 5am

Test Parameters:
    - Water Temperature: Designated by W## code (e.g., W48 = 48 degrees C)
    - Time of Day: Based on shower start time
    - Bath Fan Status: Fan running during or within 2 hours after shower

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: January 2026
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing event matching functions
from src.event_matching import match_shower_to_co2_event

# Lazy import for event_registry to avoid circular import
# These are imported inside functions that use them
_HAS_REGISTRY = False
create_synthetic_co2_event_v2 = None
infer_duration_from_neighbors = None
match_events_bidirectional = None


def _ensure_registry_imports():
    """Lazy import of event_registry module to avoid circular imports."""
    global _HAS_REGISTRY, create_synthetic_co2_event_v2
    global infer_duration_from_neighbors, match_events_bidirectional

    if _HAS_REGISTRY:
        return True

    try:
        from scripts.event_registry import (
            create_synthetic_co2_event as _create_synthetic_co2_event_v2,
        )
        from scripts.event_registry import (
            infer_duration_from_neighbors as _infer_duration_from_neighbors,
        )
        from scripts.event_registry import (
            match_events_bidirectional as _match_events_bidirectional,
        )

        create_synthetic_co2_event_v2 = _create_synthetic_co2_event_v2
        infer_duration_from_neighbors = _infer_duration_from_neighbors
        match_events_bidirectional = _match_events_bidirectional
        _HAS_REGISTRY = True
        return True
    except ImportError:
        return False


# =============================================================================
# Configuration Constants
# =============================================================================

# Experiment start date - data before this is excluded
EXPERIMENT_START_DATE = datetime(2026, 1, 15, 15, 0, 0)

# =============================================================================
# Test Configuration System
# =============================================================================
# Configuration transitions define when test conditions change.
# Each configuration parameter has a list of (transition_datetime, value) tuples.
# The value applies FROM that datetime until the next transition.
#
# To add new configurations or modify transition dates, edit these dictionaries.
# Future configurations (e.g., new door positions, additional water temps) can
# be added by simply extending the lists.

# Water temperature transitions: (datetime, code)
# Codes: "HW" (Hot Water), "CW" (Cold Water), "MW" (Mixed/Medium Water)
WATER_TEMP_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "W48"),  # Mixed waterexperiment start
    (datetime(2026, 1, 22, 14, 0, 0), "W11"),  # Cold water from Jan 22 2PM
    (datetime(2026, 2, 2, 17, 0, 0), "W25"),  # Mixed water from Feb 2 5PM
    (datetime(2026, 2, 5, 10, 0, 0), "W30"),  # Mixed water from Feb 5 10AM
    (datetime(2026, 2, 9, 10, 0, 0), "W37"),  # Mixed water from Feb 9 10AM
    (datetime(2026, 2, 11, 8, 0, 0), "W23"),  # Mixed water from Feb 11 8AM
    (datetime(2026, 2, 13, 11, 0, 0), "W22"),  # Mixed water from Feb 13 11AM
    (datetime(2026, 2, 16, 11, 0, 0), "W43"),  # Mixed water from Feb 16 11AM
    (datetime(2026, 2, 18, 10, 23, 0), "W14"),  # Mixed water from Feb 18 10:23AM
    (datetime(2026, 2, 20, 8, 0, 0), "W53"),  # Hot water from Feb 20 8AM
    (
        datetime(2026, 2, 24, 8, 0, 0),
        "W52pw",
    ),  # Pepco wide spray from Feb 24 8AM
    (
        datetime(2026, 2, 26, 10, 23, 0),
        "W49pw",
    ),  # Pepco wide spray from Feb 26 10:23AM
    (
        datetime(2026, 3, 2, 9, 23, 0),
        "W40pn",
    ),  # Pepco narrow spray from Mar 2 9:23AM
    (
        datetime(2026, 3, 4, 10, 23, 0),
        "W40pw",
    ),  # Pepco wide spray from Mar 4 10:23AM
    (
        datetime(2026, 3, 6, 8, 47, 0),
        "W40pm",
    ),  # Pepco wide spray from Mar 6 8:47AM
    (
        datetime(2026, 3, 9, 8, 47, 0),
        "W40pw2",
    ),  # Pepco wide spray from Mar 9 8:47AM
]

# Door position transitions: (datetime, position)
# Positions: "Open", "Closed", "Partial"
DOOR_POSITION_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "Open"),  # Door Open from experiment start
    # Add future transitions here, e.g.:
    # (datetime(2026, 2, 1, 0, 0, 0), "Closed"),    # Door closed from Feb 1
    # (datetime(2026, 3, 1, 0, 0, 0), "Partial"),   # Partially open from Mar 1
]

# Bath fan transitions: (datetime, status)
# Status: "On", "Off"
# Note: This is for PLANNED fan operation. Actual fan status during tests
# is still detected from the shower log via check_fan_during_test().
FAN_STATUS_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "Off"),  # Fan off from experiment start
    # Add future transitions here, e.g.:
    # (datetime(2026, 2, 15, 0, 0, 0), "On"),  # Fan on from Feb 15
]

# Time of day boundaries (hour of day)
TIME_OF_DAY_RANGES = {
    "Day": (5, 17),  # 5am - 5pm
    "Night": (17, 5),  # 5pm - 5am (wraps around midnight)
}

# Predefined exclusions: datetime -> reason
EXCLUDED_EVENTS = {
    datetime(2026, 1, 22, 15, 0, 0): "Tour in house during test",
    datetime(2026, 1, 29, 15, 0, 0): "People in house",
    datetime(2026, 2, 11, 3, 0, 0): "Confliting log entries",
    datetime(2026, 2, 11, 15, 0, 0): "Confliting log entries",
    datetime(2026, 2, 12, 3, 0, 0): "Confliting log entries",
    datetime(2026, 2, 12, 15, 0, 0): "Confliting log entries",
    datetime(2026, 2, 13, 3, 0, 0): "Confliting log entries",
}

# Expected CO2 to shower timing offset (minutes)
EXPECTED_CO2_BEFORE_SHOWER = 20

# Expected shower duration for analysis events (computer-controlled, 10 min ± 5 sec)
EXPECTED_SHOWER_DURATION_MIN = 10.0  # minutes
SHOWER_DURATION_TOLERANCE_SEC = 5  # seconds


# =============================================================================
# Helper Functions
# =============================================================================


def get_water_temp_sort_key(config_key: str) -> float:
    """
    Extract numeric water temperature from config_key for sorting.

    Extracts the numeric value from the water temperature code (e.g., "W48" -> 48,
    "W48b" -> 48) in the config_key string. Used to sort configurations from
    coldest to hottest. Letter suffixes on repeat runs (e.g., W48b) are stripped
    so that all runs at the same temperature sort together.

    Parameters:
        config_key: Configuration key (e.g., "W48_DoorOpen_FanOff", "W48b_DoorOpen_FanOff")
                    or water temp code (e.g., "W48", "W48b")

    Returns:
        Numeric sort key (water temperature in °C). Unknown values sort last.
    """
    # Handle "All" or empty strings
    if not config_key or config_key == "All":
        return float("inf")

    # Extract the water temp component (first part before _Door or first part)
    parts = config_key.split("_")
    water_temp = parts[0]

    # Extract numeric value from water temp code; strip any suffix for repeat/variant
    # runs (e.g., "W48b" -> 48, "W52pw" -> 52, "W48" -> 48). Only digits kept.
    if water_temp.startswith("W") and len(water_temp) > 1:
        numeric_str = "".join(c for c in water_temp[1:] if c.isdigit())
        if numeric_str:
            try:
                return float(numeric_str)
            except ValueError:
                pass

    return float("inf")


def sort_config_keys_by_water_temp(config_keys: list) -> list:
    """
    Sort configuration keys by water temperature from coldest to hottest.

    Parameters:
        config_keys: List of config_key strings (e.g., ["W48_DoorOpen_FanOff", "W11_DoorOpen_FanOff"])

    Returns:
        Sorted list of config_keys
    """
    return sorted(config_keys, key=get_water_temp_sort_key)


def get_time_of_day(dt: datetime) -> str:
    """
    Determine time of day category based on hour.

    Parameters:
        dt: Datetime to categorize

    Returns:
        String: "Day" or "Night"
    """
    hour = dt.hour
    day_start, day_end = TIME_OF_DAY_RANGES["Day"]

    if day_start <= hour < day_end:
        return "Day"
    else:
        return "Night"


def _get_config_value_at_time(
    dt: datetime, transitions: List[Tuple[datetime, str]]
) -> str:
    """
    Get configuration value at a given time based on transition list.

    Parameters:
        dt: Datetime to check
        transitions: List of (datetime, value) tuples, sorted by datetime

    Returns:
        Configuration value active at the given time
    """
    # Find the most recent transition before or at dt
    active_value = transitions[0][1]  # Default to first value
    for transition_time, value in transitions:
        if dt >= transition_time:
            active_value = value
        else:
            break
    return active_value


def get_water_temperature_code(dt: datetime) -> str:
    """
    Determine water temperature code based on datetime.

    Uses WATER_TEMP_TRANSITIONS to determine the water temperature.
    Supports: "HW" (Hot Water), "CW" (Cold Water), "MW" (Mixed Water)

    Parameters:
        dt: Datetime of the event

    Returns:
        String: Water temperature code (e.g., "HW", "CW", "MW")
    """
    return _get_config_value_at_time(dt, WATER_TEMP_TRANSITIONS)


def get_door_position(dt: datetime) -> str:
    """
    Determine door position based on datetime.

    Uses DOOR_POSITION_TRANSITIONS to determine the door status.
    Supports: "Open", "Closed", "Partial"

    Parameters:
        dt: Datetime of the event

    Returns:
        String: Door position (e.g., "Open", "Closed", "Partial")
    """
    return _get_config_value_at_time(dt, DOOR_POSITION_TRANSITIONS)


def get_planned_fan_status(dt: datetime) -> str:
    """
    Determine planned fan status based on datetime.

    Uses FAN_STATUS_TRANSITIONS to determine the planned fan operation.
    Note: Actual fan status during tests is detected from shower log.
    Supports: "On", "Off"

    Parameters:
        dt: Datetime of the event

    Returns:
        String: Planned fan status (e.g., "On", "Off")
    """
    return _get_config_value_at_time(dt, FAN_STATUS_TRANSITIONS)


def get_test_configuration(dt: datetime) -> Dict[str, str]:
    """
    Get complete test configuration for a given datetime.

    Returns a dictionary with all configuration parameters that can be used
    for grouping, filtering, and labeling results.

    Parameters:
        dt: Datetime of the event

    Returns:
        Dictionary with configuration keys:
            - water_temp: Temperature code (e.g., "W48", "W11", "W25")
            - door_position: "Open", "Closed", or "Partial"
            - planned_fan: "On" or "Off"
            - config_key: Combined key for grouping (e.g., "W48_DoorOpen_FanOff")
    """
    water_temp = get_water_temperature_code(dt)
    door_pos = get_door_position(dt)
    fan_status = get_planned_fan_status(dt)

    # Create combined configuration key for grouping
    fan_label = "FanOn" if fan_status == "On" else "FanOff"
    door_label = f"Door{door_pos}"
    config_key = f"{water_temp}_{door_label}_{fan_label}"

    return {
        "water_temp": water_temp,
        "door_position": door_pos,
        "planned_fan": fan_status,
        "config_key": config_key,
    }


def get_unique_configurations() -> List[Dict[str, str]]:
    """
    Get list of all unique configurations based on transition dates.

    This is useful for generating summary statistics by configuration.

    Returns:
        List of configuration dictionaries, one per unique configuration
    """
    # Collect all unique transition points
    all_transitions = set()
    for dt, _ in WATER_TEMP_TRANSITIONS:
        all_transitions.add(dt)
    for dt, _ in DOOR_POSITION_TRANSITIONS:
        all_transitions.add(dt)
    for dt, _ in FAN_STATUS_TRANSITIONS:
        all_transitions.add(dt)

    # Sort transitions
    sorted_transitions = sorted(all_transitions)

    # Get configuration at each transition point
    seen_configs = set()
    unique_configs = []

    for dt in sorted_transitions:
        config = get_test_configuration(dt)
        if config["config_key"] not in seen_configs:
            seen_configs.add(config["config_key"])
            config["start_time"] = dt
            unique_configs.append(config)

    return unique_configs


def check_fan_during_test(
    shower_on: datetime, shower_off: datetime, shower_log: pd.DataFrame
) -> bool:
    """
    Check if bath fan ran during shower or within 2 hours after shower.

    Fan running before shower is for space draw-down and not a test parameter.

    Parameters:
        shower_on: Shower start time
        shower_off: Shower end time
        shower_log: DataFrame with shower and bath_fan state changes

    Returns:
        Boolean: True if fan ran during test period, False otherwise
    """
    # Check period: from shower_on to 2 hours after shower_off
    test_start = shower_on
    test_end = shower_off + timedelta(hours=2)

    # Filter log to test period
    mask = (shower_log["datetime_EDT"] >= test_start) & (
        shower_log["datetime_EDT"] <= test_end
    )
    test_period_log = shower_log[mask]

    # Check if fan was ever on during this period
    if len(test_period_log) > 0:
        return bool((test_period_log["bath_fan"] > 0).any())

    return False


def generate_test_name(
    shower_time: datetime,
    water_temp: str,
    time_of_day: str,
    replicate_num: int,
    fan_status: bool = False,
    door_position: str = "Open",
) -> str:
    """
    Generate a test condition name following the naming convention.

    Format: MMDD_TempCode_DoorPos_TimeOfDay[_Fan]_RNN

    Parameters:
        shower_time: Datetime of shower start
        water_temp: Water temperature code (e.g., "W48", "W11", "W25")
        time_of_day: "Day" or "Night"
        replicate_num: Replicate number (1-indexed)
        fan_status: Whether bath fan ran during test (default False)
        door_position: "Open", "Closed", or "Partial" (default "Open")

    Returns:
        String: Test name (e.g., "0115_W48_Open_Day_R01")
    """
    # Format date as MMDD
    date_str = shower_time.strftime("%m%d")

    # Build name components
    components = [date_str, water_temp, door_position, time_of_day]

    # Add fan status if applicable
    if fan_status:
        components.append("Fan")

    # Add replicate number
    components.append(f"R{replicate_num:02d}")

    return "_".join(components)


# =============================================================================
# Event Filtering and Validation
# =============================================================================


def filter_events_by_date(
    events: List[Dict], start_date: datetime = EXPERIMENT_START_DATE
) -> List[Dict]:
    """
    Filter events to only include those on or after the experiment start date.

    For CO2 events, the comparison is based on the expected shower time
    (injection_start + 20 minutes) rather than the injection start itself.
    This ensures CO2 injections that occur before midnight but correspond
    to showers after midnight are correctly included.

    Parameters:
        events: List of event dictionaries
        start_date: Minimum date/time to include (default: 2026-01-14)

    Returns:
        Filtered list of events
    """
    filtered = []
    for event in events:
        if "shower_on" in event:
            # Shower event - compare shower_on directly
            event_time = event["shower_on"]
        elif "injection_start" in event:
            # CO2 event - compare expected shower time (injection + 20 min)
            # This handles cases where CO2 injection is before midnight
            # but the corresponding shower is after midnight
            event_time = event["injection_start"] + timedelta(
                minutes=EXPECTED_CO2_BEFORE_SHOWER
            )
        else:
            continue

        if event_time >= start_date:
            filtered.append(event)

    return filtered


def is_event_excluded(event_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check if an event should be excluded from analysis.

    Parameters:
        event_time: Datetime of the event

    Returns:
        Tuple of (is_excluded: bool, reason: str or None)
    """
    # Check exact match first
    if event_time in EXCLUDED_EVENTS:
        return True, EXCLUDED_EVENTS[event_time]

    # Check for events within 1 minute (to handle slight timing differences)
    for excluded_time, reason in EXCLUDED_EVENTS.items():
        if abs((event_time - excluded_time).total_seconds()) < 60:
            return True, reason

    return False, None


def is_duration_excluded(
    duration_min: Optional[float],
) -> Tuple[bool, Optional[str]]:
    """
    Check if a shower event should be excluded based on its duration.

    Analysis showers are computer-controlled at exactly 10 minutes.
    Events outside ±5 seconds of 10 min are water temperature testing
    runs and should be excluded from analysis (but retained in the log).

    Parameters:
        duration_min: Shower duration in minutes (None = unknown/synthetic)

    Returns:
        Tuple of (is_excluded: bool, reason: str or None)
    """
    if duration_min is None:
        return False, None

    tolerance_min = SHOWER_DURATION_TOLERANCE_SEC / 60.0
    lower = EXPECTED_SHOWER_DURATION_MIN - tolerance_min
    upper = EXPECTED_SHOWER_DURATION_MIN + tolerance_min

    if not (lower <= duration_min <= upper):
        return True, (
            f"Water temperature testing "
            f"(duration: {duration_min:.1f} min, expected: "
            f"{EXPECTED_SHOWER_DURATION_MIN:.0f} min "
            f"\u00b1{SHOWER_DURATION_TOLERANCE_SEC}s)"
        )

    return False, None


# =============================================================================
# Missing Event Detection and Synthetic Event Creation
# =============================================================================


def create_synthetic_co2_event(shower_time: datetime, event_number: int) -> Dict:
    """
    Create a synthetic CO2 event for a shower that has no matching CO2 data.

    The synthetic event has expected timing but no actual measurement data.

    Parameters:
        shower_time: Datetime when shower started
        event_number: Event number for this synthetic CO2 event

    Returns:
        Dictionary with synthetic CO2 event structure
    """
    # Expected CO2 injection: 20 minutes before shower
    injection_start = shower_time - timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)
    injection_end = injection_start + timedelta(minutes=4)  # 4-minute injection
    fan_off = injection_start + timedelta(minutes=5)  # Fan off at 5 minutes

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
        "fan_off": fan_off,
        "decay_start": decay_start,
        "decay_end": decay_end,
        "decay_duration_hours": 2.0,
        "is_synthetic": True,  # Flag to indicate this is not real data
    }


def detect_missing_events(
    shower_events: List[Dict],
    co2_events: List[Dict],
    time_tolerance_minutes: float = 10.0,
) -> Tuple[List[int], List[int]]:
    """
    Detect missing events in either shower or CO2 logs.

    Parameters:
        shower_events: List of shower event dictionaries
        co2_events: List of CO2 event dictionaries
        time_tolerance_minutes: Tolerance for matching (default 10 minutes)

    Returns:
        Tuple of (shower_indices_missing_co2, co2_indices_missing_shower)
    """
    showers_missing_co2 = []
    co2_missing_shower = []

    # Check each shower for matching CO2
    for i, shower_event in enumerate(shower_events):
        shower_time = shower_event["shower_on"]
        expected_co2_time = shower_time - timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)

        # Look for matching CO2 event
        found_match = False
        for co2_event in co2_events:
            co2_time = co2_event["injection_start"]
            time_diff = abs((co2_time - expected_co2_time).total_seconds() / 60.0)

            if time_diff <= time_tolerance_minutes:
                found_match = True
                break

        if not found_match:
            showers_missing_co2.append(i)

    # Check each CO2 for matching shower
    for i, co2_event in enumerate(co2_events):
        co2_time = co2_event["injection_start"]
        expected_shower_time = co2_time + timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)

        # Look for matching shower event
        found_match = False
        for shower_event in shower_events:
            shower_time = shower_event["shower_on"]
            time_diff = abs((shower_time - expected_shower_time).total_seconds() / 60.0)

            if time_diff <= time_tolerance_minutes:
                found_match = True
                break

        if not found_match:
            co2_missing_shower.append(i)

    return showers_missing_co2, co2_missing_shower


# =============================================================================
# Event Naming and Replicate Tracking
# =============================================================================


def assign_test_names(
    shower_events: List[Dict], shower_log: pd.DataFrame
) -> List[Dict]:
    """
    Assign test condition names to all shower events.

    Handles replicate numbering for tests with the same conditions.
    Adds all configuration parameters to each event for grouping and analysis.

    Parameters:
        shower_events: List of shower event dictionaries
        shower_log: DataFrame with shower and bath_fan state changes

    Returns:
        List of events with added configuration fields:
            - test_name: Full test name string
            - water_temp: Water temperature code (HW/CW/MW)
            - door_position: Door position (Open/Closed/Partial)
            - planned_fan: Planned fan status (On/Off)
            - fan_during_test: Actual fan status during test (bool)
            - time_of_day: Time of day category
            - config_key: Combined configuration key for grouping
            - replicate_num: Replicate number for this condition
    """
    # Track replicate numbers by condition (excluding replicate number)
    replicate_counters = {}

    for event in shower_events:
        shower_time = event["shower_on"]
        shower_off = event["shower_off"]

        # Check duration-based exclusion first — water temp testing events are
        # not named or counted as replicates
        duration_min = event.get("duration_min", event.get("shower_duration_min"))
        dur_excluded, dur_reason = is_duration_excluded(duration_min)
        if dur_excluded:
            event["is_excluded"] = True
            event["exclusion_reason"] = dur_reason
            event["test_name"] = ""
            event["water_temp"] = get_water_temperature_code(shower_time)
            event["door_position"] = get_door_position(shower_time)
            event["planned_fan"] = get_planned_fan_status(shower_time)
            event["fan_during_test"] = False
            event["time_of_day"] = get_time_of_day(shower_time)
            event["config_key"] = ""
            event["replicate_num"] = 0
            continue

        # Get full test configuration
        config = get_test_configuration(shower_time)
        water_temp = config["water_temp"]
        door_position = config["door_position"]
        planned_fan = config["planned_fan"]
        config_key = config["config_key"]

        # Get time of day and actual fan status during test
        time_of_day = get_time_of_day(shower_time)
        fan_during_test = check_fan_during_test(shower_time, shower_off, shower_log)

        # Create base condition key (without replicate number)
        date_str = shower_time.strftime("%m%d")
        condition_key = f"{date_str}_{water_temp}_{door_position}_{time_of_day}"
        if fan_during_test:
            condition_key += "_Fan"

        # Get next replicate number for this condition
        replicate_num = replicate_counters.get(condition_key, 0) + 1
        replicate_counters[condition_key] = replicate_num

        # Generate full test name
        test_name = generate_test_name(
            shower_time,
            water_temp,
            time_of_day,
            replicate_num,
            fan_during_test,
            door_position,
        )

        # Add all configuration fields to event
        event["test_name"] = test_name
        event["water_temp"] = water_temp
        event["door_position"] = door_position
        event["planned_fan"] = planned_fan
        event["fan_during_test"] = fan_during_test
        event["time_of_day"] = time_of_day
        event["config_key"] = config_key
        event["replicate_num"] = replicate_num

    return shower_events


# =============================================================================
# Main Processing Function
# =============================================================================


def process_events_with_management(
    shower_events: List[Dict],
    co2_events: List[Dict],
    shower_log: pd.DataFrame,
    co2_results_df: pd.DataFrame,
    output_dir: Path,
    create_synthetic: bool = True,
    prompt_user: bool = False,
) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    Process all events with filtering, matching, naming, and logging.

    This is the main entry point for the enhanced event management system.
    Supports bidirectional synthetic event creation (shower<->CO2).

    Parameters:
        shower_events: List of shower event dictionaries
        co2_events: List of CO2 event dictionaries
        shower_log: DataFrame with shower state changes
        co2_results_df: DataFrame with CO2 analysis results
        output_dir: Directory for output files
        create_synthetic: Whether to create synthetic events for missing data
        prompt_user: Whether to prompt user for duration decisions (default False)

    Returns:
        Tuple of (processed_shower_events, processed_co2_events, event_log_df)
    """
    print("\n" + "=" * 70)
    print("Event Management System")
    print("=" * 70)

    # Ensure registry imports are loaded (lazy import to avoid circular dependency)
    _ensure_registry_imports()

    # Check if CO2 processing is needed (non-empty co2_events list provided)
    process_co2 = len(co2_events) > 0 or not co2_results_df.empty

    # Step 1: Filter by date
    print(f"\nFiltering events (keeping >= {EXPERIMENT_START_DATE.date()})...")
    shower_events = filter_events_by_date(shower_events)
    if process_co2:
        co2_events = filter_events_by_date(co2_events)
    print(f"  Shower events after filtering: {len(shower_events)}")
    if process_co2:
        print(f"  CO2 events after filtering: {len(co2_events)}")

    # Step 2: Assign test names
    print("\nAssigning test condition names...")
    shower_events = assign_test_names(shower_events, shower_log)

    # Clear event_number for duration-excluded events so they are not numbered
    dur_excluded_count = 0
    for event in shower_events:
        if event.get("is_excluded", False):
            event["event_number"] = None
            dur_excluded_count += 1
    if dur_excluded_count:
        print(f"  Duration-excluded (water temp testing): {dur_excluded_count}")

    # Step 3: Detect missing events (bidirectional) - only if CO2 processing is enabled
    showers_missing_co2 = []
    co2_missing_shower = []

    if process_co2:
        print("\nDetecting missing events...")
        showers_missing_co2, co2_missing_shower = detect_missing_events(
            shower_events, co2_events
        )

        if showers_missing_co2:
            print(f"  Found {len(showers_missing_co2)} shower events without CO2 data")

            if create_synthetic:
                print("  Creating synthetic CO2 events...")
                next_co2_num = len(co2_events) + 1

                for shower_idx in showers_missing_co2:
                    shower_event = shower_events[shower_idx]
                    # Use new registry function if available (with duration inference)
                    if _HAS_REGISTRY and create_synthetic_co2_event_v2 is not None:
                        synthetic_co2 = create_synthetic_co2_event_v2(
                            shower_event["shower_on"],
                            next_co2_num,
                            co2_events,
                            prompt_user,
                        )
                    else:
                        synthetic_co2 = create_synthetic_co2_event(
                            shower_event["shower_on"], next_co2_num
                        )
                    co2_events.append(synthetic_co2)
                    next_co2_num += 1

    # Step 4: Match events (only if CO2 processing is enabled)
    matched_pairs = {}

    if process_co2:
        print("\nMatching shower events to CO2 events...")

        # Convert co2_events to DataFrame for matching if needed
        if not isinstance(co2_results_df, pd.DataFrame) or co2_results_df.empty:
            co2_results_df = pd.DataFrame(co2_events)

        for i, shower_event in enumerate(shower_events):
            shower_time = shower_event["shower_on"]

            # Skip excluded events (time-based or duration-based)
            is_excluded, _ = is_event_excluded(shower_time)
            if not is_excluded:
                is_excluded = shower_event.get("is_excluded", False)
            if is_excluded:
                matched_pairs[i] = None
                continue

            # Find matching CO2 event
            co2_idx = match_shower_to_co2_event(
                shower_time,
                co2_results_df,
                time_tolerance_before=10.0,
                time_tolerance_after=10.0,
            )

            matched_pairs[i] = co2_idx

            # Add lambda value if available (handle both old and new column names)
            lambda_col = None
            if "lambda_average_mean" in co2_results_df.columns:
                lambda_col = "lambda_average_mean"
            elif "lambda_average_mean (h-1)" in co2_results_df.columns:
                lambda_col = "lambda_average_mean (h-1)"

            if co2_idx is not None and lambda_col is not None:
                lambda_val = co2_results_df.iloc[co2_idx][lambda_col]
                shower_event["lambda_ach"] = lambda_val
                shower_event["co2_event_idx"] = co2_idx

        print(
            f"  Matched: {sum(1 for v in matched_pairs.values() if v is not None)}/{len(shower_events)}"
        )
    else:
        # No CO2 processing - set all matched pairs to None
        for i in range(len(shower_events)):
            matched_pairs[i] = None

    print("\n" + "=" * 70)
    print("Event Management Complete")
    print("=" * 70 + "\n")

    return shower_events, co2_events, pd.DataFrame()


if __name__ == "__main__":
    print("Event Manager Module")
    print("This module should be imported and used by other scripts.")
    print("\nKey functions:")
    print("  - process_events_with_management(): Main processing function")
    print("  - filter_events_by_date(): Filter events by date")
    print("  - assign_test_names(): Generate test condition names")
