#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Management System
=======================

Utility module providing event matching, filtering, naming, configuration
lookup, and exclusion logic for the EPA Legionella shower experiment. Manages
the mapping between physical test conditions (water temperature, shower head,
spray pattern, mannequin, door position, bath fan state) and the CO2/shower
events recorded in the log files, assigns structured test names (e.g.,
0115_W48_Open_R01), and identifies events that should be excluded from
analysis.

Key Functions:
    - process_events_with_management(): Main entry point; orchestrates date
      filtering, name assignment, missing-event detection, synthetic event
      creation, and shower-to-CO2 matching
    - assign_test_names(): Assign structured test condition names and replicate
      numbers to all shower events
    - filter_events_by_date(): Remove events before EXPERIMENT_START_DATE
    - is_duration_excluded(): Flag short/long showers as water-temperature
      testing runs (excluded from analysis but retained in the log)
    - is_event_excluded(): Check against predefined exclusion list (tours,
      conflicting log entries, DST misalignment, etc.) and date ranges
    - get_test_configuration(): Return water temp, shower head, spray pattern,
      mannequin, door, fan, and config_key for any datetime using the
      transition-table system
    - detect_missing_events(): Identify showers without CO2 data and vice versa
    - create_synthetic_co2_event(): Build a placeholder CO2 event with expected
      timing for unmatched shower events
    - get_water_temp_sort_key(): Extract numeric temperature for sorting

Processing Features:
    - Transition-table configuration system: each parameter (water temperature,
      shower head, spray pattern, mannequin, door position, fan status) has a
      list of (datetime, value) pairs; the active value is the most recent
      entry at or before the event time
    - Duration-based exclusion: computer-controlled analysis showers run
      exactly 10 min ± 5 s; events outside this window are flagged as water
      temperature testing runs and receive event_number=None
    - Predefined exclusion registry: EXCLUDED_EVENTS dict maps specific
      datetimes (±60 s tolerance) to human-readable exclusion reasons;
      EXCLUDED_RANGES list covers multi-event spans (e.g., conflicting logs,
      DST misalignment, instrument failures)
    - Replicate numbering: per-condition counters (keyed by date, water temp,
      shower head, spray pattern, mannequin, door position, fan) produce
      sequential R01, R02, … suffixes
    - Bidirectional synthetic event creation: uses registry module (lazy
      import) for duration inference from neighboring events when available
    - Bath fan detection: checks shower_log for fan state from shower start
      through 2 hours after shower end; pre-shower fan use is not counted

Methodology:
    1. Filter shower and CO2 event lists to EXPERIMENT_START_DATE or later
    2. Apply is_duration_excluded() to each shower; duration-excluded events
       receive is_excluded=True and event_number=None and are skipped for
       naming and replicate counting
    3. Assign water temp, shower head, spray pattern, mannequin, door position,
       fan status, config_key, replicate number, and full test_name to each
       non-excluded shower event
    4. Detect showers without a matching CO2 event (within ±10 min)
    5. Optionally create synthetic CO2 events for unmatched showers
    6. Match each non-excluded shower event to its CO2 event; attach
       lambda_ach and co2_event_idx to the shower event dict

Input Files:
    - None (all data passed as function arguments: lists of event dicts and
      pandas DataFrames loaded by the calling script)

Output Files:
    - None (results returned as modified event lists and a DataFrame;
      callers are responsible for writing event_log.csv and registry files)

Naming Convention Format:
    MMDD_W##[_ShowerHead[_SprayPattern]][_Mannequin]_DoorPos[_Fan]_RNN

    Components:
    - MMDD: Month and day (e.g., 0114 for January 14)
    - W##: Water temperature code (e.g., W48 = 48 °C); always pure numeric
    - ShowerHead: Omitted for Standard; "Pepco" for the Pepco shower head
    - SprayPattern: Omitted if no variable spray pattern; "Wide", "Narrow",
      or "Mid" for spray settings on the Pepco head
    - _Mannequin: Appended only when a mannequin was present during the test
    - DoorPos: Door position (Open, Closed, or Partial)
    - _Fan: Appended only when the bath fan ran during the test period
    - RNN: Replicate number (R01, R02, etc.)

    config_key Format:
    W##[_ShowerHead[_SprayPattern]][_Mannequin]_DoorXxx_FanXxx
    Used for grouping events across days with identical test conditions.

    Examples:
    - 0115_W48_Open_R01              (standard head, 48 °C, open door)
    - 0122_W11_Open_R03              (standard head, 11 °C)
    - 0224_W52_Pepco_Wide_Open_R01   (Pepco head, wide spray, 52 °C)
    - 0309_W40_Pepco_Narrow_Open_R01 (Pepco head, narrow spray, 40 °C)
    - 0311_W40_Pepco_Narrow_Mannequin_Open_R01

Time of Day:
    get_time_of_day() is retained for internal use (penetration factor
    averaging windows in particle_calculations.py) but is no longer included
    in test names or replicate counters, as Day/Night has not shown an
    analytical impact.

Test Parameters:
    - Water Temperature: W## code (e.g., W48 = 48 °C)
    - Shower Head: Standard or Pepco
    - Spray Pattern: Wide, Narrow, Mid, or None (for standard head)
    - Mannequin: Whether a mannequin was present during the test
    - Door Position: Open, Closed, or Partial
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

# Water temperature transitions: (datetime, W## code)
# Codes are pure numeric: W11, W22, W25, etc. Shower head and spray pattern
# are tracked separately in SHOWER_HEAD_TRANSITIONS and SPRAY_PATTERN_TRANSITIONS.
WATER_TEMP_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "W48"),  # Experiment start
    (datetime(2026, 1, 22, 14, 0, 0), "W11"),  # Cold water from Jan 22 2PM
    (datetime(2026, 2, 2, 17, 0, 0), "W25"),  # Warm water from Feb 2 5PM
    (datetime(2026, 2, 5, 10, 0, 0), "W30"),  # Warm water from Feb 5 10AM
    (datetime(2026, 2, 9, 10, 0, 0), "W37"),  # Warm water from Feb 9 10AM
    (datetime(2026, 2, 11, 8, 0, 0), "W23"),  # Feb 11 8AM (data excluded; kept for documentation)
    (datetime(2026, 2, 13, 11, 0, 0), "W22"),  # Warm water from Feb 13 11AM
    (datetime(2026, 2, 16, 11, 0, 0), "W43"),  # Hot water from Feb 16 11AM
    (datetime(2026, 2, 18, 10, 23, 0), "W14"),  # Cold water from Feb 18 10:23AM
    (datetime(2026, 2, 20, 8, 0, 0), "W53"),  # Hot water from Feb 20 8AM
    (datetime(2026, 2, 24, 8, 0, 0), "W52"),  # Pepco head installed; 52°C from Feb 24 8AM
    (datetime(2026, 2, 26, 10, 23, 0), "W49"),  # Pepco 49°C from Feb 26 10:23AM
    (datetime(2026, 3, 2, 9, 23, 0), "W40"),  # Pepco 40°C from Mar 2 9:23AM
    (datetime(2026, 3, 12, 10, 47, 0), "W38")  # 38°C from Mar 12 10:47AM
]

# Shower head transitions: (datetime, head_type)
# Types: "Standard", "Pepco"
SHOWER_HEAD_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "Standard"),  # Standard head from experiment start
    (datetime(2026, 2, 24, 8, 0, 0), "Pepco"),  # Pepco shower head installed Feb 24
    (datetime(2026, 4, 10, 8, 55, 0), "FilterWand"),  # Filtered Wand head installed Apr 10
    (datetime(2026, 4, 13, 11, 35, 0), "Used")  # Used head installed Apr 13
]

# Spray pattern transitions: (datetime, pattern)
# Patterns: None (standard head), "Wide", "Narrow", "Mid"
SPRAY_PATTERN_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), None),  # Standard head; no spray pattern variable
    (datetime(2026, 2, 24, 8, 0, 0), "Wide"),  # Pepco wide spray from Feb 24
    (datetime(2026, 3, 2, 9, 23, 0), "Narrow"),  # Pepco narrow spray from Mar 2
    (datetime(2026, 3, 4, 10, 23, 0), "Wide"),  # Pepco wide spray from Mar 4
    (datetime(2026, 3, 6, 8, 47, 0), "Mid"),  # Pepco mid spray from Mar 6
    (datetime(2026, 3, 9, 8, 47, 0), "Narrow"),  # Pepco narrow spray from Mar 9
    (datetime(2026, 3, 19, 8, 25, 0), "Wide"),  # Filtered Wand wide spray from Apr 10
    (datetime(2026, 4, 10, 8, 55, 0), None),  # Filtered Wand spray from Apr 10
    (datetime(2026, 4, 13, 11, 35, 0), "rainfall"),  # Used head rainfall spray from Apr 13
    (datetime(2026, 4, 15, 8, 25, 0), "12Nozzle"),  # Used head 12-nozzle spray from Apr 15
    (datetime(2026, 4, 17, 8, 35, 0), "SingleWide")  # Used head SingleWide spray from Apr 17
]

# Mannequin transitions: (datetime, present)
# Values: True (mannequin present), False (no mannequin)
MANNEQUIN_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), False),  # No mannequin from experiment start
    (datetime(2026, 3, 11, 9, 47, 0), True),  # Mannequin added Mar 11
    (datetime(2026, 3, 12, 10, 47, 0), False),  # Mannequin removed Mar 12
    (datetime(2026, 3, 13, 8, 45, 0), True),  # Mannequin re-added Mar 13
    (datetime(2026, 3, 17, 8, 25, 0), False),  # Mannequin removed Mar 14
    (datetime(2026, 3, 18, 9, 45, 0), True),  # Mannequin re-added Mar 18
    (datetime(2026, 3, 19, 8, 25, 0), False),  # Mannequin removed Mar 19
    (datetime(2026, 3, 22, 9, 25, 0), True),  # Mannequin re-added Mar 22
    (datetime(2026, 3, 27, 8, 25, 0), False)  # Mannequin removed Mar 23
]

# Door position transitions: (datetime, position)
# Positions: "Open", "Closed", "Partial"
DOOR_POSITION_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "Open"),  # Door open from experiment start
    (datetime(2026, 4, 8, 9, 15, 0), "Closed"),  # Door closed from Apr 8
    (datetime(2026, 4, 10, 8, 55, 0), "Open")  # Door open from Apr 10

# Bath fan transitions: (datetime, status)
# Status: "On", "Off"
# Note: This is for PLANNED fan operation. Actual fan status during tests
# is still detected from the shower log via check_fan_during_test().
FAN_STATUS_TRANSITIONS = [
    (datetime(2026, 1, 14, 0, 0, 0), "Off"),  # Fan off from experiment start
    (datetime(2026, 3, 31, 12, 45, 0), "On"),  # Fan on from March 31 for 12 min. during shower
    (datetime(2026, 4, 10, 8, 55, 0), "Off")   # Fan off from Apr 10
]

# Planned bath fan run duration (minutes). None = no planned fan operation.
# Tracks how long the fan is intended to run during the test period; if the
# protocol changes (e.g. 15 min instead of 12) add a new entry here and
# measure_fan_duration_during_test() will capture the actual runtime regardless.
FAN_DURATION_TRANSITIONS: List[Tuple[datetime, Optional[float]]] = [
    (datetime(2026, 1, 14, 0, 0, 0), None),   # No planned fan
    (datetime(2026, 3, 31, 12, 45, 0), 12.0),  # Fan planned to run 12 min from shower start
    (datetime(2026, 4, 10, 8, 55, 0), None)    # No planned fan from Apr 10
]

# Time of day boundaries (hour of day).
# Retained for penetration factor window calculations in particle_calculations.py.
# No longer included in test names or replicate counters.
TIME_OF_DAY_RANGES = {
    "Day": (5, 17),  # 5am - 5pm
    "Night": (17, 5)  # 5pm - 5am (wraps around midnight)
}

# Predefined point exclusions: datetime -> reason
# For single events identified as problematic after the fact.
# Use a tolerance of ±60 s in is_event_excluded().
EXCLUDED_EVENTS = {
    datetime(2026, 1, 22, 15, 0, 0): "Tour in house during test",
    datetime(2026, 1, 29, 15, 0, 0): "People in house"
}

# Date-range exclusions: list of (start_datetime, end_datetime, reason).
# Any event whose shower_on time falls within [start, end] (inclusive) is
# excluded. Use this instead of listing every individual event for multi-day
# or multi-event exclusion windows.
EXCLUDED_RANGES: List[Tuple[datetime, datetime, str]] = [
    (
        datetime(2026, 2, 11, 8, 0, 0),
        datetime(2026, 2, 13, 11, 0, 0),
        "Conflicting log entries",
    ),
    (
        datetime(2026, 3, 8, 0, 0, 0),
        datetime(2026, 3, 10, 0, 0, 0),
        "Daylight saving (instrument data misalignment) & elevated RH in the bedroom",
    ),
    (
        datetime(2026, 3, 15, 0, 0, 0),
        datetime(2026, 3, 16, 12, 00, 00),
        "CO2 system failure",
    )
]

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


def _get_config_value_at_time(dt: datetime, transitions: List[Tuple[datetime, str]]) -> str:
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

    Returns a pure W## code (e.g., "W48", "W11", "W40"). Shower head type
    and spray pattern are tracked separately.

    Parameters:
        dt: Datetime of the event

    Returns:
        String: Water temperature code (e.g., "W48", "W11", "W40")
    """
    return _get_config_value_at_time(dt, WATER_TEMP_TRANSITIONS)


def get_shower_head(dt: datetime) -> str:
    """
    Determine shower head type based on datetime.

    Parameters:
        dt: Datetime of the event

    Returns:
        String: "Standard" or "Pepco"
    """
    return _get_config_value_at_time(dt, SHOWER_HEAD_TRANSITIONS)


def get_spray_pattern(dt: datetime) -> Optional[str]:
    """
    Determine spray pattern setting based on datetime.

    Parameters:
        dt: Datetime of the event

    Returns:
        String: "Wide", "Narrow", "Mid", or None (for standard head)
    """
    return _get_config_value_at_time(dt, SPRAY_PATTERN_TRANSITIONS)


def get_mannequin(dt: datetime) -> bool:
    """
    Determine whether a mannequin was present based on datetime.

    Parameters:
        dt: Datetime of the event

    Returns:
        Bool: True if mannequin present, False otherwise
    """
    return _get_config_value_at_time(dt, MANNEQUIN_TRANSITIONS)


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


def get_planned_fan_duration(dt: datetime) -> Optional[float]:
    """
    Return the planned bath fan run duration (minutes) at a given datetime.

    Uses FAN_DURATION_TRANSITIONS. When the fan protocol changes, add a new
    entry to FAN_DURATION_TRANSITIONS rather than editing this function.

    Parameters:
        dt: Datetime of the event

    Returns:
        Planned duration in minutes, or None if no fan is planned.
    """
    # _get_config_value_at_time is typed -> str but returns whatever value is
    # stored in the transition list; None and float work fine at runtime.
    return _get_config_value_at_time(dt, FAN_DURATION_TRANSITIONS)  # type: ignore[return-value]


def get_test_configuration(dt: datetime) -> Dict:
    """
    Get complete test configuration for a given datetime.

    Returns a dictionary with all configuration parameters that can be used
    for grouping, filtering, and labeling results.

    Parameters:
        dt: Datetime of the event

    Returns:
        Dictionary with configuration keys:
            - water_temp: Temperature code (e.g., "W48", "W11", "W40")
            - shower_head: "Standard" or "Pepco"
            - spray_pattern: "Wide", "Narrow", "Mid", or None
            - mannequin: True or False
            - door_position: "Open", "Closed", or "Partial"
            - planned_fan: "On" or "Off"
            - config_key: Combined key for grouping
              (e.g., "W48_DoorOpen_FanOff",
                     "W40_Pepco_Narrow_DoorOpen_FanOff",
                     "W40_Pepco_Narrow_Mannequin_DoorOpen_FanOff")
    """
    water_temp = get_water_temperature_code(dt)
    shower_head = get_shower_head(dt)
    spray_pattern = get_spray_pattern(dt)
    mannequin = get_mannequin(dt)
    door_pos = get_door_position(dt)
    fan_status = get_planned_fan_status(dt)

    # Build config_key: W##[_ShowerHead[_SprayPattern]][_Mannequin]_DoorXxx_FanXxx
    key_parts = [water_temp]
    if shower_head != "Standard":
        key_parts.append(shower_head)
        if spray_pattern:
            key_parts.append(spray_pattern)
    if mannequin:
        key_parts.append("Mannequin")
    key_parts.append(f"Door{door_pos}")
    key_parts.append("FanOn" if fan_status == "On" else "FanOff")
    config_key = "_".join(key_parts)

    return {
        "water_temp": water_temp,
        "shower_head": shower_head,
        "spray_pattern": spray_pattern,
        "mannequin": mannequin,
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
    # Collect all unique transition points across every parameter
    all_transitions = set()
    for transitions in (
        WATER_TEMP_TRANSITIONS,
        SHOWER_HEAD_TRANSITIONS,
        SPRAY_PATTERN_TRANSITIONS,
        MANNEQUIN_TRANSITIONS,
        DOOR_POSITION_TRANSITIONS,
        FAN_STATUS_TRANSITIONS,
        FAN_DURATION_TRANSITIONS,
    ):
        for dt, _ in transitions:
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
    mask = (shower_log["datetime_EDT"] >= test_start) & (shower_log["datetime_EDT"] <= test_end)
    test_period_log = shower_log[mask]

    # Check if fan was ever on during this period
    if len(test_period_log) > 0:
        return bool((test_period_log["bath_fan"] > 0).any())

    return False


def measure_fan_duration_during_test(
    shower_on: datetime, shower_off: datetime, shower_log: pd.DataFrame
) -> Optional[float]:
    """
    Measure actual bath fan runtime during the test period from the shower log.

    Integrates the time-series of bath_fan state entries to compute total
    fan-on minutes from shower_on through 2 hours after shower_off.  Returns
    None when the fan was never on or when no log data exists for the window.

    Note: shower_log is a state-change log; each row records the value that
    was active *from* that timestamp until the next row.  The last row in the
    window is assumed to stay active until ``shower_off + 2 h``.

    Parameters:
        shower_on: Shower start time
        shower_off: Shower end time
        shower_log: DataFrame with columns 'datetime_EDT' and 'bath_fan'

    Returns:
        Total fan-on duration in minutes, or None if fan never ran.
    """
    test_start = shower_on
    test_end = shower_off + timedelta(hours=2)

    mask = (shower_log["datetime_EDT"] >= test_start) & (shower_log["datetime_EDT"] <= test_end)
    test_period = shower_log[mask].copy().sort_values("datetime_EDT")

    if test_period.empty:
        return None

    total_minutes = 0.0
    times = test_period["datetime_EDT"].tolist()
    states = test_period["bath_fan"].tolist()

    for i in range(len(times) - 1):
        if states[i] > 0:
            total_minutes += (times[i + 1] - times[i]).total_seconds() / 60.0

    # Last entry stays active until test_end
    if states[-1] > 0:
        total_minutes += (test_end - times[-1]).total_seconds() / 60.0

    return total_minutes if total_minutes > 0 else None


def generate_test_name(
    shower_time: datetime,
    water_temp: str,
    replicate_num: int,
    shower_head: str = "Standard",
    spray_pattern: Optional[str] = None,
    mannequin: bool = False,
    fan_status: bool = False,
    door_position: str = "Open",
) -> str:
    """
    Generate a test condition name following the naming convention.

    Format: MMDD_W##[_ShowerHead[_SprayPattern]][_Mannequin]_DoorPos[_Fan]_RNN

    Parameters:
        shower_time: Datetime of shower start
        water_temp: Water temperature code (e.g., "W48", "W40")
        replicate_num: Replicate number (1-indexed)
        shower_head: "Standard" or "Pepco" (default "Standard"; omitted from name)
        spray_pattern: "Wide", "Narrow", "Mid", or None (default None; omitted if None)
        mannequin: Whether mannequin was present (default False; omitted if False)
        fan_status: Whether bath fan ran during test (default False)
        door_position: "Open", "Closed", or "Partial" (default "Open")

    Returns:
        String: Test name (e.g., "0115_W48_Open_R01",
                               "0309_W40_Pepco_Narrow_Open_R01",
                               "0311_W40_Pepco_Narrow_Mannequin_Open_R01")
    """
    date_str = shower_time.strftime("%m%d")

    components = [date_str, water_temp]

    # Shower head: only include if not Standard
    if shower_head != "Standard":
        components.append(shower_head)
        if spray_pattern:
            components.append(spray_pattern)

    if mannequin:
        components.append("Mannequin")

    components.append(door_position)

    if fan_status:
        components.append("Fan")

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
            event_time = event["injection_start"] + timedelta(minutes=EXPECTED_CO2_BEFORE_SHOWER)
        else:
            continue

        if event_time >= start_date:
            filtered.append(event)

    return filtered


def is_event_excluded(event_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check if an event should be excluded from analysis.

    Checks both point exclusions (EXCLUDED_EVENTS, with ±60 s tolerance) and
    date-range exclusions (EXCLUDED_RANGES, inclusive on both ends).

    Parameters:
        event_time: Datetime of the event

    Returns:
        Tuple of (is_excluded: bool, reason: str or None)
    """
    # Check exact point exclusions (±60 s tolerance)
    if event_time in EXCLUDED_EVENTS:
        return True, EXCLUDED_EVENTS[event_time]
    for excluded_time, reason in EXCLUDED_EVENTS.items():
        if abs((event_time - excluded_time).total_seconds()) < 60:
            return True, reason

    # Check date-range exclusions
    for range_start, range_end, reason in EXCLUDED_RANGES:
        if range_start <= event_time <= range_end:
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
    hour_after_injection = injection_start.replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )
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


def assign_test_names(shower_events: List[Dict], shower_log: pd.DataFrame) -> List[Dict]:
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
            - water_temp: Water temperature code (e.g., "W48", "W40")
            - shower_head: "Standard" or "Pepco"
            - spray_pattern: "Wide", "Narrow", "Mid", or None
            - mannequin: True if mannequin present, False otherwise
            - door_position: Door position (Open/Closed/Partial)
            - planned_fan: Planned fan status (On/Off)
            - fan_during_test: Actual fan status during test (bool)
            - time_of_day: Time of day ("Day"/"Night"); retained for
              penetration window calculations, not used in test name
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
            config = get_test_configuration(shower_time)
            event["water_temp"] = config["water_temp"]
            event["shower_head"] = config["shower_head"]
            event["spray_pattern"] = config["spray_pattern"]
            event["mannequin"] = config["mannequin"]
            event["door_position"] = config["door_position"]
            event["planned_fan"] = config["planned_fan"]
            event["fan_during_test"] = False
            event["fan_duration_min"] = None
            event["time_of_day"] = get_time_of_day(shower_time)
            event["config_key"] = ""
            event["replicate_num"] = 0
            continue

        # Get full test configuration
        config = get_test_configuration(shower_time)
        water_temp = config["water_temp"]
        shower_head = config["shower_head"]
        spray_pattern = config["spray_pattern"]
        mannequin = config["mannequin"]
        door_position = config["door_position"]
        planned_fan = config["planned_fan"]
        config_key = config["config_key"]

        # Time of day: retained for penetration window calculations but no
        # longer included in the test name or replicate counter key
        time_of_day = get_time_of_day(shower_time)
        fan_during_test = check_fan_during_test(shower_time, shower_off, shower_log)
        fan_duration_min = (
            measure_fan_duration_during_test(shower_time, shower_off, shower_log)
            if fan_during_test
            else None
        )

        # Condition key for replicate counting: date + all test parameters
        # (time_of_day deliberately excluded — Day/Night has no analytical impact)
        date_str = shower_time.strftime("%m%d")
        condition_parts = [date_str, water_temp]
        if shower_head != "Standard":
            condition_parts.append(shower_head)
            if spray_pattern:
                condition_parts.append(spray_pattern)
        if mannequin:
            condition_parts.append("Mannequin")
        condition_parts.append(door_position)
        if fan_during_test:
            condition_parts.append("Fan")
        condition_key_for_replicates = "_".join(condition_parts)

        # Get next replicate number for this condition
        replicate_num = replicate_counters.get(condition_key_for_replicates, 0) + 1
        replicate_counters[condition_key_for_replicates] = replicate_num

        # Generate full test name
        test_name = generate_test_name(
            shower_time,
            water_temp,
            replicate_num,
            shower_head=shower_head,
            spray_pattern=spray_pattern,
            mannequin=mannequin,
            fan_status=fan_during_test,
            door_position=door_position,
        )

        # Add all configuration fields to event
        event["test_name"] = test_name
        event["water_temp"] = water_temp
        event["shower_head"] = shower_head
        event["spray_pattern"] = spray_pattern
        event["mannequin"] = mannequin
        event["door_position"] = door_position
        event["planned_fan"] = planned_fan
        event["fan_during_test"] = fan_during_test
        event["fan_duration_min"] = fan_duration_min
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
        showers_missing_co2, co2_missing_shower = detect_missing_events(shower_events, co2_events)

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
