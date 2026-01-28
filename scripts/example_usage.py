#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Event Management System
=========================================

This script demonstrates how to use the new event management system
for the EPA Legionella project.

Author: Nathan Lima, NIST
Date: January 2026
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_manager import (
    process_events_with_management,
    filter_events_by_date,
    is_event_excluded,
    assign_test_names,
    create_event_log,
)
from src.env_data_loader import load_shower_log, identify_shower_events
from src.co2_decay_analysis import load_co2_log
from src.data_paths import get_data_root


def example_basic_usage():
    """Basic example: Load and process events"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Event Processing")
    print("=" * 70)

    # Load shower log
    print("\n1. Loading shower log...")
    shower_log = load_shower_log()
    print(f"   Loaded {len(shower_log)} log entries")

    # Identify shower events
    print("\n2. Identifying shower events...")
    raw_events = identify_shower_events(shower_log)
    print(f"   Found {len(raw_events)} total shower events")

    # Filter by date
    print("\n3. Filtering events (>= 2026-01-14)...")
    filtered_events = filter_events_by_date(raw_events)
    print(f"   {len(filtered_events)} events after date filtering")

    # Assign test names
    print("\n4. Assigning test condition names...")
    named_events = assign_test_names(filtered_events, shower_log)

    print("\n   First 5 events:")
    for event in named_events[:5]:
        shower_time = event["shower_on"]
        test_name = event.get("test_name", "N/A")
        water_temp = event.get("water_temp", "N/A")
        time_of_day = event.get("time_of_day", "N/A")
        print(f"   - {test_name}")
        print(f"     Time: {shower_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"     Water: {water_temp}, Period: {time_of_day}")

    print("\n" + "=" * 70 + "\n")


def example_event_exclusion():
    """Example: Check for excluded events"""
    print("=" * 70)
    print("EXAMPLE 2: Event Exclusion Check")
    print("=" * 70)

    # Check specific times for exclusion
    test_times = [
        datetime(2026, 1, 22, 15, 0, 0),  # Should be excluded (tour)
        datetime(2026, 1, 22, 16, 0, 0),  # Should not be excluded
        datetime(2026, 1, 14, 9, 0, 0),   # Should not be excluded
    ]

    print("\nChecking exclusion status:")
    for test_time in test_times:
        is_excluded, reason = is_event_excluded(test_time)
        status = "EXCLUDED" if is_excluded else "INCLUDED"
        reason_text = f" - {reason}" if is_excluded else ""
        print(f"  {test_time.strftime('%Y-%m-%d %H:%M')}: {status}{reason_text}")

    print("\n" + "=" * 70 + "\n")


def example_full_processing():
    """Example: Full processing with event manager"""
    print("=" * 70)
    print("EXAMPLE 3: Full Event Processing Pipeline")
    print("=" * 70)

    # Setup
    output_dir = get_data_root() / "output"

    # Load data
    print("\n1. Loading data files...")
    shower_log = load_shower_log()
    raw_events = identify_shower_events(shower_log)

    # For this example, we'll use empty CO2 results
    # In real usage, load from co2_lambda_results.csv
    import pandas as pd
    co2_results = pd.DataFrame()

    print(f"   Raw shower events: {len(raw_events)}")

    # Process with event manager
    print("\n2. Running event management system...")
    print("   (This will filter, name, match, and log events)")

    try:
        events, co2_events, event_log = process_events_with_management(
            raw_events,
            [],  # Empty CO2 events for this example
            shower_log,
            co2_results,
            output_dir,
            create_synthetic=True
        )

        print(f"\n3. Results:")
        print(f"   Processed events: {len(events)}")
        print(f"   Event log rows: {len(event_log)}")
        print(f"   Event log saved to: {output_dir}/event_log.csv")

        # Show sample of results
        print("\n4. Sample processed events:")
        for i, event in enumerate(events[:3], 1):
            test_name = event.get("test_name", "N/A")
            shower_time = event["shower_on"]
            lambda_val = event.get("lambda_ach", "N/A")
            print(f"\n   Event {i}: {test_name}")
            print(f"   - Time: {shower_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   - Lambda: {lambda_val}")
            print(f"   - Water: {event.get('water_temp', 'N/A')}")
            print(f"   - Period: {event.get('time_of_day', 'N/A')}")
            print(f"   - Replicate: R{event.get('replicate_num', 0):02d}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        print("Note: This example may fail without CO2 data loaded")

    print("\n" + "=" * 70 + "\n")


def example_filtering_by_conditions():
    """Example: Filter events by test conditions"""
    print("=" * 70)
    print("EXAMPLE 4: Filtering by Test Conditions")
    print("=" * 70)

    # Load and process
    shower_log = load_shower_log()
    raw_events = identify_shower_events(shower_log)
    filtered_events = filter_events_by_date(raw_events)
    named_events = assign_test_names(filtered_events, shower_log)

    # Filter by conditions
    hot_water_events = [e for e in named_events if e.get("water_temp") == "HW"]
    cold_water_events = [e for e in named_events if e.get("water_temp") == "CW"]
    morning_events = [e for e in named_events if e.get("time_of_day") == "Morning"]
    fan_events = [e for e in named_events if e.get("fan_during_test") == True]

    print("\nEvent counts by condition:")
    print(f"  Total events: {len(named_events)}")
    print(f"  Hot water tests: {len(hot_water_events)}")
    print(f"  Cold water tests: {len(cold_water_events)}")
    print(f"  Morning tests: {len(morning_events)}")
    print(f"  Tests with fan: {len(fan_events)}")

    # Get unique test conditions
    test_conditions = set(e.get("test_name", "").rsplit("_R", 1)[0]
                          for e in named_events if e.get("test_name"))

    print(f"\n  Unique test conditions: {len(test_conditions)}")
    print("\n  Test condition breakdown:")
    for condition in sorted(test_conditions)[:10]:  # Show first 10
        count = sum(1 for e in named_events
                   if e.get("test_name", "").startswith(condition))
        print(f"    {condition}_RXX: {count} replicates")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("Event Management System - Usage Examples")
    print("*" * 70)
    print("\n")

    try:
        # Run examples
        example_basic_usage()
        example_event_exclusion()
        example_filtering_by_conditions()

        # Full processing example (may need CO2 data)
        print("\nNote: Example 3 (full processing) requires CO2 data.")
        print("      Run the main analysis scripts to see full functionality.\n")

        # example_full_processing()  # Uncomment when CO2 data is available

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all data files are available and paths are correct.\n")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 70)
    print("Examples Complete")
    print("*" * 70 + "\n")
