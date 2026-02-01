#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log File Analysis Utility
=========================

This script analyzes shower and CO2 injection log files to identify potential
data quality issues and mismatches between correlated events. It is designed
to help diagnose problems in the automated shower and CO2 injection experiment
logging system.

Key Functions:
    - Load and parse shower log and CO2 injection log files
    - Identify shower events without matching CO2 injections (within ±10 min window)
    - Identify CO2 injections without matching shower events
    - Detect duplicate consecutive ON events (missing OFF state)
    - Identify unusual shower durations (<3 min or >20 min)

Analysis Features:
    - Date range filtering (events >= 2026-01-14)
    - Temporal matching with ±10 minute tolerance around expected 20-min offset
    - Consecutive state validation for ON/OFF sequencing
    - Duration analysis for shower events

Methodology:
    1. Load shower_log_file.csv and CO2_log_file.csv
    2. Filter to events after the experiment start date
    3. For each shower ON event, search for matching CO2 injection 20 min prior
    4. For each CO2 ON event, search for matching shower 20 min later
    5. Check for consecutive ON states without intervening OFF
    6. Calculate and validate shower durations

Output Files:
    - Console output with analysis results and potential issues

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import pandas as pd
from datetime import datetime, timedelta

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import get_common_file

# Load log files using data_config.json paths
shower_log_path = get_common_file("shower_log_file")
co2_log_path = get_common_file("co2_log_file")

shower_log = pd.read_csv(shower_log_path)
co2_log = pd.read_csv(co2_log_path)

# Convert datetime
shower_log["datetime_EDT"] = pd.to_datetime(shower_log["datetime_EDT"])
co2_log["datetime_EDT"] = pd.to_datetime(co2_log["datetime_EDT"])

print("=" * 70)
print("LOG FILE ANALYSIS")
print("=" * 70)

# Shower log analysis
print("\nSHOWER LOG:")
print(f"  Total rows: {len(shower_log)}")
print(f"  Date range: {shower_log['datetime_EDT'].min()} to {shower_log['datetime_EDT'].max()}")

# Find shower ON events
shower_on_events = shower_log[shower_log["shower"] > 0]
print(f"  Shower ON events: {len(shower_on_events)}")

# Find shower events after 2026-01-14
shower_after_start = shower_on_events[shower_on_events["datetime_EDT"] >= datetime(2026, 1, 14)]
print(f"  Shower ON events >= 2026-01-14: {len(shower_after_start)}")

print("\n  First 5 shower ON events:")
for idx, row in shower_on_events.head(5).iterrows():
    print(f"    {row['datetime_EDT']}")

print("\n  Last 5 shower ON events:")
for idx, row in shower_on_events.tail(5).iterrows():
    print(f"    {row['datetime_EDT']}")

# CO2 log analysis
print("\nCO2 LOG:")
print(f"  Total rows: {len(co2_log)}")
print(f"  Date range: {co2_log['datetime_EDT'].min()} to {co2_log['datetime_EDT'].max()}")

# Find CO2 ON events
co2_on_events = co2_log[co2_log["CO2"] > 0]
print(f"  CO2 ON events: {len(co2_on_events)}")

# Find CO2 events after 2026-01-14
co2_after_start = co2_on_events[co2_on_events["datetime_EDT"] >= datetime(2026, 1, 14)]
print(f"  CO2 ON events >= 2026-01-14: {len(co2_after_start)}")

print("\n  First 5 CO2 ON events:")
for idx, row in co2_on_events.head(5).iterrows():
    print(f"    {row['datetime_EDT']}")

print("\n  Last 5 CO2 ON events:")
for idx, row in co2_on_events.tail(5).iterrows():
    print(f"    {row['datetime_EDT']}")

# Check for potential issues
print("\n" + "=" * 70)
print("POTENTIAL ISSUES:")
print("=" * 70)

# Issue 1: Check for showers without corresponding CO2 (after 2026-01-14)
print("\nChecking for shower events without matching CO2...")
showers_missing_co2 = []
for idx, shower_row in shower_after_start.iterrows():
    shower_time = shower_row["datetime_EDT"]
    expected_co2_time = shower_time - timedelta(minutes=20)

    # Look for CO2 within +/- 10 minutes
    min_time = expected_co2_time - timedelta(minutes=10)
    max_time = expected_co2_time + timedelta(minutes=10)

    matching_co2 = co2_on_events[
        (co2_on_events["datetime_EDT"] >= min_time) &
        (co2_on_events["datetime_EDT"] <= max_time)
    ]

    if len(matching_co2) == 0:
        showers_missing_co2.append(shower_time)

if showers_missing_co2:
    print(f"  Found {len(showers_missing_co2)} shower events without CO2:")
    for shower_time in showers_missing_co2[:10]:  # Show first 10
        print(f"    {shower_time}")
    if len(showers_missing_co2) > 10:
        print(f"    ... and {len(showers_missing_co2) - 10} more")
else:
    print("  All shower events have matching CO2 injections!")

# Issue 2: Check for CO2 without corresponding shower (after 2026-01-14)
print("\nChecking for CO2 events without matching shower...")
co2_missing_shower = []
for idx, co2_row in co2_after_start.iterrows():
    co2_time = co2_row["datetime_EDT"]
    expected_shower_time = co2_time + timedelta(minutes=20)

    # Look for shower within +/- 10 minutes
    min_time = expected_shower_time - timedelta(minutes=10)
    max_time = expected_shower_time + timedelta(minutes=10)

    matching_shower = shower_on_events[
        (shower_on_events["datetime_EDT"] >= min_time) &
        (shower_on_events["datetime_EDT"] <= max_time)
    ]

    if len(matching_shower) == 0:
        co2_missing_shower.append(co2_time)

if co2_missing_shower:
    print(f"  Found {len(co2_missing_shower)} CO2 events without shower:")
    for co2_time in co2_missing_shower[:10]:  # Show first 10
        print(f"    {co2_time}")
    if len(co2_missing_shower) > 10:
        print(f"    ... and {len(co2_missing_shower) - 10} more")
else:
    print("  All CO2 events have matching showers!")

# Issue 3: Check for duplicate ON events (shower turning on without turning off first)
print("\nChecking for duplicate consecutive ON events...")
duplicate_shower = []
for i in range(len(shower_log) - 1):
    if shower_log.iloc[i]["shower"] > 0 and shower_log.iloc[i + 1]["shower"] > 0:
        duplicate_shower.append(shower_log.iloc[i]["datetime_EDT"])

if duplicate_shower:
    print(f"  Found {len(duplicate_shower)} duplicate shower ON events:")
    for dt in duplicate_shower[:5]:
        print(f"    {dt}")
else:
    print("  No duplicate shower ON events found!")

duplicate_co2 = []
for i in range(len(co2_log) - 1):
    if co2_log.iloc[i]["CO2"] > 0 and co2_log.iloc[i + 1]["CO2"] > 0:
        duplicate_co2.append(co2_log.iloc[i]["datetime_EDT"])

if duplicate_co2:
    print(f"  Found {len(duplicate_co2)} duplicate CO2 ON events:")
    for dt in duplicate_co2[:5]:
        print(f"    {dt}")
else:
    print("  No duplicate CO2 ON events found!")

# Issue 4: Check for unusual shower durations
print("\nChecking for unusual shower durations...")
shower_durations = []
shower_log_sorted = shower_log.sort_values("datetime_EDT")
i = 0
while i < len(shower_log_sorted):
    if shower_log_sorted.iloc[i]["shower"] > 0:
        on_time = shower_log_sorted.iloc[i]["datetime_EDT"]
        off_time = None

        for j in range(i + 1, len(shower_log_sorted)):
            if shower_log_sorted.iloc[j]["shower"] == 0:
                off_time = shower_log_sorted.iloc[j]["datetime_EDT"]
                i = j
                break

        if off_time:
            duration_min = (off_time - on_time).total_seconds() / 60.0
            shower_durations.append((on_time, duration_min))
    i += 1

unusual_durations = [(t, d) for t, d in shower_durations if d < 3 or d > 20]
if unusual_durations:
    print(f"  Found {len(unusual_durations)} showers with unusual durations (<3 min or >20 min):")
    for time, duration in unusual_durations[:5]:
        print(f"    {time}: {duration:.1f} minutes")
else:
    print("  All shower durations look reasonable (3-20 minutes)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
