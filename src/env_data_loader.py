#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Data Loader Module
=================================

This module provides unified data loading functions for all environmental
sensors deployed in the EPA Legionella project. It abstracts instrument-specific
file formats and provides a consistent DataFrame output with datetime indexing.

Key Functions:
    - load_aranet_data: Load Aranet4 CO2/RH/Temp sensor data from Excel files
    - load_quantaq_data: Load QuantAQ MODULAIR-PM processed CSV files
    - load_daq_data: Load indoor DAQ data (Vaisala RH/Temp sensors)
    - load_aio2_data: Load AIO2 weather station data (wind speed/direction)
    - load_sensor_data: Generic loader using SENSOR_CONFIG mapping
    - load_shower_log: Load shower event state-change log
    - identify_shower_events: Parse log into individual shower events

Data Loading Features:
    - Automatic date range filtering across multiple files
    - Datetime parsing and normalization to consistent format
    - Duplicate record removal and chronological sorting
    - Error handling with informative warnings
    - Support for mixed delimiter formats (tab/comma)

Sensor Configuration:
    - SENSOR_CONFIG dict maps display names to instrument/location/column
    - Supports RH, temperature, wind speed, and wind direction variables
    - get_sensors_by_type() filters sensors by variable type

Methodology:
    1. Locate data files using data_paths module functions
    2. Parse files according to instrument-specific format
    3. Normalize datetime column to pandas datetime
    4. Filter to requested date range
    5. Combine multiple files and deduplicate
    6. Return DataFrame with 'datetime' column and sensor data

Supported Instruments:
    - Aranet4: CO2, RH, Temperature (Entry, Bedroom, Outside)
    - QuantAQ MODULAIR-PM: RH, Temperature (Inside, Outside)
    - Vaisala HMP155/HMP45A: RH, Temperature (via DAQ)
    - HOBO UX100: RH, Temperature (Bathroom, Doorway, Bedroom)
    - AIO2: Wind speed and direction

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_paths import (
    get_common_file,
    get_instrument_config,
    get_instrument_files_for_date_range,
    get_instrument_path,
)

# =============================================================================
# Sensor Configuration
# =============================================================================

# Sensor configuration - maps display names to data access details
SENSOR_CONFIG = {
    # Aranet4 RH sensors
    "Aranet4 Entry RH": {
        "instrument": "Aranet4",
        "location": "Entry",
        "column": "Relative humidity(%)",
        "variable_type": "rh",
    },
    "Aranet4 Bedroom RH": {
        "instrument": "Aranet4",
        "location": "Bedroom",
        "column": "Relative humidity(%)",
        "variable_type": "rh",
    },
    "Aranet4 Outside RH": {
        "instrument": "Aranet4",
        "location": "Mh Outside",
        "column": "Relative humidity(%)",
        "variable_type": "rh",
    },
    # Aranet4 Temperature sensors
    "Aranet4 Entry Temp": {
        "instrument": "Aranet4",
        "location": "Entry",
        "column": "Temperature(°C)",
        "variable_type": "temperature",
    },
    "Aranet4 Bedroom Temp": {
        "instrument": "Aranet4",
        "location": "Bedroom",
        "column": "Temperature(°C)",
        "variable_type": "temperature",
    },
    "Aranet4 Outside Temp": {
        "instrument": "Aranet4",
        "location": "Mh Outside",
        "column": "Temperature(°C)",
        "variable_type": "temperature",
    },
    # QuantAQ RH sensors
    "QuantAQ Inside RH": {
        "instrument": "QuantAQ_MODULAIR_PM",
        "location": "inside",
        "column": "met_rh",
        "variable_type": "rh",
    },
    "QuantAQ Outside RH": {
        "instrument": "QuantAQ_MODULAIR_PM",
        "location": "outside",
        "column": "met_rh",
        "variable_type": "rh",
    },
    # QuantAQ Temperature sensors
    "QuantAQ Inside Temp": {
        "instrument": "QuantAQ_MODULAIR_PM",
        "location": "inside",
        "column": "met_temp",
        "variable_type": "temperature",
    },
    "QuantAQ Outside Temp": {
        "instrument": "QuantAQ_MODULAIR_PM",
        "location": "outside",
        "column": "met_temp",
        "variable_type": "temperature",
    },
    # DAQ Vaisala HMP155 RH sensors
    "Vaisala Bed1 RH": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "RH_Bed1_M3_C0",
        "variable_type": "rh",
    },
    "Vaisala Liv RH": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "RH_Liv_M3_C3",
        "variable_type": "rh",
    },
    # DAQ Vaisala HMP45A RH sensors
    "Vaisala MBa RH": {
        "instrument": "Vaisala_HMP45A",
        "location": "DAQ",
        "column": "RH_MBa_M3_C5",
        "variable_type": "rh",
    },
    # DAQ Vaisala HMP155 Temperature sensors
    "Vaisala Bed1 Temp": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "T_Bed1_M4_C4",
        "variable_type": "temperature",
    },
    "Vaisala Liv Temp": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "T_Liv_M4_C7",
        "variable_type": "temperature",
    },
    # DAQ Vaisala HMP45A Temperature sensors
    "Vaisala MBa Temp": {
        "instrument": "Vaisala_HMP45A",
        "location": "DAQ",
        "column": "T_MBa_M5_C1",
        "variable_type": "temperature",
    },
    # HOBO UX100 RH sensors
    "HOBO Bathroom1 RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_D",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    "HOBO Bathroom2 RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_Bath",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    "HOBO Bath/Bed RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_E",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    "HOBO Bedroom1 RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_Bed",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    "HOBO Bedroom2 RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_F",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    "HOBO Bedroom3 RH": {
        "instrument": "HOBO_UX100",
        "location": "MB_C",
        "column": "rh_pct",
        "variable_type": "rh",
    },
    # HOBO UX100 Temperature sensors
    "HOBO Bathroom1 Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_D",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    "HOBO Bathroom2 Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_Bath",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    "HOBO Bath/Bed Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_E",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    "HOBO Bedroom1 Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_Bed",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    "HOBO Bedroom2 Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_F",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    "HOBO Bedroom3 Temp": {
        "instrument": "HOBO_UX100",
        "location": "MB_C",
        "column": "temp_c",
        "variable_type": "temperature",
    },
    # AIO2 Wind sensors
    "AIO2 Wind Speed": {
        "instrument": "AIO2",
        "location": "outdoor",
        "column": "Wind_Speed_m/s",
        "variable_type": "wind_speed",
    },
    "AIO2 Wind Direction": {
        "instrument": "AIO2",
        "location": "outdoor",
        "column": "Wind_Direction_deg",
        "variable_type": "wind_direction",
    },
}


# =============================================================================
# Shower Log Functions
# =============================================================================


def load_shower_log() -> pd.DataFrame:
    """
    Load the shower state-change log file.

    Returns:
        DataFrame with shower event timestamps
    """
    log_path = get_common_file("shower_log_file")

    if not log_path.exists():
        raise FileNotFoundError(
            f"Shower log file not found: {log_path}\n"
            "Run scripts/process_shower_log.py first to generate this file."
        )

    df = pd.read_csv(log_path)
    df["datetime_EDT"] = pd.to_datetime(df["datetime_EDT"])

    return df


def identify_shower_events(
    shower_log: pd.DataFrame,
    pre_shower_minutes: int = 30,
    post_shower_hours: int = 2,
) -> List[Dict]:
    """
    Identify individual shower events from the state-change log.

    Args:
        shower_log: DataFrame with shower state changes
        pre_shower_minutes: Minutes before shower ON for baseline
        post_shower_hours: Hours after shower OFF for analysis

    Returns:
        List of dicts with shower event details
    """
    events = []
    shower_log = shower_log.sort_values("datetime_EDT").reset_index(drop=True)

    event_number = 0  # Will be 1-indexed
    i = 0
    while i < len(shower_log):
        row = shower_log.iloc[i]

        if row["shower"] > 0:
            shower_on = row["datetime_EDT"]
            shower_off = None

            for j in range(i + 1, len(shower_log)):
                if shower_log.iloc[j]["shower"] == 0:
                    shower_off = shower_log.iloc[j]["datetime_EDT"]
                    i = j
                    break

            if shower_off is None:
                i += 1
                continue

            event_number += 1  # Increment for each complete shower event
            pre_start = shower_on - timedelta(minutes=pre_shower_minutes)
            post_end = shower_off + timedelta(hours=post_shower_hours)
            duration_min = (shower_off - shower_on).total_seconds() / 60

            # For particle analysis: penetration and deposition windows
            penetration_start = shower_on - timedelta(hours=1)
            penetration_end = shower_on
            deposition_start = shower_off
            deposition_end = shower_off + timedelta(hours=2)

            events.append(
                {
                    "event_number": event_number,
                    "shower_on": shower_on,
                    "shower_off": shower_off,
                    "duration_min": duration_min,
                    "shower_duration_min": duration_min,  # Alias for consistency
                    "pre_start": pre_start,
                    "post_end": post_end,
                    # Particle analysis specific windows
                    "penetration_start": penetration_start,
                    "penetration_end": penetration_end,
                    "deposition_start": deposition_start,
                    "deposition_end": deposition_end,
                }
            )

        i += 1

    return events


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_aranet_data(location: str, start_date, end_date) -> pd.DataFrame:
    """
    Load Aranet4 data for a specific location and date range.

    Args:
        location: Sensor location ('Bedroom', 'Entry', or 'Mh Outside')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with datetime and sensor columns
    """
    config = get_instrument_config("Aranet4")
    base_path = get_instrument_path("Aranet4")

    location_config = config["locations"].get(location, {})
    file_patterns = location_config.get(
        "file_pattern", [f"{location}_*_all.xlsx", f"{location}_*_week.xlsx"]
    )
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]
    files_set = set()
    for pattern in file_patterns:
        files_set.update(base_path.glob(pattern))
    files = sorted(files_set)

    if not files:
        return pd.DataFrame()

    all_data = []
    for filepath in files:
        try:
            df = pd.read_excel(filepath)
            datetime_col = "Time(DD/MM/YYYY h:mm:ss A)"
            if datetime_col in df.columns:
                df = df.rename(columns={datetime_col: "datetime"})
                df["datetime"] = pd.to_datetime(
                    df["datetime"], format="%d/%m/%Y %I:%M:%S %p"
                )
            else:
                for col in df.columns:
                    if "time" in col.lower():
                        df = df.rename(columns={col: "datetime"})
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        break

            if "datetime" in df.columns:
                mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
                df = df[mask]

            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    Warning: Error loading {filepath.name}: {str(e)[:50]}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return combined.reset_index(drop=True)


def load_quantaq_data(location: str, start_date, end_date) -> pd.DataFrame:
    """
    Load QuantAQ MODULAIR-PM data for a specific location and date range.

    Args:
        location: Sensor location ('inside' or 'outside')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with datetime and sensor columns
    """
    base_path = get_instrument_path("QuantAQ_MODULAIR_PM")
    config = get_instrument_config("QuantAQ_MODULAIR_PM")

    location_config = config["locations"].get(location, {})
    file_pattern = location_config.get(
        "file_pattern", f"*-quantaq-{location}-processed.csv"
    )
    files = sorted(base_path.glob(file_pattern))

    if not files:
        return pd.DataFrame()

    all_data = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            for dt_col in ["timestamp_local", "timestamp"]:
                if dt_col in df.columns:
                    df["datetime"] = pd.to_datetime(df[dt_col])
                    break

            if "datetime" in df.columns:
                mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
                df = df[mask]

            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    Warning: Error loading {filepath.name}: {str(e)[:50]}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return combined.reset_index(drop=True)


def load_daq_data(start_date, end_date) -> pd.DataFrame:
    """
    Load indoor DAQ data (Vaisala sensors) for a date range.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with datetime and all Vaisala sensor columns
    """
    files = get_instrument_files_for_date_range(
        "Vaisala_HMP155", start_date.date(), end_date.date(), must_exist=True
    )

    if not files:
        return pd.DataFrame()

    all_data = []
    for filepath in files:
        try:
            # index_col=False prevents pandas from using first column as index
            # (needed because DAQ files have trailing tabs causing column misalignment)
            df = pd.read_csv(filepath, sep="\t", index_col=False)
            if "Date" in df.columns and "Time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str),
                    format="%m/%d/%Y %H:%M:%S",
                )
            if "datetime" in df.columns:
                mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
                df = df[mask]

            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    Warning: Error loading {filepath.name}: {str(e)[:50]}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return combined.reset_index(drop=True)


def load_aio2_data(start_date, end_date) -> pd.DataFrame:
    """
    Load AIO2 weather station data for a date range.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with datetime and weather sensor columns
    """
    files = get_instrument_files_for_date_range(
        "AIO2", start_date.date(), end_date.date(), must_exist=True
    )

    if not files:
        return pd.DataFrame()

    all_data = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath, sep="\t")
            if "Date" in df.columns and "Time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str),
                    format="%m/%d/%Y %H:%M:%S",
                )
            if "datetime" in df.columns:
                mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
                df = df[mask]

            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"    Warning: Error loading {filepath.name}: {str(e)[:50]}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return combined.reset_index(drop=True)


def load_hobo_data(location: str, start_date, end_date) -> pd.DataFrame:
    """
    Load HOBO UX100 data for a specific sensor location and date range.

    Combines all CSV files matching the sensor's file suffixes, converts
    temperature from °F to °C, removes duplicates, and sorts chronologically.

    Args:
        location: Sensor location key (e.g., 'MB_D', 'MB_Bath', 'MB_E')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with datetime, temp_c, and rh_pct columns
    """
    config = get_instrument_config("HOBO_UX100")
    base_path = get_instrument_path("HOBO_UX100")

    location_config = config["locations"].get(location, {})
    file_suffixes = location_config.get("file_suffixes", [location])

    # Find all CSV files matching any of the suffixes for this sensor
    files = []
    for suffix in file_suffixes:
        pattern = f"*_{suffix}.csv"
        files.extend(base_path.glob(pattern))
    files = sorted(set(files))

    if not files:
        return pd.DataFrame()

    all_data = []
    for filepath in files:
        try:
            # Row 0 is the plot title line, row 1 is the header
            df = pd.read_csv(filepath, skiprows=1)

            if len(df.columns) < 4:
                continue

            # Rename columns by position (headers vary by serial number)
            col_names = list(df.columns)
            df = df.rename(columns={
                col_names[0]: "row_num",
                col_names[1]: "datetime_str",
                col_names[2]: "temp_f",
                col_names[3]: "rh_pct",
            })

            # Keep only the columns we need
            df = df[["datetime_str", "temp_f", "rh_pct"]].copy()

            # Parse datetime (format: MM/DD/YY HH:MM:SS AM/PM)
            df["datetime"] = pd.to_datetime(
                df["datetime_str"], format="%m/%d/%y %I:%M:%S %p"
            )

            # Convert temperature from °F to °C
            df["temp_c"] = (df["temp_f"].astype(float) - 32) * 5 / 9
            df["rh_pct"] = df["rh_pct"].astype(float)

            # Filter to requested date range
            mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
            df = df[mask]

            if len(df) > 0:
                all_data.append(df[["datetime", "temp_c", "rh_pct"]])
        except Exception as e:
            print(f"    Warning: Error loading {filepath.name}: {str(e)[:50]}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return combined.reset_index(drop=True)


def load_sensor_data(
    sensor_name: str, config: Dict, start_date: datetime, end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Load data for a specific sensor based on its configuration.

    Args:
        sensor_name: Display name of the sensor
        config: Sensor configuration dict
        start_date: Start datetime
        end_date: End datetime

    Returns:
        DataFrame with 'datetime' and 'value' column, or None if no data
    """
    instrument = config["instrument"]
    location = config["location"]
    column = config["column"]

    try:
        if instrument == "Aranet4":
            df = load_aranet_data(location, start_date, end_date)
        elif instrument == "QuantAQ_MODULAIR_PM":
            df = load_quantaq_data(location, start_date, end_date)
        elif instrument in ["Vaisala_HMP155", "Vaisala_HMP45A"]:
            df = load_daq_data(start_date, end_date)
        elif instrument == "HOBO_UX100":
            df = load_hobo_data(location, start_date, end_date)
        elif instrument == "AIO2":
            df = load_aio2_data(start_date, end_date)
        else:
            return None

        if df.empty or "datetime" not in df.columns:
            return None

        if column in df.columns:
            result = df[["datetime", column]].copy()
            result = result.rename(columns={column: "value"})
            return result

    except Exception as e:
        print(f"    Warning: Error loading {sensor_name}: {str(e)[:50]}")

    return None


def get_sensors_by_type(variable_type: str) -> Dict:
    """
    Get sensor configurations filtered by variable type.

    Args:
        variable_type: 'rh', 'temperature', 'wind_speed', or 'wind_direction'

    Returns:
        Dict of {sensor_name: config} for matching sensors
    """
    return {
        name: cfg
        for name, cfg in SENSOR_CONFIG.items()
        if cfg["variable_type"] == variable_type
    }
