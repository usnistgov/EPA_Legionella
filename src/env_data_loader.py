#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Data Loader Module
=================================

This module provides data loading functions for environmental sensors used in
the EPA Legionella project. It handles loading data from multiple instrument
types including Aranet4, QuantAQ, Vaisala (DAQ), and AIO2 weather station.

Key Functions:
    - load_aranet_data: Load Aranet4 CO2/RH/Temp sensor data
    - load_quantaq_data: Load QuantAQ MODULAIR-PM data
    - load_daq_data: Load indoor DAQ data (Vaisala sensors)
    - load_aio2_data: Load AIO2 weather station data
    - load_sensor_data: Generic loader using sensor configuration

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
    "Vaisala Fam RH": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "RH_Fam_M3_C4",
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
    "Vaisala Fam Temp": {
        "instrument": "Vaisala_HMP155",
        "location": "DAQ",
        "column": "T_Fam_M5_C0",
        "variable_type": "temperature",
    },
    # DAQ Vaisala HMP45A Temperature sensors
    "Vaisala MBa Temp": {
        "instrument": "Vaisala_HMP45A",
        "location": "DAQ",
        "column": "T_MBa_M5_C1",
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

            pre_start = shower_on - timedelta(minutes=pre_shower_minutes)
            post_end = shower_off + timedelta(hours=post_shower_hours)
            duration_min = (shower_off - shower_on).total_seconds() / 60

            events.append(
                {
                    "shower_on": shower_on,
                    "shower_off": shower_off,
                    "duration_min": duration_min,
                    "pre_start": pre_start,
                    "post_end": post_end,
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
    file_pattern = location_config.get("file_pattern", f"{location}_*_week.xlsx")
    files = sorted(base_path.glob(file_pattern))

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
