#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAIR-PM Multi-Sensor Chunk Loader
=====================================

Loads weekly raw chunk files for the IAQ&V QuantAQ MODULAIR-PM sensor fleet
(the 14 new instruments plus the original inside MOD-PM-00195 and outside
MOD-PM-00785 sensors) from the moduair-pm network share, and exposes the
12 analysis particle-size bins (opc_bin0-opc_bin11) per sensor on a common
1-minute time base.

The chunk files are produced by the separate NIST_moduair-pm repository and
written to the share configured under the ``moduair_pm_chunks`` common-file
entry in data_config.json. This module only reads them; it does not download.

Key Functions:
    - get_chunks_dir(): Resolve the chunks share path from data_config.json.
    - list_available_sensors(): Discover sensor short-IDs present in the share.
    - load_sensor_bins(): Load one sensor's raw chunks, parse the nested ``opc``
      dictionary, extract opc_bin0-opc_bin11, resample to 1 min, and apply a
      centered rolling average.
    - load_fleet_bins(): Load several sensors and align them on a shared
      1-minute datetime index, returning a dict of per-sensor DataFrames.

Processing Features:
    - Parses the QuantAQ raw schema where particle bins live inside a single
      ``opc`` column holding a Python-dict string (same format handled by
      scripts/process_quantaq_data.py).
    - Uses timestamp_local (falling back to timestamp) as the time base.
    - Resamples to a regular 1-minute grid and applies a configurable
      centered rolling average (ROLLING_WINDOW_MIN, default 10 min) so the
      smoothing matches the inside/outside pipeline.
    - Deduplicates overlapping chunk rows on the timestamp.

Input Files:
    - MOD-PM-{sn}-raw-{start}-{end}.csv on the moduair_pm_chunks share.

Output Files:
    - None; returns pandas DataFrames for downstream analysis scripts.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Created: 2026-06-25
Update log:
    2026-06-25 (Nathan Lima): Initial version for the MODULAIR-PM correction
        factor and event peak-time analyses.
"""

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 12 analysis bins (0.35-10.0 um), matching PARTICLE_BINS in
# src/particle_calculations.py. The raw chunk ``opc`` dict uses "bin0".."bin11";
# we expose them with the "opc_" prefix used elsewhere in the project.
N_BINS = 12
BIN_COLUMNS = [f"opc_bin{i}" for i in range(N_BINS)]

# Centered rolling-average window (minutes) applied after 1-min resampling.
# Matches the 10-minute window used by scripts/process_quantaq_data.py.
ROLLING_WINDOW_MIN = 10


def get_chunks_dir() -> Path:
    """
    Resolve the MODULAIR-PM chunks directory from data_config.json.

    Returns:
        Path: Directory containing the weekly per-sensor chunk CSVs.

    Raises:
        FileNotFoundError: If the configured directory does not exist.
    """
    from src.data_paths import get_common_file

    chunks_dir = get_common_file("moduair_pm_chunks")
    if not chunks_dir.exists():
        raise FileNotFoundError(
            f"MODULAIR-PM chunks directory not found: {chunks_dir}\n"
            "Check the 'moduair_pm_chunks' entry in data_config.json and that "
            "the network share is reachable."
        )
    return chunks_dir


def list_available_sensors(data_type: str = "raw") -> List[str]:
    """
    Discover sensor short-IDs that have chunk files in the share.

    Parameters:
        data_type: 'raw' or 'final'.

    Returns:
        Sorted list of zero-padded sensor IDs (e.g., ['00195', '00402', ...]).
    """
    chunks_dir = get_chunks_dir()
    pattern = re.compile(rf"MOD-PM-(\d+)-{re.escape(data_type)}-")
    sensors = set()
    for path in chunks_dir.glob(f"MOD-PM-*-{data_type}-*.csv"):
        m = pattern.match(path.name)
        if m:
            sensors.add(m.group(1))
    return sorted(sensors)


def _normalize_sn(sensor_id: str) -> str:
    """
    Normalize a sensor identifier to the 5-digit form used in filenames.

    Accepts '195', '00195', or 'MOD-PM-00195' and returns '00195'.
    """
    s = str(sensor_id).strip()
    if s.upper().startswith("MOD-PM-"):
        s = s[len("MOD-PM-"):]
    digits = "".join(c for c in s if c.isdigit())
    return digits.zfill(5)


def _parse_opc_bins(opc_str: str) -> Dict[str, float]:
    """
    Parse the nested ``opc`` dict-string and return the 12 analysis bins.

    Parameters:
        opc_str: String such as "{'bin0': 3.98, 'bin1': 0.32, ...}".

    Returns:
        Dict mapping opc_bin0..opc_bin11 to float values (NaN if absent).
    """
    if pd.isna(opc_str) or opc_str == "":
        return {col: float("nan") for col in BIN_COLUMNS}
    try:
        parsed = ast.literal_eval(opc_str)
    except (ValueError, SyntaxError):
        return {col: float("nan") for col in BIN_COLUMNS}

    out = {}
    for i in range(N_BINS):
        val = parsed.get(f"bin{i}")
        out[f"opc_bin{i}"] = float(val) if val is not None else float("nan")
    return out


def load_sensor_bins(
    sensor_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    rolling_window_min: int = ROLLING_WINDOW_MIN,
) -> pd.DataFrame:
    """
    Load one sensor's raw chunks and return its 12 analysis bins at 1-min.

    Parameters:
        sensor_id: Sensor identifier ('195', '00195', or 'MOD-PM-00195').
        start: Optional inclusive lower bound on timestamp_local.
        end: Optional inclusive upper bound on timestamp_local.
        rolling_window_min: Centered rolling-average window in minutes
            (0 disables smoothing).

    Returns:
        DataFrame indexed by 1-minute 'datetime' with columns
        opc_bin0..opc_bin11. Empty DataFrame if no chunks are found.
    """
    sn = _normalize_sn(sensor_id)
    chunks_dir = get_chunks_dir()
    files = sorted(chunks_dir.glob(f"MOD-PM-{sn}-raw-*.csv"))

    if not files:
        print(f"  [WARN] No raw chunks found for sensor {sn}")
        return pd.DataFrame(columns=["datetime"] + BIN_COLUMNS)

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=lambda c: c in ("opc", "timestamp_local", "timestamp"))
        except Exception as exc:  # noqa: BLE001 - report and skip bad chunk
            print(f"  [WARN] Could not read {path.name}: {str(exc)[:100]}")
            continue

        if "timestamp_local" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            print(f"  [WARN] No timestamp column in {path.name}; skipping")
            continue

        bins = df["opc"].apply(_parse_opc_bins)
        bins_df = pd.DataFrame(bins.tolist(), index=df.index)
        df = pd.concat([df[["datetime"]], bins_df], axis=1)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["datetime"] + BIN_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["datetime"])
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    if start is not None:
        combined = combined[combined["datetime"] >= start]
    if end is not None:
        combined = combined[combined["datetime"] <= end]

    if combined.empty:
        return pd.DataFrame(columns=["datetime"] + BIN_COLUMNS)

    # Resample to a regular 1-minute grid
    combined = combined.set_index("datetime").sort_index()
    combined = combined[BIN_COLUMNS].resample("1min").mean()

    if rolling_window_min and rolling_window_min > 0:
        combined = combined.rolling(
            window=rolling_window_min, center=True, min_periods=1
        ).mean()

    return combined.reset_index()


def load_fleet_bins(
    sensor_ids: List[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    rolling_window_min: int = ROLLING_WINDOW_MIN,
) -> Dict[str, pd.DataFrame]:
    """
    Load several sensors and align them on a shared 1-minute datetime index.

    Parameters:
        sensor_ids: Iterable of sensor identifiers.
        start: Optional inclusive lower bound on timestamp_local.
        end: Optional inclusive upper bound on timestamp_local.
        rolling_window_min: Centered rolling-average window in minutes.

    Returns:
        Dict mapping the normalized 5-digit sensor ID to its per-bin DataFrame
        (datetime + opc_bin0..opc_bin11). Sensors with no data are omitted.
    """
    fleet = {}
    for sid in sensor_ids:
        sn = _normalize_sn(sid)
        df = load_sensor_bins(sn, start=start, end=end, rolling_window_min=rolling_window_min)
        if not df.empty:
            fleet[sn] = df
            print(f"  Loaded sensor {sn}: {len(df)} 1-min rows")
    return fleet


if __name__ == "__main__":
    print("MODULAIR-PM Chunk Loader")
    print("Available raw sensors:", list_available_sensors("raw"))
