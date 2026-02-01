#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Path Utilities
===================

This module provides portable data access functions for the EPA Legionella
project. It abstracts file system paths through a JSON configuration file,
enabling the codebase to work across different machines without hardcoded paths.

Key Functions:
    - get_data_root: Get the root directory for all project data
    - get_instrument_path: Get directory path for an instrument
    - get_instrument_file: Get file path for a specific date
    - get_instrument_files_for_date_range: Get list of files for date range
    - get_all_instrument_files: Get all files using glob patterns
    - get_common_file: Get path to shared files (logs, output folders)
    - get_instrument_config: Get full configuration for an instrument
    - get_instrument_variables: Get list of variable/column names
    - get_active_instruments: List instruments with ACTIVE status

Configuration Features:
    - Machine-independent paths via data_config.json
    - Support for year-based directory structures
    - Flexible file naming templates with date formatting
    - Instrument specifications and variable definitions
    - DAQ system configuration

Methodology:
    1. Load data_config.json from project root or current directory
    2. Parse instrument configurations including paths and templates
    3. Construct full paths by combining data_root with relative paths
    4. Handle year subdirectories automatically for dated instruments
    5. Support glob patterns for finding multiple files

Configuration File (data_config.json):
    - data_root: Base directory for all data files
    - instruments: Per-instrument configuration (paths, templates, variables)
    - common_files: Shared files like log files and output directories
    - daq_system: DAQ hardware configuration

Usage:
    from src.data_paths import (
        get_data_root,
        get_instrument_path,
        get_instrument_file,
        get_instrument_files_for_date_range,
        get_common_file,
        get_instrument_config,
    )

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _load_config() -> dict:
    """Load the data configuration file."""
    possible_paths = [
        Path(__file__).parent.parent / "data_config.json",
        Path(__file__).parent / "data_config.json",
        Path.cwd() / "data_config.json",
    ]

    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Configuration file not found. Searched: {[str(p) for p in possible_paths]}"
    )


def get_data_root() -> Path:
    """
    Get the data root directory from configuration.

    Returns:
        Path: The root directory for all project data.
    """
    config = _load_config()
    return Path(config["data_root"])


def get_instrument_config(instrument_name: str) -> dict:
    """
    Get the full configuration dictionary for an instrument.

    Args:
        instrument_name: Name of the instrument (e.g., 'AIO2', 'Setra_264', 'QuantAQ_MODULAIR_PM')

    Returns:
        dict: Full instrument configuration including specs, variables, etc.
    """
    config = _load_config()
    instruments = config.get("instruments", {})

    if instrument_name not in instruments:
        # Check future_instruments
        future = config.get("future_instruments", {})
        if instrument_name in future:
            return future[instrument_name]

        available = list(instruments.keys()) + list(future.keys())
        raise KeyError(
            f"Unknown instrument: {instrument_name}. Available instruments: {available}"
        )

    return instruments[instrument_name]


def get_instrument_path(instrument_name: str, year: Optional[int] = None) -> Path:
    """
    Get the data directory path for an instrument.

    Args:
        instrument_name: Name of the instrument (e.g., 'AIO2', 'Setra_264')
        year: Optional year for year-based directory structure.
              If None and instrument has year subdirs, returns base path.

    Returns:
        Path: Directory path for the instrument's data files.
    """
    config = _load_config()
    data_root = Path(config["data_root"])
    inst_config = get_instrument_config(instrument_name)

    base_path = data_root / inst_config["base_path"]

    # Check if instrument uses year subdirectories
    has_year_subdirs = inst_config.get("has_year_subdirs", True)

    if year is not None and has_year_subdirs:
        return base_path / str(year)

    return base_path


def get_instrument_file(
    instrument_name: str,
    target_date: Union[date, str],
) -> Path:
    """
    Get the full file path for an instrument on a specific date.

    Args:
        instrument_name: Name of the instrument (e.g., 'AIO2', 'Setra_264')
        target_date: Date for the data file (date object or 'YYYY-MM-DD' string)

    Returns:
        Path: Full path to the data file.

    Raises:
        ValueError: If instrument doesn't have a file_template defined.
    """
    inst_config = get_instrument_config(instrument_name)

    if "file_template" not in inst_config:
        raise ValueError(
            f"Instrument '{instrument_name}' does not have a file_template. "
            f"Use get_instrument_path() with file_pattern instead."
        )

    # Parse date if string
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    # Get year-specific path
    year = target_date.year
    dir_path = get_instrument_path(instrument_name, year)

    # Format the filename
    date_format = inst_config.get("date_format", "%Y%m%d")
    date_str = target_date.strftime(date_format)
    filename = inst_config["file_template"].format(date_str=date_str)

    return dir_path / filename


def get_instrument_files_for_date_range(
    instrument_name: str,
    start_date: Union[date, str],
    end_date: Union[date, str],
    must_exist: bool = True,
) -> List[Path]:
    """
    Get list of data files for an instrument over a date range.

    Handles date ranges that span multiple years automatically.

    Args:
        instrument_name: Name of the instrument (e.g., 'AIO2', 'Setra_264')
        start_date: Start date (date object or 'YYYY-MM-DD' string)
        end_date: End date (date object or 'YYYY-MM-DD' string)
        must_exist: If True, only return files that exist on disk.

    Returns:
        List[Path]: List of file paths in chronological order.
    """
    # Parse dates if strings
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    files = []
    current_date = start_date

    while current_date <= end_date:
        try:
            file_path = get_instrument_file(instrument_name, current_date)
            if must_exist:
                if file_path.exists():
                    files.append(file_path)
            else:
                files.append(file_path)
        except ValueError:
            # Instrument doesn't use dated files
            break
        current_date += timedelta(days=1)

    return files


def get_all_instrument_files(
    instrument_name: str,
    years: Optional[List[int]] = None,
) -> List[Path]:
    """
    Get all data files for an instrument using glob pattern.

    Args:
        instrument_name: Name of the instrument
        years: Optional list of years to search. If None, searches all years.

    Returns:
        List[Path]: List of all matching files sorted by name.
    """
    inst_config = get_instrument_config(instrument_name)
    pattern = inst_config.get("file_pattern", "*.*")

    files = []

    if years is None:
        # Search base path and any year subdirectories
        base_path = get_instrument_path(instrument_name)

        # Direct files in base path
        if base_path.exists():
            files.extend(base_path.glob(pattern))

        # Files in year subdirectories
        if base_path.exists():
            for year_dir in base_path.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    files.extend(year_dir.glob(pattern))
    else:
        for year in years:
            year_path = get_instrument_path(instrument_name, year)
            if year_path.exists():
                files.extend(year_path.glob(pattern))

    return sorted(files)


def get_common_file(file_key: str) -> Path:
    """
    Get path to a common project file (e.g., log file, output folder).

    Args:
        file_key: Key from common_files section (e.g., 'log_file', 'output_folder',
                  'co2_log_file', 'shower_log_file')

    Returns:
        Path: Full path to the common file or directory.
    """
    config = _load_config()
    data_root = Path(config["data_root"])
    common_files = config.get("common_files", {})

    if file_key not in common_files:
        available = list(common_files.keys())
        raise KeyError(f"Unknown common file: {file_key}. Available: {available}")

    file_entry = common_files[file_key]

    # Handle object-based entries (e.g., co2_log_file, shower_log_file)
    if isinstance(file_entry, dict):
        path = file_entry.get("path", "")
        filename = file_entry.get("filename", "")
        return data_root / path / filename

    # Handle simple string entries (e.g., log_file, output_folder)
    return data_root / file_entry


def get_common_file_config(file_key: str) -> Dict[str, Any]:
    """
    Get the full configuration for a common file entry.

    Args:
        file_key: Key from common_files section (e.g., 'co2_log_file', 'shower_log_file')

    Returns:
        dict: Full configuration for the file entry, or a simple dict with 'filename'
              for string entries.
    """
    config = _load_config()
    common_files = config.get("common_files", {})

    if file_key not in common_files:
        available = list(common_files.keys())
        raise KeyError(f"Unknown common file: {file_key}. Available: {available}")

    file_entry = common_files[file_key]

    # Return dict entries directly
    if isinstance(file_entry, dict):
        return file_entry

    # Wrap string entries in a dict for consistency
    return {"filename": file_entry}


def get_instrument_variables(instrument_name: str) -> list:
    """
    Get list of variable/column names for an instrument.

    Args:
        instrument_name: Name of the instrument

    Returns:
        list: List of variable names (flattened if nested dict).
    """
    inst_config = get_instrument_config(instrument_name)
    variables = inst_config.get("variables", [])

    # Handle nested variable structure (e.g., Vaisala with RH and T groups)
    if isinstance(variables, dict):
        flat_vars = []
        for group_vars in variables.values():
            if isinstance(group_vars, list):
                flat_vars.extend(group_vars)
        return flat_vars

    return variables


def get_instrument_datetime_columns(instrument_name: str) -> list:
    """
    Get the datetime column name(s) for an instrument.

    Args:
        instrument_name: Name of the instrument

    Returns:
        list: List of datetime column names (e.g., ['Date', 'Time'])
    """
    inst_config = get_instrument_config(instrument_name)
    return inst_config.get("datetime_columns", ["Date", "Time"])


def get_instrument_specifications(instrument_name: str) -> dict:
    """
    Get the specifications dictionary for an instrument.

    Args:
        instrument_name: Name of the instrument

    Returns:
        dict: Instrument specifications (accuracy, range, etc.)
    """
    inst_config = get_instrument_config(instrument_name)
    return inst_config.get("specifications", {})


def get_instrument_status(instrument_name: str) -> str:
    """
    Get the current status of an instrument.

    Args:
        instrument_name: Name of the instrument

    Returns:
        str: Status string (e.g., 'ACTIVE', 'DISCONNECTED', 'PLANNED')
    """
    inst_config = get_instrument_config(instrument_name)
    return inst_config.get("status", "UNKNOWN")


def get_active_instruments() -> List[str]:
    """
    Get list of active instrument names.

    Returns:
        List[str]: Names of instruments with status 'ACTIVE'.
    """
    config = _load_config()
    instruments = config.get("instruments", {})

    return [name for name, cfg in instruments.items() if cfg.get("status") == "ACTIVE"]


def get_daq_system_info() -> dict:
    """
    Get DAQ system configuration information.

    Returns:
        dict: DAQ system configuration including chassis and modules.
    """
    config = _load_config()
    return config.get("daq_system", {})


# =============================================================================
# Convenience functions for specific data types
# =============================================================================


def get_indoor_data_path(year: int) -> Path:
    """Get path to indoor data directory for a specific year."""
    return get_instrument_path("Setra_264", year)


def get_outdoor_data_path(year: int) -> Path:
    """Get path to outdoor data directory for a specific year."""
    return get_instrument_path("AIO2", year)


def get_indoor_data_file(target_date: Union[date, str]) -> Path:
    """Get indoor data file for a specific date."""
    return get_instrument_file("Setra_264", target_date)


def get_outdoor_data_file(target_date: Union[date, str]) -> Path:
    """Get outdoor data file for a specific date."""
    return get_instrument_file("AIO2", target_date)


def get_quantaq_data_path() -> Path:
    """Get path to QuantAQ data directory."""
    return get_instrument_path("QuantAQ_MODULAIR_PM")
