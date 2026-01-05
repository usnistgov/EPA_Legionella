"""
Example script demonstrating data path utilities for EPA Legionella Project.

This script shows how to use the data_paths module for portable data access
across different machines and date ranges spanning multiple years.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# pylint: disable=import-error,wrong-import-position
from src.data_paths import (
    get_data_root,
    get_common_file,
    get_instrument_path,
    get_instrument_file,
    get_instrument_files_for_date_range,
    get_all_instrument_files,
    get_instrument_config,
    get_instrument_variables,
    get_instrument_datetime_columns,
    get_indoor_data_file,
    get_outdoor_data_file,
)
# pylint: enable=import-error,wrong-import-position


def main():
    """Demonstrate data path utilities."""
    
    # =========================================================================
    # Setup - Get portable paths
    # =========================================================================
    data_root = get_data_root()
    output_path = get_common_file("output_folder")
    
    print("=" * 60)
    print("EPA Legionella Project - Data Path Examples")
    print("=" * 60)
    print(f"[OK] Data root: {data_root}")
    print(f"[OK] Output path: {output_path}")
    print()

    # =========================================================================
    # Example 1: Get data for a single date
    # =========================================================================
    print("-" * 60)
    print("Example 1: Single date data access")
    print("-" * 60)
    
    target_date = date(2025, 12, 19)
    
    indoor_file = get_indoor_data_file(target_date)
    outdoor_file = get_outdoor_data_file(target_date)
    
    print(f"Target date: {target_date}")
    print(f"Indoor file: {indoor_file}")
    print(f"Outdoor file: {outdoor_file}")
    print(f"Indoor exists: {indoor_file.exists()}")
    print(f"Outdoor exists: {outdoor_file.exists()}")
    print()

    # =========================================================================
    # Example 2: Get data for a date range (potentially spanning years)
    # =========================================================================
    print("-" * 60)
    print("Example 2: Date range spanning years")
    print("-" * 60)
    
    start_date = date(2025, 12, 28)
    end_date = date(2026, 1, 5)
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Get all indoor files for the date range
    indoor_files = get_instrument_files_for_date_range(
        "Setra_264", 
        start_date, 
        end_date,
        must_exist=False  # Set True to only get existing files
    )
    
    print(f"Indoor files in range ({len(indoor_files)} total):")
    for f in indoor_files:
        exists = "✓" if f.exists() else "✗"
        print(f"  [{exists}] {f.name}")
    print()

    # =========================================================================
    # Example 3: Get instrument configuration and variables
    # =========================================================================
    print("-" * 60)
    print("Example 3: Instrument configuration")
    print("-" * 60)
    
    instrument = "AIO2"
    config = get_instrument_config(instrument)
    variables = get_instrument_variables(instrument)
    dt_cols = get_instrument_datetime_columns(instrument)
    
    print(f"Instrument: {instrument}")
    print(f"Description: {config['description']}")
    print(f"Manufacturer: {config['manufacturer']}")
    print(f"Model: {config['model']}")
    print(f"Status: {config['status']}")
    print(f"Datetime columns: {dt_cols}")
    print(f"Variables: {variables}")
    print()

    # =========================================================================
    # Example 4: Get all files for specific years
    # =========================================================================
    print("-" * 60)
    print("Example 4: All files for specific years")
    print("-" * 60)
    
    all_files_2025 = get_all_instrument_files("Setra_264", years=[2025])
    all_files_both = get_all_instrument_files("Setra_264", years=[2025, 2026])
    
    print(f"Indoor files in 2025: {len(all_files_2025)}")
    print(f"Indoor files in 2025+2026: {len(all_files_both)}")
    print()

    # =========================================================================
    # Example 5: Load and process data
    # =========================================================================
    print("-" * 60)
    print("Example 5: Load and process data")
    print("-" * 60)
    
    target_date = date(2025, 12, 19)
    
    # Load outdoor data
    outdoor_file = get_outdoor_data_file(target_date)
    if outdoor_file.exists():
        df_outdoor = pd.read_csv(outdoor_file, sep='\t')
        print(f"Loaded outdoor data: {len(df_outdoor)} rows")
        print(f"Columns: {list(df_outdoor.columns)}")
        print(f"Wind speed range: {df_outdoor['Wind_Speed_m/s'].min():.1f} - {df_outdoor['Wind_Speed_m/s'].max():.1f} m/s")
        print(f"Temperature range: {df_outdoor['Ambient_Temperature_degC'].min():.1f} - {df_outdoor['Ambient_Temperature_degC'].max():.1f} °C")
    else:
        print(f"File not found: {outdoor_file}")
    print()

    # =========================================================================
    # Example 6: Build instrument configuration dictionary (like your example)
    # =========================================================================
    print("-" * 60)
    print("Example 6: Build INSTRUMENT_CONFIG dictionary")
    print("-" * 60)
    
    # This mirrors your pattern from the burn analysis project
    INSTRUMENT_CONFIG = {}
    
    for instrument_name in ["AIO2", "Setra_264", "Vaisala_HMP155", "Vaisala_HMP45A"]:
        config = get_instrument_config(instrument_name)
        
        # Only include active instruments
        if config.get("status") != "ACTIVE":
            continue
            
        INSTRUMENT_CONFIG[instrument_name] = {
            "base_path": str(get_instrument_path(instrument_name)),
            "file_pattern": config.get("file_pattern", "*.*"),
            "datetime_columns": get_instrument_datetime_columns(instrument_name),
            "variables": get_instrument_variables(instrument_name),
            "specifications": config.get("specifications", {}),
        }
    
    print("INSTRUMENT_CONFIG = {")
    for name, cfg in INSTRUMENT_CONFIG.items():
        print(f"    '{name}': {{")
        print(f"        'base_path': '{cfg['base_path']}',")
        print(f"        'file_pattern': '{cfg['file_pattern']}',")
        print(f"        'datetime_columns': {cfg['datetime_columns']},")
        print(f"        'variables': {cfg['variables'][:3]}...,  # ({len(cfg['variables'])} total)")
        print(f"    }},")
    print("}")
    print()

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
