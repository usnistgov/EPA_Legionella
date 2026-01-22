#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to access QuantAQ API and download instrument data.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025

This module integrates with data_config.json for configuration management
and provides functions for data retrieval and storage.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def _load_config() -> dict:
    """
    Load the data configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If config file is not found.
    """
    # Look for config in multiple locations
    possible_paths = [
        Path(__file__).parent.parent / "data_config.json",  # repo root
        Path(__file__).parent / "data_config.json",  # same directory
        Path.cwd() / "data_config.json",  # current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Configuration file not found. Searched: {[str(p) for p in possible_paths]}"
    )


def get_quantaq_config() -> dict:
    """
    Get QuantAQ instrument configuration from data_config.json.

    Returns:
        dict: QuantAQ MODULAIR-PM instrument configuration.
    """
    config = _load_config()
    return config.get("instruments", {}).get("QuantAQ_MODULAIR_PM", {})


def get_data_root() -> Path:
    """
    Get the data root directory from configuration.

    Returns:
        Path: The root directory for all project data.
    """
    config = _load_config()
    return Path(config["data_root"])


def get_quantaq_data_path() -> Path:
    """
    Get the QuantAQ data storage path.

    Returns:
        Path: Directory path for QuantAQ data files.
    """
    config = _load_config()
    data_root = Path(config["data_root"])
    quantaq_config = get_quantaq_config()
    base_path = quantaq_config.get("base_path", "QuantAQ")
    return data_root / base_path


class QuantAQAPI:
    """
    Class to interact with the QuantAQ API.

    Integrates with data_config.json for configuration management.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the QuantAQ API client.

        Args:
            api_key: API key for authentication. If None, will attempt to load
                    from environment variable specified in config or 'QUANTAQ_API_KEY'.
        """
        # Load config
        self.config = get_quantaq_config()

        # Get API key
        if api_key is None:
            env_var = self.config.get("api_config", {}).get(
                "env_var_api_key", "QUANTAQ_API_KEY"
            )
            api_key = os.getenv(env_var) or os.getenv("API_KEY")

        if api_key is None:
            raise ValueError(
                "API key not provided and not found in environment variables. "
                "Set QUANTAQ_API_KEY or API_KEY in your environment or .env file."
            )

        self.api_key = api_key
        self.base_url = self.config.get("api_base_url", "https://api.quant-aq.com/v1")
        self.data_path = get_quantaq_data_path()

    def _make_request(
        self, endpoint: str, params: Optional[dict] = None, timeout: int = 30
    ) -> Optional[dict]:
        """
        Make an authenticated request to the QuantAQ API.

        Args:
            endpoint: API endpoint (will be appended to base_url).
            params: Query parameters.
            timeout: Request timeout in seconds.

        Returns:
            dict: JSON response from the API, or None if error.
        """
        url = f"{self.base_url}{endpoint}"
        auth = (self.api_key, "")

        response = None
        try:
            response = requests.get(url, auth=auth, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

    def get_devices(
        self, org_id: Optional[int] = None, network_id: Optional[int] = None
    ) -> Optional[dict]:
        """
        Retrieves a list of devices.

        Args:
            org_id: Organization ID.
            network_id: Network ID.

        Returns:
            dict: JSON response from the API, or None if error.
        """
        params = {}
        if org_id:
            params["org_id"] = org_id
        if network_id:
            params["network_id"] = network_id

        return self._make_request("/devices", params, timeout=10)

    def get_device(self, serial_number: str) -> Optional[dict]:
        """
        Retrieves information about a specific device.

        Args:
            serial_number: Serial number of the device.

        Returns:
            dict: JSON response from the API, or None if error.
        """
        return self._make_request(f"/devices/{serial_number}", timeout=10)

    def get_device_data(
        self,
        serial_number: str,
        data_type: str = "final",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        period: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Retrieves device data.

        Args:
            serial_number: Serial number of the device.
            data_type: Type of data to retrieve ('raw', 'final', or 'resampled').
            start_date: Start date for data retrieval (YYYY-MM-DD).
            end_date: End date for data retrieval (YYYY-MM-DD).
            limit: Maximum number of records to retrieve.
            period: Resampling period for resampled data (15m, 1h, 8h, 1d).

        Returns:
            dict: JSON response from the API, or None if error.
        """
        # Get endpoint from config
        endpoints = self.config.get("api_config", {}).get("data_endpoints", {})

        if data_type == "raw":
            endpoint = endpoints.get("raw", f"/devices/{serial_number}/data/raw/")
        elif data_type == "final":
            endpoint = endpoints.get("final", f"/devices/{serial_number}/data/")
        elif data_type == "resampled":
            endpoint = endpoints.get("resampled", "/data/resampled/")
        else:
            raise ValueError(
                "Invalid data type. Choose from 'raw', 'final', or 'resampled'."
            )

        # Format endpoint with serial number
        endpoint = endpoint.format(sn=serial_number)

        params = {}

        if data_type == "resampled":
            if not start_date or not end_date:
                raise ValueError("Start and end dates are required for resampled data.")
            params["sn"] = serial_number
            params["start_date"] = start_date
            params["end_date"] = end_date
            params["period"] = period if period else "1d"
        else:
            if start_date:
                params["start"] = start_date
            if end_date:
                params["end"] = end_date

        if limit:
            params["limit"] = limit

        return self._make_request(endpoint, params)

    def get_device_data_by_date(
        self, serial_number: str, target_date: Union[str, date]
    ) -> Optional[dict]:
        """
        Retrieves device data for a specific date.

        Args:
            serial_number: Serial number of the device.
            target_date: Date for data retrieval (YYYY-MM-DD string or date object).

        Returns:
            dict: JSON response from the API, or None if error.
        """
        if isinstance(target_date, date):
            target_date = target_date.strftime("%Y-%m-%d")

        endpoint = f"/devices/{serial_number}/data-by-date/{target_date}/"
        return self._make_request(endpoint)

    def get_data_as_dataframe(
        self,
        serial_number: str,
        data_type: str = "final",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        period: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves device data and returns it as a pandas DataFrame.

        Args:
            serial_number: Serial number of the device.
            data_type: Type of data to retrieve.
            start_date: Start date for data retrieval (YYYY-MM-DD).
            end_date: End date for data retrieval (YYYY-MM-DD).
            limit: Maximum number of records.
            period: Resampling period for resampled data.

        Returns:
            pd.DataFrame: Data as DataFrame, or None if error.
        """
        data = self.get_device_data(
            serial_number, data_type, start_date, end_date, limit, period
        )

        if data and "data" in data:
            df = pd.DataFrame(data["data"])

            # Convert timestamp columns to datetime
            for col in ["timestamp", "timestamp_local"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            return df

        return None

    def save_data_to_file(
        self,
        serial_number: str,
        data_type: str = "final",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        period: Optional[str] = None,
        output_format: str = "csv",
    ) -> Optional[Path]:
        """
        Retrieves device data and saves it to a file in the configured data path.

        Args:
            serial_number: Serial number of the device.
            data_type: Type of data to retrieve.
            start_date: Start date for data retrieval (YYYY-MM-DD).
            end_date: End date for data retrieval (YYYY-MM-DD).
            limit: Maximum number of records.
            period: Resampling period for resampled data.
            output_format: Output file format ('csv' or 'xlsx').

        Returns:
            Path: Path to the saved file, or None if error.
        """
        df = self.get_data_as_dataframe(
            serial_number, data_type, start_date, end_date, limit, period
        )

        if df is None or df.empty:
            print("No data to save.")
            return None

        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_range = ""
        if start_date and end_date:
            date_range = f"_{start_date}_to_{end_date}"
        elif start_date:
            date_range = f"_from_{start_date}"
        elif end_date:
            date_range = f"_to_{end_date}"

        filename = (
            f"{serial_number}_{data_type}{date_range}_{timestamp}.{output_format}"
        )
        filepath = self.data_path / filename

        # Save file
        if output_format == "csv":
            df.to_csv(filepath, index=False)
        elif output_format == "xlsx":
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"Data saved to: {filepath}")
        return filepath

    def list_available_variables(self) -> Dict[str, List[str]]:
        """
        List available variables from the instrument configuration.

        Returns:
            dict: Dictionary of variable categories and their variables.
        """
        return self.config.get("variables", {})

    def get_specifications(self) -> dict:
        """
        Get instrument specifications from configuration.

        Returns:
            dict: Instrument specifications.
        """
        return self.config.get("specifications", {})


def interactive_main():
    """
    Interactive main function to interact with the QuantAQ API.
    """
    # Try to get API key
    api_key = os.getenv("QUANTAQ_API_KEY") or os.getenv("API_KEY")
    if api_key is None:
        print("Error: API key environment variable not found.")
        print("Please set QUANTAQ_API_KEY or API_KEY in your environment or .env file.")
        return

    try:
        quantaq_api = QuantAQAPI(api_key)
        print("[OK] Connected to QuantAQ API")
        print(f"[OK] Data will be saved to: {quantaq_api.data_path}")
    except Exception as e:
        print(f"Error initializing API client: {e}")
        return

    # Option to list devices first
    list_devices = (
        input("\nWould you like to list available devices first? (y/n): ")
        .strip()
        .lower()
    )
    if list_devices == "y":
        print("\nFetching devices...")
        devices = quantaq_api.get_devices()
        if devices and "data" in devices:
            print(f"\nFound {len(devices['data'])} device(s):")
            for device in devices["data"]:
                print(
                    f"  - Serial: {device.get('sn', 'N/A')}, "
                    f"Description: {device.get('description', 'N/A')}, "
                    f"Outdoors: {device.get('outdoors', 'N/A')}"
                )
        else:
            print("No devices found or error retrieving devices.")
            return

    serial_number = input("\nEnter the serial number of the device: ").strip()

    # Verify device exists
    print(f"\nVerifying device '{serial_number}'...")
    device_info = quantaq_api.get_device(serial_number)
    if device_info is None:
        print(f"\nError: Could not access device '{serial_number}'.")
        return

    print(f"Device verified: {device_info.get('name', serial_number)}")

    # Choose data type
    print("\nData retrieval options:")
    print("  1. final - Processed data (default)")
    print("  2. raw - Raw sensor data")
    print("  3. resampled - Data resampled at specific intervals")
    print("  4. by-date - Data for a specific date")
    data_type = input("Enter the data type (1-4) or name: ").strip()

    type_map = {"1": "final", "2": "raw", "3": "resampled", "4": "by-date"}
    data_type = type_map.get(data_type, data_type)

    # Get date parameters
    start_date = None
    end_date = None

    if data_type == "by-date":
        target_date = input("Enter the date (YYYY-MM-DD): ").strip()
        print(f"\nFetching data for {target_date}...")
        data = quantaq_api.get_device_data_by_date(serial_number, target_date)
    elif data_type == "resampled":
        start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

        print("\nResampling period options: 15m, 1h, 8h, 1d")
        period = input("Enter the period (default: 1d): ").strip() or "1d"

        print(f"\nFetching resampled data from {start_date} to {end_date}...")
        data = quantaq_api.get_device_data(
            serial_number, data_type, start_date, end_date, period=period
        )
    else:
        start_date = (
            input("Enter start date (YYYY-MM-DD) or leave blank: ").strip() or None
        )
        end_date = input("Enter end date (YYYY-MM-DD) or leave blank: ").strip() or None
        limit = input("Enter max records (or leave blank): ").strip()
        limit = int(limit) if limit else None

        print(f"\nFetching {data_type} data...")
        data = quantaq_api.get_device_data(
            serial_number, data_type, start_date, end_date, limit
        )

    if data:
        print("\n" + "=" * 50)
        print("API Response Summary:")
        print("=" * 50)

        if isinstance(data, dict) and "data" in data:
            record_count = (
                len(data["data"]) if isinstance(data["data"], list) else "N/A"
            )
            print(f"Total records retrieved: {record_count}")

            # Ask to save
            save = (
                input("\nWould you like to save the data to a file? (y/n): ")
                .strip()
                .lower()
            )
            if save == "y":
                fmt = (
                    input("Output format (csv/xlsx, default: csv): ").strip().lower()
                    or "csv"
                )
                quantaq_api.save_data_to_file(
                    serial_number,
                    data_type,
                    start_date if data_type != "by-date" else None,
                    end_date if data_type != "by-date" else None,
                    output_format=fmt,
                )
        else:
            print(json.dumps(data, indent=2))
    else:
        print("\nFailed to retrieve data. Check error messages above.")


if __name__ == "__main__":
    interactive_main()
