#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download QuantAQ MODULAIR-PM data for indoor/outdoor sensors.

Downloads raw and final data for:
- MOD-PM-00785 (outside sensor) -> {date}-quantaq-outside-{type}.csv
- MOD-PM-00195 (inside sensor) -> {date}-quantaq-inside-{type}.csv

Data is downloaded from January 5, 2026 to current date.
Files are saved to the QuantAQ data path specified in data_config.json.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.quantaq_utils import get_quantaq_config, get_quantaq_data_path

load_dotenv()


# Device configuration
DEVICES = {
    "MOD-PM-00785": "quantaq-outside",
    "MOD-PM-00195": "quantaq-inside",
}

# Data start date (January 5, 2026)
START_DATE = "2026-01-05"

# Data types to download
DATA_TYPES = ["raw", "final"]


class QuantAQDownloader:
    """
    Class to download data from QuantAQ API with pagination support.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            api_key: API key for authentication. If None, loads from environment.
        """
        self.config = get_quantaq_config()

        if api_key is None:
            env_var = self.config.get("api_config", {}).get(
                "env_var_api_key", "QUANTAQ_API_KEY"
            )
            api_key = os.getenv(env_var) or os.getenv("API_KEY")

        if api_key is None:
            raise ValueError(
                "API key not found. Set QUANTAQ_API_KEY or API_KEY in environment."
            )

        self.api_key = api_key
        self.base_url = self.config.get("api_base_url", "https://api.quant-aq.com/v1")
        self.data_path = get_quantaq_data_path()

        # Rate limiting: 150 requests/minute
        self.request_delay = 0.5  # seconds between requests

    def _make_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        timeout: int = 60,
    ) -> Optional[dict]:
        """
        Make an authenticated request to the QuantAQ API.

        Args:
            endpoint: API endpoint.
            params: Query parameters.
            timeout: Request timeout in seconds.

        Returns:
            dict: JSON response, or None if error.
        """
        url = f"{self.base_url}{endpoint}"
        auth = (self.api_key, "")
        response = None

        try:
            response = requests.get(url, auth=auth, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP Error: {e}")
            if response is not None:
                print(f"  Status: {response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  Request Error: {e}")
            return None

    def download_all_data(
        self,
        serial_number: str,
        data_type: str,
        start_date: str,
        end_date: Optional[str] = None,
        per_page: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Download all data for a device with pagination.

        Args:
            serial_number: Device serial number.
            data_type: 'raw' or 'final'.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD). If None, uses today.
            per_page: Records per page (max usually 500).

        Returns:
            pd.DataFrame: All data combined, sorted by timestamp_local.
        """
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")

        # Build endpoint
        if data_type == "raw":
            endpoint = f"/devices/{serial_number}/data/raw/"
        else:
            endpoint = f"/devices/{serial_number}/data/"

        all_data = []
        page = 1
        total_pages = None

        print(f"  Downloading {data_type} data from {start_date} to {end_date}...")

        while True:
            params = {
                "page": page,
                "per_page": per_page,
                "filter": f"timestamp_local,ge,{start_date}",
                "sort": "timestamp_local,asc",
            }

            result = self._make_request(endpoint, params)

            if result is None:
                print(f"  Error fetching page {page}")
                break

            # Get pagination info from meta
            meta = result.get("meta", {})
            if total_pages is None:
                total_pages = meta.get("pages", 1)
                total_records = meta.get("total", 0)
                print(f"  Total records: {total_records}, Pages: {total_pages}")

            # Get data from response
            data = result.get("data", [])
            if not data:
                break

            # Filter data to be within date range
            for record in data:
                ts = record.get("timestamp_local", "")
                if ts:
                    record_date = ts[:10]  # Extract YYYY-MM-DD
                    if record_date >= start_date and record_date <= end_date:
                        all_data.append(record)

            print(f"  Page {page}/{total_pages} - fetched {len(data)} records")

            # Check if we've reached the end
            if page >= total_pages:
                break

            # Check for next page URL
            next_url = meta.get("next_url")
            if not next_url:
                break

            page += 1
            time.sleep(self.request_delay)  # Rate limiting

        if not all_data:
            print("  No data found in date range.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Convert timestamp columns to datetime
        for col in ["timestamp", "timestamp_local"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Sort by timestamp_local (oldest to newest)
        if "timestamp_local" in df.columns:
            df = df.sort_values("timestamp_local", ascending=True)
            df = df.reset_index(drop=True)

        print(f"  Total records downloaded: {len(df)}")
        return df

    def save_data(
        self,
        df: pd.DataFrame,
        device_name: str,
        data_type: str,
    ) -> Path:
        """
        Save DataFrame to CSV file with proper naming.

        Args:
            df: Data to save.
            device_name: Name for the file (e.g., 'quantaq-outside').
            data_type: 'raw' or 'final'.

        Returns:
            Path: Path to saved file.
        """
        # Ensure directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with today's date
        today_str = datetime.now().strftime("%Y%m%d")
        filename = f"{today_str}-{device_name}-{data_type}.csv"
        filepath = self.data_path / filename

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")

        return filepath


def main():
    """
    Main function to download all QuantAQ sensor data.
    """
    print("=" * 60)
    print("QuantAQ Data Downloader")
    print("=" * 60)
    print()

    # Initialize downloader
    try:
        downloader = QuantAQDownloader()
        print(f"Data path: {downloader.data_path}")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get end date (today)
    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Date range: {START_DATE} to {end_date}")
    print()

    # Track results
    results = []

    # Download data for each device
    for serial_number, device_name in DEVICES.items():
        print("-" * 60)
        print(f"Device: {serial_number} ({device_name})")
        print("-" * 60)

        for data_type in DATA_TYPES:
            print(f"\n[{data_type.upper()}]")

            # Download data
            df = downloader.download_all_data(
                serial_number=serial_number,
                data_type=data_type,
                start_date=START_DATE,
                end_date=end_date,
            )

            if df is not None and not df.empty:
                # Save data
                filepath = downloader.save_data(df, device_name, data_type)
                results.append(
                    {
                        "device": device_name,
                        "serial": serial_number,
                        "type": data_type,
                        "records": len(df),
                        "file": filepath.name,
                    }
                )
            else:
                results.append(
                    {
                        "device": device_name,
                        "serial": serial_number,
                        "type": data_type,
                        "records": 0,
                        "file": None,
                    }
                )

        print()

    # Summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for r in results:
        status = f"{r['records']} records -> {r['file']}" if r["file"] else "No data"
        print(f"  {r['device']} ({r['type']}): {status}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
