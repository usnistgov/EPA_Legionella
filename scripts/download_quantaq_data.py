#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantAQ Data Downloader
=======================

This script downloads raw and final data from QuantAQ MODULAIR-PM particulate
matter sensors deployed at the EPA Legionella test site. It handles pagination
and rate limiting to retrieve complete datasets from the QuantAQ cloud API.

Key Metrics Downloaded:
    - PM1, PM2.5, PM10 concentrations (raw and calibrated)
    - Meteorological data (temperature, RH from integrated sensors)
    - Optical particle counts by size bin
    - GPS and operating state information

Analysis Features:
    - Weekly chunked downloads (7-day windows) for reliability
    - Incremental downloads: skips already-downloaded chunks
    - Retry logic with exponential backoff (3 retries on timeout/error)
    - Automatic pagination for large datasets (500 records/page)
    - Rate limiting (0.5s delay) to respect API quotas
    - Date range filtering from experiment start to current date
    - Separate files for raw vs. calibrated (final) data

Methodology:
    1. Load API configuration from data_config.json
    2. Authenticate with QuantAQ API using environment variable key
    3. Break date range into 7-day chunks
    4. For each device (inside/outside), download raw and final data
    5. Skip chunks already on disk; re-download current (partial) week
    6. Handle pagination with retry logic per chunk
    7. Save each chunk to chunks/ subdirectory

Output Files (chunks/ subdirectory):
    - quantaq-outside-raw-{start}-{end}.csv: Weekly raw outside chunks
    - quantaq-outside-final-{start}-{end}.csv: Weekly calibrated outside chunks
    - quantaq-inside-raw-{start}-{end}.csv: Weekly raw inside chunks
    - quantaq-inside-final-{start}-{end}.csv: Weekly calibrated inside chunks

Configuration:
    - Device serial numbers: MOD-PM-00785 (outside), MOD-PM-00195 (inside)
    - Data start date: January 5, 2026
    - Requires QUANTAQ_API_KEY environment variable

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
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

# Chunk size for downloads (days per chunk)
CHUNK_DAYS = 7


def generate_week_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int = CHUNK_DAYS,
) -> list:
    """
    Break a date range into week-sized chunks.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        end_date: End date as YYYY-MM-DD string.
        chunk_days: Number of days per chunk (default 7).

    Returns:
        List of (chunk_start, chunk_end) tuples as YYYY-MM-DD strings.
        The final chunk may be shorter than chunk_days.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    chunks = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        chunk_start = chunk_end + timedelta(days=1)

    return chunks


def get_chunk_filename(
    device_name: str,
    data_type: str,
    chunk_start: str,
    chunk_end: str,
) -> str:
    """
    Generate the filename for a chunk file.

    Args:
        device_name: Device name (e.g., 'quantaq-outside').
        data_type: Data type ('raw' or 'final').
        chunk_start: Chunk start date as YYYY-MM-DD.
        chunk_end: Chunk end date as YYYY-MM-DD.

    Returns:
        Filename string like 'quantaq-outside-raw-20260105-20260111.csv'.
    """
    start_str = chunk_start.replace("-", "")
    end_str = chunk_end.replace("-", "")
    return f"{device_name}-{data_type}-{start_str}-{end_str}.csv"


def find_existing_chunks(
    chunks_dir: Path,
    device_name: str,
    data_type: str,
) -> set:
    """
    Scan the chunks directory to find which date ranges already have files.

    Args:
        chunks_dir: Path to the chunks/ subdirectory.
        device_name: Device name (e.g., 'quantaq-outside').
        data_type: Data type ('raw' or 'final').

    Returns:
        Set of (start_date, end_date) tuples in YYYY-MM-DD format
        for chunks that already exist on disk.
    """
    existing = set()

    if not chunks_dir.exists():
        return existing

    pattern = f"{device_name}-{data_type}-*.csv"

    for filepath in chunks_dir.glob(pattern):
        stem = filepath.stem
        parts = stem.split("-")
        # e.g. "quantaq-outside-raw-20260105-20260111" -> 5 parts
        if len(parts) >= 5:
            start_str = parts[-2]
            end_str = parts[-1]
            try:
                s = date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:8]))
                e = date(int(end_str[:4]), int(end_str[4:6]), int(end_str[6:8]))
                existing.add((
                    s.strftime("%Y-%m-%d"),
                    e.strftime("%Y-%m-%d"),
                ))
            except ValueError:
                continue

    return existing


def remove_stale_partial_chunks(
    chunks_dir: Path,
    device_name: str,
    data_type: str,
    chunk_start: str,
    chunk_end: str,
) -> list:
    """
    Remove stale chunk files that share the same start date but have an
    older (different) end date.

    When the current (partial) week chunk is re-downloaded each day, the
    end date advances. This removes yesterday's superseded file.

    Args:
        chunks_dir: Path to the chunks/ subdirectory.
        device_name: Device name (e.g., 'quantaq-outside').
        data_type: Data type ('raw' or 'final').
        chunk_start: Start date of the new chunk (YYYY-MM-DD).
        chunk_end: End date of the new chunk (YYYY-MM-DD).

    Returns:
        List of Path objects that were removed.
    """
    if not chunks_dir.exists():
        return []

    start_compact = chunk_start.replace("-", "")
    end_compact = chunk_end.replace("-", "")

    # Match files with same device, type, and start date but any end date
    pattern = f"{device_name}-{data_type}-{start_compact}-*.csv"
    removed = []

    for filepath in chunks_dir.glob(pattern):
        # Extract the end date from the filename
        stem = filepath.stem
        parts = stem.split("-")
        if len(parts) >= 5:
            file_end = parts[-1]
            # Remove if end date differs from current chunk's end date
            if file_end != end_compact:
                filepath.unlink()
                print(f"    Removed stale chunk: {filepath.name}")
                removed.append(filepath)

    return removed


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
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> Optional[dict]:
        """
        Make an authenticated request to the QuantAQ API with retry logic.

        On failure (timeout, connection error, 5xx server error, or 429 rate
        limit), retries up to max_retries times with exponential backoff.
        Does NOT retry on 4xx client errors (except 429).

        Args:
            endpoint: API endpoint.
            params: Query parameters.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds (doubles each retry).

        Returns:
            dict: JSON response, or None if all attempts fail.
        """
        url = f"{self.base_url}{endpoint}"
        auth = (self.api_key, "")

        for attempt in range(max_retries + 1):
            response = None
            try:
                response = requests.get(
                    url, auth=auth, params=params, timeout=timeout
                )

                # Handle rate limiting (429) with retry
                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Rate limited (429). Retrying in {delay:.0f}s "
                              f"(attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"  Rate limited (429). All {max_retries} retries exhausted.")
                        return None

                # Do not retry 4xx client errors (auth, not found, etc.)
                if 400 <= response.status_code < 500:
                    print(f"  HTTP Error {response.status_code}: {response.reason}")
                    return None

                # Retry 5xx server errors
                if response.status_code >= 500:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Server error ({response.status_code}). "
                              f"Retrying in {delay:.0f}s "
                              f"(attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"  Server error ({response.status_code}). "
                              f"All {max_retries} retries exhausted.")
                        return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Timeout. Retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  Timeout. All {max_retries} retries exhausted.")
                    return None

            except requests.exceptions.ConnectionError:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Connection error. Retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  Connection error. All {max_retries} retries exhausted.")
                    return None

            except requests.exceptions.RequestException as e:
                print(f"  Request Error: {e}")
                return None

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
                "filter": f"timestamp_local,ge,{start_date};timestamp_local,le,{end_date}",
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

    def save_chunk(
        self,
        df: pd.DataFrame,
        device_name: str,
        data_type: str,
        chunk_start: str,
        chunk_end: str,
    ) -> Path:
        """
        Save a DataFrame as a chunk file in the chunks/ subdirectory.

        Args:
            df: Data to save.
            device_name: Device name (e.g., 'quantaq-outside').
            data_type: 'raw' or 'final'.
            chunk_start: Chunk start date (YYYY-MM-DD).
            chunk_end: Chunk end date (YYYY-MM-DD).

        Returns:
            Path: Path to saved chunk file.
        """
        chunks_dir = self.data_path / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        filename = get_chunk_filename(device_name, data_type, chunk_start, chunk_end)
        filepath = chunks_dir / filename

        df.to_csv(filepath, index=False)
        print(f"    Saved chunk: {filepath.name}")

        return filepath

    def download_chunked(
        self,
        serial_number: str,
        device_name: str,
        data_type: str,
        start_date: str,
        end_date: str,
    ) -> list:
        """
        Download data in weekly chunks with incremental support.

        Scans for existing chunk files and skips completed chunks.
        Always re-downloads the chunk containing today (partial week).
        Saves each chunk as a separate CSV in chunks/ subdirectory.

        Args:
            serial_number: Device serial number.
            device_name: Device name (e.g., 'quantaq-outside').
            data_type: 'raw' or 'final'.
            start_date: Overall start date (YYYY-MM-DD).
            end_date: Overall end date (YYYY-MM-DD).

        Returns:
            List of Paths to all chunk files (both existing and newly downloaded).
        """
        chunks_dir = self.data_path / "chunks"
        today = date.today()

        # Generate all chunks for the full date range
        all_chunks = generate_week_chunks(start_date, end_date)
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Total chunks: {len(all_chunks)}")

        # Find existing chunk files
        existing = find_existing_chunks(chunks_dir, device_name, data_type)
        print(f"  Existing chunks on disk: {len(existing)}")

        saved_paths = []

        for i, (chunk_start, chunk_end) in enumerate(all_chunks, 1):
            chunk_end_date = date.fromisoformat(chunk_end)
            is_current_chunk = chunk_end_date >= today

            filename = get_chunk_filename(
                device_name, data_type, chunk_start, chunk_end
            )
            filepath = chunks_dir / filename

            # Skip if chunk exists AND chunk is fully in the past
            if (chunk_start, chunk_end) in existing and not is_current_chunk:
                print(f"  Chunk {i}/{len(all_chunks)} [{chunk_start} to {chunk_end}]: "
                      f"SKIP (already downloaded)")
                saved_paths.append(filepath)
                continue

            # Download this chunk
            action = "RE-DOWNLOAD (current week)" if is_current_chunk else "DOWNLOAD"
            print(f"  Chunk {i}/{len(all_chunks)} [{chunk_start} to {chunk_end}]: "
                  f"{action}")

            df = self.download_all_data(
                serial_number=serial_number,
                data_type=data_type,
                start_date=chunk_start,
                end_date=chunk_end,
            )

            if df is not None and not df.empty:
                path = self.save_chunk(
                    df, device_name, data_type, chunk_start, chunk_end
                )
                saved_paths.append(path)

                # Remove stale partial chunks with same start but older end date
                if is_current_chunk:
                    remove_stale_partial_chunks(
                        chunks_dir, device_name, data_type,
                        chunk_start, chunk_end,
                    )
            else:
                print(f"    No data for this chunk.")

        return saved_paths


def main():
    """
    Main function to download all QuantAQ sensor data in weekly chunks.
    """
    print("=" * 60)
    print("QuantAQ Data Downloader (Chunked)")
    print("=" * 60)
    print()

    # Initialize downloader
    try:
        downloader = QuantAQDownloader()
        print(f"Data path: {downloader.data_path}")
        print(f"Chunks path: {downloader.data_path / 'chunks'}")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get end date (today)
    end_date = date.today().strftime("%Y-%m-%d")
    print(f"Date range: {START_DATE} to {end_date}")
    print(f"Chunk size: {CHUNK_DAYS} days")
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

            chunk_paths = downloader.download_chunked(
                serial_number=serial_number,
                device_name=device_name,
                data_type=data_type,
                start_date=START_DATE,
                end_date=end_date,
            )

            results.append(
                {
                    "device": device_name,
                    "serial": serial_number,
                    "type": data_type,
                    "chunks": len(chunk_paths),
                }
            )

        print()

    # Summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['device']} ({r['type']}): {r['chunks']} chunks")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
