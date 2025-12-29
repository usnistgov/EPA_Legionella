"""
Utility script to access QuantAQ API and download instrument data.

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
"""

import os
from datetime import datetime, timedelta
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class QuantAQAPI:
    """
    Class to interact with the QuantAQ API.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.quant-aq.com/v1"

    def get_devices(self, org_id=None, network_id=None):
        """
        Retrieves a list of devices.

        Args:
            org_id (int, optional): Organization ID. Defaults to None.
            network_id (int, optional): Network ID. Defaults to None.

        Returns:
            dict: JSON response from the API.
        """
        url = f"{self.base_url}/devices"
        auth = (self.api_key, "")
        params = {}
        if org_id:
            params["org_id"] = org_id
        if network_id:
            params["network_id"] = network_id

        response = None
        try:
            response = requests.get(url, auth=auth, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response is not None:
                print(f"Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

    def get_device(self, serial_number):
        """
        Retrieves information about a specific device.

        Args:
            serial_number (str): Serial number of the device.

        Returns:
            dict: JSON response from the API, or None if error.
        """
        url = f"{self.base_url}/devices/{serial_number}"
        auth = (self.api_key, "")

        response = None
        try:
            response = requests.get(url, auth=auth, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError:
            print(f"HTTP Error: Device '{serial_number}' not found or access denied.")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

    def get_device_data(
        self, serial_number, data_type="final", start_date=None,
        end_date=None, limit=None, period=None
    ):
        """
        Retrieves device data.

        Args:
            serial_number (str): Serial number of the device.
            data_type (str, optional): Type of data to retrieve. Defaults to "final".
            start_date (str, optional): Start date for data retrieval
                (YYYY-MM-DD). Defaults to None.
            end_date (str, optional): End date for data retrieval (YYYY-MM-DD). Defaults to None.
            limit (int, optional): Maximum number of records to retrieve. Defaults to None.
            period (str, optional): Resampling period for resampled data.
                Options: 15m, 1h, 8h, 1d. Defaults to "1d".

        Returns:
            dict: JSON response from the API, or None if error.
        """
        if data_type == "raw":
            url = f"{self.base_url}/devices/{serial_number}/data/raw/"
        elif data_type == "final":
            url = f"{self.base_url}/devices/{serial_number}/data/"
        elif data_type == "resampled":
            url = f"{self.base_url}/data/resampled/"
        else:
            raise ValueError("Invalid data type. Choose from 'raw', 'final', or 'resampled'.")

        auth = (self.api_key, "")
        params = {}

        if data_type == "resampled":
            if not start_date or not end_date:
                raise ValueError("Start and end dates are required for resampled data.")
            params["sn"] = serial_number
            params["start_date"] = start_date
            params["end_date"] = end_date
            params["period"] = period if period else "1d"  # Default to 1d if not specified
        else:
            # For raw and final data, use query parameters instead of custom filter
            if start_date:
                params["start"] = start_date
            if end_date:
                params["end"] = end_date

        if limit:
            params["limit"] = limit

        response = None
        try:
            response = requests.get(url, auth=auth, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error retrieving data: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

    def get_device_data_by_date(self, serial_number, date):
        """
        Retrieves device data for a specific date using the data-by-date endpoint.

        Args:
            serial_number (str): Serial number of the device.
            date (str): Date for data retrieval (YYYY-MM-DD format).

        Returns:
            dict: JSON response from the API, or None if error.
        """
        url = f"{self.base_url}/devices/{serial_number}/data-by-date/{date}/"
        auth = (self.api_key, "")

        response = None
        try:
            response = requests.get(url, auth=auth, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error retrieving data for date {date}: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

def main():
    """
    Main function to interact with the QuantAQ API.
    """
    api_key = os.getenv("API_KEY")
    if api_key is None:
        print("Error: API_KEY environment variable not found.")
        print("Please create a .env file with your API key: API_KEY=your_key_here")
        return

    quantaq_api = QuantAQAPI(api_key)

    # Option to list devices first
    list_devices = input("Would you like to list available devices first? (y/n): ").strip().lower()
    if list_devices == 'y':
        print("\nFetching devices...")
        devices = quantaq_api.get_devices()
        if devices and 'data' in devices:
            print(f"\nFound {len(devices['data'])} device(s):")
            for device in devices['data']:
                print(f"  - Serial: {device.get('sn', 'N/A')}, Description: {device.get('description', 'N/A')}, Located outdoor: {device.get('outdoors', 'N/A')}")
        else:
            print("No devices found or error retrieving devices.")
            return

    serial_number = input("\nEnter the serial number of the device: ").strip()

    # Verify device exists before retrieving data
    print(f"\nVerifying device '{serial_number}'...")
    device_info = quantaq_api.get_device(serial_number)
    if device_info is None:
        print(f"\nError: Could not access device '{serial_number}'.")
        print("Please check:")
        print("  1. The serial number is correct")
        print("  2. Your API key has permission to access this device")
        print("  3. The device exists in your organization")
        return

    print(f"Device verified: {device_info.get('name', serial_number)}")

    # Choose endpoint type
    print("\nData retrieval options:")
    print("  1. final - Processed data (default)")
    print("  2. raw - Raw sensor data")
    print("  3. resampled - Data resampled at specific intervals")
    print("  4. by-date - Data for a specific date")
    data_type = input("Enter the data type (1-4) or name: ").strip()

    # Map numeric choices to names
    type_map = {"1": "final", "2": "raw", "3": "resampled", "4": "by-date"}
    data_type = type_map.get(data_type, data_type)

    if data_type == "by-date":
        date = input("Enter the date (YYYY-MM-DD): ").strip()
        print(f"\nFetching data for {date}...")
        data = quantaq_api.get_device_data_by_date(serial_number, date)
    elif data_type == "resampled":
        start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

        print("\nResampling period options:")
        print("  1. 15m - 15 minutes")
        print("  2. 1h - 1 hour")
        print("  3. 8h - 8 hours")
        print("  4. 1d - 1 day (default)")
        period_choice = input("Enter the period (1-4) or specify (15m/1h/8h/1d): ").strip()

        # Map numeric choices to period values
        period_map = {"1": "15m", "2": "1h", "3": "8h", "4": "1d"}
        period = period_map.get(period_choice, period_choice) if period_choice else "1d"

        print(f"\nFetching resampled data from {start_date} to {end_date} with {period} intervals...")
        data = quantaq_api.get_device_data(serial_number, data_type, start_date, end_date, period=period)
    else:
        start_date = input(
            "Enter the start date (YYYY-MM-DD) or leave blank for recent data: "
        ).strip()
        end_date = input("Enter the end date (YYYY-MM-DD) or leave blank for recent data: ").strip()
        limit = input("Enter maximum number of records (or leave blank for default): ").strip()

        start_date = start_date if start_date else None
        end_date = end_date if end_date else None
        limit = int(limit) if limit else None

        print(f"\nFetching {data_type} data...")
        data = quantaq_api.get_device_data(serial_number, data_type, start_date, end_date, limit)

    if data:
        print("\n" + "="*50)
        print("API Response:")
        print("="*50)
        print(json.dumps(data, indent=2))

        # Show data summary if it's a list
        if isinstance(data, dict) and 'data' in data:
            record_count = len(data['data']) if isinstance(data['data'], list) else 'N/A'
            print(f"\nTotal records retrieved: {record_count}")
    else:
        print("\nFailed to retrieve data. Check error messages above.")

if __name__ == "__main__":
    main()
