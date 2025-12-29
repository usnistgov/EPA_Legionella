"""
Utility script to access QuantAQ API and download instrument data.
"""

import os
from datetime import datetime, timedelta
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
        response = requests.get(url, auth=auth, params=params, timeout=10)
        return response.json()

    def get_device_data(self, serial_number, data_type="final", start_date=None, end_date=None):
        """
        Retrieves device data.

        Args:
            serial_number (str): Serial number of the device.
            data_type (str, optional): Type of data to retrieve. Defaults to "final".
            start_date (str, optional): Start date for data retrieval. Defaults to None.
            end_date (str, optional): End date for data retrieval. Defaults to None.

        Returns:
            dict: JSON response from the API.
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
            params["period"] = "1d"  # Default period, can be changed
        else:
            if start_date and end_date:
                params["filter"] = f"timestamp,ge,{start_date};timestamp,le,{end_date}"

        response = requests.get(url, auth=auth, params=params, timeout=10)
        return response.json()

def main():
    """
    Main function to interact with the QuantAQ API.
    """
    api_key = os.getenv("API_KEY")
    if api_key is None:
        print("Error: API_KEY environment variable not found.")
        return
    quantaq_api = QuantAQAPI(api_key)

    serial_number = input("Enter the serial number of the device: ")
    data_type = input("Enter the data type (raw, final, or resampled): ")

    if data_type == "resampled":
        start_date = input("Enter the start date (YYYY-MM-DD): ")
        end_date = input("Enter the end date (YYYY-MM-DD): ")
        data = quantaq_api.get_device_data(serial_number, data_type, start_date, end_date)
    else:
        start_date = input("Enter the start date (YYYY-MM-DD) or leave blank for all data: ")
        end_date = input("Enter the end date (YYYY-MM-DD) or leave blank for all data: ")
        start_date = start_date if start_date else None
        end_date = end_date if end_date else None
        data = quantaq_api.get_device_data(serial_number, data_type, start_date, end_date)

    print(data)

if __name__ == "__main__":
    main()