# src/macro_fetcher.py

"""
macro_fetcher.py

This module fetches macroeconomic indicators relevant to the stock's sector.
"""

import os
import json
import yfinance as yf
from fredapi import Fred  # Ensure fredapi is installed
from logger import get_logger  # Import the centralized logger

# Initialize logger for this module
logger = get_logger('macro_fetcher', 'macro_fetcher.log')

# Replace with your actual FRED API key
FRED_API_KEY = 'YOUR_FRED_API_KEY'


def fetch_and_save_macro_data(ticker: str, sector: str) -> None:
    """
    Fetches macroeconomic indicators relevant to the stock's sector and saves them as macro_indicators.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    :param sector: Sector of the stock.
    """
    try:
        fred = Fred(api_key=FRED_API_KEY)
        macro_data = {}

        if sector == 'Financials':
            interest_rates = fred.get_series('DGS10')
            macro_data['interest_rates'] = interest_rates.tail(30).to_dict()

        gdp = fred.get_series('GDP')
        macro_data['GDP'] = gdp.tail(10).to_dict()

        save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        macro_file_path = os.path.join(save_dir, f"{ticker}_macro_indicators.json")
        with open(macro_file_path, 'w') as json_file:
            json.dump(macro_data, json_file, indent=4, default=str)

        # Validate the saved JSON file
        with open(macro_file_path, 'r') as json_file:
            try:
                json.load(json_file)
                logger.info(f"Macroeconomic data saved to '{macro_file_path}'.")
            except json.JSONDecodeError as e:
                logger.error(f"Validation failed for '{macro_file_path}': {e}")

    except Exception as e:
        logger.error(f"Error fetching macroeconomic data for ticker '{ticker}': {e}")