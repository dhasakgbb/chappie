# src/fundamentals_fetcher.py

"""
fundamentals_fetcher.py

This module provides functionality to fetch and save fundamental data for a given stock ticker.
It uses yfinance to retrieve fundamental information and saves it as a JSON file in the
designated ticker directory. Robust error handling ensures that missing or invalid attributes
are handled gracefully.
"""

import os
import json
from typing import Any, Dict
import yfinance as yf

from logger import get_logger  # Import the centralized logger

# Initialize logger for this module
logger = get_logger('fundamentals_fetcher')


def fetch_and_save_fundamentals(ticker: str) -> None:
    """
    Fetches fundamental data for the given ticker using yfinance and saves it as fundamentals.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        # Retrieve fundamental data from the 'info' attribute
        info = stock.info

        if not info:
            logger.warning(f"No fundamental data found for ticker '{ticker}'.")
            return

        # Define the fields to extract
        fundamental_fields = [
            'marketCap',
            'enterpriseValue',
            'trailingPE',
            'forwardPE',
            'pegRatio',
            'priceToBook',
            'beta',
            'profitMargins',
            'enterpriseToRevenue',
            'enterpriseToEbitda',
            '52WeekChange',
            'shortPercentOfFloat',
            'shortRatio',
            'bookValue',
            'priceToSalesTrailing12Months',
            'dividendYield',
            'fiveYearAvgDividendYield',
            'payoutRatio',
            'trailingEps',
            'forwardEps',
            'revenuePerShare',
            'grossProfits',
            'freeCashflow',
            'operatingCashflow',
            'revenueGrowth',
            'grossMargins',
            'ebitdaMargins',
            'operatingMargins',
            'financialCurrency',
            'dividendRate',
            'dividendYield',
            'payoutRatio',
            'fiveYearAvgDividendYield',
            'dividendDate',
            'exDividendDate'
        ]

        # Extract the fundamental data, handling missing attributes
        fundamentals = {}
        for field in fundamental_fields:
            value = info.get(field)
            if value is None:
                logger.debug(f"Field '{field}' is missing for ticker '{ticker}'.")
            fundamentals[field] = value

        # Get the absolute path to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Define the save directory
        save_dir = os.path.join(project_root, 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        # Define the file path
        file_path = os.path.join(save_dir, f"{ticker}_fundamentals.json")

        # Save the fundamentals to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(fundamentals, json_file, indent=4)

        # Validate the saved JSON file
        with open(file_path, 'r') as json_file:
            try:
                json.load(json_file)
                logger.info(f"Fundamental data saved to '{file_path}'.")
            except json.JSONDecodeError as e:
                logger.error(f"Validation failed for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Unexpected error in fetch_and_save_fundamentals for '{ticker}': {e}")