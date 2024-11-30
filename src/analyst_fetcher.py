# src/analyst_fetcher.py

"""
analyst_fetcher.py

This module fetches analyst ratings and recommendations for a given stock ticker.
It retrieves information such as consensus ratings, price targets, and changes in analyst sentiment.
"""

import os
import json
import yfinance as yf
from typing import Dict, Any
from logger import get_logger  # Import the centralized logger

# Initialize logger for this module
logger = get_logger('analyst_fetcher', 'analyst_fetcher.log')

def fetch_and_save_analyst_data(ticker: str) -> None:
    """
    Fetches analyst ratings and recommendations for the given ticker and saves them as analyst_ratings.json
    in the corresponding data/{TICKER}/ directory.

    :param ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        # Retrieve analyst recommendations
        rec_data = stock.recommendations

        if rec_data is None or rec_data.empty:
            logger.warning(f"No analyst recommendation data found for ticker '{ticker}'.")
            return

        # Convert the DataFrame to a dictionary
        rec_data.reset_index(inplace=True)
        rec_list = rec_data.to_dict(orient='records')

        # Define the save directory
        save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        # Define the file path
        file_path = os.path.join(save_dir, f"{ticker}_analyst_ratings.json")

        # Save analyst data to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(rec_list, json_file, indent=4, default=str)

        # Validate the saved JSON file
        with open(file_path, 'r') as json_file:
            try:
                json.load(json_file)
                logger.info(f"Analyst ratings data saved to '{file_path}'.")
            except json.JSONDecodeError as e:
                logger.error(f"Validation failed for '{file_path}': {e}")

    except Exception as e:
        logger.error(f"Error fetching analyst data for ticker '{ticker}': {e}")