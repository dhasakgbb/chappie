# src/institutional_fetcher.py

"""
institutional_fetcher.py

This module fetches institutional holdings and insider trading activities for a given stock ticker.
It retrieves data on top institutional investors, percentage ownership, and recent changes in holdings.
"""

import os
import json
import yfinance as yf
from typing import Dict, Any
from logger import get_logger  # Import the centralized logger
from pandas import Timestamp  # Add this import

# Initialize logger for this module
logger = get_logger('institutional_fetcher', 'institutional_fetcher.log')

def fetch_and_save_institutional_data(ticker: str) -> None:
    """
    Fetches institutional holdings and insider trading data for the given ticker and saves them as institutional_holdings.json
    and insider_trading.json in the corresponding data/{TICKER}/ directory.
    :param ticker: Stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        major_holders = stock.major_holders
        if major_holders is not None:
            major_holders_list = major_holders.to_dict(orient='records')
        else:
            major_holders_list = []
            logger.warning(f"No major holders data found for ticker '{ticker}'.")

        institutional_holders = stock.institutional_holders
        if institutional_holders is not None:
            institutional_holders_list = institutional_holders.fillna('').to_dict(orient='records')
        else:
            institutional_holders_list = []
            logger.warning(f"No institutional holders data found for ticker '{ticker}'.")

        holdings_data = {
            'major_holders': major_holders_list,
            'institutional_holders': institutional_holders_list
        }

        try:
            insider_transactions = stock.insider_transactions
            if insider_transactions is not None:
                insider_transactions.reset_index(inplace=True)
                insider_transactions_list = insider_transactions.fillna('').to_dict(orient='records')
            else:
                insider_transactions_list = []
                logger.warning(f"No insider trading data found for ticker '{ticker}'.")
        except AttributeError:
            insider_transactions_list = []
            logger.warning(f"No insider trading data available for ticker '{ticker}'.")

        save_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', ticker.upper())
        os.makedirs(save_dir, exist_ok=True)

        def convert_timestamp(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            raise TypeError("Type not serializable")

        holdings_file_path = os.path.join(save_dir, f"{ticker}_institutional_holdings.json")
        with open(holdings_file_path, 'w') as json_file:
            json.dump(holdings_data, json_file, indent=4, default=convert_timestamp)

        insider_file_path = os.path.join(save_dir, f"{ticker}_insider_trading.json")
        with open(insider_file_path, 'w') as json_file:
            json.dump(insider_transactions_list, json_file, indent=4, default=convert_timestamp)

        # Validate the saved JSON files
        for path in [holdings_file_path, insider_file_path]:
            with open(path, 'r') as json_file:
                try:
                    json.load(json_file)
                    logger.info(f"Data saved to '{path}'.")
                except json.JSONDecodeError as e:
                    logger.error(f"Validation failed for '{path}': {e}")

    except Exception as e:
        logger.error(f"Error fetching institutional data for ticker '{ticker}': {e}")